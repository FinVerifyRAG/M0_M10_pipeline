"""
signals/s_mech.py  —  Mechanistic (Attention) Signal
------------------------------------------------------
s_mech implements a "lookback ratio" style mechanistic signal inspired by
the ReDeEP / lookback-ratio line of work.

Intuition
---------
When generating the atom's tokens, does the model attend MORE to the
evidence (prompt tokens) or to previously generated tokens (context)?

    lookback_ratio = sum(attn to prompt tokens) / sum(attn to all tokens)

    s_mech = 1 - lookback_ratio
           = fraction of attention going to generated tokens, not evidence.

High s_mech → model relies on its OWN prior (not evidence) → higher risk.
Low  s_mech → model heavily attends to evidence          → lower risk.

Implementation
--------------
Requires running Qwen with `output_attentions=True`. This is expensive,
so results are CACHED to disk in `.signal_cache/s_mech/<answer_id>.json`.

If attentions are not available, s_mech is missing (NaN) and
`s_mech_skipped` is 1. The missing value is not filled with 0.5.
The model id comes from configs/experiment.yaml, never from the API
model name stored on the answer. A config-named fallback model is tried
once. Strict mode raises if both fail to load.

The attention is averaged across all heads and all decoder layers at the
atom's token span positions.

Cache
-----
Cache key: hash(answer.answer_text + query)[:16]
Cache path: .signal_cache/s_mech/<key>.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from common.errors import StrictFailure
from common.schemas import VerifiedAtom, GeneratedAnswer, RetrievalResult

logger = logging.getLogger("signals.s_mech")

_CACHE_DIR = Path(".signal_cache/s_mech")
_CHARS_PER_TOKEN = 4
_LOAD_ERROR: Optional[str] = None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(answer_text: str, query: str) -> str:
    raw = (answer_text + query).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_load(key: str) -> Optional[dict]:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _cache_save(key: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Token span helper (shared with s_ent)
# ---------------------------------------------------------------------------

def _token_span(char_span: Optional[Tuple[int, int]], answer_text: str) -> Tuple[int, int]:
    if char_span is None:
        return (0, max(1, len(answer_text) // _CHARS_PER_TOKEN))
    start = max(0, char_span[0] // _CHARS_PER_TOKEN)
    end   = max(start + 1, char_span[1] // _CHARS_PER_TOKEN)
    return (start, end)


# ---------------------------------------------------------------------------
# Attention extraction (heavy — requires transformers + GPU)
# ---------------------------------------------------------------------------

def _mech_model_ids() -> Tuple[str, Optional[str], bool]:
    """Primary and explicit fallback ids from experiment config."""
    try:
        from common.experiment import load_experiment
        cfg = (load_experiment().get("s_mech") or {})
    except Exception as exc:
        logger.warning("s_mech: could not read experiment config: %s", exc)
        cfg = {}
    primary = cfg.get("model_id") or "Qwen/Qwen2.5-0.5B-Instruct"
    fallback = cfg.get("fallback_model_id")
    return primary, fallback, bool(cfg.get("enabled", False))


def _load_causal_lm(model_id: str):
    """
    Load a local causal LM for attention.

    `device_map='auto'` needs the `accelerate` package. The previous run
    passed the Groq API id `qwen/qwen3.8-27b`, which is not a local
    Hugging Face repo, and failed that check 20 times. On a device_map /
    accelerate error we retry the same id on CPU without device_map.
    """
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            output_attentions=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as exc:
        message = str(exc)
        device_map_failed = "device_map" in message or "accelerate" in message.lower()
        if not device_map_failed:
            raise
        logger.warning(
            "s_mech: device_map failed for %s (%s). Retrying on CPU without device_map.",
            model_id, exc,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            output_attentions=True,
            torch_dtype=torch.float32,
        )
    model.eval()
    return tokenizer, model


def _compute_lookback_from_model(
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    atom_tok_start: int,
    atom_tok_end: int,
) -> Optional[float]:
    """
    Run the configured local model with output_attentions=True.
    Returns None if every configured model fails to load.
    Does not read answer.model_id (that field is the API generator, not a local checkpoint).
    """
    global _LOAD_ERROR
    try:
        import torch  # type: ignore
    except ImportError:
        _LOAD_ERROR = "torch/transformers not available"
        logger.error("s_mech: %s", _LOAD_ERROR)
        return None

    primary, fallback_id, enabled = _mech_model_ids()
    if not enabled:
        _LOAD_ERROR = "s_mech disabled in configs/experiment.yaml"
        return None
    candidates = [primary]
    if fallback_id and fallback_id != primary:
        candidates.append(fallback_id)

    last_error = None
    tokenizer = model = None
    for model_id in candidates:
        try:
            tokenizer, model = _load_causal_lm(model_id)
            logger.info("s_mech: loaded local model %s", model_id)
            _LOAD_ERROR = None
            break
        except Exception as exc:
            last_error = exc
            logger.error("s_mech: failed to load configured model %s: %s", model_id, exc)
    if model is None or tokenizer is None:
        _LOAD_ERROR = f"all configured s_mech models failed: {last_error}"
        return None

    # Build the prompt (simplified — just context + answer)
    ctx = "\n\n".join(c.text[:500] for c in rr.chunks[:3])
    full_text = f"Evidence:\n{ctx}\n\nAnswer:\n{answer.answer_text}"
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (num_layers,) each shape [batch, heads, seq, seq]
    # Average across layers and heads → [seq, seq]
    attn_stack = torch.stack(outputs.attentions, dim=0)   # [L, 1, H, S, S]
    attn_avg = attn_stack.mean(dim=(0, 1, 2))              # [S, S]

    # For the atom's generated token positions, compute lookback ratio
    gen_start = prompt_len   # first generated token index
    a_start = gen_start + atom_tok_start
    a_end   = gen_start + atom_tok_end
    a_end   = min(a_end, attn_avg.shape[0])

    if a_start >= attn_avg.shape[0]:
        return None

    attn_slice = attn_avg[a_start:a_end, :]   # [atom_toks, seq]
    if attn_slice.numel() == 0:
        return None

    # prompt attention = attention paid to prompt tokens [0:prompt_len]
    attn_to_prompt = attn_slice[:, :prompt_len].sum(dim=-1).mean().item()
    attn_total     = attn_slice.sum(dim=-1).mean().item()

    if attn_total == 0:
        return None

    lookback_ratio = attn_to_prompt / attn_total
    return round(lookback_ratio, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_s_mech(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    use_cache: bool = True,
) -> float:
    """
    Compute mechanistic (attention lookback) signal.
    s_mech = 1 - lookback_ratio ∈ [0, 1].
    Missing attentions yield NaN plus s_mech_skipped=1, not a constant 0.5.
    """
    feats = s_mech_features(verified_atom, answer, rr, use_cache=use_cache)
    return feats["s_mech"]


def s_mech_features(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    use_cache: bool = True,
    strict: bool = False,
) -> dict:
    """
    Return:
        {
          "s_mech":         float or NaN — 1 - lookback_ratio,
          "s_mech_skipped": 0/1  — flag when attentions unavailable,
          "s_mech_error":   str or "" — set when the model failed to load,
        }
    """
    atom = verified_atom.atom
    tok_start, tok_end = _token_span(atom.span, answer.answer_text)

    # Check disk cache first
    cache_key = _cache_key(answer.answer_text, rr.query)
    if use_cache:
        cached = _cache_load(cache_key)
        if cached and atom.atom_id in cached:
            val = cached[atom.atom_id]
            return {"s_mech": val, "s_mech_skipped": 0, "s_mech_error": ""}

    # Try computing from model
    lookback = _compute_lookback_from_model(answer, rr, tok_start, tok_end)

    if lookback is None:
        if strict:
            raise StrictFailure(_LOAD_ERROR or "s_mech model unavailable")
        logger.error(
            "s_mech: missing for atom %s (%s). Not substituting 0.5.",
            atom.atom_id, _LOAD_ERROR,
        )
        return {
            "s_mech": float("nan"),
            "s_mech_skipped": 1,
            "s_mech_error": _LOAD_ERROR or "model_unavailable",
        }

    s_mech_val = round(1.0 - lookback, 4)

    # Save to cache
    if use_cache:
        existing = _cache_load(cache_key) or {}
        existing[atom.atom_id] = s_mech_val
        _cache_save(cache_key, existing)

    return {"s_mech": s_mech_val, "s_mech_skipped": 0, "s_mech_error": ""}
