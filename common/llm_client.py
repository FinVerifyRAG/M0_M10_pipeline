import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import openai
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
import logging

from common.rate_limit import get_bucket

# Load .env from project root so LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

logger = logging.getLogger("llm_client")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".llm_cache")

def get_prompt_hash(model: str, system: str, user: str, kwargs: dict) -> str:
    """Generates a stable hash for a prompt and its parameters."""
    payload = {
        "model": model,
        "system": system,
        "user": user,
        "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"} 
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

class LLMClient:
    """
    OpenAI-compatible client wrapper for vLLM and API models, 
    with retries and a disk cache.
    Reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from environment (loaded via
    python-dotenv at startup). Falls back to local Ollama if env vars absent.
    """
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        default_model: str = None,
    ):
        base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        api_key  = api_key  or os.getenv("LLM_API_KEY",  "ollama")
        default_model = default_model or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = default_model
        os.makedirs(CACHE_DIR, exist_ok=True)
        
    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
        
    def _write_cache(self, cache_key: str, data: Dict[str, Any]):
        path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @retry(
        wait=wait_exponential_jitter(initial=1, max=20, jitter=1),
        stop=stop_after_attempt(int(os.getenv("LLM_MAX_RETRIES", "3"))),
        reraise=True,
    )
    def chat(self, user: str, system: Optional[str] = None, model: Optional[str] = None, logprobs: bool = False, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Sends a chat request with caching and retries.
        """
        model = model or self.default_model
        system_msg = system or "You are a helpful assistant."
        
        cache_key = get_prompt_hash(model, system_msg, user, {"temperature": temperature, "logprobs": logprobs, **kwargs})
        
        cached = self._read_cache(cache_key)
        if cached:
            cached = dict(cached)
            cached["cache_hit"] = True
            return cached

        get_bucket().acquire(1.0)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            logprobs=logprobs,
            **kwargs
        )
        
        result = {
            "text": response.choices[0].message.content,
            "model": response.model,
            "cache_hit": False,
        }
        
        logger.info(f"LLM Generation completed - Model: {model}, Prompt Hash: {cache_key}")
        
        if logprobs and response.choices[0].logprobs:
            # Depending on the backend (vLLM/Ollama), parse the logprobs array
            result["token_logprobs"] = [
                token.logprob for token in response.choices[0].logprobs.content
            ]
            
        self._write_cache(cache_key, result)
        return result
