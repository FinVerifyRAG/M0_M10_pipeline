import os
import re
from typing import List, Dict, Any, Tuple
from common.schemas import RetrievalResult, GeneratedAnswer
from common.llm_client import LLMClient

class Generator:
    """Answer generator using LLM with logprobs and citation parsing."""

    def __init__(self, llm_client: LLMClient, prompt_path: str = "generation/prompts/answer_v1.txt"):
        self.llm = llm_client
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def generate(self, rr: RetrievalResult) -> GeneratedAnswer:
        # 1. Format evidence
        ctx = "\n\n".join(f"[{c.chunk_id}] ({c.regulator}, {c.issue_date}) {c.text}" for c in rr.chunks)
        user_prompt = f"Evidence:\n{ctx}\n\nQuestion: {rr.query}"
        
        # 2. Main generation with context
        # Note: logprobs disabled — Groq API does not support it for this model.
        # M6 s_ent signal will fall back to its 0.5 default when token_logprobs is empty.
        response = self.llm.chat(
            user=user_prompt,
            system=self.system_prompt,
            temperature=0.0,
        )
        
        answer_text = response["text"]
        model_id = response["model"]
        token_logprobs = response.get("token_logprobs", [])
        
        # 3. Parse citations
        citations = self.parse_citations(answer_text)
        
        # 4. Generate no-context answer (for M6 s_div divergence signal)
        no_context_prompt = f"Question: {rr.query}"
        no_ctx_response = self.llm.chat(
            user=no_context_prompt,
            system="Answer the question directly.",
            temperature=0.0,
            logprobs=False
        )
        
        return GeneratedAnswer(
            query=rr.query,
            answer_text=answer_text,
            citations=citations,
            token_logprobs=token_logprobs,
            model_id=model_id,
            metadata={
                "no_context_answer": no_ctx_response["text"],
                "prompt_version": "v1",
                "temperature": 0.0
            }
        )

    def parse_citations(self, text: str) -> List[str]:
        """Extracts [chunk_id] citations from text."""
        # Finds all strings inside brackets that look like chunk IDs
        matches = re.findall(r'\[([A-Za-z0-9_\-]+)\]', text)
        return list(set(matches))
