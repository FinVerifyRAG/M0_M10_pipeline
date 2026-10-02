import json
import re
from typing import List, Dict, Any, Optional
import requests

class RegulatoryGenerator:
    """CPU-friendly generation module for regulatory QA with explicit section citations and SLM claim extraction."""

    def __init__(self, model_name: str = "qwen2.5:7b-instruct-q4_K_M", ollama_host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip('/')

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesizes answer with citations and discrete claims from retrieved context chunks."""
        if not context_chunks:
            return {
                "answer": "No relevant RBI/SEBI regulatory context was found to answer your query.",
                "citations": [],
                "claims": ["No relevant regulatory guidelines available."],
                "model_used": "None"
            }

        # Format context block
        formatted_contexts = []
        citations = []

        for idx, chunk in enumerate(context_chunks, start=1):
            meta = chunk.get("metadata", {})
            doc = meta.get("doc_title", "Doc")
            sec = meta.get("section", "Section")
            yr = meta.get("year", "")
            citation_str = f"[{doc} ({yr}), {sec}]"
            if citation_str not in citations:
                citations.append(citation_str)

            text_snippet = chunk.get("text", "")
            formatted_contexts.append(f"Context [{idx}] {citation_str}:\n{text_snippet}\n")

        context_str = "\n".join(formatted_contexts)

        # Build System & User Prompts
        system_prompt = (
            "You are an expert RBI and SEBI financial regulatory compliance officer. "
            "Answer the user's question using ONLY the provided regulatory context snippets. "
            "You must cite document names, sections, or clauses. "
            "At the end of your response, output a section titled 'DISCRETE_CLAIMS:' listing "
            "each factual assertion as an atomic bullet point for validation by an SLM verifier."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Retrieved Regulatory Context:\n{context_str}\n\n"
            f"Provide a thorough, grounded answer with citations followed by DISCRETE_CLAIMS."
        )

        # Attempt generation via Ollama local service
        ollama_response = self._call_ollama(system_prompt, user_prompt)
        if ollama_response:
            answer_text, claims = self._parse_claims(ollama_response)
            return {
                "answer": answer_text,
                "citations": citations,
                "claims": claims,
                "model_used": f"{self.model_name} (Ollama)"
            }

        # Fallback Extractive CPU Generator
        print("[Generator] Ollama endpoint unavailable/model not found. Using CPU Extractive Fallback Engine.")
        fallback_answer, claims = self._fallback_generate(query, context_chunks, citations)

        return {
            "answer": fallback_answer,
            "citations": citations,
            "claims": claims,
            "model_used": "Extractive Regulatory CPU Engine (Fallback)"
        }

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Calls local Ollama API server if running."""
        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 4096
                }
            }
            resp = requests.post(url, json=payload, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "")
        except Exception as e:
            print(f"[Generator] Ollama error: {e}")
        return None

    def _parse_claims(self, raw_response: str) -> tuple[str, List[str]]:
        """Extracts the main answer text and discrete claims list from raw LLM output."""
        if "DISCRETE_CLAIMS:" in raw_response:
            parts = raw_response.split("DISCRETE_CLAIMS:")
            answer_text = parts[0].strip()
            claims_block = parts[1].strip()
            claims = [re.sub(r'^[\-\*\d\.\s]+', '', line).strip() for line in claims_block.splitlines() if line.strip()]
        else:
            answer_text = raw_response.strip()
            sentences = [s.strip() for s in re.split(r'[\.\!\?]\s+', answer_text) if len(s.strip()) > 15]
            claims = sentences[:5]

        return answer_text, claims if claims else ["Generated claim supported by retrieved regulatory text."]

    def _fallback_generate(self, query: str, chunks: List[Dict[str, Any]], citations: List[str]) -> tuple[str, List[str]]:
        """Extractive fallback generator extracting top relevant sentences and formatting claims."""
        sentences = []
        for chunk in chunks[:3]:
            text = chunk.get("text", "")
            # Filter breadcrumb line if present
            lines = [l for l in text.splitlines() if not l.startswith("[Issuer:")]
            text_body = " ".join(lines)
            s_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_body) if len(s.strip()) > 20]
            sentences.extend(s_list[:2])

        summary_body = " ".join(sentences[:4])
        citation_suffix = " Grounded in: " + ", ".join(citations[:3]) if citations else ""
        answer = f"Based on RBI and SEBI guidelines: {summary_body}{citation_suffix}"

        claims = [s for s in sentences[:4]]
        if not claims:
            claims = [f"Retrieved regulation addresses query: {query}"]

        return answer, claims
