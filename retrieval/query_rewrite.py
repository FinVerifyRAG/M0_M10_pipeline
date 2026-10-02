from typing import List, Dict
from common.llm_client import LLMClient

def rewrite_query(llm: LLMClient, current_query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Optional module to resolve follow-up questions using chat history.
    Uses a small prompt to Qwen to contextualize the query.
    """
    if not chat_history:
        return current_query
        
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    system_prompt = "You are a query rewriter. Given a chat history and the latest user query, rewrite the user query to be fully self-contained, resolving any pronouns or missing context. Output ONLY the rewritten query text."
    user_prompt = f"Chat History:\n{history_str}\n\nLatest Query: {current_query}\n\nRewritten Query:"
    
    response = llm.chat(user=user_prompt, system=system_prompt, temperature=0.0)
    return response["text"].strip()
