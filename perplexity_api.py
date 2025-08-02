import os
import httpx
from typing import List, Dict, Any

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

async def perplexity_chat(messages: List[Dict[str, str]], model: str = "sonar-medium-online", max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call Perplexity LLM for chat completion."""
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

async def perplexity_semantic_search(query: str, context_chunks: List[str], model: str = "sonar-medium-online") -> Dict[str, Any]:
    """
    Use Perplexity LLM to select the best matching chunk(s) for a query.
    Returns dict with selected chunk(s) and reasoning.
    """
    system_prompt = (
        "You are an expert insurance/legal document analyst. "
        "Given a user query and a list of document chunks, select the most relevant chunk(s) and explain your reasoning. "
        "Return both the selected chunk(s) and a short rationale."
    )
    user_prompt = (
        f"Query: {query}\n\nDocument Chunks:\n" + "\n---\n".join(context_chunks[:10]) + "\n\nSelect the most relevant chunk(s) and explain why."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    answer = await perplexity_chat(messages, model=model, max_tokens=512)
    return {"selected": answer}
