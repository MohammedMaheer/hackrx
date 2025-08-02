from typing import List, Dict, Any
from perplexity_api import perplexity_chat

async def generate_explainable_answer(question: str, matched_chunks: List[str], model: str = "sonar-medium-online") -> Dict[str, Any]:
    """
    Given a question and matched document chunks, use Perplexity to generate an answer, rationale, matched clauses, and confidence.
    """
    system_prompt = (
        "You are an expert insurance/legal/HR document analyst. "
        "Given a user question and matched document clauses, answer the question, cite the relevant clause(s), provide a rationale, and estimate a confidence score (1-100). "
        "Format your response as JSON with keys: answer, rationale, matched_clauses, confidence."
    )
    user_prompt = (
        f"Question: {question}\n\nMatched Clauses:\n" + "\n---\n".join(matched_chunks[:5]) + "\n\nRespond in JSON as specified."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = await perplexity_chat(messages, model=model, max_tokens=512)
    # Try to parse JSON, fallback to raw string if not valid
    import json
    try:
        return json.loads(response)
    except Exception:
        return {"answer": response, "rationale": "Could not parse rationale.", "matched_clauses": matched_chunks[:1], "confidence": 50}
