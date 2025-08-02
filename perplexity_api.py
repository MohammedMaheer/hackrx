import os
import logging
from typing import List, Dict, Any
import httpx
import json

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

async def perplexity_chat(
    messages: List[Dict[str, str]], 
    model: str = "llama-3.1-sonar-small-128k-online", 
    max_tokens: int = 512, 
    temperature: float = 0.2
) -> str:
    """Call Perplexity LLM for chat completion with error handling"""
    
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY not set")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(PERPLEXITY_API_URL, headers=headers, json=payload)
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Perplexity API error {response.status_code}: {error_detail}")
                raise httpx.HTTPStatusError(
                    f"Perplexity API returned {response.status_code}", 
                    request=response.request, 
                    response=response
                )
            
            data = response.json()
            
            if "choices" not in data or not data["choices"]:
                raise ValueError("No choices in Perplexity response")
            
            content = data["choices"][0]["message"]["content"]
            return content
            
    except httpx.TimeoutException:
        logger.error("Perplexity API timeout")
        raise RuntimeError("Perplexity API request timeout")
    except httpx.HTTPError as e:
        logger.error(f"Perplexity HTTP error: {e}")
        raise RuntimeError(f"Perplexity API error: {str(e)}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON response from Perplexity")
        raise RuntimeError("Invalid response from Perplexity API")
    except Exception as e:
        logger.error(f"Unexpected error in Perplexity API call: {e}")
        raise RuntimeError(f"Perplexity API error: {str(e)}")

async def perplexity_semantic_search(
    query: str, 
    context_chunks: List[str], 
    model: str = "llama-3.1-sonar-small-128k-online"
) -> Dict[str, Any]:
    """
    Use Perplexity LLM to select the best matching chunk(s) for a query.
    Returns dict with selected chunk(s) and reasoning.
    """
    
    if not context_chunks:
        return {"selected": [], "reasoning": "No context chunks provided"}
    
    # Limit number of chunks to avoid token limits
    limited_chunks = context_chunks[:10]
    
    system_prompt = (
        "You are an expert document analyst specializing in insurance, legal, and HR documents. "
        "Your task is to identify the most relevant document chunk(s) that can answer the user's question. "
        "Analyze each chunk carefully and select the 1-3 most relevant ones. "
        "Respond with a JSON object containing 'selected' (array of chunk texts) and 'reasoning' (explanation)."
    )
    
    # Create numbered chunks for easier reference
    numbered_chunks = []
    for i, chunk in enumerate(limited_chunks):
        # Truncate very long chunks
        truncated_chunk = chunk[:800] + "..." if len(chunk) > 800 else chunk
        numbered_chunks.append(f"[Chunk {i+1}]: {truncated_chunk}")
    
    user_prompt = (
        f"Query: {query}\n\n"
        f"Document Chunks:\n" + "\n\n".join(numbered_chunks) + "\n\n"
        f"Select the most relevant chunk(s) that can answer the query. "
        f"Return JSON with 'selected' (array of full chunk texts) and 'reasoning'."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = await perplexity_chat(messages, model=model, max_tokens=1024)
        
        # Try to parse JSON response
        try:
            parsed_response = json.loads(response)
            
            # Validate response structure
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a JSON object")
            
            selected = parsed_response.get("selected", [])
            reasoning = parsed_response.get("reasoning", "No reasoning provided")
            
            # Ensure selected is a list
            if isinstance(selected, str):
                selected = [selected]
            elif not isinstance(selected, list):
                selected = []
            
            # If no valid selection, use first chunk as fallback
            if not selected and limited_chunks:
                selected = [limited_chunks[0]]
                reasoning = "Fallback: selected first chunk due to parsing issues"
            
            return {
                "selected": selected,
                "reasoning": reasoning
            }
            
        except json.JSONDecodeError:
            # Fallback: treat entire response as selected text
            logger.warning("Could not parse JSON from Perplexity, using raw response")
            return {
                "selected": [response] if response else limited_chunks[:1],
                "reasoning": "Raw response used due to JSON parsing failure"
            }
    
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        # Return first chunk as emergency fallback
        return {
            "selected": limited_chunks[:1] if limited_chunks else [],
            "reasoning": f"Error in semantic search: {str(e)}"
        }

async def perplexity_answer_question(
    question: str, 
    context: str, 
    model: str = "llama-3.1-sonar-small-128k-online"
) -> str:
    """Generate a direct answer to a question given context"""
    
    system_prompt = (
        "You are an expert analyst for insurance, legal, and HR documents. "
        "Answer the user's question based strictly on the provided context. "
        "Be precise, factual, and cite specific parts of the context when possible. "
        "If the context doesn't contain enough information, say so clearly."
    )
    
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based on the context above:"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        return await perplexity_chat(messages, model=model, max_tokens=512)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"Unable to generate answer: {str(e)}"