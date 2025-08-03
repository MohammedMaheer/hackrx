#!/usr/bin/env python3
"""
Google Gemini API integration for HackRx LLM Query-Retrieval System
"""

import os
import logging
from typing import List, Dict, Any
import httpx
import json

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")

async def gemini_chat(
    messages: List[Dict[str, str]], 
    model: str = "gemini-pro", 
    max_tokens: int = 512, 
    temperature: float = 0.2
) -> str:
    """Call Google Gemini LLM for chat completion with error handling"""
    
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    
    # Convert messages to Gemini format
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            # Gemini doesn't have a system role, so we prepend the system message to the user message
            system_content = msg["content"]
        else:
            gemini_messages.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })
    
    # If we have a system message, we need to incorporate it into the user message
    if "system_content" in locals():
        if gemini_messages and gemini_messages[0]["role"] == "user":
            gemini_messages[0]["parts"][0]["text"] = system_content + "\n\n" + gemini_messages[0]["parts"][0]["text"]
        else:
            gemini_messages.insert(0, {
                "role": "user",
                "parts": [{"text": system_content}]
            })
    
    # Create the prompt for Gemini (it expects a single prompt, not a conversation history)
    prompt_parts = []
    for msg in gemini_messages:
        if msg["role"] == "user":
            prompt_parts.append(msg["parts"][0]["text"])
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "\n\n".join(prompt_parts)
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Gemini API error {response.status_code}: {error_detail}")
                raise httpx.HTTPStatusError(
                    f"Gemini API returned {response.status_code}", 
                    request=response.request, 
                    response=response
                )
            
            data = response.json()
            
            if "candidates" not in data or not data["candidates"]:
                raise ValueError("No candidates in Gemini response")
            
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return content
            
    except httpx.TimeoutException:
        logger.error("Gemini API timeout")
        raise RuntimeError("Gemini API request timeout")
    except httpx.HTTPError as e:
        logger.error(f"Gemini HTTP error: {e}")
        raise RuntimeError(f"Gemini API error: {str(e)}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON response from Gemini")
        raise RuntimeError("Invalid response from Gemini API")
    except Exception as e:
        logger.error(f"Unexpected error in Gemini API call: {e}")
        raise RuntimeError(f"Gemini API error: {str(e)}")

async def gemini_semantic_search(
    query: str, 
    context_chunks: List[str], 
    model: str = "gemini-pro"
) -> Dict[str, Any]:
    """
    Use Google Gemini LLM to select the best matching chunk(s) for a query.
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
        {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
    ]
    
    try:
        response = await gemini_chat(messages, model=model, max_tokens=1024)
        
        # Try to parse JSON response
        try:
            # Gemini might wrap the JSON in markdown code blocks
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
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
            logger.warning("Could not parse JSON from Gemini, using raw response")
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

async def gemini_answer_question(
    question: str, 
    context: str, 
    model: str = "gemini-pro"
) -> str:
    """Generate a direct answer to a question given context using Google Gemini"""
    
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
        {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
    ]
    
    try:
        return await gemini_chat(messages, model=model, max_tokens=512)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"Unable to generate answer: {str(e)}"
