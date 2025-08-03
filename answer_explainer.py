#!/usr/bin/env python3
"""
Answer generation with rationale, clause traceability, and confidence scoring
"""

import json
import logging
import os
from typing import List, Dict, Any
from openai_api import openai_chat

logger = logging.getLogger(__name__)

async def generate_explainable_answer(
    question: str, 
    matched_chunks: List[str], 
    model: str = "gpt-3.5-turbo"
) -> Dict[str, Any]:
    """
    Generate an explainable answer with rationale, matched clauses, and confidence score (OpenAI only)
    """
    if not matched_chunks:
        return {
            "answer": "No relevant information found in the document.",
            "rationale": "No matching content was found for this question.",
            "matched_clauses": [],
            "confidence": 0
        }
    # Deduplicate and limit context size for OpenAI input
    seen = set()
    context_chunks = []
    for chunk in matched_chunks:
        chunk_clean = chunk.strip()
        if chunk_clean not in seen:
            seen.add(chunk_clean)
            context_chunks.append(chunk_clean)
        if len(context_chunks) >= 2:
            break
    combined_context = "\n\n".join([f"Clause {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])

    system_prompt = (
        "You are an expert insurance, legal, and HR document analyst.\n"
        "Given a user question and relevant document clauses, answer ONLY using the provided clauses.\n"
        "If the answer is not present, say: 'Not found in document.'\n"
        "Your response MUST be a valid JSON object with these exact keys:\n"
        "- 'answer': Direct answer to the question (string)\n"
        "- 'rationale': Explanation of how you arrived at the answer (string)\n"
        "- 'matched_clauses': List of relevant clause texts (array of strings)\n"
        "- 'confidence_score': Numerical confidence in your answer (float 0.0-1.0)\n\n"
        "Be concise, factual, and cite specific parts of the document clauses.\n"
        "NEVER make up information. If unsure, say 'Not found in document.'\n"
        "Your response MUST be valid JSON parsable by Python's json.loads."
    )

    user_prompt = (
        f"Document Clauses:\n{combined_context}\n\n"
        f"Question: {question}\n\n"
        f"Respond in JSON format only."
    )

    openai_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        openai_response = await openai_chat(openai_messages, model=model, temperature=0, max_tokens=128)
        try:
            data = json.loads(openai_response)
            required_keys = ["answer", "rationale", "matched_clauses", "confidence_score"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing key in response: {key}")
            logger.info("Successfully got valid response from OpenAI API")
            return {
                "answer": data["answer"],
                "rationale": data["rationale"],
                "matched_clauses": data["matched_clauses"],
                "confidence": data["confidence_score"]
            }
        except json.JSONDecodeError as json_err:
            logger.warning(f"Failed to parse JSON from OpenAI, attempting to parse raw response: {json_err}")
            return _parse_raw_response(openai_response, context_chunks)
    except Exception as e:
        logger.error(f"OpenAI API failed with error: {str(e)}", exc_info=True)
        return {
            "answer": "Not found in document.",
            "rationale": "System encountered an error during answer generation with OpenAI API.",
            "matched_clauses": context_chunks[:1] if 'context_chunks' in locals() else [],
            "confidence": 0
        }

# Async batch version for leaderboard: answers multiple questions efficiently
async def generate_explainable_answers(
    questions: List[str],
    matched_chunks_list: List[List[str]],
    model: str = "gpt-3.5-turbo"
) -> List[Dict[str, Any]]:
    import asyncio
    tasks = [
        generate_explainable_answer(q, c, model=model)
        for q, c in zip(questions, matched_chunks_list)
    ]
    return await asyncio.gather(*tasks)

def _parse_raw_response(response: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Fallback parser for non-JSON responses
    """
    
    # Try to extract structured information from the response
    answer = ""
    rationale = ""
    confidence = 50
    
    # Simple heuristic parsing
    lines = response.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for section headers
        if any(keyword in line.lower() for keyword in ['answer:', 'response:', 'conclusion:']):
            current_section = 'answer'
            # Extract text after the colon
            if ':' in line:
                answer += line.split(':', 1)[1].strip() + " "
        elif any(keyword in line.lower() for keyword in ['rationale:', 'reasoning:', 'explanation:']):
            current_section = 'rationale'
            if ':' in line:
                rationale += line.split(':', 1)[1].strip() + " "
        elif any(keyword in line.lower() for keyword in ['confidence:', 'certainty:']):
            # Try to extract confidence score
            import re
            numbers = re.findall(r'\d+', line)
            if numbers:
                try:
                    confidence = min(100, max(0, int(numbers[0])))
                except ValueError:
                    pass
        else:
            # Add to current section
            if current_section == 'answer':
                answer += line + " "
            elif current_section == 'rationale':
                rationale += line + " "
    
    # If still empty, use the entire response as answer
    if not answer:
        answer = response[:500] + "..." if len(response) > 500 else response
    
    if not rationale:
        rationale = "Answer extracted from document analysis."
    
    return {
        "answer": answer.strip(),
        "rationale": rationale.strip(),
        "matched_clauses": context_chunks[:2],
        "confidence": confidence
    }

async def generate_summary(chunks: List[str], model: str = "llama-3.1-sonar-small-128k-online") -> str:
    """Generate a summary of document chunks"""
    
    if not chunks:
        return "No content to summarize."
    
    # Combine chunks with size limit
    combined_text = "\n\n".join(chunks[:10])
    if len(combined_text) > 3000:
        combined_text = combined_text[:3000] + "..."
    
    system_prompt = (
        "You are an expert document analyst. "
        "Create a concise summary of the provided document content, "
        "highlighting key topics, important clauses, and main themes."
    )
    
    user_prompt = f"Document Content:\n{combined_text}\n\nProvide a comprehensive summary:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        return await gemini_chat(messages, model=model, max_tokens=512)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Unable to generate summary: {str(e)}"