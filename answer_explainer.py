#!/usr/bin/env python3
"""
Answer generation with rationale, clause traceability, and confidence scoring
"""

import json
import logging
from typing import List, Dict, Any
from perplexity_api import perplexity_chat

# Try to import Gemini API as fallback
try:
    from gemini_api import gemini_chat
    GEMINI_AVAILABLE = True
except ImportError:
    logging.warning("Gemini API module not found")
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

async def generate_explainable_answer(
    question: str, 
    matched_chunks: List[str], 
    model: str = "llama-3.1-sonar-small-128k-online"
) -> Dict[str, Any]:
    """
    Generate an explainable answer with rationale, matched clauses, and confidence score
    """
    
    if not matched_chunks:
        return {
            "answer": "No relevant information found in the document.",
            "rationale": "No matching content was found for this question.",
            "matched_clauses": [],
            "confidence": 0
        }
    
    # Limit context size to avoid token limits
    context_chunks = matched_chunks[:5]
    combined_context = "\n\n".join([f"Clause {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    system_prompt = (
        "You are an expert insurance, legal, and HR document analyst. "
        "Given a user question and relevant document clauses, provide a comprehensive answer. "
        "Your response must be a valid JSON object with these exact keys:\n"
        "- 'answer': Direct answer to the question (string)\n"
        "- 'rationale': Explanation of how you arrived at the answer (string)\n"
        "- 'matched_clauses': List of relevant clause texts (array of strings)\n"
        "- 'confidence_score': Numerical confidence in your answer (float 0.0-1.0)\n\n"
        "Be precise, factual, and cite specific parts of the document clauses when possible. "
        "If the clauses don't contain enough information, say so clearly. "
        "Ensure your response is valid JSON that can be parsed directly."
    )
    
    user_prompt = (
        f"Document Clauses:\n{combined_context}\n\n"
        f"Question: {question}\n\n"
        f"Answer in JSON format:"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        logger.info("Attempting to get answer from Perplexity API")
        response = await perplexity_chat(messages, model=model)
        
        # Parse JSON response
        if response and isinstance(response, str):
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            try:
                data = json.loads(response)
                
                # Validate response structure
                required_keys = ["answer", "rationale", "matched_clauses", "confidence_score"]
                for key in required_keys:
                    if key not in data:
                        raise ValueError(f"Missing key in response: {key}")
                
                logger.info("Successfully got valid response from Perplexity API")
                return {
                    "answer": data["answer"],
                    "rationale": data["rationale"],
                    "matched_clauses": data["matched_clauses"],
                    "confidence": data["confidence_score"]
                }
                
            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse JSON from Perplexity, attempting to parse raw response: {json_err}")
                return _parse_raw_response(response, context_chunks)
        else:
            raise ValueError("Empty or invalid response from Perplexity API")
            
    except Exception as e:
        logger.error(f"Perplexity API failed with error: {str(e)}", exc_info=True)
        raise  # Re-raise to trigger Gemini fallback
    
    except Exception as e:
        logger.error(f"Perplexity API failed: {e}")
        
        # Try Gemini API as fallback if available
        if GEMINI_AVAILABLE:
            try:
                logger.info("Attempting fallback to Gemini API")
                gemini_system_prompt = (
                    "You are an expert insurance, legal, and HR document analyst. "
                    "Given a user question and relevant document clauses, provide a comprehensive answer. "
                    "Respond with a valid JSON object containing these keys: "
                    "'answer' (direct answer), 'rationale' (explanation), "
                    "'matched_clauses' (array of relevant clauses), 'confidence_score' (0.0-1.0). "
                    "Be precise and cite specific document parts. If insufficient, say so clearly."
                )
                
                gemini_user_prompt = (
                    f"Document Clauses:\n{combined_context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer in JSON format:"
                )
                
                gemini_messages = [
                    {"role": "user", "content": gemini_system_prompt + "\n\n" + gemini_user_prompt}
                ]
                
                gemini_response = await gemini_chat(gemini_messages, model="gemini-pro")
                
                # Parse JSON response
                if gemini_response.startswith("```json"):
                    gemini_response = gemini_response[7:]
                if gemini_response.endswith("```"):
                    gemini_response = gemini_response[:-3]
                
                try:
                    gemini_data = json.loads(gemini_response)
                except json.JSONDecodeError:
                    # Fallback to raw response parsing
                    return _parse_raw_response(gemini_response, context_chunks)
                gemini_data = json.loads(gemini_response)
                
                # Validate response structure
                required_keys = ["answer", "rationale", "matched_clauses", "confidence_score"]
                for key in required_keys:
                    if key not in gemini_data:
                        raise ValueError(f"Missing key in Gemini response: {key}")
                
                return {
                    "answer": gemini_data["answer"],
                    "rationale": gemini_data["rationale"],
                    "matched_clauses": gemini_data["matched_clauses"],
                    "confidence": gemini_data["confidence_score"]
                }
            
            except Exception as gemini_e:
                logger.error(f"Gemini API also failed: {gemini_e}")
                # Return error fallback
                return {
                    "answer": f"Unable to generate answer due to system error: {str(gemini_e)}",
                    "rationale": "System encountered an error during answer generation with both Perplexity and Gemini APIs.",
                    "matched_clauses": context_chunks[:1],
                    "confidence": 0
                }
        else:
            # Return error fallback
            return {
                "answer": f"Unable to generate answer due to system error: {str(e)}",
                "rationale": "System encountered an error during answer generation.",
                "matched_clauses": context_chunks[:1],
                "confidence": 0
            }

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
        return await perplexity_chat(messages, model=model, max_tokens=512)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Unable to generate summary: {str(e)}"