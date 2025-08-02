import json
import logging
from typing import List, Dict, Any
from perplexity_api import perplexity_chat

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
        "- 'matched_clauses': List of the most relevant clause texts that support your answer (array of strings)\n"
        "- 'confidence': Your confidence in the answer from 0-100 (number)\n\n"
        "Be precise, factual, and base your answer strictly on the provided clauses. "
        "If the clauses don't fully answer the question, indicate what information is missing."
    )
    
    user_prompt = (
        f"Question: {question}\n\n"
        f"Document Clauses:\n{combined_context}\n\n"
        f"Provide your analysis as a JSON object with the required keys."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Get response from Perplexity
        response = await perplexity_chat(messages, model=model, max_tokens=1024, temperature=0.1)
        
        # Try to parse JSON response
        try:
            parsed_response = json.loads(response)
            
            # Validate response structure
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a JSON object")
            
            # Extract and validate fields
            answer = parsed_response.get("answer", "")
            rationale = parsed_response.get("rationale", "")
            matched_clauses = parsed_response.get("matched_clauses", [])
            confidence = parsed_response.get("confidence", 50)
            
            # Ensure matched_clauses is a list
            if isinstance(matched_clauses, str):
                matched_clauses = [matched_clauses]
            elif not isinstance(matched_clauses, list):
                matched_clauses = context_chunks[:2]
            
            # Ensure confidence is a number between 0-100
            try:
                confidence = float(confidence)
                confidence = max(0, min(100, confidence))
            except (ValueError, TypeError):
                confidence = 50
            
            # Fallback values if fields are empty
            if not answer:
                answer = "Answer could not be determined from the provided clauses."
            if not rationale:
                rationale = "Analysis based on document content review."
            if not matched_clauses:
                matched_clauses = context_chunks[:1]
            
            return {
                "answer": answer,
                "rationale": rationale,
                "matched_clauses": matched_clauses,
                "confidence": confidence
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            # Fallback: extract information from raw response
            return _parse_raw_response(response, context_chunks)
    
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
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