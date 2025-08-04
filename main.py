import sys
import os
import logging
import traceback
import asyncio
import time
import tempfile
from typing import List, Optional, Dict, Any

# Railway environment fixes
def setup_railway_environment():
    """Setup environment variables and paths for Railway deployment"""
    port = int(os.environ.get("PORT", 8000))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if not os.getenv("PYTHONPATH"):
        os.environ["PYTHONPATH"] = current_dir
    return port

port = setup_railway_environment()

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx
import aiofiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils import verify_bearer_token
from semantic_chunker import semantic_chunk_text

app = FastAPI(title="HackRx 6.0 LLM Query–Retrieval System")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Exception on {request.url}: {tb}")
    return JSONResponse(
        status_code=500, 
        content={"error": str(exc), "traceback": tb}
    )

# --- Request/Response Schemas ---
class RunRequest(BaseModel):
    documents: str  # URL to PDF/DOCX/email
    questions: List[str]

class ClauseMatch(BaseModel):
    text: str
    page: Optional[int] = None
    start_idx: Optional[int] = None

class AnswerWithExplain(BaseModel):
    answer: str
    matched_clauses: List[ClauseMatch]
    rationale: str
    confidence_score: float

class RunResponse(BaseModel):
    answers: List[AnswerWithExplain]

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "HackRx 6.0 LLM Query-Retrieval System",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(
    req: RunRequest,
    authorization: str = Header(..., description="Bearer <api_key>")
):
    """Main endpoint for document analysis and question answering"""
    
    # Auth check
    if not verify_bearer_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token.")

    try:
        from document_parser import parse_document
        from hybrid_retriever import HybridRetriever

        from answer_explainer import generate_explainable_answer
        from cache_utils import acache_result

        # Decorate once, outside process_question

        cached_answer = acache_result(generate_explainable_answer)

        leaderboard_log = []
        start_time = time.time()
        
        # 1. Download document
        logger.info(f"Downloading document from: {req.documents}")
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.get(req.documents)
                response.raise_for_status()
            except httpx.TimeoutException:
                raise HTTPException(status_code=408, detail="Document download timeout")
            except httpx.HTTPError as e:
                raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
            
            # Determine file extension
            suffix = os.path.splitext(req.documents)[-1]
            if not suffix:
                # Try to determine from content-type
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type:
                    suffix = '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    suffix = '.docx'
                else:
                    suffix = '.txt'
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

        # 2. Parse and chunk document
        logger.info("Parsing document...")
        try:
            parsed = parse_document(tmp_path)
            # Use semantic/sentence-aware chunking
            raw_text = parsed.text if hasattr(parsed, 'text') else '\n'.join(parsed.chunks)
            chunks = semantic_chunk_text(raw_text, max_tokens=350, overlap=60)
            if not chunks:
                raise HTTPException(status_code=400, detail="No content could be extracted from document")
            logger.info(f"Extracted {len(chunks)} semantic chunks from document")
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Document parsing failed: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

        # 3. Index chunks (hybrid Pinecone/FAISS)
        logger.info("Indexing document chunks...")
        try:
            retriever = HybridRetriever()
            await retriever.index(chunks)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Document indexing failed: {str(e)}")

        # 4. Batch retrieve top chunks for all questions
        logger.info(f"Processing {len(req.questions)} questions...")
        all_retrieved = await retriever.batch_search(req.questions, top_k=8, re_rank=True)

        # 5. Generate answers for all questions in batch
        answers = []
        for q, retrieved in zip(req.questions, all_retrieved):
            if not retrieved:
                answers.append(AnswerWithExplain(
                    answer="No relevant information found in the document.",
                    matched_clauses=[],
                    rationale="No matching content found for this question.",
                    confidence_score=0.0
                ))
                continue
            # Directly use retrieved chunks for answer pipeline
            answer_dict = await cached_answer(q, retrieved[:2])
            matched_clauses_text = answer_dict.get("matched_clauses", retrieved[:2])
            if isinstance(matched_clauses_text, str):
                matched_clauses_text = [matched_clauses_text]
            clause_objs = [
                ClauseMatch(text=cl[:500], page=None, start_idx=None)
                for cl in matched_clauses_text[:5]
            ]
            answers.append(AnswerWithExplain(
                answer=answer_dict.get("answer", "Unable to generate answer"),
                matched_clauses=clause_objs,
                rationale=answer_dict.get("rationale", "Answer generated from document analysis"),
                confidence_score=float(answer_dict.get("confidence", 50))
            ))

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f}s for {len(req.questions)} questions")
        logger.info(f"Leaderboard log: {leaderboard_log}")
        
        return RunResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hackrx_run: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)