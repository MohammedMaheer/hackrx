import os
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

from utils import verify_bearer_token

app = FastAPI(title="HackRx 6.0 LLM Query–Retrieval System")

import logging, traceback
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s %(message)s')
from fastapi.responses import JSONResponse
from fastapi.requests import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logging.error(f"Exception on {request.url}: {tb}")
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})

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
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(
    req: RunRequest,
    authorization: str = Header(..., description="Bearer <api_key>")
):
    # Auth check
    if not verify_bearer_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token.")

    import aiofiles, tempfile, httpx, asyncio, time
    from document_parser import parse_document
    from hybrid_retriever import HybridRetriever
    from perplexity_api import perplexity_semantic_search
    from answer_explainer import generate_explainable_answer
    from cache_utils import acache_result

    leaderboard_log = []
    start_time = time.time()
    # 1. Download document
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(req.documents)
        r.raise_for_status()
        suffix = os.path.splitext(req.documents)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
    # 2. Parse and chunk document
    parsed = parse_document(tmp_path)
    chunks = parsed.chunks
    # 3. Index chunks (hybrid Pinecone/FAISS)
    retriever = HybridRetriever()
    retriever.index(chunks)

    # 4. Async batch for questions with caching
    async def process_question(q):
        # Hybrid semantic retrieval (embedding)
        retrieved = retriever.search(q, top_k=8)
        # LLM clause selection (Perplexity, cached)
        cached_semantic = await acache_result(perplexity_semantic_search)
        llm_match = await cached_semantic(q, retrieved)
        selected_chunks = [llm_match["selected"]] if isinstance(llm_match["selected"], str) else llm_match["selected"]
        # LLM answer, rationale, confidence (cached)
        cached_answer = await acache_result(generate_explainable_answer)
        answer_dict = await cached_answer(q, selected_chunks)
        clause_objs = [ClauseMatch(text=cl, page=None, start_idx=None) for cl in answer_dict.get("matched_clauses", selected_chunks)]
        leaderboard_log.append({"question": q, "latency": time.time() - start_time})
        return AnswerWithExplain(
            answer=answer_dict.get("answer", ""),
            matched_clauses=clause_objs,
            rationale=answer_dict.get("rationale", ""),
            confidence_score=float(answer_dict.get("confidence", 50))
        )
    answers = await asyncio.gather(*(process_question(q) for q in req.questions))
    os.remove(tmp_path)
    # Optionally: log leaderboard analytics (latency, etc.)
    print("Leaderboard log:", leaderboard_log)
    return RunResponse(answers=list(answers))

