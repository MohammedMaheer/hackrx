# HackRx 6.0 LLM Query–Retrieval System

## Overview
A leaderboard-optimized, explainable, and hybrid semantic retrieval system for HackRx 6.0. Features Perplexity API as the sole LLM, with hybrid Pinecone and FAISS vector search, delivering rationale-rich answers for insurance/legal/HR/compliance queries.

## Features
- Perplexity API integration for LLM capabilities
- Hybrid Pinecone (cloud) and FAISS (local) vector search
- Answer generation with rationale, clause traceability, and confidence scoring
- FastAPI backend with `/hackrx/run` endpoint (Bearer token authentication)
- Advanced document parsing (PDF, DOCX, email)
- Async, batch, and caching optimizations for leaderboard performance

## LLM Provider

This system uses **Perplexity API** exclusively for all LLM answer generation. No OpenAI or Gemini code remains in the codebase.

## Environment Variables
- `PERPLEXITY_API_KEY`: Your Perplexity API key (required)
- `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_REGION`: Pinecone vector DB config
- `HACKRX_API_KEY`: API key for authentication

## Usage
1. Add your Perplexity API key to `.env` as `PERPLEXITY_API_KEY`.
2. Start the FastAPI server: `python -m uvicorn main:app --reload --port 8000`
3. Use `python test_api.py http://localhost:8000` to test the `/hackrx/run` endpoint.

## Output Structure
Each answer includes:
- `answer`
- `rationale`
- `matched_clauses`
- `confidence_score`

## Production
- Only Perplexity is used for answer generation; no fallback to OpenAI/Gemini.
- All code, documentation, and environment files are Perplexity-only.

## Authors
- Maahir (sir)
- Cascade AI
