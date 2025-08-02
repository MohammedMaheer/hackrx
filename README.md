# HackRx 6.0 LLM Query–Retrieval System

## Overview
A leaderboard-optimized, explainable, and hybrid semantic retrieval system for HackRx 6.0. Supports both Pinecone and FAISS, integrates Perplexity and Cohere APIs, and provides rationale-rich answers for insurance/legal/HR/compliance queries.

## Features
- FastAPI backend with `/hackrx/run` endpoint
- Bearer token authentication
- PDF/DOCX/email parsing with layout/context awareness
- Hybrid vector DB: Pinecone (cloud) + FAISS (local)
- Perplexity & Cohere API integration for semantic search
- LLM-powered answer generation with rationale, clause traceability, and confidence scoring
- Optimized for accuracy, latency, and explainability

## Quickstart
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your API keys
3. `uvicorn main:app --reload`

## API Usage
- **POST** `/hackrx/run`
  - Auth: `Authorization: Bearer <api_key>`
  - Body: `{ "documents": "<url>", "questions": ["..."] }`
  - Response: `{ "answers": [ ... ] }`

## Advanced
- Modular design for rapid extension
- Caching and async for leaderboard performance

## Authors
- Maahir (sir)
- Cascade AI
