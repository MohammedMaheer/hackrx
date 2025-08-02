import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import faiss
import cohere

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hackrx-llm-index")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-west-2")

# --- Embedding Model (Cohere for speed/token efficiency) ---
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# --- Pinecone Setup ---
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)

# --- FAISS Setup (in-memory, for fallback/local) ---
class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
    def add(self, vectors: np.ndarray, texts: List[str]):
        self.index.add(vectors)
        self.texts.extend(texts)
    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        D, I = self.index.search(vector, top_k)
        return [(self.texts[i], float(D[0][k])) for k, i in enumerate(I[0])]

# --- Embedding Utility ---
def embed_texts(texts: List[str], model: str = "embed-english-v3.0") -> np.ndarray:
    if co:
        res = co.embed(texts=texts, model=model)
        return np.array(res.embeddings, dtype=np.float32)
    raise RuntimeError("Cohere API key not set for embedding.")

# --- Hybrid Retriever ---
class HybridRetriever:
    def __init__(self, use_pinecone: bool = True, use_faiss: bool = True, dim: int = 1024):
        self.use_pinecone = use_pinecone and PINECONE_API_KEY is not None
        self.use_faiss = use_faiss
        self.faiss_index = FaissIndex(dim) if use_faiss else None
        self.pinecone_index = None
        if self.use_pinecone:
            # Create or connect to Pinecone index
            index_name = PINECONE_INDEX
            region = PINECONE_REGION
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing_indexes:
                pc.create_index(
                    name=index_name,
                    dimension=dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=region)
                )
            self.pinecone_index = pc.Index(index_name)
    def index(self, chunks: List[str]):
        vectors = embed_texts(chunks)
        if self.use_faiss:
            self.faiss_index.add(vectors, chunks)
        if self.use_pinecone:
            ids = [f"chunk-{i}" for i in range(len(chunks))]
            meta = [{"text": c} for c in chunks]
            self.pinecone_index.upsert(list(zip(ids, vectors.tolist(), meta)))
    def search(self, query: str, top_k: int = 5) -> List[str]:
        qvec = embed_texts([query])
        results = []
        if self.use_pinecone:
            pc_res = self.pinecone_index.query(vector=qvec[0].tolist(), top_k=top_k, include_metadata=True)
            results.extend([m["metadata"]["text"] for m in pc_res["matches"]])
        if self.use_faiss:
            faiss_res = self.faiss_index.search(qvec, top_k)
            results.extend([t for t, _ in faiss_res])
        # Deduplicate, preserve order
        seen = set()
        deduped = []
        for r in results:
            if r not in seen:
                deduped.append(r)
                seen.add(r)
        return deduped[:top_k]
