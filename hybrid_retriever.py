import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from semantic_chunker import semantic_chunk_text

# Vector database imports with error handling
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Warning: Pinecone not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available")

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("Warning: Cohere not available")

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hackrx-llm-index")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

logger = logging.getLogger(__name__)

# --- Embedding Models ---
co = None
if COHERE_AVAILABLE and COHERE_API_KEY:
    try:
        co = cohere.Client(COHERE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Cohere client: {e}")

# Pinecone client
pc = None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")

# --- FAISS Setup (in-memory, for fallback/local) ---
class FaissIndex:
    def __init__(self, dim: int):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.dim = dim
    
    def add(self, vectors: np.ndarray, texts: List[str]):
        """Add vectors and corresponding texts to the index"""
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dim}")
        
        self.index.add(vectors.astype(np.float32))
        self.texts.extend(texts)
    
    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors and return texts with distances"""
        if len(self.texts) == 0:
            return []
        
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        top_k = min(top_k, len(self.texts))
        D, I = self.index.search(vector.astype(np.float32), top_k)
        
        results = []
        for k in range(len(I[0])):
            idx = I[0][k]
            if 0 <= idx < len(self.texts):
                distance = float(D[0][k])
                results.append((self.texts[idx], distance))
        
        return results

# --- Embedding Utilities ---
async def embed_texts_cohere(texts: List[str], model: str = "embed-english-v3.0") -> np.ndarray:
    """Get embeddings using Cohere API"""
    if not co:
        raise RuntimeError("Cohere client not initialized")
    
    try:
        # Cohere has limits on batch size
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = co.embed(texts=batch, model=model, input_type="search_document")
            all_embeddings.extend(response.embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    except Exception as e:
        logger.error(f"Cohere embedding failed: {e}")
        raise

async def embed_texts_fallback(texts: List[str]) -> np.ndarray:
    """Fallback embedding using random vectors"""
    # Return random embeddings as fallback
    import numpy as np
    return np.random.rand(len(texts), 768).astype(np.float32)

async def embed_texts(texts: List[str], model: str = "embed-english-v3.0") -> np.ndarray:
    """Get embeddings with fallback options"""
    if COHERE_AVAILABLE and co is not None:
        return await embed_texts_cohere(texts, model)
    else:
        logger.warning("No embedding service available, using random fallback")
        return await embed_texts_fallback(texts)

# --- Hybrid Retriever ---
class HybridRetriever:
    def __init__(self, use_pinecone: bool = True, use_faiss: bool = True, dim: int = 1024):
        self.use_pinecone = use_pinecone and PINECONE_AVAILABLE and pc is not None
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.dim = dim
        self.faiss_index = None
        self.pinecone_index = None
        
        # Initialize FAISS
        if self.use_faiss:
            try:
                self.faiss_index = FaissIndex(dim)
                logger.info("FAISS index initialized")
            except Exception as e:
                logger.error(f"FAISS initialization failed: {e}")
                self.use_faiss = False
        
        # Initialize Pinecone
        if self.use_pinecone:
            try:
                self._setup_pinecone_index()
                logger.info("Pinecone index initialized")
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {e}")
                self.use_pinecone = False
        
        if not self.use_faiss and not self.use_pinecone:
            logger.warning("No vector databases available - using simple text matching")
    
    def _setup_pinecone_index(self):
        """Setup Pinecone index"""
        if not pc:
            raise Exception("Pinecone client not available")
        
        index_name = PINECONE_INDEX
        
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=self.dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
                )
                # Wait for index to be ready
                import time
                time.sleep(10)
            
            self.pinecone_index = pc.Index(index_name)
            
        except Exception as e:
            logger.error(f"Pinecone setup failed: {e}")
            raise
    
    async def index(self, chunks: List[str]):
        """Index document chunks in both vector databases"""
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        try:
            # Get embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            vectors = await embed_texts(chunks)
            
            if vectors.size == 0:
                logger.error("No embeddings generated")
                return
            
            # Index in FAISS
            if self.use_faiss and self.faiss_index:
                try:
                    self.faiss_index.add(vectors, chunks)
                    logger.info(f"Indexed {len(chunks)} chunks in FAISS")
                except Exception as e:
                    logger.error(f"FAISS indexing failed: {e}")
            
            # Index in Pinecone
            if self.use_pinecone and self.pinecone_index:
                try:
                    # Prepare data for Pinecone
                    ids = [f"chunk-{i}-{hash(chunk[:50])}" for i, chunk in enumerate(chunks)]
                    metadata = [{"text": chunk[:1000]} for chunk in chunks]  # Limit metadata size
                    
                    # Upsert in batches
                    batch_size = 100
                    for i in range(0, len(chunks), batch_size):
                        batch_ids = ids[i:i + batch_size]
                        batch_vectors = vectors[i:i + batch_size].tolist()
                        batch_metadata = metadata[i:i + batch_size]
                        
                        upsert_data = list(zip(batch_ids, batch_vectors, batch_metadata))
                        self.pinecone_index.upsert(vectors=upsert_data)
                    
                    logger.info(f"Indexed {len(chunks)} chunks in Pinecone")
                except Exception as e:
                    logger.error(f"Pinecone indexing failed: {e}")
        
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 5, re_rank: bool = True) -> List[str]:
        """Search for relevant chunks using hybrid approach, with optional semantic re-ranking."""
        if not query.strip():
            return []
        try:
            # Get query embedding
            query_vector = await embed_texts([query])
            if query_vector.size == 0:
                logger.error("Failed to generate query embedding")
                return self._fallback_text_search(query, top_k)
            results = []
            # Search Pinecone
            if self.use_pinecone and self.pinecone_index:
                try:
                    pc_response = self.pinecone_index.query(
                        vector=query_vector[0].tolist(),
                        top_k=top_k*2,  # retrieve more for re-ranking
                        include_metadata=True
                    )
                    for match in pc_response.get("matches", []):
                        if match.get("metadata", {}).get("text"):
                            results.append(match["metadata"]["text"])
                    logger.info(f"Pinecone returned {len(results)} results")
                except Exception as e:
                    logger.error(f"Pinecone search failed: {e}")
            # Search FAISS
            if self.use_faiss and self.faiss_index:
                try:
                    faiss_results = self.faiss_index.search(query_vector, top_k*2)
                    for text, distance in faiss_results:
                        results.append(text)
                    logger.info(f"FAISS returned {len(faiss_results)} results")
                except Exception as e:
                    logger.error(f"FAISS search failed: {e}")
            # Deduplicate results while preserving order
            seen = set()
            deduped_results = []
            for result in results:
                if result not in seen:
                    deduped_results.append(result)
                    seen.add(result)
            candidates = deduped_results[:max(top_k*2, 8)]
            # Semantic re-ranking
            if re_rank and candidates:
                candidate_embeddings = await embed_texts(candidates)
                query_vec = query_vector[0]
                similarities = np.dot(candidate_embeddings, query_vec) / (
                    np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8)
                ranked = sorted(zip(candidates, similarities), key=lambda x: -x[1])
                final_results = [x[0] for x in ranked[:top_k]]
            else:
                final_results = candidates[:top_k]
            if not final_results:
                logger.warning("No vector search results, using text fallback")
                return self._fallback_text_search(query, top_k)
            return final_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._fallback_text_search(query, top_k)

    async def batch_search(self, queries: List[str], top_k: int = 5, re_rank: bool = True) -> List[List[str]]:
        """Batch search for multiple queries, async and with semantic re-ranking."""
        import asyncio
        results = await asyncio.gather(*[self.search(q, top_k=top_k, re_rank=re_rank) for q in queries])
        return results
    
    def _fallback_text_search(self, query: str, top_k: int) -> List[str]:
        """Simple text-based search fallback when vector search fails"""
        # This would require access to original chunks
        # For now, return empty list
        logger.warning("Text fallback search not implemented")
        return []