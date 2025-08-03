import re
from typing import List

# Optional: import nltk or spacy for sentence splitting
try:
    import nltk
    nltk.download('punkt', quiet=True)
    sent_tokenize = nltk.sent_tokenize
except ImportError:
    sent_tokenize = None


def semantic_chunk_text(text: str, max_tokens: int = 350, overlap: int = 60) -> List[str]:
    """
    Split text into semantic/sentence-aware chunks, each up to max_tokens (approximate, by word count).
    Uses NLTK if available, otherwise falls back to regex sentence splitting.
    """
    # Preprocess: normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Sentence split
    if sent_tokenize:
        sentences = sent_tokenize(text)
    else:
        # Simple regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Chunk assembly
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        tokens = sent.split()
        if current_len + len(tokens) > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                # Overlap: keep last N tokens
                if overlap > 0:
                    overlap_tokens = current_chunk[-overlap:] if len(current_chunk) >= overlap else current_chunk
                    current_chunk = list(overlap_tokens)
                    current_len = len(current_chunk)
                else:
                    current_chunk = []
                    current_len = 0
        current_chunk.extend(tokens)
        current_len += len(tokens)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
