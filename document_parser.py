import os
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import pdfplumber
from docx import Document
from tika import parser as tika_parser

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".eml"]

class ParsedDocument:
    def __init__(self, chunks: List[str], meta: Optional[Dict] = None):
        self.chunks = chunks
        self.meta = meta or {}

def chunk_text(text: str, max_chunk_size: int = 1024, overlap: int = 128) -> List[str]:
    """Chunk text into overlapping segments for semantic search."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        start += max_chunk_size - overlap
    return chunks

def parse_pdf_pymupdf(path: str) -> List[str]:
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    return chunk_text(text)

def parse_pdf_pdfplumber(path: str) -> List[str]:
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return chunk_text(text)

def parse_docx(path: str) -> List[str]:
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return chunk_text(text)

def parse_email(path: str) -> List[str]:
    raw = tika_parser.from_file(path)
    text = raw.get("content", "")
    return chunk_text(text)

def parse_document(path: str) -> ParsedDocument:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        try:
            return ParsedDocument(parse_pdf_pymupdf(path), {"parser": "pymupdf"})
        except Exception:
            return ParsedDocument(parse_pdf_pdfplumber(path), {"parser": "pdfplumber"})
    elif ext == ".docx":
        return ParsedDocument(parse_docx(path), {"parser": "docx"})
    elif ext == ".eml":
        return ParsedDocument(parse_email(path), {"parser": "tika"})
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
