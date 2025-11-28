"""
BM25-based document retrieval for RAG.
"""

from pathlib import Path
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi


# Global BM25 index
_bm25_index = None
_chunks = []
_chunk_ids = []


def _load_and_chunk_documents(docs_dir: str = "docs/"):
    """Load documents and split into chunks."""
    global _chunks, _chunk_ids
    
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return
    
    for doc_path in docs_path.glob("*.md"):
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by paragraphs or sections
        chunks = _split_into_chunks(content)
        
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                chunk_id = f"{doc_path.stem}::chunk{i}"
                _chunks.append(chunk_text.strip())
                _chunk_ids.append(chunk_id)


def _split_into_chunks(text: str, chunk_size: int = 300) -> List[str]:
    """Split text into chunks by paragraphs or size."""
    # First try to split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    # Split by whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def build_bm25(docs_dir: str = "docs/"):
    """Build BM25 index from documents."""
    global _bm25_index, _chunks, _chunk_ids
    
    _load_and_chunk_documents(docs_dir)
    
    if not _chunks:
        print("⚠️ No documents found to index")
        return
    
    # Tokenize chunks
    tokenized_chunks = [_tokenize(chunk) for chunk in _chunks]
    
    # Build BM25 index
    _bm25_index = BM25Okapi(tokenized_chunks)
    print(f"✓ BM25 index built with {len(_chunks)} chunks")


def retrieve(query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
    """
    Retrieve top-k relevant chunks using BM25.
    
    Returns:
        Tuple of (chunks, chunk_ids)
    """
    global _bm25_index, _chunks, _chunk_ids
    
    if _bm25_index is None or not _chunks:
        return [], []
    
    # Tokenize query
    query_tokens = _tokenize(query)
    
    # Get BM25 scores
    scores = _bm25_index.get_scores(query_tokens)
    
    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    # Filter out zero scores
    results_chunks = []
    results_ids = []
    
    for idx in top_indices:
        if scores[idx] > 0:
            results_chunks.append(_chunks[idx])
            results_ids.append(_chunk_ids[idx])
    
    return results_chunks, results_ids
