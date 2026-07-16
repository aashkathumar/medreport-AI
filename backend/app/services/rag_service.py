"""
Lightweight local RAG retrieval using FAISS.
Requires data/rag_chunks.json and data/rag_index.faiss to already exist --
build them with data/build_rag_chunks.py then scripts/build_rag_index.py.
"""
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

_MODEL = None
_INDEX = None
_CHUNKS = None


def _load():
    global _MODEL, _INDEX, _CHUNKS
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _INDEX = faiss.read_index(str(_DATA_DIR / "rag_index.faiss"))
        _CHUNKS = json.loads((_DATA_DIR / "rag_chunks.json").read_text())


def retrieve_context(query: str, k: int = 3) -> list[dict]:
    _load()
    query_vec = _MODEL.encode([query])
    distances, indices = _INDEX.search(np.array(query_vec, dtype="float32"), k)
    return [_CHUNKS[i] for i in indices[0] if 0 <= i < len(_CHUNKS)]
