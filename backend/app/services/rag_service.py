import os
import json
import faiss
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

_MODEL = None
_INDEX = None
_CHUNKS = None


def _load():
    global _MODEL, _INDEX, _CHUNKS
    if _MODEL is None:
        torch.set_default_dtype(torch.float32)
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        index_path = _DATA_DIR / "rag_index.faiss"
        chunks_path = _DATA_DIR / "rag_chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            return

        _INDEX = faiss.read_index(str(index_path))
        _CHUNKS = json.loads(chunks_path.read_text())


def retrieve_context(query: str, k: int = 3) -> list[dict]:
    try:
        _load()
        if _MODEL is None or _INDEX is None or _CHUNKS is None:
            return []
            
        query_vec = _MODEL.encode([query])
        distances, indices = _INDEX.search(np.array(query_vec, dtype="float32"), k)
        return [_CHUNKS[i] for i in indices[0] if 0 <= i < len(_CHUNKS)]
    except Exception as e:
        print(f"RAG Retrieval warning: {e}")
        return []