"""
Builds the FAISS vector index used for RAG retrieval.
Run this ONCE (and again any time data/rag_chunks.json changes),
AFTER running data/build_rag_chunks.py.

Usage (from repo root):
    python scripts/build_rag_index.py

First run downloads the embedding model (~90MB) from Hugging Face Hub
automatically -- no manual download needed, just an internet connection.
"""
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

_DATA_DIR = Path(__file__).parent.parent / "data"


def build_index():
    chunks_path = _DATA_DIR / "rag_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            "data/rag_chunks.json not found. Run data/build_rag_chunks.py first."
        )

    chunks = json.loads(chunks_path.read_text())
    texts = [c["text"] for c in chunks]

    print("Loading embedding model (first run downloads ~90MB from Hugging Face)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, str(_DATA_DIR / "rag_index.faiss"))
    print(f"Index built with {index.ntotal} vectors -> data/rag_index.faiss")


if __name__ == "__main__":
    build_index()
