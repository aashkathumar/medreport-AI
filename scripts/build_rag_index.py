"""Builds the FAISS vector index used for RAG retrieval. Run this once,
and again any time data/rag_chunks.json changes, after
data/build_rag_chunks.py.

Usage (from repo root): python scripts/build_rag_index.py
First run downloads the embedding model (~90MB) automatically.

BUG FOUND: the index used to be raw, un-normalised embeddings filtered
on an uninterpretable L2-distance threshold. Embeddings are now
L2-normalised in an inner-product index, so a search score is cosine
similarity in [-1, 1], a number that can actually be reasoned about, and
the metric this embedding model is trained for. Also writes
rag_index_meta.json so rag_service can verify the index and the chunks
file are in sync, since a stale index paired with a rebuilt corpus would
otherwise return text for the wrong test.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_DATA_DIR = Path(__file__).parent.parent / "data"

EMBED_MODEL = "all-MiniLM-L6-v2"


def build_index():
    chunks_path = _DATA_DIR / "rag_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            "data/rag_chunks.json not found. Run data/build_rag_chunks.py first."
        )

    chunks = json.loads(chunks_path.read_text())
    if not chunks:
        raise ValueError("data/rag_chunks.json is empty, re-run data/build_rag_chunks.py.")

    # Embed topic + section + body (see "embed_text" in build_rag_chunks.py):
    # a passage like "What do the results mean?" is not retrievable on its own
    # without the topic it belongs to.
    texts = [c.get("embed_text") or c["text"] for c in chunks]

    print(f"Loading embedding model {EMBED_MODEL} (first run downloads ~90MB)...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via inner product
        batch_size=32,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(_DATA_DIR / "rag_index.faiss"))

    meta = {
        "embed_model": EMBED_MODEL,
        "metric": "cosine",           # normalised vectors + inner product
        "dimensions": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
    }
    (_DATA_DIR / "rag_index_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Index built with {index.ntotal} vectors -> data/rag_index.faiss")
    print(f"Metadata -> data/rag_index_meta.json ({meta['dimensions']}d, cosine)")


if __name__ == "__main__":
    build_index()
