"""FAISS-based semantic retrieval over the NHS UK / NIH MedlinePlus corpus
built by data/build_rag_chunks.py + scripts/build_rag_index.py.

BUG FOUND: _DATA_DIR resolved to a path that doesn't exist, so _load()
returned early and retrieve_context() silently returned [] for every
query ever made. The path is now derived the same way reference_db.py
derives it, and index_health() exposes the load state instead of failing
quietly. Scores are also cosine similarity now, an interpretable 0-1
number rather than raw L2 distance, and retrieval is test_id-aware: a
page covering the analyte being explained gets a ranking bonus so a name
retrieves its own page rather than a semantically similar one about a
different analyte.
"""
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# The only sources this system is permitted to ground explanations in.
ALLOWED_SOURCE_DOMAINS = {"www.nhs.uk", "nhs.uk", "medlineplus.gov"}

# IMPORT ORDER IS LOAD-BEARING, do not let an import sorter reshuffle these.
# faiss-cpu and torch each ship their own OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402  (must precede faiss)
from sentence_transformers import SentenceTransformer  # noqa: E402
import faiss  # noqa: E402

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

EMBED_MODEL = "all-MiniLM-L6-v2"

# Cosine-similarity floor.

# internal values (P2 Peak / P3 Peak from an HbA1c HPLC assay, not real
# NHS/NIH-covered tests) match onto the *nearest available* chunk in the
# corpus - Prostate- Specific Antigen content, at 0.45-0.48 similarity - which
MIN_SIMILARITY = 0.55

# Ranking bonus for a chunk scraped from a page that explicitly covers this
# test_id. Large enough to promote the right page above a generically similar
# one, small enough that a strong semantic match still wins on its own.
TEST_ID_BONUS = 0.15

_MODEL = None
_INDEX = None
_CHUNKS = None
_LOAD_ERROR: Optional[str] = None
_TAG_INDEX: Optional[Dict[str, list]] = None
_LOADED = False
_LOCK = threading.Lock()


def _load() -> None:
    """Loads model + index once. Records why it failed rather than leaving the
    caller to guess from empty results forever."""
    global _MODEL, _INDEX, _CHUNKS, _LOAD_ERROR, _LOADED

    if _LOADED:
        return

    with _LOCK:
        if _LOADED:
            return
        try:
            index_path = _DATA_DIR / "rag_index.faiss"
            chunks_path = _DATA_DIR / "rag_chunks.json"

            if not index_path.exists() or not chunks_path.exists():
                _LOAD_ERROR = (
                    f"RAG index not built: expected {index_path} and {chunks_path}. "
                    "Run `python data/build_rag_chunks.py` then "
                    "`python scripts/build_rag_index.py` from the repo root."
                )
                return

            _CHUNKS = json.loads(chunks_path.read_text())

            # Project constraint: explanations are grounded EXCLUSIVELY in NHS
            # UK and NIH MedlinePlus.
            foreign = sorted({
                urlparse(c.get("url", "")).netloc
                for c in _CHUNKS
                if urlparse(c.get("url", "")).netloc not in ALLOWED_SOURCE_DOMAINS
            })
            if foreign:
                _LOAD_ERROR = (
                    f"RAG corpus contains non-approved source domains: {foreign}. "
                    f"Only {sorted(ALLOWED_SOURCE_DOMAINS)} are permitted."
                )
                _CHUNKS = None
                return

            _INDEX = faiss.read_index(str(index_path))

            if _INDEX.ntotal != len(_CHUNKS):
                # A stale index paired with a rebuilt corpus returns text for
                # the wrong test, refuse rather than ground on mismatched IDs.
                _LOAD_ERROR = (
                    f"RAG index is stale: {_INDEX.ntotal} vectors vs "
                    f"{len(_CHUNKS)} chunks. Re-run scripts/build_rag_index.py."
                )
                _INDEX = None
                return

            torch.set_default_dtype(torch.float32)
            _MODEL = SentenceTransformer(EMBED_MODEL, device="cpu")
            _LOAD_ERROR = None
        except Exception as e:
            _LOAD_ERROR = f"{type(e).__name__}: {e}"
            _INDEX = None
        finally:
            _LOADED = True

    if _LOAD_ERROR:
        print(f"RAG unavailable, {_LOAD_ERROR}")


def index_health() -> dict[str, Any]:
    """Load state for the /rag/health endpoint and startup diagnostics."""
    _load()
    return {
        "available": _INDEX is not None and bool(_CHUNKS),
        "chunk_count": len(_CHUNKS) if _CHUNKS else 0,
        "vector_count": _INDEX.ntotal if _INDEX is not None else 0,
        "data_dir": str(_DATA_DIR),
        "embed_model": EMBED_MODEL,
        "min_similarity": MIN_SIMILARITY,
        "error": _LOAD_ERROR,
    }


# Words that appear in almost every lab-reference page title and so prove
# nothing about topical relatedness ("Blood Glucose Test" vs "Blood Urea
# Nitrogen" share "blood" without being related).

# no NHS/NIH page discusses urine specimen volume - passed this guard anyway
# and grounded on "Blood in Urine", an unrelated topic, because "urine" is the
# only word the two share and it wasn't excluded here the way "blood" already
_ANCHOR_STOPWORDS = frozenset({
    "test", "tests", "testing", "blood", "level", "levels", "count", "counts",
    "panel", "result", "results", "screening", "screen", "serum", "plasma",
    "total", "normal", "range", "ranges", "what", "mean", "means", "your",
    "the", "and", "for", "with", "this", "that", "measure", "measures",
    "urine", "urinary",
})


def _chunks_tagged(test_id: str) -> list:
    """Indices of every chunk explicitly curated as covering `test_id`.

    Built lazily once, so retrieve_context can pull a correctly-tagged page in
    by id without paying a full scan of the corpus on every call.
    """
    global _TAG_INDEX
    if _TAG_INDEX is None:
        index: Dict[str, list] = {}
        for i, chunk in enumerate(_CHUNKS or []):
            for tag in (chunk.get("test_ids") or []):
                index.setdefault(tag, []).append(i)
        _TAG_INDEX = index
    return _TAG_INDEX.get(test_id, [])


def _anchor_tokens(text: str) -> set:
    """Lowercased words of >=3 letters, crudely de-pluralised, minus the
    generic lab vocabulary above."""
    words = re.findall(r"[A-Za-z]{3,}", (text or "").lower())
    return {w[:-1] if w.endswith("s") else w for w in words} - _ANCHOR_STOPWORDS


def _has_lexical_anchor(test_name: str, test_id: Optional[str], chunk: dict) -> bool:
    """Requires a passage to be tied to the test by something other than
    raw embedding proximity: either the page is explicitly tagged with
    this test_id, or the test's name shares a real word with the page's
    title.

    BUG FOUND: for a name with little semantic signal of its own ("Colour",
    "pH"), the query's boilerplate suffix dominated the embedding, so it
    landed near whichever generic lab page was closest and cleared the
    similarity floor on that alone, e.g. "pH" matching an unrelated
    Alkaline Phosphatase page at a plausible-looking score. A higher floor
    can't fix this since genuine and spurious matches score similarly; what
    separates them is lexical overlap with the page title, which this
    checks directly. The test_id clause keeps aliased tests working where
    the printed name legitimately differs from the page title.
    """
    if test_id and test_id in (chunk.get("test_ids") or []):
        return True
    name_tokens = _anchor_tokens(test_name)
    if not name_tokens:
        return False
    page_tokens = _anchor_tokens(f"{chunk.get('topic', '')} {chunk.get('section', '')}")
    if name_tokens & page_tokens:
        return True
    # Title alone is too strict for analytes whose reference page is filed
    # under a broader name: MCH/MCHC live on the MCV page, VLDL and the
    # cholesterol ratios on "Cholesterol Levels", Hb A/Hb A2 on "Hemoglobin
    # Test".
    return bool(name_tokens & _anchor_tokens(chunk.get("text", "")))


def retrieve_context(
    test_name: str,
    k: int = 3,
    test_id: Optional[str] = None,
    min_similarity: float = MIN_SIMILARITY,
) -> list[dict]:
    """Retrieves the k most relevant reference passages for a test.

    Each returned chunk carries `text`, `source`, `url`, `topic`, `section`
    and the `similarity` it matched at, so downstream prompts can cite the
    real source per passage rather than a generic label.
    """
    _load()
    if _INDEX is None or not _CHUNKS or not test_name:
        return []

    query = test_name if not test_id else f"{test_name} ({test_id}) blood test result meaning"

    query_vector = _MODEL.encode(
        [query], normalize_embeddings=True
    ).astype("float32")

    # Over-fetch, then re-rank with the test_id bonus before applying the
    # cutoff, the right page may not be in the raw top-k on name alone.
    fetch_k = min(max(k * 5, 15), _INDEX.ntotal)
    similarities, indices = _INDEX.search(query_vector, fetch_k)

    candidates = list(zip(similarities[0], indices[0]))

    # Vector rank alone can miss a page that is EXPLICITLY curated for this
    # analyte: the CBC page is tagged MCH/MCHC but embeds at only 0.233
    # against the query "MCH", so it sat far outside the top fetch_k and
    # MCH/MCHC were reported as uncovered.
    if test_id:
        seen_idx = {int(i) for _, i in candidates if i != -1}
        for extra_idx in _chunks_tagged(test_id):
            if extra_idx in seen_idx:
                continue
            try:
                vector = _INDEX.reconstruct(extra_idx)
            except Exception:
                continue
            candidates.append((float(query_vector[0] @ vector), extra_idx))

    scored = []
    for similarity, idx in candidates:
        if idx == -1:
            continue
        chunk = _CHUNKS[idx]
        score = float(similarity)
        tagged = bool(test_id) and test_id in (chunk.get("test_ids") or [])
        if tagged:
            score += TEST_ID_BONUS
        # A page EXPLICITLY curated in build_rag_chunks.py as covering this
        # analyte is relevant by editorial decision, which is better evidence
        # than embedding proximity, so the similarity floor does not apply to
        if score < min_similarity and not tagged:
            continue
        scored.append((score, float(similarity), chunk))

    scored.sort(key=lambda row: row[0], reverse=True)

    scored = [row for row in scored if _has_lexical_anchor(test_name, test_id, row[2])]

    results = []
    seen_sections = set()
    for score, raw_similarity, chunk in scored:
        # One passage per (topic, section): near-duplicate sections from
        # overlapping pages would otherwise crowd out a second perspective.
        key = (chunk.get("topic"), chunk.get("section"))
        if key in seen_sections:
            continue
        seen_sections.add(key)
        results.append({**chunk, "similarity": round(raw_similarity, 4),
                        "score": round(score, 4)})
        if len(results) >= k:
            break

    return results
