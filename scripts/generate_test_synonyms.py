"""One-time batch job: asks an LLM for common lab-report names/
abbreviations for each RAG-covered test_id, so an unfamiliar phrasing
still resolves on the first try instead of waiting for a human to patch
ALIAS_MAP by hand.

This never touches what's shown to a patient, only reference_db.ALIAS_MAP,
the internal naming lookup used before any grounding or explanation
happens; the NHS UK / NIH MedlinePlus-only rule governs explanation
content, not this naming plumbing.

Two safety checks guard against conflating genuinely different tests that
share a name (Ferritin and Iron are related but not synonyms): a
candidate is dropped if it collides with an existing alias pointing to a
different test_id, and dropped if it was also proposed for a different
test_id within the same run, since an LLM proposing one name for two
tests is exactly the ambiguous case to refuse rather than guess on.

Usage (from repo root): backend/.venv/bin/python scripts/generate_test_synonyms.py
Safe to re-run: skips any test_id already covered in the output file.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "backend"))
os.chdir(_ROOT / "backend")  # see precompute_what_it_measures.py, .env loads relative to cwd

from app.services.llm_service import call_with_fallback  # noqa: E402
from app.services.reference_db import ALIAS_MAP  # noqa: E402

_DATA_DIR = _ROOT / "data"
_OUT_FILE = _DATA_DIR / "llm_generated_aliases.json"

SYSTEM = (
    "You identify alternate real-world names for a laboratory test, as they "
    "would literally appear as a field label on a lab report. Only list "
    "names/abbreviations that refer to the EXACT SAME test, never a "
    "related-but-different test, a component of a panel, or a different "
    "measurement unit of the same analyte. If you are not certain something "
    "is a true synonym, omit it. Respond ONLY with valid JSON: "
    '{"synonyms": ["...", "..."]}'
)


def _alias_key(raw_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw_name.strip().lower()).strip("_")


def covered_test_topics() -> dict:
    """test_id -> set of page topics (titles) already known for it."""
    chunks = json.loads((_DATA_DIR / "rag_chunks.json").read_text())
    topics = defaultdict(set)
    for c in chunks:
        for tid in c["test_ids"]:
            topics[tid].add(c["topic"])
    return topics


def generate_synonyms(test_id: str, topics: set) -> list:
    topic_str = "; ".join(sorted(topics))
    prompt = (
        f"Canonical test: {test_id}\n"
        f"Known page title(s) for this test: {topic_str}\n\n"
        "List other names, abbreviations, or lab-report field labels a "
        "diagnostic lab might print for this EXACT same test.\n"
        'Return valid JSON: {"synonyms": ["...", "..."]}'
    )
    try:
        data = call_with_fallback(system=SYSTEM, prompt=prompt, max_tokens=300, capability="text")
    except Exception as e:
        print(f"  FAILED: {e}")
        return []
    syns = data.get("synonyms", []) if isinstance(data, dict) else []
    return [s for s in syns if isinstance(s, str) and s.strip()]


def load_existing() -> dict:
    if not _OUT_FILE.exists():
        return {}
    try:
        return json.loads(_OUT_FILE.read_text())
    except Exception:
        return {}


def save(candidates_by_id: dict) -> None:
    _OUT_FILE.write_text(json.dumps(candidates_by_id, indent=2, sort_keys=True))


def main() -> int:
    force = "--force" in sys.argv
    topics_by_id = covered_test_topics()
    existing = {} if force else load_existing()
    todo = sorted(t for t in topics_by_id if t not in existing)

    print(f"{len(topics_by_id)} covered test_ids, {len(todo)} to process "
          f"({len(existing)} already done)")

    # Raw synonym candidates, before cross-test collision filtering.
    raw_candidates: dict = dict(existing)  # test_id -> [names]
    for i, test_id in enumerate(todo, 1):
        print(f"[{i:>3}/{len(todo)}] {test_id}")
        syns = generate_synonyms(test_id, topics_by_id[test_id])
        raw_candidates[test_id] = syns
        if i % 10 == 0:
            save(raw_candidates)
        time.sleep(0.1)
    save(raw_candidates)

    # --- Safety pass: resolve which alias_key -> test_id assignments are UNAMBIGUOUS ---
    key_to_ids = defaultdict(set)
    for test_id, names in raw_candidates.items():
        for name in names:
            key = _alias_key(name)
            if key:
                key_to_ids[key].add(test_id)

    accepted: dict = {}
    dropped_ambiguous = []
    dropped_collision = []
    for key, ids in key_to_ids.items():
        if len(ids) > 1:
            dropped_ambiguous.append((key, sorted(ids)))
            continue
        test_id = next(iter(ids))
        existing_alias = ALIAS_MAP.get(key)
        if existing_alias and existing_alias != test_id:
            dropped_collision.append((key, existing_alias, test_id))
            continue
        accepted[key] = test_id

    print(f"\n{len(accepted)} aliases accepted")
    print(f"{len(dropped_ambiguous)} dropped, same name proposed for multiple different tests:")
    for key, ids in dropped_ambiguous[:20]:
        print(f"   {key!r} -> {ids}")
    print(f"{len(dropped_collision)} dropped, collided with an existing DIFFERENT alias:")
    for key, existing_id, proposed_id in dropped_collision[:20]:
        print(f"   {key!r}: existing={existing_id!r} vs proposed={proposed_id!r}")

    accepted_file = _DATA_DIR / "llm_generated_aliases_accepted.json"
    accepted_file.write_text(json.dumps(accepted, indent=2, sort_keys=True))
    print(f"\nWrote {len(accepted)} accepted aliases to {accepted_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
