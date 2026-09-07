"""One-time batch job: generates and caches `what_it_measures` for every
test_id in the RAG corpus, so reports never need an LLM call for that
field again. explain_all_test_results_batched() checks
local_data/what_it_measures_cache.json first (see llm_service._wim_get)
and only asks the LLM to fill in whatever fields it doesn't already have
cached for that test.

Usage (from repo root): backend/.venv/bin/python scripts/precompute_what_it_measures.py
Safe to re-run: skips any test_id already cached. Pass --force to regenerate.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "backend"))
# Settings.model_config points env_file=".env" relative to the CURRENT WORKING
# DIRECTORY, not this file's location, running this script from the repo root
# (rather than backend/) silently loaded zero provider keys and fell through
os.chdir(_ROOT / "backend")

from app.models.schemas import RangeStatus, TestResult  # noqa: E402
from app.services.llm_service import (  # noqa: E402
    _build_grounding, _verify_grounded, call_with_fallback,
)

_DATA_DIR = _ROOT / "data"
_CACHE_FILE = _ROOT / "local_data" / "what_it_measures_cache.json"

SYSTEM = (
    "You are an expert health literacy assistant. For the given lab test, "
    "explain in 2-3 sentences what it measures and what it does in the body, "
    "based ONLY on the facts in the reference context provided. Do not "
    "mention any specific patient value, range, or status, this is a "
    "general explanation reused across every report, not tied to one "
    "result. Write in your own words, do not copy the reference text "
    "verbatim. Respond ONLY with valid JSON: {\"what_it_measures\": \"...\"}"
)


def load_cache() -> dict:
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text())
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    _CACHE_FILE.parent.mkdir(exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def covered_test_ids() -> list[str]:
    chunks = json.loads((_DATA_DIR / "rag_chunks.json").read_text())
    ids = set()
    for c in chunks:
        ids.update(c["test_ids"])
    return sorted(ids)


def precompute_one(test_id: str) -> str | None:
    """Returns grounded what_it_measures text for test_id, or None if there's
    no usable reference context or the LLM's output fails verification."""
    # A synthetic TestResult, precomputing this field needs no real patient
    # value, only enough shape for _build_grounding()'s retrieval to work.
    stub = TestResult(
        test_id=test_id, raw_name=test_id.replace("_", " ").title(),
        value="", unit="", status=RangeStatus.UNKNOWN,
    )
    grounding = _build_grounding(stub)
    if not grounding.get("text"):
        return None

    prompt = (
        f"Test: {test_id}\n"
        f"Reference context: {grounding['text']}\n\n"
        'Return valid JSON: {"what_it_measures": "..."}'
    )
    try:
        data = call_with_fallback(system=SYSTEM, prompt=prompt, max_tokens=400, capability="text")
    except Exception as e:
        print(f"  FAILED (all providers): {e}")
        return None

    text = (data.get("what_it_measures") or "").strip() if isinstance(data, dict) else ""
    if not text:
        print("  FAILED: empty response")
        return None

    ok, reason = _verify_grounded(text, grounding["text"], allowed_numbers=set())
    if not ok:
        print(f"  REJECTED (ungrounded): {reason}")
        return None
    return text


def main() -> int:
    force = "--force" in sys.argv
    cache = {} if force else load_cache()
    ids = covered_test_ids()
    todo = ids if force else [t for t in ids if t not in cache]

    print(f"{len(ids)} test_ids covered by the corpus, {len(todo)} to precompute "
          f"({len(ids) - len(todo)} already cached)")

    done = 0
    for i, test_id in enumerate(todo, 1):
        print(f"[{i:>3}/{len(todo)}] {test_id}")
        text = precompute_one(test_id)
        if text:
            cache[test_id] = text
            done += 1
            if done % 10 == 0:
                save_cache(cache)  # periodic checkpoint, so a crash doesn't lose progress
        time.sleep(0.1)  # light pacing across providers, mirrors build_rag_chunks.py

    save_cache(cache)
    print(f"\nWrote {len(cache)} total entries ({done} new this run) to {_CACHE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
