"""
One-time batch job: generates and caches ALL THREE test-general fields
(what_it_measures, lifestyle_suggestions, gp_question) for every test_id
covered by the RAG corpus, writing directly into the same exact-match
cache (local_data/explanation_cache.json, keyed (test_id, specimen)) that
_cache_get()/_explain_batch() already check first, before any LLM call.

Why this exists alongside precompute_what_it_measures.py: that script only
ever covers ONE of the four fields. _EXPLANATION_CACHE already covers all
four -- keyed on (test_id, specimen) alone in this branch, since nothing
generated here depends on the patient's specific value (see _cache_key's
docstring) -- but it is only ever populated REACTIVELY, the first time a
real report happens to mention a given test. This script does the same
work proactively, for the whole known-covered set, so the FIRST real
report to ever mention any of these ~407 tests gets a cache hit (zero LLM
calls, zero chance of an empty/degraded field) instead of needing one
live "warm-up" generation per test_id first.

Usage (from repo root):
    backend/.venv/bin/python scripts/precompute_full_explanations.py

Safe to re-run: skips any (test_id, None) key already in the cache, so a
partial run (rate limit, interrupted) just picks up where it left off.
Pass --force to regenerate every entry from scratch.
"""
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "backend"))
# Same cwd fix as precompute_what_it_measures.py -- Settings.model_config
# loads .env relative to the CURRENT working directory, not this file's
# location, so running from the repo root would silently load zero
# provider keys otherwise.
os.chdir(_ROOT / "backend")

from app.models.schemas import RangeStatus, TestResult  # noqa: E402
from app.services.llm_service import (  # noqa: E402
    _build_grounding, _verify_grounded, _cache_key, call_with_fallback,
)

_DATA_DIR = _ROOT / "data"
_CACHE_FILE = _ROOT / "local_data" / "explanation_cache.json"

# Mirrors llm_service.SYSTEM's instructions exactly, so precomputed text is
# held to the identical standard as a live batch call would be -- this is
# not a looser or stricter prompt, just the same one issued once per test_id
# instead of once per report.
SYSTEM = """You are an expert health literacy assistant helping patients understand what their laboratory tests are and how to support their health, based on NHS UK and NIH MedlinePlus standards.

For the given test, write GENERAL, test-level guidance -- never referencing any specific patient value, whether it is high/low/normal, or any reference range:
1. what_it_measures: Explain clearly what the biomarker does in the body.
2. lifestyle_suggestions: Provide 2-3 evidence-based dietary, hydration, or activity habits that generally support keeping this measurement in a healthy range, grounded in NHS/NIH guidance -- phrased as good general practice for anyone, not as a response to any particular result.
3. gp_question: Formulate a general, constructive question a patient could ask their GP about this test.

STRICT RULES:
- Do NOT mention, infer, or imply any specific value, or whether it is elevated, low, normal, or abnormal, anywhere in your answer.
- Base every explanation ONLY on the facts given in the reference context provided. Do NOT introduce any medical fact, cause, condition, or figure not present there.
- Write in your own words - paraphrase the reference context, do not copy its sentences verbatim.
- Never diagnose, prescribe, or make clinical recommendations beyond general lifestyle guidance.

Respond ONLY with valid JSON."""


def load_cache() -> dict:
    """Returns {(test_id, specimen): details_dict}, reassembled from the
    persisted [{"key": [...], "value": {...}}] list format."""
    if not _CACHE_FILE.exists():
        return {}
    try:
        entries = json.loads(_CACHE_FILE.read_text())
        return {tuple(e["key"]): e["value"] for e in entries}
    except Exception as e:
        print(f"Could not load existing cache, starting fresh: {e}")
        return {}


def save_cache(cache: dict) -> None:
    _CACHE_FILE.parent.mkdir(exist_ok=True)
    entries = [{"key": list(k), "value": v} for k, v in cache.items()]
    _CACHE_FILE.write_text(json.dumps(entries))


def covered_test_ids() -> list[str]:
    chunks = json.loads((_DATA_DIR / "rag_chunks.json").read_text())
    ids = set()
    for c in chunks:
        ids.update(c["test_ids"])
    return sorted(ids)


def precompute_one(test_id: str) -> dict | None:
    """Returns a details dict ({what_it_measures, lifestyle_suggestions,
    gp_question, disclaimer}) for test_id, or None if there's no usable
    reference context, the call fails, or verification rejects the result."""
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
        'Return valid JSON: {"what_it_measures": "...", '
        '"lifestyle_suggestions": ["...", "..."], "gp_question": "..."}'
    )
    try:
        data = call_with_fallback(system=SYSTEM, prompt=prompt, max_tokens=700, capability="text")
    except Exception as e:
        print(f"  FAILED (all providers): {e}")
        return None

    if not isinstance(data, dict):
        print("  FAILED: non-dict response")
        return None

    wim = (data.get("what_it_measures") or "").strip()
    gp_q = (data.get("gp_question") or "").strip()
    suggestions = data.get("lifestyle_suggestions") or []
    if not isinstance(suggestions, list):
        suggestions = []
    suggestions = [s.strip() for s in suggestions if isinstance(s, str) and s.strip()]

    if not wim or not gp_q or not suggestions:
        print(f"  FAILED: missing field(s) (wim={bool(wim)}, gp_q={bool(gp_q)}, suggestions={len(suggestions)})")
        return None

    # Verify EACH free-text field independently against the same grounding
    # context, the same check a live batch call's output goes through --
    # no reason for a precomputed entry to be held to a looser standard.
    for label, text in (("what_it_measures", wim), ("gp_question", gp_q)):
        ok, reason = _verify_grounded(text, grounding["text"], allowed_numbers=set())
        if not ok:
            print(f"  REJECTED ({label}): {reason}")
            return None
    joined_suggestions = " ".join(suggestions)
    ok, reason = _verify_grounded(joined_suggestions, grounding["text"], allowed_numbers=set())
    if not ok:
        print(f"  REJECTED (lifestyle_suggestions): {reason}")
        return None

    return {
        "what_it_measures": wim,
        "lifestyle_suggestions": suggestions,
        "gp_question": gp_q,
        "disclaimer": "Educational information only. Consult your GP.",
    }


def main() -> int:
    force = "--force" in sys.argv
    cache = {} if force else load_cache()
    ids = covered_test_ids()
    # specimen=None matches the default TestResult.specimen used for the
    # vast majority of tests; specimen-redirected variants (e.g. urine
    # glucose vs blood glucose, both aliasing to GLUCOSE) still fall back to
    # a live call the first time, same as today -- a deliberately smaller
    # scope than trying to enumerate every specimen variant up front.
    todo = [t for t in ids if (t, None) not in cache]

    print(f"{len(ids)} test_ids covered by the corpus, {len(todo)} to precompute "
          f"({len(ids) - len(todo)} already cached)")

    done = 0
    for i, test_id in enumerate(todo, 1):
        print(f"[{i:>3}/{len(todo)}] {test_id}")
        details = precompute_one(test_id)
        if details:
            cache[(test_id, None)] = details
            done += 1
            if done % 10 == 0:
                save_cache(cache)
        time.sleep(0.1)

    save_cache(cache)
    print(f"\nWrote {len(cache)} total entries ({done} new this run) to {_CACHE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
