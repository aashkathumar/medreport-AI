"""
Provider comparison benchmark.

Runs the SAME fixed clinical cases through each LLM provider, so any
difference in the output is attributable to the model rather than the input.
Measures latency, readability, and (new) whether the explanation actually
passed source verification against its NHS/NIH grounding.

Usage (from backend/):
    python evaluation/compare_providers.py
    python evaluation/plot_rename.py        # summary table + chart

CHANGED: this imported explain_test_result(), which no longer exists -- the
per-test call was replaced by explain_all_test_results_batched(). The script
raised ImportError at startup, so the benchmark could not run at all. It now
calls the batched function with a single test per call, which keeps the
per-(test, provider) timing the original design depends on.

Two columns are new:
  * grounded          -- did the explanation pass post-generation source
                         verification? An unverified explanation is replaced
                         by a GP referral in production, so a provider that
                         is fast and readable but frequently ungrounded is
                         NOT the better provider. Without this column the
                         benchmark would rank providers on fluency alone.
  * source            -- which NHS/NIH page(s) grounded the explanation.
"""
import csv
import sys
import time
from pathlib import Path

import textstat

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import RangeStatus, TestResult, UserProfile
from app.services.llm_providers import provider_has_key
from app.services.llm_service import explain_all_test_results_batched
from app.services.reference_db import flag_result, get_normal_range

# A small fixed set of test cases -- same inputs across all providers,
# so any difference in output is attributable to the model, not the data.
TEST_CASES = [
    {"test_id": "HGB", "raw_name": "Haemoglobin", "value": 10.2, "unit": "g/dL"},
    {"test_id": "HBA1C", "raw_name": "HbA1c", "value": 58.0, "unit": "mmol/mol"},
    {"test_id": "FERRITIN", "raw_name": "Ferritin", "value": 9.0, "unit": "ng/mL"},
    {"test_id": "TSH", "raw_name": "TSH", "value": 5.8, "unit": "mIU/L"},
]

PROFILE = UserProfile(user_id="eval_user", age=35, sex="female", diet_type="vegetarian")

# Every provider registered in llm_providers. Ones without an API key are
# skipped with a note rather than producing a row of error text -- the old
# run wasted 4 of its 12 rows on Gemini 404s.
PROVIDERS_TO_TEST = ["gemini", "groq_llama", "mistral", "nvidia", "openrouter"]

FIELDNAMES = [
    "test",
    "provider",
    "status",
    "grounded",
    "source",
    "what_it_measures",
    "what_your_result_means",
    "lifestyle_suggestions",
    "gp_question",
    "response_time_sec",
    "error",
    "fk_grade_what_it_measures",
    "fk_grade_what_result_means",
]


def _probe_provider_error(provider: str) -> str:
    """Calls the provider directly with a trivial prompt to recover the real
    failure message, which the batched explanation path absorbs."""
    from app.services.llm_providers import PROVIDERS

    fn = PROVIDERS.get(provider)
    if not fn:
        return f"unknown provider {provider!r}"
    try:
        fn("Return strict JSON only.",
           'Reply with a JSON object having key "ok" set to boolean true.',
           max_tokens=50, model=None)
        return ""   # provider is reachable; the failure was elsewhere
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _fk_grade(text: str):
    """Flesch-Kincaid grade, or "" for text too short to score meaningfully."""
    if not text or len(text.split()) < 5:
        return ""
    return round(textstat.flesch_kincaid_grade(text), 2)


def run_comparison():
    rows = []
    skipped = []

    for provider in PROVIDERS_TO_TEST:
        if not provider_has_key(provider):
            skipped.append(provider)
            continue

        for case in TEST_CASES:
            normal_range = get_normal_range(case["test_id"], sex=PROFILE.sex)
            status = flag_result(case["value"], normal_range)
            test = TestResult(
                test_id=case["test_id"],
                raw_name=case["raw_name"],
                value=case["value"],
                unit=case["unit"],
                status=status,
                normal_range_min=normal_range["min"] if normal_range else None,
                normal_range_max=normal_range["max"] if normal_range else None,
            )

            print(f"Running {case['raw_name']} through {provider}...")
            start = time.time()
            try:
                explained, degraded, ungrounded = explain_all_test_results_batched(
                    [test], PROFILE, provider=provider
                )
                elapsed = round(time.time() - start, 2)

                if not explained:
                    raise RuntimeError("no explanation returned")

                result = explained[0]
                # A provider that failed outright is recorded as an error, not
                # as a fast row with placeholder text -- otherwise a dead
                # provider looks like the fastest one in the chart.
                if result.test_id in degraded:
                    raise RuntimeError("all providers in chain failed for this call")

                rows.append({
                    "test": case["raw_name"],
                    "provider": provider,
                    "status": status.value,
                    "grounded": result.test_id not in ungrounded,
                    "source": result.source,
                    "what_it_measures": result.what_it_measures,
                    "what_your_result_means": result.what_your_result_means,
                    "lifestyle_suggestions": " | ".join(result.lifestyle_suggestions),
                    "gp_question": result.gp_question,
                    "response_time_sec": elapsed,
                    "error": "",
                    "fk_grade_what_it_measures": _fk_grade(result.what_it_measures),
                    "fk_grade_what_result_means": _fk_grade(result.what_your_result_means),
                })
            except Exception as e:
                # explain_all_test_results_batched() handles provider failures
                # internally, so all this level sees is "chain failed". For a
                # benchmark the actual cause is the interesting part -- "model
                # retired", "quota exhausted" and "library incompatibility" are
                # very different findings -- so probe the provider directly to
                # recover the real error text.
                reason = _probe_provider_error(provider) or f"{type(e).__name__}: {e}"
                rows.append({
                    "test": case["raw_name"],
                    "provider": provider,
                    "status": status.value,
                    "grounded": "",
                    "source": "",
                    "what_it_measures": "",
                    "what_your_result_means": "",
                    "lifestyle_suggestions": "",
                    "gp_question": "",
                    "response_time_sec": "",
                    "error": reason,
                    "fk_grade_what_it_measures": "",
                    "fk_grade_what_result_means": "",
                })

    out_path = Path(__file__).parent / "comparison_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    print(f"\nDone -- {len(rows)} rows written to {out_path}")
    print(f"  succeeded: {len(ok)}   failed: {len(rows) - len(ok)}")
    if skipped:
        print(f"  skipped (no API key configured): {', '.join(skipped)}")
    print("\nNext: python evaluation/plot_rename.py")


if __name__ == "__main__":
    run_comparison()
