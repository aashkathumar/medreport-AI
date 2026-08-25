"""
Regression tests for the RAG grounding + provider fallback defects.

Every test here corresponds to a bug that was live and silent -- each one
failed by producing plausible-looking output rather than an error, which is
why none of them were caught by the existing suite.

Run (from backend/, no API keys and no network needed):
    python tests/test_rag_and_fallback.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.schemas import RangeStatus, TestResult
from app.services.llm_providers import PROVIDERS, supports_vision
from app.services.reference_db import get_normal_range, resolve_test_id


def test_fallback_chains_exist():
    """call_with_fallback() read settings.text_fallback_chain, which was never
    defined -- so every default-path LLM call raised AttributeError before
    reaching a model, and the callers swallowed it."""
    assert isinstance(settings.text_fallback_chain, list)
    assert settings.text_fallback_chain, "text fallback chain must not be empty"
    assert isinstance(settings.vision_fallback_chain, list)
    assert settings.vision_fallback_chain, "vision fallback chain must not be empty"

    for entry in settings.text_fallback_chain + settings.vision_fallback_chain:
        assert entry.split(":", 1)[0] in PROVIDERS, f"unknown provider in chain: {entry}"


def test_vision_chain_only_contains_vision_providers():
    """groq_llama and mistral raise on image payloads by design. Routing a
    vision request to them is a guaranteed failure, not a fallback."""
    for entry in settings.vision_fallback_chain:
        provider = entry.split(":", 1)[0]
        assert supports_vision(provider), (
            f"{provider} is in vision_fallback_chain but cannot accept images"
        )


def test_vision_chain_rejects_text_only_provider():
    from app.services.llm_service import LLMUnavailableError, _build_chain

    try:
        _build_chain("vision", "groq_llama", None)
    except LLMUnavailableError:
        return
    raise AssertionError("a text-only provider must be refused for vision work")


def test_multiword_alias_resolution():
    """ALIAS_MAP is keyed with underscores but resolve_test_id() looked up the
    raw spaced name, so ONLY single-word names ever resolved. These 7 tests
    silently lost their reference range and their curated grounding."""
    cases = {
        "Haemoglobin": "HGB",                      # worked before (single word)
        "White Blood Cell Count": "WBC",           # all of the following did not
        "Total Cholesterol": "CHOL",
        "LDL Cholesterol": "LDL",
        "Vitamin D": "VIT_D",
        "Vitamin B12": "B12",
        "Blood Glucose": "GLUCOSE",
        "Urine Protein": "URINE_PROTEIN",
    }
    for raw_name, expected in cases.items():
        assert resolve_test_id(raw_name) == expected, f"{raw_name!r} did not resolve"
        assert get_normal_range(expected, sex="female", age=35), (
            f"{expected} resolved but has no normal range"
        )


def test_alias_resolution_strips_sample_qualifiers():
    assert resolve_test_id("Creatinine, Serum") == "CREATININE"
    assert resolve_test_id("Glucose (Plasma)") == "GLUCOSE"


def test_rag_index_loads():
    """_DATA_DIR pointed at backend/data, which does not exist -- retrieval
    returned [] for every query ever made, permanently and silently."""
    from app.services.rag_service import index_health

    health = index_health()
    assert health["available"], (
        f"RAG index unavailable: {health['error']}. "
        "Run data/build_rag_chunks.py then scripts/build_rag_index.py."
    )
    assert health["chunk_count"] == health["vector_count"], "index is stale vs chunks"
    assert health["chunk_count"] > 100, "corpus is suspiciously small"


def test_rag_retrieves_the_right_page():
    from app.services.rag_service import retrieve_context

    hits = retrieve_context("Serum Creatinine", k=2, test_id="CREATININE")
    assert hits, "no passages retrieved for creatinine"
    assert any("creatinine" in h["topic"].lower() for h in hits), (
        f"retrieved the wrong pages: {[h['topic'] for h in hits]}"
    )
    assert all(h["url"].startswith("http") for h in hits), "chunks must carry a source URL"


def test_rag_rejects_irrelevant_query():
    """The similarity floor has to actually reject: grounding an explanation
    in an unrelated passage is worse than having no grounding."""
    from app.services.rag_service import retrieve_context

    assert retrieve_context("Zorbleflax Index", k=3) == []


def test_grounding_covers_uncurated_tests():
    """ALP is not in the curated JSON. Before, it had no grounding at all and
    was emitted as 'Unlisted biomarker / Requires Clinical Review'; it should
    now be grounded from the scraped corpus."""
    from app.services.llm_service import _build_grounding

    test = TestResult(test_id="ALP", raw_name="Alkaline Phosphatase", value=210,
                      unit="U/L", status=RangeStatus.HIGH)
    grounding = _build_grounding(test)
    assert grounding["has_rag"], "ALP should be grounded from the RAG corpus"
    assert grounding["text"], "grounding text must not be empty"
    assert grounding["sources"], "grounding must carry citable source URLs"



def test_grounding_verifier_rejects_invented_figures():
    """The dangerous class: a model stating a threshold, cutoff or dosage that
    appears nowhere in the NHS/NIH context. Caught exactly, not statistically."""
    from app.services.llm_service import _verify_grounded

    context = ("Haemoglobin is an iron-rich protein in red blood cells that "
               "carries oxygen around the body. A low result may relate to anaemia.")
    ok, reason = _verify_grounded(
        "Haemoglobin below 7.3 g/dL requires immediate transfusion.",
        context, allowed_numbers=set())
    assert not ok and "unsourced figure" in reason, reason

    ok, _ = _verify_grounded(
        "Take 325 mg ferrous sulphate three times daily.", context, set())
    assert not ok


def test_grounding_verifier_rejects_topic_drift():
    from app.services.llm_service import _verify_grounded

    context = ("Haemoglobin is an iron-rich protein in red blood cells that "
               "carries oxygen around the body.")
    ok, reason = _verify_grounded(
        "Your pancreas produces insulin which regulates glucose uptake in "
        "muscle tissue during exercise.", context, set())
    assert not ok and "overlap" in reason, reason


def test_grounding_verifier_allows_patient_own_numbers():
    """The patient's own value and reference bounds legitimately appear in an
    explanation. A string-based check turned "40" into "4" via rstrip("0") and
    wrongly rejected real reference bounds."""
    from app.services.llm_service import _verify_grounded

    context = "Neutrophils are a type of white blood cell that fights infection."
    ok, reason = _verify_grounded(
        "Your neutrophil percentage is 73, within the 40 to 80 range.",
        context, allowed_numbers={"73", "40.0", "80.0"})
    assert ok, reason


def test_corpus_uses_only_approved_domains():
    """Project constraint: NHS UK and NIH MedlinePlus exclusively."""
    import json
    from pathlib import Path
    from urllib.parse import urlparse
    from app.services.rag_service import ALLOWED_SOURCE_DOMAINS, _DATA_DIR

    chunks = json.loads((_DATA_DIR / "rag_chunks.json").read_text())
    domains = {urlparse(c["url"]).netloc for c in chunks}
    assert domains, "corpus is empty"
    assert domains <= ALLOWED_SOURCE_DOMAINS, f"non-approved sources: {domains - ALLOWED_SOURCE_DOMAINS}"


def test_extraction_guardrail_fails_closed_without_source_text():
    """A scanned PDF with no text layer and no OCR is exactly where the vision
    model is most likely to fabricate -- the guardrail used to switch itself
    off there and pass results through unverified."""
    from app.services.pdf_parser import verify_results_against_source

    fabricated = [{"raw_name": "Tumour Marker CA-125", "value": "42.0"}]
    assert verify_results_against_source(fabricated, "") == []



def test_grounding_verifier_handles_thousands_separators():
    """A platelet count written "150,000" was split into the figures 150 and
    000, neither matching the source value 150000, so a correct explanation
    was rejected."""
    from app.services.llm_service import _verify_grounded

    context = "Platelets are small cells that help blood clot after injury."
    ok, reason = _verify_grounded(
        "Your platelet count is 150,000 which sits within the usual range.",
        context, allowed_numbers={"150000"})
    assert ok, reason


def test_grounding_verifier_ignores_digits_inside_analyte_names():
    """T3, T4, B12, P24 and CO2 are names, not quantities."""
    from app.services.llm_service import _verify_grounded

    context = "Thyroid stimulating hormone controls the thyroid gland."
    ok, reason = _verify_grounded(
        "TSH tells the thyroid gland to make T3 and T4 hormones.",
        context, allowed_numbers=set())
    assert ok, reason


def test_grounding_verifier_allows_digits_inside_units():
    """"10^9/L" is a unit, not two invented figures.

    _NUM_RE reads "10^9/L" as the standalone quantities 10 and 9 ("^" and
    "/" are the non-alphanumeric boundaries it looks for), so a correctly
    grounded WBC/PLT explanation was rejected for citing unsourced figures
    and replaced with a GP referral. explain_all_test_results_batched()
    therefore has to seed allowed_numbers with the unit's digits.
    """
    from app.services.llm_service import _verify_grounded, _NUM_RE

    context = ("White blood cells help the body fight infection. "
               "A normal count is 4.0 to 11.0.")
    generated = ("Your white blood cell count is 9.8 10^9/L, which is "
                 "within the normal range and helps fight infection.")

    allowed = {9.8, 4.0, 11.0} | {float(n) for n in _NUM_RE.findall("10^9/L")}
    ok, reason = _verify_grounded(generated, context, allowed_numbers=allowed)
    assert ok, reason

    # Without the unit's digits the same explanation is thrown away -- this
    # is the regression being guarded against, so assert it still would be.
    ok_without, _ = _verify_grounded(
        generated, context, allowed_numbers={9.8, 4.0, 11.0})
    assert not ok_without, "unit digits no longer needed -- update this test"


def test_grounding_verifier_still_rejects_invented_figures_with_units():
    """Widening for units must not let a real invented figure through."""
    from app.services.llm_service import _verify_grounded, _NUM_RE

    context = ("White blood cells help the body fight infection. "
               "A normal count is 4.0 to 11.0.")
    allowed = {9.8, 4.0, 11.0} | {float(n) for n in _NUM_RE.findall("10^9/L")}

    ok, reason = _verify_grounded(
        "Your count of 9.8 10^9/L means a 73% higher risk of infection.",
        context, allowed_numbers=allowed)
    assert not ok, "invented figure 73 should still be rejected"
    assert "73" in reason, reason


def test_grounding_verifier_ignores_digits_in_hyphenated_names():
    """"omega-3" is a name, not the quantity 3 -- same class as T3/B12, but a
    hyphen is non-alphanumeric so the original boundary rule missed it and
    threw away correct NHS dietary advice ("increase your intake of omega-3
    fatty acids"). A digit after "letter-" is a name; after "digit-" it is a
    real range bound, which must still be checked."""
    from app.services.llm_service import _verify_grounded, _NUM_RE

    assert _NUM_RE.findall("omega-3 fatty acids") == []
    assert _NUM_RE.findall("COVID-19") == []
    assert _NUM_RE.findall("6-8 glasses") == ["6", "8"]

    context = "Triglycerides are a type of fat found in the blood."
    ok, reason = _verify_grounded(
        "Increase your intake of omega-3 fatty acids found in oily fish.",
        context, allowed_numbers=set())
    assert ok, reason

    # a genuine invented range must still be caught
    ok, _ = _verify_grounded(
        "Keep your level between 2-4 by taking supplements.", context, set())
    assert not ok


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
