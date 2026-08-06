import time
from typing import Union, List
from app.config import settings
from app.models.schemas import TestResult, ExplainedResult, UserProfile, RangeStatus
from app.services.rag_service import retrieve_context
from app.services.reference_db import get_reference_data
from app.services.llm_providers import PROVIDERS

SYSTEM = """You are a health literacy assistant helping patients understand their blood and urine test results.
You must always:
- Use simple plain English (reading age 14)
- Cite the source (NHS UK or NIH MedlinePlus, or General Medical Consensus if dynamic)
- Include a GP referral prompt
- Be calm and non-alarmist
You must never diagnose, prescribe, or replace medical advice.
Respond ONLY with valid JSON. No markdown, no preamble."""

# CHANGED: explanations are now generated in bounded batches rather than one
# giant call for the whole report. A single call covering 100+ tests was
# observed to silently omit most of them, which is why nearly every result
# in a real report ended up with the generic "Maintain a healthy balanced
# diet. Stay well hydrated." fallback - the LLM just didn't return per-test
# detail for most of an oversized request.
EXPLAIN_BATCH_SIZE = 15


def call_with_fallback(
    system: str, 
    prompt: Union[str, list], 
    max_tokens: int = 4000, 
    provider: str = None, 
    model_name: str = None
) -> dict:
    primary = provider or settings.default_llm_provider
    # Primary -> Groq -> Mistral -> Gemini -> OpenRouter
    fallback_sequence = [primary, "groq_llama", "mistral", "gemini", "openrouter"]
    
    seen = set()
    sequence = [p for p in fallback_sequence if not (p in seen or seen.add(p))]

    for prov in sequence:
        fn = PROVIDERS.get(prov)
        if not fn:
            continue
            
        target_model = model_name if prov == primary else None

        try:
            return fn(system, prompt, max_tokens=max_tokens, model=target_model)
        except Exception as e:
            print(f"⚠️ Provider '{prov}' failed: {e}. Moving immediately to next fallback...")
            continue

    raise RuntimeError("All LLM providers failed in fallback pipeline.")


def _build_grounding(test: TestResult) -> str:
    """
    Returns a short grounding snippet for ONE test, used both to reduce the
    LLM inventing explanation content from nothing, and - the actual ask
    here - to let ANY test get a grounded explanation, not just the two
    (HGB/HBA1C) hand-curated in reference_db.REFERENCE_DB.

    Priority:
      1. Curated REFERENCE_DB entry, if this test has one (highest quality,
         human-reviewed).
      2. RAG-retrieved NHS/NIH context via rag_service.retrieve_context()
         (this existed already but was never called from anywhere - wiring
         it in here is the actual fix for "explain any test").
      3. Nothing found - the LLM is told this explicitly in the prompt, so
         it gives a general, clearly-hedged explanation instead of a
         confidently-sourced one it made up.

    NOTE: assumes retrieve_context() returns chunks shaped roughly like
    {"text": "...", "source": "..."} - adjust the .get() keys below if your
    actual rag_chunks.json schema differs.
    """
    ref = get_reference_data(test.test_id)
    if ref:
        parts = [ref.get("plain_english", "")]
        status = test.status.value if hasattr(test.status, "value") else str(test.status)
        if status == "low" and ref.get("low_means"):
            parts.append(f"Low result context: {ref['low_means']}")
        elif status == "high" and ref.get("high_means"):
            parts.append(f"High result context: {ref['high_means']}")
        return " ".join(p for p in parts if p)

    try:
        chunks = retrieve_context(test.raw_name, k=2)
    except Exception as e:
        print(f"⚠️ RAG retrieval failed for '{test.raw_name}': {e}")
        chunks = []

    if chunks:
        snippets = [
            (c.get("text") or c.get("content") or "").strip()
            for c in chunks if isinstance(c, dict)
        ]
        snippets = [s for s in snippets if s]
        if snippets:
            return " ".join(snippets)[:600]  # keep prompt size bounded per test

    return ""


def explain_all_test_results_batched(
    test_results: List[TestResult], 
    profile: UserProfile, 
    provider: str = None,
    model_name: str = None
) -> List[ExplainedResult]:
    """
    Explains ALL extracted test results, in bounded batches (see
    EXPLAIN_BATCH_SIZE), each test grounded via REFERENCE_DB or RAG where
    available (see _build_grounding). Each batch failure is isolated - one
    bad batch falls back to generic defaults for just that batch's tests,
    not the whole report.
    """
    if not test_results:
        return []

    all_explained: List[ExplainedResult] = []

    for batch_start in range(0, len(test_results), EXPLAIN_BATCH_SIZE):
        batch = test_results[batch_start:batch_start + EXPLAIN_BATCH_SIZE]

        tests_summary = []
        for t in batch:
            status_val = t.status.value if hasattr(t.status, "value") else str(t.status)
            min_val = getattr(t, 'normal_range_min', None)
            max_val = getattr(t, 'normal_range_max', None)
            range_str = f"{min_val}-{max_val}" if (min_val is not None or max_val is not None) else "not available"
            grounding = _build_grounding(t)

            entry = (
                f"- Test ID: {t.test_id} | Name: {t.raw_name} | Value: {t.value} {t.unit} | "
                f"Range: {range_str} {t.unit} | Status: {status_val}"
            )
            entry += (f"\n  Reference context: {grounding}" if grounding
                      else "\n  Reference context: none available - give a general, "
                           "clearly-hedged explanation, do not invent a specific source or claim.")
            tests_summary.append(entry)

        tests_text = "\n".join(tests_summary)

        prompt = f"""Patient Profile: {profile.age} year old {profile.sex}, {profile.diet_type} diet.

Extracted Diagnostic Tests (batch {batch_start // EXPLAIN_BATCH_SIZE + 1}, {len(batch)} of {len(test_results)} total):
{tests_text}

For EACH test listed above, generate a plain English, non-alarmist explanation suitable for a reading age of 14.
If "Reference context" is provided for a test, base your explanation on it. If it says "none available",
say so plainly rather than inventing a specific claim or citing a specific source you don't actually have.

Return valid JSON in this exact structure:
{{
  "explanations": [
    {{
      "test_id": "TEST_ID_HERE",
      "what_it_measures": "2-3 sentence explanation of what this test evaluates.",
      "what_your_result_means": "2-3 sentences explaining this specific finding for the patient.",
      "lifestyle_suggestions": ["Practical tip 1", "Practical tip 2"],
      "gp_question": "One specific question to ask their GP.",
      "disclaimer": "This explanation is for information only and does not replace medical advice."
    }}
  ]
}}"""

        try:
            data = call_with_fallback(
                system=SYSTEM,
                prompt=prompt,
                max_tokens=4000,
                provider=provider,
                model_name=model_name
            )
            explanations_list = data.get("explanations", []) if isinstance(data, dict) else []
        except Exception as e:
            print(f"⚠️ Explanation batch ({batch_start + 1}-{batch_start + len(batch)}) failed: {e}")
            explanations_list = []

        exp_map = {item.get("test_id", ""): item for item in explanations_list if isinstance(item, dict)}

        for t in batch:
            details = exp_map.get(t.test_id, {})
            all_explained.append(
                ExplainedResult(
                    test_id=t.test_id,
                    raw_name=t.raw_name,
                    value=t.value,
                    unit=t.unit,
                    status=t.status,
                    normal_range_min=getattr(t, 'normal_range_min', None),
                    normal_range_max=getattr(t, 'normal_range_max', None),
                    source="NHS UK / Medical Consensus",
                    what_it_measures=details.get("what_it_measures", f"Evaluates {t.raw_name} level in sample."),
                    what_your_result_means=details.get("what_your_result_means", f"Your result is {t.value} {t.unit}."),
                    lifestyle_suggestions=details.get("lifestyle_suggestions", ["Maintain a healthy balanced diet.", "Stay well hydrated."]),
                    gp_question=details.get("gp_question", "What does this result mean for my overall health?"),
                    disclaimer=details.get("disclaimer", "This explanation is for educational purposes only.")
                )
            )

    return all_explained


def generate_summary(
    explained: list, 
    profile: UserProfile, 
    provider: str = None,
    model_name: str = None
) -> dict:
    flagged = [r for r in explained if hasattr(r, 'status') and r.status in (RangeStatus.LOW, RangeStatus.HIGH)]
    normal = [r for r in explained if hasattr(r, 'status') and r.status == RangeStatus.NORMAL]
    dynamic = [r for r in explained if hasattr(r, 'status') and r.status == RangeStatus.UNKNOWN]

    prompt = f"""Patient: {profile.age} year old {profile.sex}, {profile.diet_type}.

Normal results: {', '.join(r.raw_name for r in normal) or 'None'}
Flagged results (out of range): {', '.join(f'{r.raw_name} ({r.status.value})' for r in flagged) or 'None'}
Dynamic results (analyzed dynamically): {', '.join(r.raw_name for r in dynamic) or 'None'}

Provide an overall encouraging summary covering all the tests above.

Return JSON:
{{
"overall_summary": "2-3 sentence calm summary summarizing overall health findings",
"top_gp_topics": ["topic 1", "topic 2"],
"top_lifestyle_change": "Single most impactful change",
"closing_message": "Encouraging closing statement"
}}"""

    return call_with_fallback(
        system=SYSTEM, 
        prompt=prompt, 
        max_tokens=1000, 
        provider=provider, 
        model_name=model_name
    )