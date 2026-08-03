import time
from typing import Union, List
from app.config import settings
from app.models.schemas import TestResult, ExplainedResult, UserProfile, RangeStatus
from app.services.rag_service import retrieve_context
from app.services.llm_providers import PROVIDERS

SYSTEM = """You are a health literacy assistant helping patients understand their blood and urine test results.
You must always:
- Use simple plain English (reading age 14)
- Cite the source (NHS UK or NIH MedlinePlus, or General Medical Consensus if dynamic)
- Include a GP referral prompt
- Be calm and non-alarmist
You must never diagnose, prescribe, or replace medical advice.
Respond ONLY with valid JSON. No markdown, no preamble."""


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
    

def explain_all_test_results_batched(
    test_results: List[TestResult], 
    profile: UserProfile, 
    provider: str = None,
    model_name: str = None
) -> List[ExplainedResult]:
    """
    Explains ALL extracted test results in a SINGLE batched LLM call.
    Eliminates TPM/RPM rate limits and speeds up execution from 180s to ~2s.
    """
    if not test_results:
        return []

    # Build structured list of tests for prompt
    tests_summary = []
    for t in test_results:
        status_val = t.status.value if hasattr(t.status, "value") else str(t.status)
        min_val = getattr(t, 'normal_range_min', None) or "N/A"
        max_val = getattr(t, 'normal_range_max', None) or "N/A"
        tests_summary.append(
            f"- Test ID: {t.test_id} | Name: {t.raw_name} | Value: {t.value} {t.unit} | "
            f"Range: {min_val}-{max_val} {t.unit} | Status: {status_val}"
        )

    tests_text = "\n".join(tests_summary)

    prompt = f"""Patient Profile: {profile.age} year old {profile.sex}, {profile.diet_type} diet.

Extracted Diagnostic Tests ({len(test_results)} total):
{tests_text}

For EACH test listed above, generate a plain English, non-alarmist explanation suitable for a reading age of 14.

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

    data = call_with_fallback(
        system=SYSTEM,
        prompt=prompt,
        max_tokens=4000,
        provider=provider,
        model_name=model_name
    )

    explanations_list = data.get("explanations", []) if isinstance(data, dict) else []
    exp_map = {item.get("test_id", ""): item for item in explanations_list if isinstance(item, dict)}

    explained_results = []
    for t in test_results:
        details = exp_map.get(t.test_id, {})
        explained_results.append(
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

    return explained_results


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