from app.config import settings
from app.models.schemas import TestResult, ExplainedResult, UserProfile, RangeStatus
from app.services.rag_service import retrieve_context
from app.services.llm_providers import PROVIDERS

SYSTEM = """You are a health literacy assistant helping patients understand their blood test results.
You must always:
- Use simple plain English (reading age 14)
- Cite the source (NHS UK or NIH MedlinePlus)
- Include a GP referral prompt
- Be calm and non-alarmist
You must never diagnose, prescribe, or replace medical advice.
Respond ONLY with valid JSON. No markdown, no preamble."""


def call_with_fallback(
    system: str, 
    prompt: str, 
    max_tokens: int, 
    provider: str = None, 
    model_name: str = None
) -> dict:
    """
    Attempts to call the primary LLM provider. If it hits a rate limit,
    quota error, or network failure, it automatically falls back to other 
    configured providers in sequence.
    """
    primary = provider or settings.default_llm_provider
    fallback_order = [primary, "groq_llama", "gemini", "mistral"]
    providers_to_try = list(dict.fromkeys(fallback_order))

    for current_provider in providers_to_try:
        if current_provider not in PROVIDERS:
            continue
        try:
            call_fn = PROVIDERS[current_provider]
            kwargs = {"max_tokens": max_tokens}
            if model_name and current_provider == primary:
                kwargs["model"] = model_name
                
            return call_fn(system, prompt, **kwargs)
        except Exception as e:
            print(f"⚠️ Provider '{current_provider}' failed: {e}. Attempting fallback...")
            continue
            
    raise RuntimeError("All LLM providers failed or rate limits were reached.")


def explain_test_result(
    test: TestResult, 
    profile: UserProfile, 
    ref: dict, 
    provider: str = None,
    model_name: str = None
) -> ExplainedResult:
    status_str = "NORMAL"
    if hasattr(test, "status") and test.status and hasattr(test.status, "value"):
        status_str = test.status.value

    retrieved = retrieve_context(f"{test.raw_name} {status_str}", k=2)
    retrieved_text = "\n".join(f"- {c['text']}" for c in retrieved) if retrieved else "No additional context retrieved."

    prompt = f"""Patient: {profile.age} year old {profile.sex}, {profile.diet_type} diet.

Test: {test.raw_name}
Value: {test.value} {test.unit}
Normal range: {getattr(test, 'normal_range_min', 'N/A')} -- {getattr(test, 'normal_range_max', 'N/A')} {test.unit}
Status: {status_str}

Reference (source: {ref.get('source','NHS UK')}):
- Definition: {ref.get('plain_english','')}
- Low means: {ref.get('low_means','N/A')}
- High means: {ref.get('high_means','N/A')}

Additional retrieved context:
{retrieved_text}

Return JSON with exactly these keys:
{{
"what_it_measures": "2-3 sentence plain English explanation",
"what_your_result_means": "2-3 sentences specific to this patient's result",
"lifestyle_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
"gp_question": "One specific question to ask their GP",
"disclaimer": "This explanation is for information only and does not replace medical advice. Please discuss your results with your GP."
}}"""

    data = call_with_fallback(
        system=SYSTEM, 
        prompt=prompt, 
        max_tokens=1000, 
        provider=provider, 
        model_name=model_name
    )
    
    # Remove 'source' from LLM dictionary if present to prevent keyword collisions
    data.pop("source", None)
    
    current_status = test.status if (hasattr(test, "status") and test.status) else RangeStatus.UNKNOWN

    return ExplainedResult(
        test_id=test.test_id,
        raw_name=test.raw_name,
        value=test.value,
        unit=test.unit,
        status=current_status,
        normal_range_min=getattr(test, 'normal_range_min', None),
        normal_range_max=getattr(test, 'normal_range_max', None),
        source=ref.get("source", "NHS UK"),
        **data,
    )


def generate_summary(
    explained: list, 
    profile: UserProfile, 
    provider: str = None,
    model_name: str = None
) -> dict:
    flagged = [r for r in explained if hasattr(r, 'status') and r.status != RangeStatus.NORMAL]
    normal = [r for r in explained if hasattr(r, 'status') and r.status == RangeStatus.NORMAL]

    prompt = f"""Patient: {profile.age} year old {profile.sex}, {profile.diet_type}.

Normal results: {', '.join(r.raw_name for r in normal) or 'None'}
Flagged results: {', '.join(f'{r.raw_name} ({r.status.value if hasattr(r.status, "value") else r.status})' for r in flagged) or 'None'}

Return JSON:
{{
"overall_summary": "2-3 sentence calm summary",
"top_gp_topics": ["topic 1", "topic 2"],
"top_lifestyle_change": "Single most impactful change",
"closing_message": "Encouraging closing statement"
}}"""

    return call_with_fallback(
        system=SYSTEM, 
        prompt=prompt, 
        max_tokens=1200, 
        provider=provider, 
        model_name=model_name
    )