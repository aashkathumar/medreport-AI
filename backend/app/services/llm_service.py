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


def explain_test_result(
    test: TestResult, profile: UserProfile, ref: dict, provider: str = None
) -> ExplainedResult:
    provider = provider or settings.default_llm_provider
    call = PROVIDERS[provider]

    retrieved = retrieve_context(f"{test.raw_name} {test.status.value}", k=2)
    retrieved_text = "\n".join(f"- {c['text']}" for c in retrieved) or "No additional context retrieved."

    prompt = f"""Patient: {profile.age} year old {profile.sex}, {profile.diet_type} diet.

Test: {test.raw_name}
Value: {test.value} {test.unit}
Normal range: {test.normal_range_min} -- {test.normal_range_max} {test.unit}
Status: {test.status.value}

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

    data = call(SYSTEM, prompt, max_tokens=1000)
    return ExplainedResult(
        test_id=test.test_id, raw_name=test.raw_name,
        value=test.value, unit=test.unit, status=test.status,
        normal_range_min=test.normal_range_min,
        normal_range_max=test.normal_range_max,
        source=ref.get("source", "NHS UK"), **data,
    )


def generate_summary(explained: list, profile: UserProfile, provider: str = None) -> dict:
    provider = provider or settings.default_llm_provider
    call = PROVIDERS[provider]

    flagged = [r for r in explained if r.status != RangeStatus.NORMAL]
    normal = [r for r in explained if r.status == RangeStatus.NORMAL]

    prompt = f"""Patient: {profile.age} year old {profile.sex}, {profile.diet_type}.

Normal results: {', '.join(r.raw_name for r in normal) or 'None'}
Flagged results: {', '.join(f'{r.raw_name} ({r.status.value})' for r in flagged) or 'None'}

Return JSON:
{{
"overall_summary": "2-3 sentence calm summary",
"top_gp_topics": ["topic 1", "topic 2"],
"top_lifestyle_change": "Single most impactful change",
"closing_message": "Encouraging closing statement"
}}"""

    return call(SYSTEM, prompt, max_tokens=600)
