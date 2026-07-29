"""
LLM-based extraction -- an alternative/fallback to the regex parser in
pdf_parser.py. Handles report layouts regex can't (e.g. dense single-space
table formats common in real lab PDFs), at the cost of one extra LLM call.
"""
from app.config import settings
from app.services.llm_providers import PROVIDERS
from app.services.reference_db import resolve_test_id

EXTRACTION_SYSTEM = """You extract structured lab test data from medical report text.
Rules:
- Only extract values that are clearly numeric test RESULTS (not dates, ages,
  phone numbers, accession numbers, patient IDs, or the reference/normal
  range values -- e.g. "13.0 - 17.0" is a range, not a result).
- Use the exact test name as it appears in the report.
- If the same test appears more than once, only include it once.
- Return ONLY valid JSON, no markdown, no commentary."""

EXTRACTION_PROMPT = """Extract lab test results from this medical report text.

CRITICAL RULES:
- Extract ONLY the patient's actual result value, not reference/normal range values
- Reference ranges appear AFTER the result (e.g. "14.5 g/dL 13.0 - 16.5" means result=14.5, range=13.0-16.5, extract 14.5 only)
- Ignore dates, IDs, percentages in headers, and any non-numeric qualitative results (Negative, Absent, Non Reactive)
- If a result shows "H" or "L" flag before the number (e.g. "H 141.0"), the number is still the result
- Do not include the same test twice

Report text:
---
{text}
---

Return format: {{"results": [{{"raw_name": "...", "value": 0.0, "unit": "..."}}]}}"""


def extract_with_llm(text: str, provider: str = None) -> list[dict]:
    provider = provider or settings.default_llm_provider
    call = PROVIDERS[provider]
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])

    try:
        data = call(EXTRACTION_SYSTEM, prompt, max_tokens=2000)
        raw_results = data.get("results", [])
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return []

    matched, seen = [], set()
    for r in raw_results:
        test_id = resolve_test_id(r.get("raw_name", ""))
        if test_id and test_id not in seen:
            try:
                matched.append({
                    "raw_name": r["raw_name"], "test_id": test_id,
                    "value": float(r["value"]), "unit": r.get("unit", ""),
                })
                seen.add(test_id)
            except (ValueError, TypeError):
                continue
    return matched