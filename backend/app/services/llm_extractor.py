"""
LLM-based extraction -- an alternative/fallback to the regex parser in
pdf_parser.py. Handles report layouts regex can't (e.g. dense single-space
table formats or prose-embedded values common in real lab PDFs), at the
cost of one extra LLM call per chunk.
"""
import concurrent.futures
from typing import List
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

EXTRACTION_PROMPT = """Extract lab test results (Blood or Urine tests) from this medical report.

CRITICAL EXTRACTION RULES:
1. Extract numerical values (e.g. 5.5, 1.025, 10.2) AND semi-quantitative grade values (e.g., "1+", "2+", "3+", "Negative", "NIL").
2. Match test names with their corresponding results across rows or columns.
   - Example: If 'Urinary Glucose' matches with '1+', extract value: "1+" (or numeric 1.0).
   - Example: If 'Urinary pH' matches with '5.5', extract value: 5.5.
3. Ignore reference ranges (e.g., '6.0-8.0', '1.005-1.030'), dates, method names (e.g. 'bromothymol blue'), and hospital metadata.
4. Extract ONLY the test results. Do not output duplicate test names.
5. This may be a fragment of a longer report -- only extract what is present in this excerpt, do not guess at values that are cut off.

Report text:
---
{text}
---

Return JSON with format:
{{"results": [{{"raw_name": "...", "value": 0.0, "unit": "..."}}]}}"""

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 200
MAX_CHUNK_WORKERS = 4


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _extract_chunk(call, chunk: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(text=chunk)
    try:
        data = call(EXTRACTION_SYSTEM, prompt, max_tokens=2000)
        return data.get("results", [])
    except Exception as e:
        print(f"LLM extraction failed on chunk: {e}")
        return []


def extract_with_llm(text: str, provider: str = None) -> list[dict]:
    provider = provider or settings.default_llm_provider
    call = PROVIDERS[provider]
    chunks = _chunk_text(text)

    # Chunks are independent LLM calls -- running them in parallel via thread pool
    if len(chunks) == 1:
        all_raw_results = _extract_chunk(call, chunks[0])
    else:
        all_raw_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), MAX_CHUNK_WORKERS)) as executor:
            for raw_results in executor.map(lambda c: _extract_chunk(call, c), chunks):
                all_raw_results.extend(raw_results)

    matched, seen = [], set()
    for r in all_raw_results:
        raw_name = r.get("raw_name", "")
        if not raw_name:
            continue

        test_id = resolve_test_id(raw_name) or raw_name.upper().strip().replace(" ", "_")

        if test_id in seen:
            continue

        # Safely attempt float conversion; retain original string value ("Negative", "1+", etc.) if float fails
        raw_val = r.get("value")
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            val = str(raw_val) if raw_val is not None else ""

        matched.append({
            "raw_name": raw_name,
            "test_id": test_id,
            "value": val,
            "unit": r.get("unit", ""),
        })
        seen.add(test_id)
        
    return matched