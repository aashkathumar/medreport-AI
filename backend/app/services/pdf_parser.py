import re
import io
from typing import List
from app.services.reference_db import resolve_test_id

# NOTE on digits in the name-capture groups:
# - Colon-delimited (pattern 1) and tab-delimited (pattern 3) lines have an
#   unambiguous delimiter, so digits are safe in the name (needed for names
#   like "HbA1c" and "Vitamin B12").
# - The plain multi-space pattern (pattern 2) has NO such delimiter, so
#   allowing digits in the name group previously caused it to greedily
#   swallow the numeric value itself on lines like
#   "Platelets      250    10^9/L" (matched "Platelets      250" as the
#   name and "10" as the value). That pattern's name group stays digit-free.
PATTERNS = [
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+):\s*([\d\.]+)\s*([a-zA-Z0-9\/\%\u00b5u\u03bc\*\^]+)?",
    r"([A-Za-z][A-Za-z\s\(\)\/\-]{1,30}?)\s{2,}([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+)\t+([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
]


def parse_structured_pdf(file_bytes: bytes) -> List[dict]:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return _extract_from_text(text)
    except Exception as e:
        print(f"PDF parse error: {e}")
        return []


def parse_report(file_bytes: bytes) -> dict:
    results = parse_structured_pdf(file_bytes)
    if len(results) >= 2:
        return {"results": results, "method": "structured"}
    return {"results": results, "method": "manual_required"}


def _extract_from_text(text: str) -> List[dict]:
    results, seen = [], set()
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 4:
            continue
        for pattern in PATTERNS:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                raw_name = m.group(1).strip()
                try:
                    value = float(m.group(2).strip())
                except ValueError:
                    continue
                unit = m.group(3).strip() if m.lastindex >= 3 and m.group(3) else ""
                test_id = resolve_test_id(raw_name)
                if test_id and test_id not in seen:
                    seen.add(test_id)
                    results.append({
                        "raw_name": raw_name, "test_id": test_id,
                        "value": value, "unit": unit,
                    })
                break
    return results
