import re
import io
from typing import List
from app.services.reference_db import resolve_test_id

# NOTE: the name-capture groups deliberately exclude digits (0-9). The
# original version of these patterns allowed digits in the test-name group,
# which meant it greedily swallowed the numeric value itself on
# multi-space/tab-separated lines (e.g. "Platelets      250    10^9/L"
# matched "Platelets      250" as the name and "10" as the value). Unit
# groups include digits/^ so units like "10^9/L" match correctly.
PATTERNS = [
    r"([A-Za-z][A-Za-z\s\(\)\/\-]+):\s*([\d\.]+)\s*([a-zA-Z0-9\/\%\u00b5u\u03bc\*\^]+)?",
    r"([A-Za-z][A-Za-z\s\(\)\/\-]{1,30}?)\s{2,}([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
    r"([A-Za-z][A-Za-z\s\(\)\/\-]+)\t+([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
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
