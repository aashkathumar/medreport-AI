import re
import io
from typing import List
from app.services.reference_db import resolve_test_id
from app.services.llm_extractor import extract_with_llm

PATTERNS = [
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+):\s*([\d\.]+)\s*([a-zA-Z0-9\/\%\u00b5u\u03bc\*\^]+)?",
    r"([A-Za-z][A-Za-z\s\(\)\/\-]{1,30}?)\s{2,}([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+)\t+([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
]

# Below this many characters of extracted text, assume the PDF has no real
# text layer (i.e. it's a scanned image) and OCR is needed before anything
# else can work.
MIN_TEXT_LENGTH_FOR_TEXT_LAYER = 30


def _extract_pypdf_text(file_bytes: bytes) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_ocr_text(file_bytes: bytes) -> str:
    """
    OCR fallback for scanned/image-based PDFs with no real text layer.
    Requires: pip install pytesseract pdf2image
    Plus system binaries: brew install tesseract poppler (Mac)
                           apt install tesseract-ocr poppler-utils (Linux)
    """
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(file_bytes)
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""


def parse_structured_pdf(file_bytes: bytes) -> List[dict]:
    """Regex-only extraction, using whatever text pypdf can find."""
    try:
        text = _extract_pypdf_text(file_bytes)
        return _extract_from_text(text)
    except Exception as e:
        print(f"PDF parse error: {e}")
        return []


def parse_report(file_bytes: bytes, method: str = "auto") -> dict:
    """
    method:
      "auto"  -- full pipeline: pypdf text -> (OCR if no text layer) ->
                 regex -> LLM fallback. Default, and what /upload-pdf uses.
      "regex" -- regex only, no OCR, no LLM. Useful to reproduce/measure the
                 baseline failure rate on real-world reports.
      "llm"   -- skip regex, extract via LLM directly (still OCRs first if
                 there's no text layer).
      "ocr"   -- force OCR text extraction regardless of whether a text
                 layer exists, then run regex on the OCR'd text. Useful for
                 testing OCR quality in isolation.
    """
    # Step 1: get text -- try the real text layer first, OCR only if needed
    text = _extract_pypdf_text(file_bytes)
    used_ocr = False

    if method == "ocr" or len(text.strip()) < MIN_TEXT_LENGTH_FOR_TEXT_LAYER:
        ocr_text = _extract_ocr_text(file_bytes)
        if ocr_text.strip():
            text = ocr_text
            used_ocr = True

    if not text.strip():
        return {"results": [], "method": "manual_required"}

    # Step 2: extraction method
    if method == "llm":
        results = extract_with_llm(text)
        return {"results": results, "method": "ocr_llm" if used_ocr else "llm_extraction"}

    results = _extract_from_text(text)
    if len(results) >= 2:
        return {"results": results, "method": "ocr_structured" if used_ocr else "structured"}

    if method in ("regex", "ocr"):
        return {"results": results, "method": "manual_required"}

    # method == "auto": regex found too little -- try LLM as a fallback
    llm_results = extract_with_llm(text)
    if len(llm_results) >= 2:
        return {"results": llm_results, "method": "ocr_llm" if used_ocr else "llm_extraction"}

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