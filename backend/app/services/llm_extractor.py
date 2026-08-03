import fitz  # PyMuPDF
import base64
import json
from app.config import settings
from app.services.llm_service import call_with_fallback
from app.services.reference_db import resolve_test_id

# In app/services/llm_extractor.py

EXTRACTION_SYSTEM = """You are an expert clinical laboratory data extraction system.
Extract ONLY actual medical diagnostic tests and their measured results.
IGNORE ALL administrative metadata, patient info, doctor names, and barcodes.

CRITICAL JSON SYNTAX RULES:
1. Output MUST be strictly valid JSON.
2. Every key and string value MUST be enclosed in double quotes (e.g., "unit": "%").
3. NEVER use parentheses or non-standard syntax for JSON keys (e.g., DO NOT write "unit("%")", write "unit": "%").
4. For special symbols like %, μg, or μmol/L, write clean plain-text strings like "%", "ug/L", or "umol/L".

For each extracted result, evaluate its medical status based on standard clinical context:
- "normal": Value is within expected limits or indicates a healthy qualitative finding (e.g., 'Clear', 'Pale Yellow', 'Negative', 'Nil', '0-1').
- "high": Value is elevated above normal range or indicates an abnormal presence (e.g., '1+', '2+', 'Positive', 'Cloudy').
- "low": Value is below normal expected range.
- "unknown": Only if context is genuinely ambiguous.

Return valid JSON in this EXACT structure:
{
  "results": [
    {"raw_name": "Glycosylated Haemoglobin (HbA1c)", "value": "5.7", "unit": "%", "status": "normal"},
    {"raw_name": "Serum Creatinine", "value": "80", "unit": "umol/L", "status": "normal"}
  ]
}"""


def pdf_to_base64_images(pdf_bytes: bytes, max_pages: int = 5) -> list[str]:
    """Helper to render PDF pages as Base64 data URLs for Vision models."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            images.append(f"data:image/png;base64,{b64_str}")
        return images
    except Exception as e:
        print(f"⚠️ PyMuPDF image conversion failed: {e}")
        return []


def extract_with_vision(pdf_bytes: bytes, provider: str = None) -> list[dict]:
    """TIER 1: Multimodal Vision Extraction using call_with_fallback"""
    if not pdf_bytes:
        return []

    images = pdf_to_base64_images(pdf_bytes, max_pages=5)
    if not images:
        return []

    content_blocks = [{"type": "text", "text": "Extract all clinical lab test results visible in these images."}]
    for img_url in images:
        content_blocks.append({"type": "image_url", "image_url": {"url": img_url}})

    try:
        data = call_with_fallback(
            system=EXTRACTION_SYSTEM,
            prompt=content_blocks,
            max_tokens=4000,
            provider=provider
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        return _format_extracted_results(results)
    except Exception as e:
        print(f"⚠️ Tier 1 (Vision) failed across all providers: {e}")
        return []


def extract_with_llm_text(text: str, provider: str = None) -> list[dict]:
    """TIER 2: Text-Based LLM Extraction using call_with_fallback"""
    if not text or not text.strip():
        return []

    prompt = f"Extract all medical test results from this text:\n---\n{text}\n---"

    try:
        data = call_with_fallback(
            system=EXTRACTION_SYSTEM,
            prompt=prompt,
            max_tokens=4000,
            provider=provider
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        return _format_extracted_results(results)
    except Exception as e:
        print(f"⚠️ Tier 2 (LLM Text) failed: {e}.")
        return []


def _format_extracted_results(raw_results: list) -> list[dict]:
    cleaned = []
    for r in raw_results:
        name = str(r.get("raw_name", "")).strip()
        val = str(r.get("value", "")).strip()
        if not name or not val:
            continue
        
        test_id = resolve_test_id(name) or name.upper().strip().replace(" ", "_")
        cleaned.append({
            "test_id": test_id,
            "raw_name": name,
            "value": val,
            "unit": str(r.get("unit", "")).strip(),
            "status": str(r.get("status", "unknown")).lower().strip()
        })
    return cleaned


def extract_holistic(pdf_bytes: bytes, raw_text: str = None, provider: str = None) -> tuple[list[dict], str]:
    # 1. Try Tier 1: Vision LLM
    if pdf_bytes:
        results = extract_with_vision(pdf_bytes, provider=provider)
        if results:
            return results, "vision"

    # 2. Try Tier 2: LLM Text
    if raw_text:
        results = extract_with_llm_text(raw_text, provider=provider)
        if results:
            return results, "llm_text"

    return [], "failed"