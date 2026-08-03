import re
import io
from typing import List, Optional, Tuple
import pdfplumber
import pandas as pd
from app.services.reference_db import resolve_test_id
from app.services.llm_extractor import extract_holistic

PATTERNS = [
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+):\s*([\d\.]+)\s*([a-zA-Z0-9\/\%\u00b5u\u03bc\*\^]+)?",
    r"([A-Za-z][A-Za-z\s\(\)\/\-]{1,30}?)\s{2,}([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+)\t+([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
]

MIN_TEXT_LENGTH_FOR_TEXT_LAYER = 30


def _extract_pypdf_text(file_bytes: bytes) -> str:
    """
    Extracts structured layout text using pdfplumber, keeping both layout-preserved
    text and extracted markdown tables to prevent column desynchronization.
    """
    output_lines = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                output_lines.append(f"--- PAGE {page_num} ---")

                # 1. First, extract raw text with layout=True (preserves horizontal positioning)
                layout_text = page.extract_text(layout=True)
                if layout_text:
                    output_lines.append("=== LAYOUT TEXT ===")
                    output_lines.append(layout_text)

                # 2. Extract tables as Markdown grid structures
                tables = page.extract_tables()
                if tables:
                    output_lines.append("=== DETECTED TABLES ===")
                    for table in tables:
                        df = pd.DataFrame(table).dropna(how="all")
                        output_lines.append(df.to_markdown(index=False))

    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")

    return "\n\n".join(output_lines)


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


def _resolve_or_synthesize_id(raw_name: str) -> Tuple[str, bool]:
    """
    Returns (test_id, is_known).
    Resolves known IDs or synthesizes uppercase string identifiers for novel tests.
    """
    test_id = resolve_test_id(raw_name)
    if test_id:
        return test_id, True
    synthetic = re.sub(r"[^A-Z0-9]+", "_", raw_name.upper().strip()).strip("_")
    return (synthetic or "UNKNOWN_TEST"), False


def _parse_pipe_row(line: str) -> Optional[dict]:
    """
    Parses Markdown table rows produced by pdfplumber's extract_tables().
    """
    if not line.startswith("|"):
        return None
    cells = [c.strip() for c in line.strip("|").split("|")]
    cells = [c for c in cells if c and not re.fullmatch(r"-{2,}", c)]
    if len(cells) < 2:
        return None

    name_idx = None
    for i, cell in enumerate(cells):
        if re.match(r"^[A-Za-z][A-Za-z0-9\s\(\)\/\-]{1,40}$", cell):
            name_idx = i
            break
    if name_idx is None:
        return None

    for cell in cells[name_idx + 1:]:
        m = re.fullmatch(r"([\d.]+)\s*([a-zA-Z0-9/%\u00b5u\u03bc\^]*)", cell)
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            return {"raw_name": cells[name_idx], "value": value, "unit": m.group(2) or ""}
    return None


def parse_structured_pdf(file_bytes: bytes) -> List[dict]:
    """Regex-only extraction, using whatever text pdfplumber can find."""
    try:
        text = _extract_pypdf_text(file_bytes)
        return _extract_from_text(text)
    except Exception as e:
        print(f"PDF parse error: {e}")
        return []


def parse_report(file_bytes: bytes, method: str = "auto", provider: str = None) -> dict:
    """
    Safely parses PDF reports using the 4-Tier Fallback Pipeline:
      Tier 1: Multimodal Vision LLM (Processes file_bytes natively)
      Tier 2: Text-Based LLM (Processes raw layout text)
      Tier 3: OCR Text (For scanned/image-only PDFs)
      Tier 4: Deterministic Regex Parser
    """
    # Step 1: Pre-extract layout text (or OCR text if standard layer is empty)
    text = _extract_pypdf_text(file_bytes) or ""
    used_ocr = False

    if method == "ocr" or len(text.strip()) < MIN_TEXT_LENGTH_FOR_TEXT_LAYER:
        ocr_text = _extract_ocr_text(file_bytes)
        if ocr_text.strip():
            text = ocr_text
            used_ocr = True

    # --- EXPLICIT REGEX / OCR ONLY MODES ---
    if method in ("regex", "ocr"):
        structured_results = _extract_from_text(text)
        if structured_results:
            method_tag = "ocr_structured" if used_ocr else "structured"
            return {"results": structured_results, "method": method_tag}
        return {"results": [], "method": "manual_required"}

    # --- AUTO / HYBRID / VISION / LLM MODES ---
    # Try Tier 1 Vision & Tier 2 Text via extract_holistic
    try:
        llm_results, method_used = extract_holistic(
            pdf_bytes=file_bytes,
            raw_text=text,
            provider=provider
        )
        
        if llm_results:
            # Ensure every result is safely mapped with a test_id
            processed_llm_results = []
            for r in llm_results:
                if isinstance(r, dict) and "raw_name" in r:
                    if "test_id" not in r:
                        tid, is_known = _resolve_or_synthesize_id(r["raw_name"])
                        r["test_id"] = tid
                        r["known"] = is_known
                    processed_llm_results.append(r)
            
            method_tag = f"ocr_{method_used}" if (used_ocr and "ocr" not in method_used) else method_used
            return {"results": processed_llm_results, "method": method_tag}

    except Exception as e:
        print(f"LLM Holistic Extraction Error (falling back to structured): {e}")

    # --- FINAL SAFETY FALLBACK (Deterministic Regex / Pipe Table Parsing) ---
    structured_results = _extract_from_text(text)
    if structured_results:
        return {"results": structured_results, "method": "ocr_structured" if used_ocr else "structured"}

    return {"results": [], "method": "manual_required"}


def _extract_from_text(text: str) -> List[dict]:
    results, seen = [], set()

    def _add(raw_name: str, value: float, unit: str):
        test_id, is_known = _resolve_or_synthesize_id(raw_name)
        if test_id in seen:
            return
        seen.add(test_id)
        results.append({
            "raw_name": raw_name, "test_id": test_id,
            "value": value, "unit": unit, "known": is_known,
        })

    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 4:
            continue

        if line.startswith("|"):
            row = _parse_pipe_row(line)
            if row:
                _add(row["raw_name"], row["value"], row["unit"])
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
                _add(raw_name, value, unit)
                break
    return results