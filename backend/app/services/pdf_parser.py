import re
import io
from typing import List, Optional, Tuple
import pdfplumber
import pandas as pd
from app.services.reference_db import resolve_test_id, finalize_result, dedupe_results, canonicalize_test_name
from app.services.llm_extractor import extract_holistic

# --------------------------------------------------------------------------
# NEW: source-verification guardrail against LLM hallucination
# --------------------------------------------------------------------------
# Observed in production: on reports with no ruled tables (so Tier 1 never
# fires), the vision/text LLM tier can fabricate entire test panels that
# were never in the source document at all (confirmed by diffing an actual
# app output against its real 19-page source: ~100+ tests were returned for
# a report that genuinely contains ~40, including full tumor-marker and
# autoantibody panels the source never mentions). This is a correctness/
# safety issue, not just a missing-reference-range issue - fix it before
# broadening explanation coverage.
#
# This check is deliberately cheap and deterministic (no extra LLM call):
# every extracted test name's substantive words must actually appear in the
# pre-extracted document text, and any numeric value must appear verbatim
# somewhere in the source too. It will not catch a subtly-wrong number
# attached to a real test name, but it reliably kills wholesale fabrication.

_STRUCTURAL_STOPWORDS = {
    "test", "tests", "result", "results", "unit", "units", "biological",
    "reference", "interval", "range", "normal", "value", "values", "method",
    "count", "level", "levels", "laboratory", "report", "sample", "patient",
    "name", "date", "page",
}


def _normalize_for_match(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", s or "")  # drop parentheticals like "(HbA1c)"
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def verify_results_against_source(results: List[dict], source_text: str,
                                   min_ratio: float = 0.6) -> List[dict]:
    """Drops results whose test name / value can't be found in the
    deterministically pre-extracted document text. See module note above.

    CHANGED: matching used to be whole-document - name-words checked for
    presence anywhere in the doc, and value digits checked with a bare
    `value_digits in source_text` substring test anywhere in the doc. Two
    real fabrication cases slipped past that:
      1. A real test name (e.g. "Creatinine", "Potassium") legitimately
         appears elsewhere in the report attached to its REAL value, so
         name-coverage passed even when the value paired with it by the
         LLM was fabricated.
      2. The value substring check had no word boundaries, so e.g. a
         fabricated "44" matched inside the unrelated "44109" printed in
         a peak-area table on a completely different page.
    Now: split the source into lines, and only accept a result if its
    value appears - as a whole number, not a substring - on one of the
    SAME lines where enough of the test name's significant words appear.
    This ties the value to the row it claims to come from, not the
    document at large. It still won't catch a wrong number attached to
    the right test IF that wrong number happens to also appear on the
    same line for some other reason, but that's a much narrower gap than
    before."""
    if not source_text:
        return results  # nothing to verify against - don't block on an empty text layer

    lines = [l for l in source_text.split("\n") if l.strip()]
    # Keep both forms per line: normalized (for word-coverage matching on
    # the test name) and raw (for value matching - normalization strips
    # decimal points, which would break a check like "44.0").
    line_pairs = [(_normalize_for_match(l), l) for l in lines]

    verified, dropped = [], []
    for r in results:
        name_norm = _normalize_for_match(r.get("raw_name", ""))
        sig_words = [w for w in name_norm.split() if len(w) > 2 and w not in _STRUCTURAL_STOPWORDS]
        if not sig_words:
            sig_words = [w for w in name_norm.split() if w not in _STRUCTURAL_STOPWORDS] or name_norm.split()

        value_digits = re.sub(r"[^\d.]", "", str(r.get("value", "")))
        value_pattern = re.compile(rf"(?<!\d){re.escape(value_digits)}(?!\d)") if value_digits else None

        found = False
        for norm_line, raw_line in line_pairs:
            line_words = set(norm_line.split())
            matched = sum(1 for w in sig_words if w in line_words)
            coverage = matched / max(len(sig_words), 1)
            if coverage < min_ratio:
                continue
            if value_pattern is None or value_pattern.search(raw_line):
                found = True
                break

        if found:
            r["verified"] = True
            verified.append(r)
        else:
            r["verified"] = False
            dropped.append(r)

    if dropped:
        names = [d.get("raw_name") for d in dropped][:10]
        suffix = " ..." if len(dropped) > 10 else ""
        print(f"⚠️ Dropped {len(dropped)} unverified/likely-hallucinated result(s): {names}{suffix}")

    return verified

PATTERNS = [
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+):\s*([\d\.]+)\s*([a-zA-Z0-9\/\%\u00b5u\u03bc\*\^]+)?",
    r"([A-Za-z][A-Za-z\s\(\)\/\-]{1,30}?)\s{2,}([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-]+)\t+([\d\.]+)\s*([a-zA-Z0-9\/\%\^]+)?",
]

MIN_TEXT_LENGTH_FOR_TEXT_LAYER = 30

# CHANGED: below this many table-extracted results, we don't trust the
# "table" (could be a signature block, letterhead grid, or a single stray
# line) and fall through to the LLM tiers instead of returning too little.
MIN_STRUCTURED_RESULTS = 3


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
    # CHANGED: previously slugged the full raw name verbatim, so
    # "Creatinine" and "Creatinine (24 hour)" got different synthetic IDs
    # and never deduped against each other. canonicalize_test_name() strips
    # only non-distinguishing sample/method/timeframe qualifiers - see
    # reference_db.py for what it keeps vs strips.
    return canonicalize_test_name(raw_name), False


# --------------------------------------------------------------------------
# NEW: Tier 1 - deterministic, LLM-free extraction from real detected tables
# --------------------------------------------------------------------------
# This existed before only as a last-resort fallback (via markdown-ified
# pipe-row regex parsing in _extract_from_text). Promoted to Tier 1 and
# rewritten to map columns by their HEADER labels (Test/Result/Unit/
# Reference), so ref_range is captured directly from the document with zero
# LLM calls whenever the report has a real ruled/structured table - which
# is the majority case for digital lab report PDFs.

def _extract_tables_with_pages(file_bytes: bytes) -> List[Tuple[int, list]]:
    """CHANGED: many real lab report PDFs align columns with whitespace
    only - no visible ruled/bordered grid lines (confirmed on the Sterling
    Accuris sample report this pipeline was diffed against). pdfplumber's
    default table detection uses a "lines" strategy and finds nothing on
    pages like that, so table_results stays below MIN_STRUCTURED_RESULTS
    and the report gets routed to the LLM vision tier - with its
    hallucination risk - even though the data is cleanly laid out and
    extractable for free. Now falls back to a text-position strategy
    (columns/rows inferred from character alignment) whenever the default
    ruled-line strategy finds nothing on a page."""
    out = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if not tables:
                    try:
                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                        })
                    except Exception as e:
                        print(f"pdfplumber text-strategy table extraction failed on page {page_num}: {e}")
                        tables = []
                for table in tables:
                    if table:
                        out.append((page_num, table))
    except Exception as e:
        print(f"pdfplumber table extraction failed: {e}")
    return out


def _table_to_structured_rows(table: list, page_num: int) -> List[dict]:
    if not table or len(table) < 2:
        return []

    header = [(h or "").strip().lower() for h in table[0]]

    def find_col(*keywords):
        for i, h in enumerate(header):
            if any(k in h for k in keywords):
                return i
        return None

    idx_test = find_col("test", "observation")
    idx_result = find_col("result")
    idx_unit = find_col("unit")
    idx_range = find_col("ref", "biological", "interval", "normal")

    if idx_test is None or idx_result is None:
        return []

    rows = []
    for row in table[1:]:
        if idx_test >= len(row) or not row[idx_test]:
            continue
        raw_name = (row[idx_test] or "").strip()
        value = (row[idx_result] or "").strip() if idx_result < len(row) else ""
        if not raw_name or not value:
            continue

        test_id, is_known = _resolve_or_synthesize_id(raw_name)
        rows.append({
            "test_id": test_id,
            "raw_name": raw_name,
            "value": value,
            "unit": (row[idx_unit].strip() if idx_unit is not None and idx_unit < len(row) and row[idx_unit] else ""),
            "ref_range": (row[idx_range].strip() if idx_range is not None and idx_range < len(row) and row[idx_range] else None),
            "status": None,   # no LLM guess to store - computed deterministically in finalize_result
            "known": is_known,
            "page": page_num,
        })
    return rows


def parse_structured_tables(file_bytes: bytes) -> List[dict]:
    """TIER 1: free, instant, LLM-free. When it works, it's also the most
    trustworthy tier - ref_range comes straight from a header-mapped column,
    not an LLM transcription."""
    tables = _extract_tables_with_pages(file_bytes)
    results = []
    for page_num, table in tables:
        results.extend(_table_to_structured_rows(table, page_num))
    return dedupe_results(results)


def _parse_pipe_row(line: str) -> Optional[dict]:
    """
    Parses Markdown table rows produced by pdfplumber's extract_tables().
    Used only in the Tier 4 regex safety net below (positional guessing,
    no header mapping - ref_range is not captured here by design; Tier 1
    above is the header-aware path for ref_range extraction).
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


def parse_report(
    file_bytes: bytes,
    method: str = "auto",
    provider: str = None,
    patient_sex: str = "unknown",
    patient_age: int = 30,
) -> dict:
    """
    Parses PDF reports using a cost-ordered, cheapest-first pipeline:
      Tier 1: Deterministic table extraction (pdfplumber, free, instant, no LLM)
      Tier 2: Multimodal Vision LLM   (only if Tier 1 found too little)
      Tier 3: Text-Based LLM          (only if Tier 2 found nothing)
      Tier 4: Deterministic Regex     (final safety net)

    CHANGED from the previous version: table/regex extraction used to be the
    LAST resort, meaning every PDF - even ones with clean ruled tables -
    burned an LLM call. Tier 1 now runs first and, when confident, returns
    with ZERO LLM calls spent.

    Every result, from every tier, is passed through
    reference_db.finalize_result() so status/normal-range are computed the
    same deterministic way everywhere, and merged via dedupe_results() so
    the same test never appears twice across tiers.
    """
    # --- explicit single-mode overrides (unchanged behavior, now also
    #     routed through finalize_result/dedupe_results for consistency) ---
    if method in ("regex", "ocr"):
        text = _extract_pypdf_text(file_bytes) or ""
        used_ocr = False
        if method == "ocr" or len(text.strip()) < MIN_TEXT_LENGTH_FOR_TEXT_LAYER:
            ocr_text = _extract_ocr_text(file_bytes)
            if ocr_text.strip():
                text = ocr_text
                used_ocr = True

        structured_results = dedupe_results(_extract_from_text(text))
        if structured_results:
            finalized = [finalize_result(r, patient_sex, patient_age) for r in structured_results]
            method_tag = "ocr_structured" if used_ocr else "structured"
            return {"results": finalized, "method": method_tag}
        return {"results": [], "method": "manual_required"}

    # --- TIER 1: deterministic tables, zero LLM calls spent ---
    table_results = parse_structured_tables(file_bytes)
    if len(table_results) >= MIN_STRUCTURED_RESULTS:
        finalized = [finalize_result(r, patient_sex, patient_age) for r in table_results]
        return {"results": finalized, "method": "table_structured"}

    # --- Pre-extract layout text once, reused by Tiers 2-4 ---
    text = _extract_pypdf_text(file_bytes) or ""
    used_ocr = False
    if len(text.strip()) < MIN_TEXT_LENGTH_FOR_TEXT_LAYER:
        ocr_text = _extract_ocr_text(file_bytes)
        if ocr_text.strip():
            text = ocr_text
            used_ocr = True

    # --- TIER 2 & 3: Vision LLM, then Text LLM (via extract_holistic) ---
    try:
        llm_results, method_used = extract_holistic(
            pdf_bytes=file_bytes,
            raw_text=text,
            provider=provider
        )
        if llm_results:
            # Guardrail: reject any LLM-extracted result that can't be found
            # in the document itself before trusting it further.
            llm_results = verify_results_against_source(llm_results, text)

            # Tier 1 may have found a few results just below the confidence
            # threshold - merge rather than discard them, they cost nothing
            # and may cover tests the LLM missed.
            merged = dedupe_results(table_results + llm_results)

            # If verification rejected everything and Tier 1 found nothing
            # either, don't report a false "success" with zero results -
            # fall through to Tier 4 instead.
            if merged:
                finalized = [finalize_result(r, patient_sex, patient_age) for r in merged]
                method_tag = f"ocr_{method_used}" if (used_ocr and "ocr" not in method_used) else method_used
                return {"results": finalized, "method": method_tag}

    except Exception as e:
        print(f"LLM Holistic Extraction Error (falling back to structured): {e}")

    # --- TIER 4: final safety net ---
    structured_results = _extract_from_text(text)
    merged = dedupe_results(table_results + structured_results)
    if merged:
        finalized = [finalize_result(r, patient_sex, patient_age) for r in merged]
        method_tag = "ocr_structured" if used_ocr else "structured"
        return {"results": finalized, "method": method_tag}

    return {"results": [], "method": "manual_required"}


def _extract_from_text(text: str) -> List[dict]:
    """CHANGED: now tracks page number via the "--- PAGE N ---" markers
    _extract_pypdf_text() already inserts, and includes ref_range/status
    keys (as None) so every tier's dicts share the same shape going into
    finalize_result()."""
    results, seen = [], set()
    current_page = 0

    def _add(raw_name: str, value: float, unit: str):
        test_id, is_known = _resolve_or_synthesize_id(raw_name)
        if test_id in seen:
            return
        seen.add(test_id)
        results.append({
            "raw_name": raw_name, "test_id": test_id,
            "value": value, "unit": unit, "known": is_known,
            "ref_range": None, "status": None, "page": current_page,
        })

    for line in text.split("\n"):
        line = line.strip()

        page_marker = re.match(r"^-{3}\s*PAGE\s+(\d+)\s*-{3}$", line, re.IGNORECASE)
        if page_marker:
            current_page = int(page_marker.group(1))
            continue

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