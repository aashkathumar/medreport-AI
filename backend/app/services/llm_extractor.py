import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import re
from app.config import settings
from app.services.llm_service import call_with_fallback
from app.services.reference_db import resolve_test_id, dedupe_results, canonicalize_test_name

# In app/services/llm_extractor.py

# CHANGED: the prompt now demands the printed reference range verbatim.
EXTRACTION_SYSTEM = """You are an expert clinical laboratory data extraction system.
Extract ONLY actual medical diagnostic tests and their measured results.
IGNORE ALL administrative metadata, patient info, doctor names, and barcodes.

CRITICAL JSON SYNTAX RULES:
1. Output MUST be strictly valid JSON.
2. Every key and string value MUST be enclosed in double quotes (e.g., "unit": "%").
3. NEVER use parentheses or non-standard syntax for JSON keys (e.g., DO NOT write "unit("%")", write "unit": "%").
4. For special symbols like %, μg, or μmol/L, write clean plain-text strings like "%", "ug/L", or "umol/L".

For each test, ALSO extract the "Reference Range" / "Biological Ref. Interval" / "Normal Range"
column EXACTLY as printed next to that test on the report (e.g. "13.0 - 16.5", "< 16.7", "Negative").
Do NOT compute, round, or guess this value - copy it verbatim from the document.
If no reference range is printed for a test, use null for "ref_range".

Only rely on your OWN clinical judgement for "status" (normal/high/low/unknown) when the printed
range is purely qualitative/non-numeric (e.g. "Negative", "Clear") and can't be range-checked
programmatically. For tests with a numeric printed range, still fill in your best guess for
"status", but getting "ref_range" exactly right matters far more - it will be used to compute
the authoritative status deterministically, overriding your guess.

CRITICAL - DO NOT CONFUSE THE RESULT WITH A REFERENCE-RANGE BOUNDARY:
Some reference ranges are printed as several named bands across multiple lines, e.g.
"For Screening: Diabetes: >6.5% / Pre-Diabetes: 5.7% - 6.4% / Non-Diabetes: < 5.7%" or
"Deficiency: <10 / Insufficiency: 10-30 / Sufficiency: 30-100 / Toxicity: >100".
Every number in a block like that is a BAND BOUNDARY, not a result. Copy the whole block into
"ref_range" verbatim. NEVER take one of those boundary numbers and use it as "value". The
patient's actual result is a single number printed in the "Result" column, directly beside the
test name, often flagged with an "H" or "L" next to it if abnormal - it is a DIFFERENT number
from anything in the reference-range block, even when a boundary number looks similar.

Return valid JSON in this EXACT structure:
{
  "results": [
    {"raw_name": "Glycosylated Haemoglobin (HbA1c)", "value": "5.7", "unit": "%", "ref_range": "20.0 - 42.0", "status": "normal"},
    {"raw_name": "Serum Creatinine", "value": "80", "unit": "umol/L", "ref_range": "62 - 106", "status": "normal"}
  ]
}"""

# CHANGED: pages are now batched instead of hard-capped at 5. A 15-20 page
# multi-panel pathology report (thyroid/lipid/vitamin/HbA1c/HIV panels etc.
VISION_BATCH_SIZE = 4  # pages per LLM call - keeps each request's payload/
                        # token size reasonable and avoids degraded accuracy
                        # from cramming too many images into one call.
RENDER_DPI = 200        # raised from 150 for better small-print fidelity
                        # (units/method footnotes are often tiny grey text);


def pdf_to_base64_images(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[str]:
    """Renders EVERY page of the PDF as a Base64 PNG data URL for vision models.
    No page cap - see VISION_BATCH_SIZE / RENDER_DPI comments above."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            images.append(f"data:image/png;base64,{b64_str}")
        return images
    except Exception as e:
        print(f"PyMuPDF image conversion failed: {e}")
        return []


def extract_with_vision(pdf_bytes: bytes, provider: str = None) -> list[dict]:
    """TIER 2 (deterministic table extraction in pdf_parser.py is now Tier 1):
    Multimodal Vision Extraction using call_with_fallback.

    CHANGED: processes ALL pages in small batches rather than the old
    max_pages=5 cutoff. Each batch failure is isolated (one bad batch no
    longer kills results from the rest of the report), and results carry a
    "page_range" tag for provenance/audit."""
    if not pdf_bytes:
        return []

    images = pdf_to_base64_images(pdf_bytes)
    if not images:
        return []

    def _run_batch(batch_start: int) -> list[dict]:
        batch = images[batch_start:batch_start + VISION_BATCH_SIZE]
        page_range = f"{batch_start + 1}-{batch_start + len(batch)}"

        content_blocks = [{
            "type": "text",
            "text": f"Extract all clinical lab test results visible in these images "
                    f"(report pages {page_range})."
        }]
        for img_url in batch:
            content_blocks.append({"type": "image_url", "image_url": {"url": img_url}})

        try:
            data = call_with_fallback(
                system=EXTRACTION_SYSTEM,
                prompt=content_blocks,
                max_tokens=4000,
                provider=provider,
                capability="vision",   # routes to vision_fallback_chain only
            )
            results = data.get("results", []) if isinstance(data, dict) else []
            formatted = _format_extracted_results(results)
            for r in formatted:
                r["page_range"] = page_range
            return formatted
        except Exception as e:
            print(f"Vision batch (pages {page_range}) failed across all providers: {e}")
            return []  # one bad batch shouldn't cost the rest of the report

    # CHANGED: batches were issued strictly serially. A 19-page report is 5
    # batches carrying ~2.6 MB of base64 page images each (12.9 MB total);
    starts = list(range(0, len(images), VISION_BATCH_SIZE))
    workers = max(1, min(settings.llm_max_concurrency, len(starts)))
    batched: list[list[dict]] = [[] for _ in starts]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_batch, start): i for i, start in enumerate(starts)}
        for future in as_completed(futures):
            batched[futures[future]] = future.result()

    all_results = [r for group in batched for r in group]

    return dedupe_results(all_results)


def extract_with_llm_text(text: str, provider: str = None) -> list[dict]:
    """TIER 3: Text-Based LLM Extraction using call_with_fallback"""
    if not text or not text.strip():
        return []

    prompt = f"Extract all medical test results from this text:\n---\n{text}\n---"

    try:
        data = call_with_fallback(
            system=EXTRACTION_SYSTEM,
            prompt=prompt,
            max_tokens=4000,
            provider=provider,
            capability="text",
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        return dedupe_results(_format_extracted_results(results))
    except Exception as e:
        print(f"Tier 3 (LLM Text) failed: {e}.")
        return []


def _value_matches_range_boundary(value, ref_range) -> bool:
    """CHANGED: safety net for the class of bug seen in production on an
    HbA1c result - true value 7.10% (flagged High) was extracted as 5.7%,
    which is actually the "Pre-Diabetes: 5.7% - 6.4%" band boundary from
    inside the reference-range text, not the printed Result. That single
    mix-up flipped the reported status from High to "below normal" - the
    most dangerous direction for a health-literacy tool to be wrong in.
    This doesn't fully prevent the mistake (the prompt change above is the
    real fix), but it catches it after the fact: if the numeric value is
    identical to one of the boundary numbers embedded in ref_range, that's
    suspicious enough to flag for review rather than trust silently."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    if not ref_range:
        return False
    boundary_numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", str(ref_range))]
    return any(abs(val - b) < 1e-9 for b in boundary_numbers)


def _format_extracted_results(raw_results: list) -> list[dict]:
    """CHANGED: now passes through ref_range (previously dropped entirely),
    and keeps the LLM's raw status separate from the final computed one -
    finalize_result() in reference_db.py computes the authoritative status
    later, uniformly across every extraction tier."""
    cleaned = []
    for r in raw_results:
        name = str(r.get("raw_name", "")).strip()
        val = str(r.get("value", "")).strip()
        if not name or not val:
            continue

        resolved = resolve_test_id(name)
        # CHANGED: the prompt now demands the printed reference range
        # verbatim. "status" is still requested as a fallback for purely
        # qualitative results (Negative/Clear/etc.), but for anything numeric
        test_id = resolved or canonicalize_test_name(name)
        raw_ref_range = r.get("ref_range")

        cleaned.append({
            "test_id": test_id,
            "raw_name": name,
            "value": val,
            "unit": str(r.get("unit", "")).strip(),
            "ref_range": (str(raw_ref_range).strip() if raw_ref_range else None),
            "status": str(r.get("status", "unknown")).lower().strip(),  # LLM's guess; may be overridden later
            "known": resolved is not None,
            "needs_review": _value_matches_range_boundary(val, raw_ref_range),
        })
    return cleaned


def extract_holistic(pdf_bytes: bytes, raw_text: str = None, provider: str = None) -> tuple[list[dict], str]:
    """Called only when Tier 1 (deterministic table extraction, in
    pdf_parser.py) found too little to trust. Tries vision first, then
    falls back to text-based LLM extraction on the pre-extracted layout text."""
    # 1. Try Tier 2: Vision LLM
    if pdf_bytes:
        results = extract_with_vision(pdf_bytes, provider=provider)
        if results:
            return results, "vision"

    # 2. Try Tier 3: LLM Text
    if raw_text:
        results = extract_with_llm_text(raw_text, provider=provider)
        if results:
            return results, "llm_text"

    return [], "failed"