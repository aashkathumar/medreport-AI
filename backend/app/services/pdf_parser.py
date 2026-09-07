import re
import io
from typing import List, Optional, Tuple
import pdfplumber
import pandas as pd
from app.services.reference_db import resolve_test_id, finalize_result, dedupe_results, canonicalize_test_name
from app.services.llm_extractor import extract_holistic

# --------------------------------------------------------------------------
# Guardrail against LLM hallucination: without this, the vision/text tier
# could fabricate entire test panels never in the source document at all.
# --------------------------------------------------------------------------
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

    BUG FOUND: matching used to check name-words and value digits anywhere
    in the whole document, so a real test name paired with a fabricated
    value still passed if that name appeared elsewhere with its real value,
    and a bare substring check let a fabricated "44" match inside an
    unrelated "44109" on a different page. Now the source is split into
    lines, and a result is only accepted if its value appears as a whole
    number on one of the same lines as its test name, tying it to the row
    it claims to come from rather than the document at large."""
    if not source_text:
        # CHANGED: this used to return `results` unverified. An empty source
        # text means the PDF had no usable text layer AND OCR produced
        # nothing, i.e.
        print("No source text to verify LLM extraction against "
              "(no text layer and no OCR output), rejecting unverifiable "
              "results rather than trusting them.")
        return []

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
        print(f"Dropped {len(dropped)} unverified/likely-hallucinated result(s): {names}{suffix}")

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
    # CHANGED: previously slugged the raw name verbatim, so "Creatinine" and
    # "Creatinine (24 hour)" never deduped, this strips non-distinguishing
    # qualifiers first (see reference_db.py for what it keeps vs strips).
    return canonicalize_test_name(raw_name), False


# --------------------------------------------------------------------------
# Tier 1: deterministic, LLM-free extraction from real detected tables.
# --------------------------------------------------------------------------

def _extract_tables_with_pages(file_bytes: bytes) -> Tuple[List[Tuple[int, list, Optional[str]]], set]:
    """Extracts ruled tables (pdfplumber's default "lines" strategy). Many
    specimen PDFs align columns with whitespace only, with no ruled grid; a
    text-position fallback for those pages was removed after it shredded
    words across an inferred whole-page grid and swept up letterhead as if
    it were results, junk rows that then blocked the positional extractor
    (see _extract_columnar_rows) from running at all. Returns (tables,
    pages that yielded a ruled table) so the caller knows which pages
    still need that positional pass.
    """
    out = []
    ruled_pages = set()
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    tables = page.extract_tables()
                except Exception as e:
                    print(f"pdfplumber table extraction failed on page {page_num}: {e}")
                    continue
                if not tables:
                    continue
                specimen = _page_specimen(page.extract_text() or "")
                # BUG FOUND (QuantiFERON-TB panel): pdfplumber sometimes
                # splits one visual table into a header-only table plus a
                # separate data table with no header, losing every row.
                pending_header = None
                for table in tables:
                    if not table:
                        continue
                    if pending_header is not None:
                        table = [pending_header] + table
                        pending_header = None
                    if len(table) == 1 and _find_header_row(table) == 0:
                        pending_header = table[0]
                        continue
                    out.append((page_num, table, specimen))
                    ruled_pages.add(page_num)
    except Exception as e:
        print(f"pdfplumber table extraction failed: {e}")
    return out, ruled_pages


# BUG FOUND: "Bilirubin"/"Red Cells" appear on both a blood panel and the
# urinalysis panel; ungrounded, every occurrence cited the blood page.
_URINE_SECTION_RE = re.compile(
    r"\b(urinalysis|urine\s+(examination|analysis|routine|r\s*/\s*e)|"
    r"routine\s+urine\s+examination|"
    # Real-world phrasing confirmed on the Sterling Accuris report: the page
    # header prints "Sample Type : Urine", not any of the section-heading
    # words above - the original patterns matched nothing on real data.
    r"sample\s+type\s*:?\s*urine|"
    r"physical\s*&?\s*chemical\s+examination)\b",
    re.IGNORECASE,
)


def _page_specimen(page_text: str) -> Optional[str]:
    """Returns 'urine' if this page's text contains a urinalysis section
    heading, else None (caller defaults ambiguous cases to blood/serum,
    which is the common case and preserves existing behaviour)."""
    if page_text and _URINE_SECTION_RE.search(page_text):
        return "urine"
    return None


_HEAD_TEST = ("test", "observation", "investigation", "parameter", "analyte")
_HEAD_RESULT = ("result", "value", "observed")
_HEAD_UNIT = ("unit",)
_HEAD_REF = ("ref", "biological", "interval", "normal range", "range")


def _find_header_row(table: list) -> Optional[int]:
    """BUG FOUND: the header index used to be hardcoded to table[0], but on
    specimen PDFs the column header is often several rows down, past
    letterhead and demographics. find_col() then returned None on row 0
    and the whole table was discarded, sending every such report to the
    expensive, hallucination-prone vision tier for data that was sitting
    right there in the text layer.

    Scans for the first row that looks like a column header instead. Cells
    are joined before matching since the text-strategy grid sometimes
    fragments words across cells, which broke substring matching on
    individual cells.
    """
    for i, row in enumerate(table[:40]):
        joined = " ".join((c or "") for c in row).lower()
        joined = re.sub(r"\s+", " ", joined)
        has_test = any(k in joined for k in _HEAD_TEST)
        has_result = any(k in joined for k in _HEAD_RESULT)
        has_ref = any(k in joined for k in _HEAD_REF)
        if has_test and (has_result or has_ref):
            return i
    return None


def _table_to_structured_rows(table: list, page_num: int, specimen: Optional[str] = None) -> List[dict]:
    if not table or len(table) < 2:
        return []

    header_idx = _find_header_row(table)
    if header_idx is None:
        return []

    header = [(h or "").strip().lower() for h in table[header_idx]]

    def find_col(*keywords):
        for i, h in enumerate(header):
            if any(k in h for k in keywords):
                return i
        return None

    idx_test = find_col(*_HEAD_TEST)
    idx_result = find_col(*_HEAD_RESULT)
    idx_unit = find_col(*_HEAD_UNIT)
    idx_range = find_col(*_HEAD_REF)

    if idx_test is None or idx_result is None:
        return []

    table = table[header_idx:]

    rows = []
    for row in table[1:]:
        if idx_test >= len(row) or not row[idx_test]:
            continue
        built = _build_row(
            raw_name=(row[idx_test] or ""),
            value=(row[idx_result] or "") if idx_result < len(row) else "",
            unit=(row[idx_unit] or "") if idx_unit is not None and idx_unit < len(row) else "",
            ref_range=(row[idx_range] or "") if idx_range is not None and idx_range < len(row) else "",
            page_num=page_num,
            specimen=specimen,
        )
        if built:
            rows.append(built)
    return rows


# --------------------------------------------------------------------------
# Positional (columnar) extraction for whitespace-aligned reports.
# --------------------------------------------------------------------------
_ROW_TOLERANCE = 2.5      # points; words within this vertical span are one line
_COL_TOLERANCE = 6.0      # points of slack when assigning a word to a column

# "H"/"L" abnormal-flag markers printed beside the value.
_FLAG_RE = re.compile(r"(?:^|\s)([HL])(?:\s|$)")

_QUALITATIVE_VALUES = {
    "negative", "positive", "nil", "absent", "present", "trace", "clear",
    "reactive", "non-reactive", "normal", "abnormal", "detected",
    "not detected", "pale yellow", "yellow", "straw", "cloudy", "turbid",
}


def _page_lines(page) -> List[List[dict]]:
    """Groups a page's words into visual lines, left to right.

    BUG FOUND: a specimen PDF contained a hidden word from a signature
    elsewhere in the document, extracted off the left edge of the visible
    page, rotated vertically and non-upright, likely a leftover artifact
    from the source PDF being password-protected and then unlocked. Its
    coordinates happened to fall on the same row as a real result line, so
    it got prepended to that test's name. Excluding non-upright words and
    words outside the page's visible bounds keeps line-grouping to the
    actual printed left-to-right table content.
    """
    try:
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    except Exception:
        return []
    page_width = getattr(page, "width", None)
    visible = [
        w for w in words
        if w.get("upright", True)
        and w["x0"] >= 0
        and (page_width is None or w["x1"] <= page_width)
    ]
    buckets: dict = {}
    for w in visible:
        buckets.setdefault(round(w["top"] / _ROW_TOLERANCE), []).append(w)
    return [sorted(buckets[k], key=lambda w: w["x0"]) for k in sorted(buckets)]


def _header_columns(line_words: List[dict]) -> Optional[dict]:
    """If this line is a column header, returns {column_name: x_start}."""
    text = " ".join(w["text"] for w in line_words).lower()
    has_test = any(k in text for k in _HEAD_TEST)
    has_result = any(k in text for k in _HEAD_RESULT)
    has_ref = any(k in text for k in _HEAD_REF)
    if not (has_test and (has_result or has_ref)):
        return None

    cols: dict = {}
    for w in line_words:
        token = w["text"].lower().strip(".:()")
        if token.startswith(_HEAD_TEST):
            cols.setdefault("test", w["x0"])
        elif token.startswith(_HEAD_RESULT):
            cols.setdefault("result", w["x0"])
        elif token.startswith(_HEAD_UNIT):
            cols.setdefault("unit", w["x0"])
        elif token.startswith(("biological", "ref", "interval", "normal", "range")):
            cols.setdefault("ref", w["x0"])
    return cols if "test" in cols and "result" in cols else None


def _bucket_line(line_words: List[dict], cols: dict) -> dict:
    """Assigns each word on a line to the column it sits under."""
    ordered = sorted(cols.items(), key=lambda kv: kv[1])
    out = {name: [] for name, _ in ordered}
    for w in line_words:
        mid = (w["x0"] + w["x1"]) / 2
        target = ordered[0][0]
        for name, x_start in ordered:
            if mid >= x_start - _COL_TOLERANCE:
                target = name
        out[target].append(w["text"])
    return {name: " ".join(parts).strip() for name, parts in out.items()}


# Lines that are structurally in the table but are not results: signature
# blocks, qualifications, interpretive prose. Before this filter they were
# emitted as tests ("Dr. Purvish Darji = Dr. Sanjee", "MD(Path) = MD Path").
_NON_TEST_NAME_RE = re.compile(
    r"^\s*(dr\.?|prof\.?|m\.?d\.?|mbbs|dnb|md\s*\(|consultant|pathologist|"
    r"technologist|signature|verified|authoris|authoriz|approved|"
    r"end of report|interpretation|note|comment|remark|"
    # BUG FOUND (LabReport.pdf, QuantiFERON-TB panel): "Final Result" ->
    # "Negative" was extracted as its own test row and, having no NHS/NIH page
    # of its own (it isn't an analyte, it's the report's own summary verdict
    r"final result|overall result|overall interpretation|test result|"
    # "(Urine )?Quantity" alone, specimen volume submitted for testing, not a
    # diagnostic measurement with health information behind it (there is no
    # "what your urine quantity means" NHS/NIH page, nor should there be).
    r"(urine\s+)?quantity\s*$|"
    # BUG FOUND (live testing, an image-embedded report table that forced the
    # the OCR/vision fallback): with no clean text layer to anchor on, that
    # tier also picked up the surrounding letterhead, field labels and column
    # headers as if they were result rows -- "Age", "Registered On", "PID",
    r"age|sex|gender|pid|patient\s*(id|name|location)?\s*$|"
    r"client(\s*(name|code|address|add|no))?\s*$|"
    r"(registered|collected|reported|generated|approved|received|ordered)\s*"
    r"(on|by|location)?\s*$|"
    r"sample\s*collection(\s*by)?\s*$|"
    r"ref(erence)?\s*(id\s*#?\d*|by|doctor)?\s*$|"
    r"mrn|uhid|visit\s*type\s*$|"
    r"investigation\s+result(\s+reference\s*value)?\s*$)",
    re.IGNORECASE,
)


def _is_test_name(name: str) -> bool:
    name = (name or "").strip()
    if len(name) < 2 or not re.search(r"[A-Za-z]{2}", name):
        return False
    if _NON_TEST_NAME_RE.match(name):
        return False
    # Interpretive prose rather than a test label.
    return len(name.split()) <= 8


# BUG FOUND (PK0016.pdf, "P-LCC" / "Neutrophil-Lymphocyte Ratio (NLR)"): the
# same glued-multi-line-cell mechanism documented above for units
# (_GLUED_UNIT_PREFIX_RE) also corrupts the NAME column -- a stray character
def _clean_glued_name_prefix(raw_name: str) -> str:
    while "\n" in raw_name:
        first, _, rest = raw_name.partition("\n")
        if _is_test_name(first.strip()) or not _is_test_name(rest.strip()):
            break
        raw_name = rest.strip()
    return raw_name


def _build_row(raw_name: str, value: str, unit: str, ref_range: str,
               page_num: int, specimen: Optional[str] = None) -> Optional[dict]:
    """Shared validation + cleaning for BOTH deterministic strategies.

    Previously only the grid path existed and it accepted any non-empty
    value, which is how signature blocks and prose became "tests", and how
    'L 18.0' was stored verbatim as a value (leaving status 'unknown'
    because it would not parse as a number).
    """
    raw_name = (raw_name or "").strip(" .:-")
    raw_name = _clean_glued_name_prefix(raw_name)
    value, flag = _clean_value(value or "")
    raw_name, name_flag = _split_name_flag(raw_name)
    flag = flag or name_flag

    if not _is_test_name(raw_name) or not _is_result_value(value, raw_name):
        return None

    # Convert the accepted-but-not-plain-numeric forms into their final stored
    # value: a small count range takes its upper (conservative) bound;
    range_match = _SMALL_COUNT_RANGE_RE.match(value)
    if range_match:
        value = range_match.group(2)
    elif _CATEGORICAL_TEST_NAME_RE.search(raw_name):
        value = _clean_categorical_value(value)

    unit = _clean_unit(_strip_footer(unit))
    ref_range = _strip_footer(ref_range)

    # Recover the range when the sub-table's columns are offset from the page
    # page header and the unit cell has swallowed it (differential counts:
    # unit="% 40 - 80", ref="7716 /cmm 2000 - 6700").
    unit_match = _UNIT_WITH_RANGE_RE.match(unit)
    if unit_match:
        unit = unit_match.group("unit")
        ref_range = unit_match.group("range")

    test_id, is_known = _resolve_or_synthesize_id(raw_name)
    return {
        "test_id": test_id,
        "raw_name": raw_name,
        "value": value,
        "unit": unit,
        "ref_range": ref_range or None,
        "status": None,   # no LLM guess - computed in finalize_result()
        "known": is_known,
        "page": page_num,
        "flag": flag,     # printed H/L marker, kept for audit
        "specimen": specimen,  # 'urine' if this row's page had a urinalysis
                                # section heading, else None (assume blood/serum)
    }


# Page furniture that sits on the same visual line as data and otherwise gets
# swallowed into the reference-range column ("0 - 14 Dr.Yash Shah MD Path
# Page 1 of 19"), corrupting the range before it reaches parse_ref_range_string().
_FOOTER_RE = re.compile(
    r"\s*(page\s+\d+\s+of\s+\d+|#\s*referred\s+test|dr\.?\s*[a-z]|"
    r"\bmd\s*path\b|\bm\.?d\.?\b\s*\(|end of report).*$",
    re.IGNORECASE,
)

# A unit cell that has absorbed the reference range, e.g. unit="% 40 - 80"
# on the differential-count sub-table, whose columns don't line up with the
# page's main header.
_UNIT_WITH_RANGE_RE = re.compile(
    r"^(?P<unit>[^\s\d]{1,12})\s+(?P<range>-?\d+\.?\d*\s*-\s*-?\d+\.?\d*)$"
)


def _strip_footer(text: str) -> str:
    return _FOOTER_RE.sub("", text or "").strip()


# BUG FOUND (a specimen report, Epithelial Cells / Urinary RBC): pdfplumber
# joins a multi-line table cell with an embedded "\n", and on this report's
# layout the unit cell for these rows picked up a single stray character from
_GLUED_UNIT_PREFIX_RE = re.compile(r"^[A-Za-z]\s*\n\s*")


def _clean_unit(unit: str) -> str:
    return _GLUED_UNIT_PREFIX_RE.sub("", unit or "")


# BUG FOUND: "H10570" (flag glued to the number, no space) failed the
# numeric check and silently dropped an abnormal WBC count.
_GLUED_FLAG_RE = re.compile(r"^([HL])(\d)")

# BUG FOUND (same report, Urine Glucose): the printed value "Present (+)"
# carries a trailing qualitative marker that isn't in _QUALITATIVE_VALUES, so
# the exact-match check failed and an abnormal urinalysis result (glucose in
_TRAILING_QUALITATIVE_MARKER_RE = re.compile(r"\s*[\(\[][+-][\)\]]\s*$")


def _clean_value(value: str) -> tuple:
    """Splits a printed result into (value, abnormal_flag).
    Lab reports print the flag beside the number ('H 141.0'), and it must not
    end up inside the value or glued onto the test name."""
    value = _TRAILING_QUALITATIVE_MARKER_RE.sub("", value)
    flag_match = _FLAG_RE.search(value)
    flag = flag_match.group(1).upper() if flag_match else None
    cleaned = _FLAG_RE.sub(" ", value).strip()

    glued = _GLUED_FLAG_RE.match(cleaned)
    if glued:
        flag = flag or glued.group(1)
        cleaned = cleaned[1:]

    return cleaned, flag


def _split_name_flag(name: str) -> tuple:
    """The printed H/L marker sits between the name and the value, so when it
    falls left of the Result column it buckets into the NAME ('HbA1c H',
    'Urea L', 'Fasting Blood Sugar H'). Pull it back off."""
    match = re.search(r"\s+([HL])$", name)
    if match:
        return name[: match.start()].strip(), match.group(1).upper()
    return name, None


# BUG FOUND (same report, Pus Cells / Epithelial Cells): microscopy fields are
# conventionally printed as a small count range ("1-2 /hpf"), not a single
# number, so these rows failed the single-number check and were dropped.
_SMALL_COUNT_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")

# BUG FOUND (same report, ABO Type): printed as a quoted categorical letter
# ('"A"'), which is neither numeric nor in _QUALITATIVE_VALUES, so the row was
# dropped.
_CATEGORICAL_TEST_NAME_RE = re.compile(r"\b(abo|blood\s*group|rh\s*\(?d\)?)\b", re.IGNORECASE)


def _clean_categorical_value(value: str) -> str:
    return value.strip().strip('"\'').strip()


# BUG FOUND (live testing, an image-embedded report table that forced the
# OCR/vision fallback): with no clean text layer to anchor on, that tier also
# picked up the surrounding letterhead, field labels and column headers as if
_MAX_PLAUSIBLE_RESULT_MAGNITUDE = 1e15


def _is_result_value(value: str, raw_name: str = "") -> bool:
    if not value:
        return False
    if re.fullmatch(r"-?\d+\.?\d*", value):
        try:
            if abs(float(value)) > _MAX_PLAUSIBLE_RESULT_MAGNITUDE:
                return False
        except ValueError:
            return False
        return True
    if _SMALL_COUNT_RANGE_RE.match(value):
        return True
    if value.lower() in _QUALITATIVE_VALUES:
        return True
    if _CATEGORICAL_TEST_NAME_RE.search(raw_name):
        token = _clean_categorical_value(value)
        return bool(token) and len(token) <= 4 and token.replace(" ", "").isalpha()
    return False


def _columnar_rows_for_page(page, page_num: int) -> List[dict]:
    lines = _page_lines(page)
    rows: List[dict] = []
    cols = None
    specimen = _page_specimen(page.extract_text() or "")

    for line_words in lines:
        # A page can carry several sub-tables (e.g. the differential count
        # under the CBC), each with its own column offsets - so keep watching
        # for new header lines rather than locking onto the first one.
        maybe_header = _header_columns(line_words)
        if maybe_header:
            cols = maybe_header
            continue
        if not cols:
            continue

        cells = _bucket_line(line_words, cols)
        ref_range = cells.get("ref", "").strip()
        built = _build_row(
            raw_name=cells.get("test", ""),
            value=cells.get("result", ""),
            unit=cells.get("unit", ""),
            ref_range=ref_range,
            page_num=page_num,
            specimen=specimen,
        )

        if built:
            rows.append(built)
        elif rows and not cells.get("result", "").strip():
            # Continuation of a multi-line reference band (e.g. Cholesterol's
            # bands printed across three lines), without this it truncates
            # to the first line and the healthy band can't be found.
            continuation = _strip_footer(ref_range)
            # Only genuine band text - a bare footer or a method annotation is
            # not part of the range.
            if continuation and re.search(r"[<>:]|\d", continuation):
                previous = rows[-1]
                previous["ref_range"] = f"{previous['ref_range'] or ''} {continuation}".strip()

    return rows


def _extract_columnar_rows(file_bytes: bytes) -> List[dict]:
    out: List[dict] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    out.extend(_columnar_rows_for_page(page, page_num))
                except Exception as e:
                    print(f"columnar extraction failed on page {page_num}: {e}")
    except Exception as e:
        print(f"columnar extraction failed: {e}")
    return out


def parse_structured_tables(file_bytes: bytes) -> List[dict]:
    """Tier 1: free, instant, LLM-free, and the most trustworthy tier since
    ref_range comes straight from a header-mapped column, not an LLM
    transcription. Two complementary deterministic strategies, header-mapped
    grid tables and positional columnar extraction, are merged so a report
    that only one strategy understands is still handled."""
    tables, ruled_pages = _extract_tables_with_pages(file_bytes)
    results = []
    for page_num, table, specimen in tables:
        results.extend(_table_to_structured_rows(table, page_num, specimen))

    # Positional pass for every page that had no ruled table. Done per page
    # (not "only if the whole document came up short") because a report can
    # mix both: the Sterling report has one ruled table on page 1 and 18
    # whitespace-aligned pages behind it, and judging by the document total
    # let those 18 pages fall through to the vision LLM.
    columnar = [r for r in _extract_columnar_rows(file_bytes)
                if r["page"] not in ruled_pages]
    results.extend(columnar)
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
    """Cost-ordered pipeline: table extraction, vision LLM, text LLM, then
    regex, each tried only if the previous tier found too little. Every
    result is merged through the same finalize/dedupe path across tiers.
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