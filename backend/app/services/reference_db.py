"""
Reference Database Service.
Handles test alias resolution, normal range lookups by demographic (sex/age),
and status flagging for lab values.

CHANGED: flag_result() priority is now:
  1. The reference range PRINTED ON THIS REPORT (most authoritative - it's
     lab/method/instrument/demographic-specific, unlike a static lookup table).
  2. Our static REFERENCE_DB (fallback for tests the document didn't print
     a numeric range for, e.g. regex/OCR-only extraction with no range column).
  3. The LLM's own status guess (a judgement call, not a comparison against
     a specific number - lowest-trust numeric-adjacent source).
  4. Qualitative string matching, unchanged.

Also added: finalize_result() and dedupe_results(), shared helpers so every
extraction tier (table / vision LLM / text LLM / regex) in pdf_parser.py
computes status the exact same deterministic way, rather than each tier
producing its own inconsistent answer.
"""
import re
from typing import Dict, Any, Optional, List, Tuple
from app.models.schemas import RangeStatus

# Canonical alias mapping table
ALIAS_MAP = {
    "hb": "HGB",
    "haemoglobin": "HGB",
    "hemoglobin": "HGB",
    "hgb": "HGB",
    "hba1c": "HBA1C",
    "glycated_haemoglobin": "HBA1C",
    "wbc": "WBC",
    "white_blood_cell": "WBC",
    "rbc": "RBC",
    "plt": "PLT",
    "platelets": "PLT",
}

# Reference ranges database by test_id
REFERENCE_DB = {
    "HGB": {
        "female": {"min": 11.5, "max": 16.0},
        "male": {"min": 13.0, "max": 17.5},
        "default": {"min": 11.5, "max": 17.5},
        "source": "NHS UK",
        "plain_english": "Haemoglobin is the protein in red blood cells that carries oxygen throughout the body.",
        "low_means": "May indicate anaemia or blood loss.",
        "high_means": "May indicate dehydration or polycythaemia.",
    },
    "HBA1C": {
        "default": {"min": 20.0, "max": 42.0},
        "source": "NHS UK",
        "plain_english": "HbA1c measures your average blood sugar levels over the past 2 to 3 months.",
        "low_means": "Generally normal, but very low levels can occur in certain blood conditions.",
        "high_means": "Indicates prediabetes or diabetes.",
    },
}


def resolve_test_id(raw_name: str) -> Optional[str]:
    """
    Resolves raw test names/aliases to canonical test IDs in the database.
    Returns canonical ID string (e.g. 'HGB') or None if unmapped/unknown.
    """
    if not raw_name or not isinstance(raw_name, str):
        return None
    cleaned = raw_name.strip().lower()
    return ALIAS_MAP.get(cleaned)


def get_normal_range(test_id: str, sex: str = "unknown", age: int = 30) -> Optional[Dict[str, float]]:
    """
    Retrieves the normal range for a test based on patient demographics.
    """
    entry = REFERENCE_DB.get(test_id)
    if not entry:
        return None

    # Check sex-specific ranges first, fallback to default range
    sex_key = sex.lower() if sex else "unknown"
    if sex_key in entry:
        return entry[sex_key]
    return entry.get("default")


def get_reference_data(test_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves full reference metadata for a test ID.
    """
    return REFERENCE_DB.get(test_id)


# --------------------------------------------------------------------------
# NEW: canonical name normalization, shared by every tier's ID synthesis
# --------------------------------------------------------------------------
# CHANGED: pdf_parser.py and llm_extractor.py each used to slug the FULL raw
# test name verbatim into a synthetic test_id when resolve_test_id() found
# no alias. That meant "Creatinine", "Creatinine, Serum" and "Creatinine
# (24 hour)" produced three different synthetic IDs, and dedupe_results()
# below (which keys on test_id) never merged them - the same analyte showed
# up multiple times with different values in the output. This strips only
# qualifiers that describe HOW/WHERE the sample was taken, not WHAT is
# being measured - it deliberately leaves clinically-distinguishing
# qualifiers alone, e.g. "Direct Bilirubin" / "Total Bilirubin" /
# "Unconjugated Bilirubin" are genuinely different values and must stay
# separate test_ids.
_NON_DISTINGUISHING_QUALIFIERS = re.compile(
    r"\b(serum|plasma|urine|edta\s*blood|whole\s*blood|fluoride\s*plasma|"
    r"\d+\s*hour(?:s)?|qualitative|quantitative|random\s*sample)\b",
    re.IGNORECASE,
)


def canonicalize_test_name(raw_name: str) -> str:
    """Produces a stable synthetic test_id for tests with no known alias,
    so the same analyte extracted with slightly different wording across
    tiers/batches still dedupes to one entry. NOT a substitute for growing
    ALIAS_MAP - genuinely different phrasings of the same well-known test
    (e.g. "Fasting Glucose" vs "Glucose (Fasting)") still need a real alias
    entry to merge; this only strips sample-collection/method noise."""
    if not raw_name or not isinstance(raw_name, str):
        return "UNKNOWN_TEST"
    cleaned = _NON_DISTINGUISHING_QUALIFIERS.sub(" ", raw_name)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned.upper()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "UNKNOWN_TEST"


# --------------------------------------------------------------------------
# NEW: parsing the reference range as actually printed on the report
# --------------------------------------------------------------------------
# Real reports use several formats, and real OCR/text extraction sometimes
# inserts stray whitespace (e.g. "6 .0 - 8.0 pH" was seen verbatim in a real
# report), so we normalize that first.

# CHANGED: allow an optional unit symbol (e.g. "%") directly after a number
# and before the dash - printed bands like "5.7% - 6.4%" wouldn't match
# the old pattern at all, since "%" sat between the digits and the dash.
_RANGE_RE = re.compile(r"(-?\d+\.?\d*)\s*%?\s*-\s*(-?\d+\.?\d*)\s*%?")
_LT_RE = re.compile(r"[<\u2264]\s*(-?\d+\.?\d*)")
_GT_RE = re.compile(r"[>\u2265]\s*(-?\d+\.?\d*)")
_STRAY_DECIMAL_RE = re.compile(r"(\d)\s+\.\s*(\d)")  # fixes "6 .0" -> "6.0"

# CHANGED: many printed reference ranges are actually several NAMED bands,
# not one simple low-high pair - e.g. Cholesterol's "Desirable: <200 /
# Borderline High: 200-239 / High: >240", or HbA1c's "Non-Diabetes: <5.7% /
# Pre-Diabetes: 5.7-6.4% / Diabetes: >6.5%". The naive single-pass regex
# above just grabs whichever "number - number" pattern it finds FIRST in
# the whole string - for Cholesterol that's "200-239" (the *borderline
# high* band), so a genuinely healthy 189 mg/dL got flagged "Below normal
# range" in production. _parse_labeled_bands looks for a band whose label
# reads as the healthy/reference band and uses ONLY that band's numbers.
_NORMAL_BAND_KEYWORDS = (
    "desirable", "normal", "optimal", "sufficiency", "negative",
    "non-reactive", "non reactive", "non-diabetes", "non diabetes",
    "not diabetic", "good control", "within normal",
)

_BAND_SEGMENT_RE = re.compile(
    r"([A-Za-z][A-Za-z \-/]{1,40}?)\s*:\s*([^:]*?)"
    r"(?=(?:[A-Za-z][A-Za-z \-/]{1,40}?\s*:)|$)"
)


def _parse_labeled_bands(ref_range: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Returns (low, high) from the band labeled as the healthy/reference
    range, or None if this text isn't multi-band (fewer than 2 "label:
    value" segments) or no band reads as the healthy one - callers should
    fall back to the plain single-range parsing in that case."""
    if not ref_range:
        return None
    segments = _BAND_SEGMENT_RE.findall(ref_range)
    if len(segments) < 2:
        return None

    for label, expr in segments:
        label_l = label.strip().lower()
        # CHANGED: plain substring containment ("sufficiency" in label)
        # false-matches "Insufficiency" too, since "sufficiency" sits
        # inside it with no word break - which would have picked Vitamin
        # D's Insufficiency band (10-30) instead of Sufficiency (30-100).
        # Word-boundary matching avoids that.
        if not any(re.search(rf"\b{re.escape(k)}\b", label_l) for k in _NORMAL_BAND_KEYWORDS):
            continue
        m = _RANGE_RE.search(expr)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                pass
        m = _LT_RE.search(expr)
        if m:
            try:
                return None, float(m.group(1))
            except ValueError:
                pass
        m = _GT_RE.search(expr)
        if m:
            try:
                return float(m.group(1)), None
            except ValueError:
                pass
    return None  # multi-band but no clearly-labeled healthy band - fall through


def parse_ref_range_string(ref_range: Optional[str]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """
    Parses a reference range EXACTLY as printed on a lab report into
    (low, high) bounds, where either side can be None for an open-ended
    range (e.g. "< 16.7" -> (None, 16.7)).

    Returns None if the string is purely categorical/unparseable (e.g.
    "Negative", "Non Reactive : <1.0" without extractable digits, "Pale
    Yellow") - those cases fall through to qualitative handling elsewhere.

    Handles the formats actually observed in real reports:
      "13.0 - 16.5"            -> (13.0, 16.5)
      "6 .0 - 8.0 pH"          -> (6.0, 8.0)   (stray-whitespace artifact)
      "< 16.7"                 -> (None, 16.7)
      "> 60.0"                 -> (60.0, None)
      "Non Reactive : <1.0"    -> (None, 1.0)
    """
    if not ref_range or not isinstance(ref_range, str):
        return None

    cleaned = _STRAY_DECIMAL_RE.sub(r"\1.\2", ref_range.strip())

    # CHANGED: try labeled-band parsing first (see _parse_labeled_bands
    # above). Only takes effect when the text actually has 2+ "label:
    # value" bands; otherwise returns None immediately and everything
    # below runs exactly as before.
    banded = _parse_labeled_bands(cleaned)
    if banded is not None:
        return banded

    m = _RANGE_RE.search(cleaned)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    m = _LT_RE.search(cleaned)
    if m:
        try:
            return None, float(m.group(1))
        except ValueError:
            pass

    m = _GT_RE.search(cleaned)
    if m:
        try:
            return float(m.group(1)), None
        except ValueError:
            pass

    return None


def flag_result(
    value: Any,
    normal_range: Optional[Dict[str, float]] = None,
    llm_status: Optional[str] = None,
    document_ref_range: Optional[str] = None,
) -> RangeStatus:
    """
    Determines the clinical status of a result. Priority order (highest
    trust first) - see module docstring for rationale:
      1. document_ref_range  (printed on THIS report)
      2. normal_range        (our static REFERENCE_DB lookup)
      3. llm_status           (LLM's own qualitative guess)
      4. qualitative string matching
    """
    # 1. Highest priority: the range this specific lab actually printed.
    if document_ref_range:
        bounds = parse_ref_range_string(document_ref_range)
        if bounds is not None:
            try:
                val = float(value)
                low, high = bounds
                if low is not None and val < low:
                    return RangeStatus.LOW
                if high is not None and val > high:
                    return RangeStatus.HIGH
                return RangeStatus.NORMAL
            except (ValueError, TypeError):
                pass  # value wasn't numeric (e.g. "Present (+)") - fall through

    # 2. Static reference DB, only reached if the document's own range
    #    couldn't be parsed numerically above.
    if value is not None:
        try:
            val = float(value)
            if normal_range:
                if "min" in normal_range and val < normal_range["min"]:
                    return RangeStatus.LOW
                if "max" in normal_range and val > normal_range["max"]:
                    return RangeStatus.HIGH
                return RangeStatus.NORMAL
        except (ValueError, TypeError):
            pass

    # 3. LLM's own qualitative judgement - used only once both deterministic
    #    numeric comparisons above were unavailable or inapplicable.
    if llm_status and llm_status in RangeStatus.__members__.values():
        return RangeStatus(llm_status)

    if value is None:
        return RangeStatus.UNKNOWN

    # 4. Qualitative fallback safety net
    val_str = str(value).strip().lower()
    if val_str in ["negative", "nil", "clear", "absent", "normal", "pale yellow", "straw"]:
        return RangeStatus.NORMAL
    elif val_str in ["positive", "1+", "2+", "3+", "4+", "reactive", "cloudy"]:
        return RangeStatus.HIGH

    return RangeStatus.UNKNOWN


# --------------------------------------------------------------------------
# NEW: shared post-processing helpers used by every extraction tier
# --------------------------------------------------------------------------

def finalize_result(result: dict, patient_sex: str = "unknown", patient_age: int = 30) -> dict:
    """
    Takes a raw extracted result dict from ANY tier (deterministic table,
    vision LLM, text LLM, or regex) and computes the final, authoritative
    status + normal-range bounds the same way every time.

    Mutates and returns `result`, adding/overwriting:
      - status              final RangeStatus.value string
      - llm_status          the tier's original raw guess, kept for audit/debugging
      - normal_range_min/max  resolved bounds, for downstream display or explain_all_test_results_batched
    """
    test_id = result.get("test_id") or "UNKNOWN_TEST"
    value = result.get("value")
    doc_range = result.get("ref_range")
    llm_status = result.get("status")

    range_min, range_max = None, None
    bounds = parse_ref_range_string(doc_range) if doc_range else None
    if bounds:
        range_min, range_max = bounds
    else:
        static_range = get_normal_range(test_id, sex=patient_sex, age=patient_age)
        if static_range:
            range_min = static_range.get("min")
            range_max = static_range.get("max")

    normal_range = None
    if range_min is not None or range_max is not None:
        normal_range = {}
        if range_min is not None:
            normal_range["min"] = range_min
        if range_max is not None:
            normal_range["max"] = range_max

    status = flag_result(value, normal_range=normal_range, llm_status=llm_status,
                          document_ref_range=doc_range)

    result["status"] = status.value if hasattr(status, "value") else str(status)
    result["llm_status"] = llm_status
    result["normal_range_min"] = range_min
    result["normal_range_max"] = range_max
    return result


def dedupe_results(results: List[dict]) -> List[dict]:
    """
    De-duplicates extracted results by test_id (results can otherwise repeat
    across table + LLM tiers, or across vision batches on long reports).
    Keeps the most informative entry per test_id: prefers one with a
    parseable printed ref_range, then one with a recognized/known test_id,
    then first-seen.
    """
    best: Dict[str, dict] = {}
    order: List[str] = []

    def score(r: dict) -> tuple:
        return (bool(parse_ref_range_string(r.get("ref_range"))), bool(r.get("known")))

    for r in results:
        tid = r.get("test_id") or "UNKNOWN_TEST"
        if tid not in best:
            best[tid] = r
            order.append(tid)
            continue
        if score(r) > score(best[tid]):
            best[tid] = r

    return [best[tid] for tid in order]