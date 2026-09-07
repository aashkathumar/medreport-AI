"""Reference database service: test alias resolution, normal-range lookups
by demographic, and status flagging.

flag_result()'s priority order, highest trust first: the range printed on
that specific report, then the static REFERENCE_DB lookup, then the LLM's
own qualitative guess, then plain string matching as a last resort.

finalize_result() and dedupe_results() are shared helpers so every
extraction tier in pdf_parser.py computes status the same deterministic
way instead of each producing its own inconsistent answer.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from app.models.schemas import RangeStatus

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# Canonical alias mapping table.
ALIAS_MAP = {
    # BUG FOUND (PK0016.pdf): these had no alias, so they canonicalized to a
    # synthetic id that never matched the corpus's real tag for the analyte.
    "leukocyte_esterase": "URINE_LEUKOCYTES",
    "absolute_lymphocyte_count": "LYMPHOCYTES",
    "absolute_monocyte_count": "MONOCYTES",
    "absolute_eosinophil_count": "EOSINOPHILS",
    "absolute_basophil_count": "BASOPHILS",
    "blood_sugar_fasting": "GLUCOSE",
    "chloride": "CL",
    # "Urinary Transparency" and "Clearity" are the same measurement under
    # two labs' naming, safe to merge since they never share one report.
    "urinary_transparency": "CLEARITY",
    # BUG FOUND (PK0016.pdf): both labels were missing an alias entirely,
    # bypassing the precomputed cache and forcing a fresh live generation.
    "total_iron_binding_capacity": "TOTAL_IRON_BINDING_CAPACITY_TIBC",
    "25_oh_vitamin_d_total": "VIT_D",

    # BUG FOUND (QuantiFERON-TB Gold panel): none of these 4 tube names had
    # an alias, even though the corpus already covers this exact IGRA test.
    "tb_nil_tube": "TUBERCULOSIS",
    "tb_antigen_tube": "TUBERCULOSIS",
    "tb_mitogen_tube": "TUBERCULOSIS",
    "tb_ag_minus_nil": "TUBERCULOSIS",

    "hb": "HGB", "haemoglobin": "HGB", "hemoglobin": "HGB", "hgb": "HGB",
    "hba1c": "HBA1C", "glycated_haemoglobin": "HBA1C", "a1c": "HBA1C",
    "wbc": "WBC", "white_blood_cell": "WBC", "white_blood_cell_count": "WBC",
    "plt": "PLT", "platelets": "PLT", "platelet_count": "PLT",
    "chol": "CHOL", "cholesterol": "CHOL", "total_cholesterol": "CHOL",
    "ldl": "LDL", "ldl_cholesterol": "LDL",
    "hdl": "HDL", "hdl_cholesterol": "HDL",
    "tsh": "TSH",
    "ferritin": "FERRITIN",
    "vitamin_d": "VIT_D", "vit_d": "VIT_D", "25_oh_vitamin_d": "VIT_D",
    "vitamin_b12": "B12", "b12": "B12", "vit_b12": "B12",
    "creatinine": "CREATININE",
    "alt": "ALT", "alanine_aminotransferase": "ALT",
    "glucose": "GLUCOSE", "blood_glucose": "GLUCOSE", "fasting_glucose": "GLUCOSE",
    "crp": "CRP", "c_reactive_protein": "CRP",
    "urine_protein": "URINE_PROTEIN", "protein_urine": "URINE_PROTEIN",
    "urine_glucose": "URINE_GLUCOSE", "glucose_urine": "URINE_GLUCOSE",
    "urine_ph": "URINE_PH", "ph_urine": "URINE_PH",
    "urine_leukocytes": "URINE_LEUKOCYTES", "leukocytes_urine": "URINE_LEUKOCYTES",
    "urine_nitrites": "URINE_NITRITES", "nitrites_urine": "URINE_NITRITES",
    "urine_ketones": "URINE_KETONES", "ketones_urine": "URINE_KETONES",
    # BUG FOUND: SGPT/SGOT are the older names for ALT/AST, unaliased before,
    # so real liver-enzyme results were wrongly reported as "not available".
    "sgpt": "ALT", "alt_sgpt": "ALT", "sgpt_alt": "ALT",
    "sgot": "AST", "ast_sgot": "AST", "sgot_ast": "AST",
    "ast": "AST", "aspartate_aminotransferase": "AST",
    "direct_ldl": "LDL", "ldl_direct": "LDL", "ldl_cholesterol_direct": "LDL",
    "fasting_blood_sugar": "GLUCOSE", "fasting_blood_glucose": "GLUCOSE",
    "blood_sugar": "GLUCOSE", "random_blood_sugar": "GLUCOSE",
    # Red-cell indices: curated on the CBC page (see build_rag_chunks.py).
    "mch": "MCH", "mchc": "MCHC", "mcv": "MCV",
    "rdw_cv": "RDW_CV", "rdw": "RDW",
    "abo_type": "ABO_TYPE", "blood_group": "ABO_TYPE",
    "rh_d_type": "RH_D_TYPE", "rh_type": "RH_D_TYPE", "rhesus_factor": "RH_D_TYPE",
    # Resolves the report's exact wording to the corpus's "HIV" tag, without
    # this it canonicalised to a unique id no source could ever match.
    "hiv_i_ii_ab_ag_with_p24_ag": "HIV", "hiv_ab_ag": "HIV", "hiv_screening": "HIV",
    "hbsag": "HBSAG", "hepatitis_b_surface_antigen": "HBSAG",
}

# BUG FOUND (Rh (D) Type): the qualitative fallback flagged blood-type
# fields as "above normal range", a category error, not a real finding.
BLOOD_TYPE_TEST_IDS = {"ABO_TYPE", "RH_D_TYPE"}


# Grounding-only redirects: test_id -> the id whose page also explains this
# analyte. Deliberately not in ALIAS_MAP, since that key also drives dedupe.
GROUNDING_ALIASES = {
    # BUG FOUND: used to redirect HB_A/HB_A2/FOETAL_HB here on the assumption
    # a general haemoglobin page covers all fractions, it doesn't.
    "TIBC": "IRON",
    "TOTAL_IRON_BINDING_CAPACITY": "IRON",
    "TOTAL_IRON_BINDING_CAPACITY_TIBC": "IRON",
    "TRANSFERRIN_SATURATION": "IRON",
    # No dedicated "ratio" page exists, so these ground on the component
    # analyte's page instead.
    "SGOT_SGPT_RATIO": "AST",
    "TC_HDL_RATIO": "CHOL",
    "90_DAY_AVERAGE_BLOOD_GLUCOSE": "HBA1C",
    # BUG FOUND (PK0016.pdf): urine RBC and blood RBC count are different
    # tests; without this it grounded on the wrong (blood) page.
    "URINARY_RBC": "URINE_BLOOD",
}

# BUG FOUND (urinalysis panel): bare labels like "Bilirubin"/"Red Cells"
# exist on both blood and urine panels; without a specimen tag, grounding
# always cited the blood page even for a urine dipstick result.
SPECIMEN_GROUNDING_ALIASES = {
    ("BILIRUBIN", "urine"): "URINE_BILIRUBIN",
    ("RED_CELLS", "urine"): "URINE_BLOOD",
    ("PUS_CELLS", "urine"): "URINE_LEUKOCYTES",
    ("COLOUR", "urine"): "URINE_COLOUR",
    # Same bug class as COLOUR above: bare "pH" missed the curated URINE_PH
    # entry entirely.
    ("PH", "urine"): "URINE_PH",
    # BUG FOUND (a different lab): "Blood [In Urine]" needs its own redirect,
    # distinct from Sterling's "Red Cells" naming for the same test.
    ("BLOOD", "urine"): "URINE_BLOOD",
    # BUG FOUND (PK0016.pdf): ungrounded, fell back to the blood RBC page.
    ("RBC", "urine"): "URINE_BLOOD",
    # A lab printing bare "Protein"/"Ketones"/"Nitrites" (no "Urine" prefix)
    # needs the specimen tag to find the same curated page.
    ("PROTEIN", "urine"): "URINE_PROTEIN",
    ("KETONES", "urine"): "URINE_KETONES",
    ("NITRITES", "urine"): "URINE_NITRITES",
    # BUG FOUND: "Urinary Glucose" canonicalised to bare GLUCOSE, silently
    # matching the blood-glucose curated entry instead.
    ("GLUCOSE", "urine"): "URINE_GLUCOSE",
}

# Other curated blood-test ids (blood_tests.json) that a "Urinary X"-style
# name would canonicalise onto identically, if a report used that phrasing:


def grounding_test_id(test_id: str, specimen: Optional[str] = None) -> str:
    """The test_id to retrieve reference passages under. Identity-preserving
    for everything except the redirects above."""
    if specimen:
        specimen_redirect = SPECIMEN_GROUNDING_ALIASES.get((test_id, specimen))
        if specimen_redirect:
            return specimen_redirect
    return GROUNDING_ALIASES.get(test_id, test_id)


def _load_reference_db() -> Dict[str, Dict[str, Any]]:
    """Builds REFERENCE_DB from the curated NHS UK / NIH MedlinePlus JSON
    files rather than a small hand-maintained dict, so every test those
    files cover has real grounding available."""
    db: Dict[str, Dict[str, Any]] = {}
    for filename in ("blood_tests.json", "urine_tests.json"):
        path = _DATA_DIR / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for test in data.get("tests", []):
            ranges = test.get("normal_range", {}) or {}
            entry = {
                "source": test.get("source", "NHS UK"),
                "plain_english": test.get("plain_english", ""),
                "low_means": test.get("low_means", ""),
                "high_means": test.get("high_means", ""),
                "unit": test.get("unit", ""),
            }
            if "male" in ranges:
                entry["male"] = ranges["male"]
            if "female" in ranges:
                entry["female"] = ranges["female"]
            entry["default"] = ranges.get("all") or ranges.get("male") or ranges.get("female")
            db[test["id"]] = entry

            alias_key = re.sub(r"[^a-z0-9]+", "_", test["name"].strip().lower()).strip("_")
            ALIAS_MAP.setdefault(alias_key, test["id"])
    return db


REFERENCE_DB = _load_reference_db()


def _load_generated_aliases() -> None:
    """Merges LLM-generated synonym candidates (see
    scripts/generate_test_synonyms.py) into ALIAS_MAP. The generator
    already excludes anything ambiguous or colliding with an existing
    alias; setdefault() here is a second, defense-in-depth guarantee that
    a hand-curated entry can never be overwritten by a generated one."""
    path = _DATA_DIR / "llm_generated_aliases_accepted.json"
    if not path.exists():
        return
    try:
        generated = json.loads(path.read_text())
    except Exception as e:
        print(f"Could not load generated aliases from {path}: {e}")
        return
    for key, test_id in generated.items():
        ALIAS_MAP.setdefault(key, test_id)


_load_generated_aliases()


def _alias_key(raw_name: str) -> str:
    """Normalises a name to the same shape ALIAS_MAP is keyed by:
    lowercase, non-alphanumerics collapsed to single underscores."""
    return re.sub(r"[^a-z0-9]+", "_", raw_name.strip().lower()).strip("_")


def resolve_test_id(raw_name: str) -> Optional[str]:
    """Resolves raw test names/aliases to canonical test IDs. Returns a
    canonical ID string (e.g. 'HGB') or None if unmapped.

    BUG FOUND: the lookup used to compare `raw_name.strip().lower()`
    against ALIAS_MAP's underscore-separated keys, so it only ever matched
    single-word names; multi-word ones like "White Blood Cell Count" or
    "Total Cholesterol" always returned None and silently lost both their
    static range and their curated grounding text. Normalising the lookup
    the same way the keys are built fixed 8 of the 21 curated tests.

    Also falls back to the qualifier-stripped form, so "Cholesterol, Serum"
    and "Creatinine (24 hour)" resolve to the same entry as the bare name.
    """
    if not raw_name or not isinstance(raw_name, str):
        return None

    direct = ALIAS_MAP.get(_alias_key(raw_name))
    if direct:
        return direct

    # canonicalize_test_name() strips only non-distinguishing qualifiers
    # (serum/plasma/urine/24 hour/...), see its docstring for what it keeps.
    stripped = _alias_key(canonicalize_test_name(raw_name))
    return ALIAS_MAP.get(stripped)


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
# Strips only how/where a sample was taken, not what's measured, so
# "Creatinine (24 hour)" and "Creatinine, Serum" dedupe to one test_id.
_NON_DISTINGUISHING_QUALIFIERS = re.compile(
    r"\b(serum|plasma|urine|urinary|edta\s*blood|whole\s*blood|fluoride\s*plasma|"
    r"\d+\s*hour(?:s)?|qualitative|quantitative|random\s*sample)\b",
    re.IGNORECASE,
)

# BUG FOUND (PK0016.pdf): strips "[In Urine]"-style bracketed notes some
# labs append, which produced a different synthetic id than other labs'
# bare naming for the same test.
_BRACKETED_SPECIMEN_NOTE = re.compile(
    r"[\[\(]\s*in\s+(urine|blood|serum|plasma|stool|csf)\s*[\]\)]",
    re.IGNORECASE,
)


# BUG FOUND (PK0016.pdf): stripping "urine" collapsed "Urinary RBC" onto
# bare "RBC", a blood-context id, describing the wrong test entirely.
_AMBIGUOUS_WITHOUT_SPECIMEN = {"RBC", "WBC"}
_URINE_QUALIFIER_RE = re.compile(r"\b(urine|urinary)\b", re.IGNORECASE)


def canonicalize_test_name(raw_name: str) -> str:
    """Produces a stable synthetic test_id for tests with no known alias, so
    the same analyte extracted with slightly different wording still
    dedupes to one entry. Not a substitute for growing ALIAS_MAP: genuinely
    different phrasings still need a real alias to merge; this only strips
    sample-collection/method noise."""
    if not raw_name or not isinstance(raw_name, str):
        return "UNKNOWN_TEST"
    cleaned = _BRACKETED_SPECIMEN_NOTE.sub(" ", raw_name)
    stripped_further = _NON_DISTINGUISHING_QUALIFIERS.sub(" ", cleaned)
    candidate = re.sub(r"[^A-Za-z0-9]+", "_", stripped_further.upper()).strip("_")
    candidate = re.sub(r"_+", "_", candidate)
    if candidate in _AMBIGUOUS_WITHOUT_SPECIMEN and _URINE_QUALIFIER_RE.search(raw_name):
        # Keep the urine qualifier instead of stripping it, so this stays a
        # distinct id from the blood-context "RBC"/"WBC".
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned.upper()).strip("_")
        return re.sub(r"_+", "_", cleaned) or "UNKNOWN_TEST"
    return candidate or "UNKNOWN_TEST"


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

# CHANGED: some ranges are several named bands, not one low-high pair (e.g.
# Cholesterol's Desirable/Borderline/High). A naive regex grabbed whichever
# band came first, flagging a healthy result as abnormal.
_NORMAL_BAND_KEYWORDS = (
    "desirable", "normal", "optimal", "sufficiency", "negative",
    "non-reactive", "non reactive", "non-diabetes", "non diabetes",
    "not diabetic", "good control", "within normal",
)

_BAND_SEGMENT_RE = re.compile(
    r"([A-Za-z][A-Za-z \-/]{1,40}?)\s*:\s*([^:]*?)"
    r"(?=(?:[A-Za-z][A-Za-z \-/]{1,40}?\s*:)|$)"
)

# BUG FOUND (HDL Cholesterol): HDL is protective (higher is good), so a lab
# labels "High" as the desirable band, not the abnormal one, which the
# generic parser above didn't recognise. Handles that specific shape.
def _parse_low_high_labeled_bands(ref_range: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    segments = _BAND_SEGMENT_RE.findall(ref_range)
    if len(segments) != 2:
        return None

    labels = [label.strip().lower() for label, _ in segments]
    if sorted(labels) != ["high", "low"]:
        return None

    low_expr = next(expr for label, expr in segments if label.strip().lower() == "low")
    m = _LT_RE.search(low_expr) or _RANGE_RE.search(low_expr)
    if m:
        try:
            return float(m.group(1)), None
        except ValueError:
            pass
    return None


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
        # CHANGED: word-boundary match, plain substring check false-matched
        # "Insufficiency" as containing "sufficiency".
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
    """Parses a reference range exactly as printed on a lab report into
    (low, high) bounds, where either side can be None for an open-ended
    range (e.g. "< 16.7" -> (None, 16.7)). Returns None if the string is
    purely categorical/unparseable ("Negative", "Pale Yellow"), which
    falls through to qualitative handling elsewhere. Handles the formats
    actually seen on specimen reports, including stray-whitespace OCR
    artifacts like "6 .0 - 8.0 pH".
    """
    if not ref_range or not isinstance(ref_range, str):
        return None

    cleaned = _STRAY_DECIMAL_RE.sub(r"\1.\2", ref_range.strip())

    # CHANGED: try labeled-band parsing first (see _parse_labeled_bands
    # above). Only takes effect when the text actually has 2+ "label: value"
    # bands;
    banded = _parse_labeled_bands(cleaned)
    if banded is not None:
        return banded

    banded = _parse_low_high_labeled_bands(cleaned)
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
    """Determines the clinical status of a result. Priority order, highest
    trust first (see module docstring): the range printed on this report,
    then the static REFERENCE_DB lookup, then the LLM's own qualitative
    guess, then plain string matching."""
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
    elif val_str in ["positive", "1+", "2+", "3+", "4+", "reactive", "cloudy",
                      "present", "detected"]:
        return RangeStatus.HIGH

    return RangeStatus.UNKNOWN


# --------------------------------------------------------------------------
# NEW: shared post-processing helpers used by every extraction tier
# --------------------------------------------------------------------------

def finalize_result(result: dict, patient_sex: str = "unknown", patient_age: int = 30) -> dict:
    """Takes a raw extracted result dict from any tier (table, vision LLM,
    text LLM, or regex) and computes the final status and normal-range
    bounds the same way every time. Mutates and returns `result` with
    status, the tier's original raw guess (kept for audit), and the
    resolved range bounds."""
    test_id = result.get("test_id") or "UNKNOWN_TEST"
    value = result.get("value")
    doc_range = result.get("ref_range")
    llm_status = result.get("status")

    if test_id in BLOOD_TYPE_TEST_IDS:
        result["status"] = RangeStatus.NORMAL.value
        result["llm_status"] = llm_status
        result["normal_range_min"] = None
        result["normal_range_max"] = None
        return result

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
    """De-duplicates extracted results by (test_id, raw_name), since
    results can otherwise repeat across tiers or vision batches. Keeps the
    most informative entry per key: prefers one with a parseable printed
    ref_range, then a recognized test_id, then first-seen.

    BUG FOUND (a QuantiFERON-TB panel): keying on test_id alone collapsed
    "TB NIL Tube", "TB Antigen Tube", and "TB Ag Minus NIL", three genuinely
    distinct measurements that all resolve to the same "TUBERCULOSIS"
    reference entry, silently discarding two of the three. test_id-alone
    dedup is still correct for its original case (the same row found twice
    across tiers); it just needed raw_name as a second key to tell that
    apart from different rows sharing one reference page.
    """
    best: Dict[tuple, dict] = {}
    order: List[tuple] = []

    def score(r: dict) -> tuple:
        return (bool(parse_ref_range_string(r.get("ref_range"))), bool(r.get("known")))

    for r in results:
        tid = r.get("test_id") or "UNKNOWN_TEST"
        name_key = (r.get("raw_name") or "").strip().lower()
        key = (tid, name_key)
        if key not in best:
            best[key] = r
            order.append(key)
            continue
        if score(r) > score(best[key]):
            best[key] = r

    return [best[key] for key in order]