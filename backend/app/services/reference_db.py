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
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from app.models.schemas import RangeStatus

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# Canonical alias mapping table. Includes the JSON-sourced test names/IDs
# themselves (lowercased) plus common clinical abbreviations/synonyms, so
# raw names extracted from a report resolve to the same curated entry
# regardless of exact wording.
ALIAS_MAP = {
    # BUG FOUND (PK0016.pdf): these raw field labels had NO alias entry at
    # all, so they canonicalized to a synthetic id (e.g. "CHLORIDE") that
    # never matches the corpus's real tag for the same analyte (e.g. "CL")
    # -- 11 of PK0016's 19 "not covered" rejections were this, not a
    # missing-source problem: the corpus already had grounding content for
    # every one of these, just filed under a different canonical id.
    #
    # BUG FOUND (2026-08-30, medreport_ai-2/3.pdf): tc_hdl_ratio and
    # sgot_sgpt_ratio used to live HERE, in ALIAS_MAP -- which doesn't just
    # steer grounding, it sets the RESULT's own test_id (resolve_test_id()
    # is what pdf_parser.py calls to identity-tag each row). Aliasing
    # "SGOT/SGPT Ratio" straight to "AST" meant that row's test_id became
    # literally identical to the real "SGOT (AST)" row on the same report --
    # not just for grounding, but for the per-test_id what_it_measures
    # cache (_wim_get) AND the batch LLM response dict (both keyed by
    # test_id). Two genuinely different rows collapsed onto one shared
    # identity, so the ratio row silently received the AST row's entire
    # cached explanation, verbatim. Same thing for "90 Day Average Blood
    # Glucose" aliased to "HBA1C". Moved both (plus tc_hdl_ratio) down to
    # GROUNDING_ALIASES instead, which redirects retrieval ONLY -- the row
    # keeps its own distinct synthetic test_id (SGOT_SGPT_RATIO /
    # TC_HDL_RATIO / 90_DAY_AVERAGE_BLOOD_GLUCOSE via canonicalize_test_name)
    # while still pulling the component analyte's reference material to
    # explain from, exactly the intended fallback.
    "leukocyte_esterase": "URINE_LEUKOCYTES",
    "absolute_lymphocyte_count": "LYMPHOCYTES",
    "absolute_monocyte_count": "MONOCYTES",
    "absolute_eosinophil_count": "EOSINOPHILS",
    "absolute_basophil_count": "BASOPHILS",
    "blood_sugar_fasting": "GLUCOSE",
    "chloride": "CL",
    # Unlike the ratio tests above, "Urinary Transparency" and "Clearity" are
    # genuinely the SAME single measurement under two different labs'
    # naming conventions -- they never both appear on one report, so merging
    # their identity here (unlike SGOT/SGPT Ratio's) causes no same-report
    # collision. Belongs in ALIAS_MAP, not GROUNDING_ALIASES.
    "urinary_transparency": "CLEARITY",
    # BUG FOUND (PK0016.pdf): both raw labels are the exact same measurement
    # as an already-covered, already-cached id, just missing the alias --
    # resolve_test_id() returned None, so each fell through to its own
    # synthetic id instead of the real, corpus-covered one. Merging identity
    # here is safe the same way urinary_transparency/CLEARITY is: these are
    # alternate labels for one test, not a distinct sub-fraction like
    # HB_A2, so sharing the cached explanation is correct, not lossy.
    "total_iron_binding_capacity": "TOTAL_IRON_BINDING_CAPACITY_TIBC",
    "25_oh_vitamin_d_total": "VIT_D",

    # BUG FOUND (LabReport.pdf, QuantiFERON-TB Gold panel): none of these
    # 4 component-tube names have an alias, so all 5 rows on this report
    # showed "not covered" -- but the corpus already has a "Tuberculosis
    # Screening" page (test_id TUBERCULOSIS) that explicitly covers IGRA
    # blood tests (this exact methodology) and explains positive/negative
    # results. "Final Result" deliberately NOT aliased here despite being
    # on the same report -- it's too generic a label (other multi-row
    # panels likely reuse "Final Result" for an unrelated summary line),
    # so a global alias would risk grounding a different test's result on
    # TB content. The 4 TB-specific tube names are unique enough to be safe.
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
    # BUG FOUND on the Sterling Accuris report: these are simply the names real
    # pathology labs print for tests the RAG corpus already covers under their
    # canonical names, but no alias mapped them -- so resolve_test_id() returned
    # None, retrieval on the raw name missed, and the patient was told
    # "NHS UK and NIH MedlinePlus reference material was not available" for
    # their LIVER ENZYMES, which is simply untrue. SGPT/SGOT are the older
    # (and still standard in Indian labs) names for ALT/AST.
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
    # Lets "HIV I & II Ab/Ag with P24 Ag" (Sterling Accuris's exact wording)
    # resolve to the same "HIV" id the corpus's hiv-screening-test page is
    # already tagged with - without this it canonicalised to the report's
    # entire wording as one synthetic id, which no source could ever match.
    "hiv_i_ii_ab_ag_with_p24_ag": "HIV", "hiv_ab_ag": "HIV", "hiv_screening": "HIV",
    "hbsag": "HBSAG", "hepatitis_b_surface_antigen": "HBSAG",
}

# BUG FOUND (Sterling Accuris sample report, Rh (D) Type): flag_result()'s
# qualitative fallback maps the string "positive" to RangeStatus.HIGH, which
# is correct for a result like "Urine Protein: Positive" (that IS abnormal),
# but blood-group/antigen-typing fields aren't a "range" concept at all -
# "Positive"/"Negative" or "A"/"B"/"O"/"AB" just state a fact about the
# patient, never a clinical abnormality. The qualitative fallback flagged
# "Rh (D) Type: Positive" as "Above normal range", which is a category
# error, not a real finding. These test_ids are excluded from status
# flagging entirely in finalize_result() below.
BLOOD_TYPE_TEST_IDS = {"ABO_TYPE", "RH_D_TYPE"}


# Grounding-only redirects: test_id -> the test_id whose curated reference page
# also explains this analyte.
#
# These are deliberately NOT in ALIAS_MAP. ALIAS_MAP decides IDENTITY, and
# identity is the key dedupe_results() de-duplicates on -- so putting them
# there made four genuinely different analytes (Hemoglobin, Hb A, Hb A2,
# Foetal Hb) collapse into one row and SILENTLY DROPPED three of the
# patient's results, plus two more across Iron/TIBC/Transferrin Saturation.
# Each of these keeps its own identity and its own reported value; only the
# page consulted for GROUNDING is redirected.
GROUNDING_ALIASES = {
    # BUG FOUND (2026-09-01, live-tested on Sterling Accuris): HB_A/HB_A2/
    # FOETAL_HB used to redirect here on the assumption that "a type of
    # haemoglobin" made the general haemoglobin page a valid reference.
    # Checked directly against the actual chunks tagged HGB before trusting
    # that assumption -- none of them mention haemoglobin electrophoresis,
    # fractions, Hb A, Hb A2, or fetal haemoglobin at all; the content is
    # entirely about the total haemoglobin count and iron-deficiency
    # anaemia. On this branch specifically, the redirect produced a
    # confirmed, serious hallucination: the model filled in fabricated
    # reference-range numbers (get_normal_range('HB_A', ...) returns None
    # -- there is no real range for this test anywhere in the system) and,
    # for FOETAL_HB, invented content about "monitoring fetal health during
    # pregnancy" -- a genuine clinical error, since this is a haemoglobin-
    # fraction test on the patient's own blood, unrelated to an actual
    # pregnancy. Removed so these fall through to the same honest "not
    # covered" path as P2 Peak/P3 Peak/Amorphous Material.
    # Iron studies: both are reported on the MedlinePlus iron-tests page.
    "TIBC": "IRON",
    "TOTAL_IRON_BINDING_CAPACITY": "IRON",
    "TOTAL_IRON_BINDING_CAPACITY_TIBC": "IRON",
    "TRANSFERRIN_SATURATION": "IRON",
    # Computed ratios/derived values: no dedicated "ratio" reference page
    # exists, so ground on the component analyte's page and let the LLM
    # synthesize a ratio-specific explanation from it -- same pattern
    # already proven to work for CHOL/HDL Ratio and LDL/HDL Ratio (no entry
    # needed there; semantic RAG retrieval alone finds the right page).
    # These three needed an explicit redirect because their synthetic ids
    # (SGOT_SGPT_RATIO, TC_HDL_RATIO, 90_DAY_AVERAGE_BLOOD_GLUCOSE) don't
    # share enough vocabulary with the corpus's AST/CHOL/HBA1C pages for
    # retrieval to find them unaided.
    "SGOT_SGPT_RATIO": "AST",
    "TC_HDL_RATIO": "CHOL",
    "90_DAY_AVERAGE_BLOOD_GLUCOSE": "HBA1C",
}

# BUG FOUND (Sterling Accuris report, urinalysis panel): "Bilirubin" and
# "Red Cells" are printed with those bare names on BOTH the liver panel
# (blood) and the urinalysis panel (urine) - genuinely different tests with
# genuinely different reference pages. With no specimen information,
# grounding always cited the blood page for both, producing "No bilirubin
# was detected in your blood" as the explanation for a urine dipstick
# result. pdf_parser.py now tags each row with the specimen its page's
# section heading indicated ('urine' or None/blood) - this redirects
# grounding for the ambiguous ids to the urine-specific corpus page ONLY
# when that tag says urine, leaving the blood case (the default) untouched.
SPECIMEN_GROUNDING_ALIASES = {
    ("BILIRUBIN", "urine"): "URINE_BILIRUBIN",
    ("RED_CELLS", "urine"): "URINE_BLOOD",
    ("PUS_CELLS", "urine"): "URINE_LEUKOCYTES",
    ("COLOUR", "urine"): "URINE_COLOUR",
    # "pH" on a urinalysis panel synthesises to the bare id "PH" (no ALIAS_MAP
    # entry for the unqualified word), missing the existing curated
    # URINE_PH entry entirely - same class of bug as COLOUR above, just a
    # different id, and initially missed in the same fix pass.
    ("PH", "urine"): "URINE_PH",
    # BUG FOUND (PK0016.pdf, a different lab): "Blood [In Urine]" -> BLOOD -
    # a different synthesized id than Sterling's "Red Cells" -> RED_CELLS,
    # same underlying test, needs its own redirect entry.
    ("BLOOD", "urine"): "URINE_BLOOD",
    # BUG FOUND (PK0016.pdf): "Urinary RBC" -> RBC without a redirect
    # grounded on the BLOOD RBC-count page (a semantic near-match, since
    # both are literally "counting red blood cells") - but urine microscopy
    # RBC and a blood RBC count are different tests measuring different
    # things. The correct reference for RBC-in-urine is the same
    # "blood in urine" page as the dipstick blood test above.
    ("RBC", "urine"): "URINE_BLOOD",
    # These three already have curated corpus pages tagged URINE_PROTEIN/
    # URINE_KETONES/URINE_NITRITES (Sterling's report resolves to them via
    # a direct ALIAS_MAP entry keyed on "urine_protein" etc, since it prints
    # "Urine Protein"). A lab printing bare "Protein"/"Ketones"/"Nitrites"
    # on a urinalysis panel has no such alias and needs the specimen tag
    # to find the same page.
    ("PROTEIN", "urine"): "URINE_PROTEIN",
    ("KETONES", "urine"): "URINE_KETONES",
    ("NITRITES", "urine"): "URINE_NITRITES",
    # BUG FOUND (found while auditing for this exact risk class): "Urinary
    # Glucose" canonicalises to bare "GLUCOSE" - the same id as blood
    # glucose, which HAS a curated entry (blood_tests.json) that matched
    # BEFORE any redirect ran, silently substituting "Blood glucose measures
    # the amount of sugar in your blood..." for a urine dipstick result,
    # where mere presence (not a concentration) is the abnormal finding.
    ("GLUCOSE", "urine"): "URINE_GLUCOSE",
}

# Other curated blood-test ids (blood_tests.json) that a "Urinary X"-style
# name would canonicalise onto identically, if a report used that phrasing:
# CREATININE, HGB, TSH, ALT, CHOL, HDL, LDL, WBC, PLT, CRP, FERRITIN, VIT_D,
# B12, HBA1C. None of these has been observed in a real report yet (unlike
# GLUCOSE above, which was), and none currently has a matching URINE_*
# curated entry to redirect to even if it were - so a fix here would be
# speculative rather than verified. Flagging as a known risk class: if a
# future report shows e.g. "Urinary Creatinine" grounding on the blood
# creatinine page, this dict is where the fix belongs, the same way
# GLUCOSE was fixed above.


def grounding_test_id(test_id: str, specimen: Optional[str] = None) -> str:
    """The test_id to retrieve reference passages under. Identity-preserving
    for everything except the redirects above."""
    if specimen:
        specimen_redirect = SPECIMEN_GROUNDING_ALIASES.get((test_id, specimen))
        if specimen_redirect:
            return specimen_redirect
    return GROUNDING_ALIASES.get(test_id, test_id)


def _load_reference_db() -> Dict[str, Dict[str, Any]]:
    """
    Builds REFERENCE_DB from the curated NHS UK / NIH MedlinePlus-sourced
    JSON files, rather than a small hand-maintained dict, so every test
    those files cover has real grounding available (avoiding LLM
    hallucination for tests that otherwise had no reference data).
    """
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
    """Merges LLM-generated synonym candidates (scripts/generate_test_synonyms.py)
    into ALIAS_MAP.

    The generator script already excludes anything ambiguous (proposed for
    two different test_ids) or colliding with an existing alias pointing
    elsewhere -- what's in this file passed both checks. setdefault() here
    is a second, defense-in-depth guarantee of the same rule: a hand-curated
    or earlier-loaded entry can never be silently overwritten by a
    generated one, regardless of what the file contains.
    """
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
    """
    Resolves raw test names/aliases to canonical test IDs in the database.
    Returns canonical ID string (e.g. 'HGB') or None if unmapped/unknown.

    CHANGED: the lookup used `raw_name.strip().lower()` while every key in
    ALIAS_MAP is underscore-separated ("white_blood_cell_count") -- including
    the keys _load_reference_db() generates from the curated JSON names. So a
    lookup only ever hit for SINGLE-WORD names: "Haemoglobin" and "Platelets"
    resolved, while "White Blood Cell Count", "Total Cholesterol", "LDL
    Cholesterol", "Vitamin D", "Vitamin B12", "Blood Glucose" and "Urine
    Protein" all returned None. Those tests then had no curated entry, so
    they lost BOTH their static reference range (shown as "None-None" in the
    UI and PDF) and their curated grounding text. Normalising the lookup the
    same way the keys are built fixes 8 of the 21 curated tests.

    Falls back to the sample/method-qualifier-stripped form, so "Cholesterol,
    Serum" and "Creatinine (24 hour)" resolve to the same entry as the bare
    analyte name.
    """
    if not raw_name or not isinstance(raw_name, str):
        return None

    direct = ALIAS_MAP.get(_alias_key(raw_name))
    if direct:
        return direct

    # canonicalize_test_name() strips only non-distinguishing qualifiers
    # (serum/plasma/urine/24 hour/...) -- see its docstring for what it keeps.
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
    r"\b(serum|plasma|urine|urinary|edta\s*blood|whole\s*blood|fluoride\s*plasma|"
    r"\d+\s*hour(?:s)?|qualitative|quantitative|random\s*sample)\b",
    re.IGNORECASE,
)

# BUG FOUND (PK0016.pdf, a different lab's urinalysis format): this lab
# prints "Urinary pH" / "Urinary Specific Gravity" where Sterling Accuris
# printed bare "pH" / "Specific Gravity" and "Blood [In Urine]" where
# Sterling printed bare "Red Cells" - two different naming conventions for
# the exact same tests, producing different synthetic ids (URINARY_PH vs
# PH; BLOOD_IN vs RED_CELLS) that neither ALIAS_MAP nor
# SPECIMEN_GROUNDING_ALIASES recognised as the same thing. Stripping
# "urinary" (above) collapses the first case onto the ids already handled.
# This strips the "[In Urine]" / "(In Blood)" bracketed specimen note some
# labs append, collapsing the second case the same way.
_BRACKETED_SPECIMEN_NOTE = re.compile(
    r"[\[\(]\s*in\s+(urine|blood|serum|plasma|stool|csf)\s*[\]\)]",
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
    cleaned = _BRACKETED_SPECIMEN_NOTE.sub(" ", raw_name)
    cleaned = _NON_DISTINGUISHING_QUALIFIERS.sub(" ", cleaned)
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

# BUG FOUND (Sterling Accuris sample report, HDL Cholesterol): printed range
# was "Low: <40.0 / High: >60.0" - HDL is a protective marker where higher is
# GOOD, so the lab prints "High" as the desirable band, not the "abnormal"
# one. _parse_labeled_bands above only recognises bands labelled with
# desirable/normal/optimal/etc, so neither "Low" nor "High" matched and this
# fell through to the plain single-range regex, which grabbed "<40.0" as a
# lone upper bound - flagging a healthy 60.0 mg/dL result as "Above normal
# range" (HIGH), directly contradicting the app's own explanation text that
# higher HDL is protective.
# This handles the specific "Low: <X / High: >Y" two-band shape: only the
# Low threshold is treated as a flagging bound (below it is genuinely LOW);
# the High band is the protective/desirable extreme for this shape and is
# never flagged as abnormal by this generic parser.
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
    elif val_str in ["positive", "1+", "2+", "3+", "4+", "reactive", "cloudy",
                      "present", "detected"]:
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
    """
    De-duplicates extracted results by (test_id, raw_name) (results can
    otherwise repeat across table + LLM tiers, or across vision batches on
    long reports). Keeps the most informative entry per key: prefers one
    with a parseable printed ref_range, then one with a recognized/known
    test_id, then first-seen.

    BUG FOUND (LabReport.pdf, a QuantiFERON-TB panel): keying on test_id
    ALONE used to collapse genuinely different rows that happen to share
    one canonical test_id -- confirmed directly on this report, where "TB
    NIL Tube", "TB Antigen Tube", and "TB Ag Minus NIL" are three distinct
    real measurements that all resolve to the single "TUBERCULOSIS"
    reference entry (there's one generic screening page for the whole
    panel, not one per sub-component). Deduping by test_id alone kept only
    one of the three and silently discarded the other two, even though
    they were never duplicates of each other -- test_id-alone dedup is
    still correct and needed for the case this function was built for (the
    SAME row found twice, once per extraction tier), it just needed a
    second key to tell "the same row, twice" apart from "two different
    rows that share a reference page".
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