"""
Regression tests for Tier 1 (deterministic, LLM-free) extraction.

Tier 1 previously returned ZERO rows on whitespace-aligned lab reports, the
common case, so those reports fell through to the vision LLM, which is both
the expensive tier and the one that hallucinates. These tests lock in the
behaviour that keeps them on the free, deterministic path.

Run (from backend/, no API keys and no network needed):
    python tests/test_tier1_extraction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.pdf_parser import (
    MIN_STRUCTURED_RESULTS,
    _clean_value,
    _is_test_name,
    _split_name_flag,
    _strip_footer,
    parse_report,
    parse_structured_tables,
)

TESTS_DIR = Path(__file__).resolve().parent
STERLING = TESTS_DIR / "sterling-accuris-pathology-sample-report-unlocked.pdf"
RULED = TESTS_DIR / "my_real_report.pdf"


def _by_name(results, needle):
    for r in results:
        if needle.lower() in r["raw_name"].lower():
            return r
    return None


def test_whitespace_aligned_report_stays_on_tier1():
    """The header row on this report is at index 13-19, not 0. Tier 1 used to
    give up and hand the whole 19-page report to the vision LLM."""
    if not STERLING.exists():
        print("  SKIP  sterling sample not present")
        return
    rows = parse_structured_tables(STERLING.read_bytes())
    assert len(rows) >= MIN_STRUCTURED_RESULTS, "would fall through to the LLM tier"
    assert len(rows) > 50, f"expected the full panel, got {len(rows)}"


def test_ruled_table_report_still_works():
    """The grid path must not regress while fixing the positional path."""
    if not RULED.exists():
        print("  SKIP  ruled sample not present")
        return
    rows = parse_structured_tables(RULED.read_bytes())
    assert len(rows) >= MIN_STRUCTURED_RESULTS
    assert _by_name(rows, "Urinary pH"), "expected a known row from the ruled table"


def test_no_llm_call_for_whitespace_report():
    if not STERLING.exists():
        print("  SKIP  sterling sample not present")
        return
    # provider=None with no network would raise if any LLM tier were reached.
    result = parse_report(STERLING.read_bytes(), method="auto",
                          patient_sex="female", patient_age=30)
    assert result["method"] == "table_structured", (
        f"expected the deterministic tier, got {result['method']}"
    )


def test_hba1c_value_is_the_result_not_a_band_boundary():
    """The safety case. The vision tier read 5.7, the Pre-Diabetes band
    boundary printed inside the reference range, instead of the printed
    result 7.10, which flipped the reported status from High to below-normal.
    Deterministic extraction must get this right."""
    if not STERLING.exists():
        print("  SKIP  sterling sample not present")
        return
    rows = parse_structured_tables(STERLING.read_bytes())
    hba1c = _by_name(rows, "HbA1c")
    assert hba1c, "HbA1c not extracted"
    assert float(hba1c["value"]) == 7.10, f"got {hba1c['value']}, expected 7.10"


def test_cholesterol_keeps_full_multiline_band():
    """The printed range spans three lines; truncating it to the first line
    stops _parse_labeled_bands() finding the healthy band."""
    if not STERLING.exists():
        print("  SKIP  sterling sample not present")
        return
    rows = parse_structured_tables(STERLING.read_bytes())
    chol = _by_name(rows, "Cholesterol")
    assert chol and chol["ref_range"], "cholesterol range missing"
    assert "borderline" in chol["ref_range"].lower(), (
        f"band truncated: {chol['ref_range']!r}"
    )


def test_differential_uses_percentage_range_not_absolute_count():
    """The differential sub-table's columns are offset from the page header,
    so the unit cell absorbed the % range and the ref cell held the ABSOLUTE
    count, range-checking 73% against 2000-6700 flagged it low."""
    if not STERLING.exists():
        print("  SKIP  sterling sample not present")
        return
    rows = parse_structured_tables(STERLING.read_bytes())
    neut = _by_name(rows, "Neutrophils")
    assert neut, "Neutrophils not extracted"
    assert neut["unit"] == "%", f"unit polluted: {neut['unit']!r}"
    assert neut["ref_range"].startswith("40 - 80"), f"got {neut['ref_range']!r}"


def test_abnormal_flag_is_split_off_the_name_and_value():
    assert _split_name_flag("HbA1c H") == ("HbA1c", "H")
    assert _split_name_flag("Urea L") == ("Urea", "L")
    assert _split_name_flag("Creatinine, Serum") == ("Creatinine, Serum", None)
    assert _clean_value("H 141.0")[0] == "141.0"
    assert _clean_value("H 141.0")[1] == "H"


def test_page_furniture_stripped_from_ranges():
    assert _strip_footer("0 - 14 Dr.Yash Shah MD Path Page 1 of 19") == "0 - 14"
    assert _strip_footer("74 - 106 Page 4 of 19 # Referred Test") == "74 - 106"
    assert _strip_footer("13.0 - 16.5") == "13.0 - 16.5"


def test_signature_blocks_are_not_tests():
    assert not _is_test_name("Dr. Purvish Darji")
    assert not _is_test_name("MD(Path)")
    assert not _is_test_name("")
    assert _is_test_name("Haemoglobin")
    assert _is_test_name("Total Iron Binding Capacity (TIBC)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
