"""Regression tests for Tier 1's row-cleaning helpers: splitting an
abnormal flag off a name/value, stripping page furniture from a printed
range, and telling a real test name apart from a signature block.

Run (from backend/, no API keys and no network needed):
    python tests/test_tier1_extraction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.pdf_parser import (
    _clean_value,
    _is_test_name,
    _split_name_flag,
    _strip_footer,
)


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
