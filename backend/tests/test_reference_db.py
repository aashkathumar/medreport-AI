import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.reference_db import resolve_test_id, get_normal_range, flag_result
from app.models.schemas import RangeStatus


def test_alias_resolution():
    assert resolve_test_id("Hb") == "HGB"
    assert resolve_test_id("HbA1c") == "HBA1C"
    assert resolve_test_id("nonsense_test") is None


def test_normal_range_by_sex():
    r = get_normal_range("HGB", sex="female")
    assert r == {"min": 11.5, "max": 16.0}


def test_flagging():
    assert flag_result(10.0, {"min": 11.5, "max": 17.5}) == RangeStatus.LOW
    assert flag_result(14.0, {"min": 11.5, "max": 17.5}) == RangeStatus.NORMAL
    assert flag_result(20.0, {"min": 11.5, "max": 17.5}) == RangeStatus.HIGH


if __name__ == "__main__":
    test_alias_resolution()
    test_normal_range_by_sex()
    test_flagging()
    print("All reference_db tests passed")
