import json
import os
from app.models.schemas import RangeStatus

_DIR = os.path.join(os.path.dirname(__file__), "../../../data")

with open(os.path.join(_DIR, "blood_tests.json")) as f:
    _BLOOD = {t["id"]: t for t in json.load(f)["tests"]}

with open(os.path.join(_DIR, "urine_tests.json")) as f:
    _URINE = {t["id"]: t for t in json.load(f)["tests"]}

_ALL = {**_BLOOD, **_URINE}

ALIAS_MAP = {
    "hb": "HGB", "haemoglobin": "HGB", "hemoglobin": "HGB", "hgb": "HGB",
    "wbc": "WBC", "wcc": "WBC", "white blood cells": "WBC", "white cell count": "WBC",
    "white blood cell count": "WBC",
    "plt": "PLT", "platelets": "PLT", "platelet count": "PLT",
    "hba1c": "HBA1C", "a1c": "HBA1C", "glycated haemoglobin": "HBA1C",
    "cholesterol": "CHOL", "total cholesterol": "CHOL",
    "ldl": "LDL", "ldl cholesterol": "LDL",
    "hdl": "HDL", "hdl cholesterol": "HDL",
    "tsh": "TSH", "thyroid stimulating hormone": "TSH",
    "ferritin": "FERRITIN", "serum ferritin": "FERRITIN",
    "vitamin d": "VIT_D", "vit d": "VIT_D", "25-oh vitamin d": "VIT_D",
    "b12": "B12", "vitamin b12": "B12", "serum b12": "B12",
    "creatinine": "CREATININE", "serum creatinine": "CREATININE",
    "alt": "ALT", "alanine aminotransferase": "ALT",
    "glucose": "GLUCOSE", "blood glucose": "GLUCOSE", "fasting glucose": "GLUCOSE",
    "crp": "CRP", "c-reactive protein": "CRP",
    "protein": "URINE_PROTEIN", "urine protein": "URINE_PROTEIN",
    "urine glucose": "URINE_GLUCOSE",
    "ph": "URINE_PH", "urine ph": "URINE_PH",
    "leukocytes": "URINE_LEUKOCYTES", "wbc urine": "URINE_LEUKOCYTES",
    "nitrites": "URINE_NITRITES", "nitrite": "URINE_NITRITES",
    "ketones": "URINE_KETONES", "ketone": "URINE_KETONES",
}


def resolve_test_id(raw_name: str) -> str | None:
    key = raw_name.strip().lower()
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    upper = key.upper()
    return upper if upper in _ALL else None


def get_normal_range(test_id: str, sex: str = "unknown", age: int = 30) -> dict | None:
    test = _ALL.get(test_id)
    if not test:
        return None
    r = test.get("normal_range", {})
    # Check sex-specific range FIRST -- several tests (HGB, HDL, FERRITIN,
    # CREATININE) define distinct male/female ranges, and always falling
    # back to "all" first (the original ordering) meant those distinctions
    # were silently ignored for every request.
    if sex == "male" and "male" in r:
        return r["male"]
    if sex == "female" and "female" in r:
        return r["female"]
    if "all" in r:
        return r["all"]
    return r.get("male") or r.get("female")


def flag_result(value: float, normal_range: dict | None) -> RangeStatus:
    if normal_range is None:
        return RangeStatus.UNKNOWN
    if value < normal_range["min"]:
        return RangeStatus.LOW
    if value > normal_range["max"]:
        return RangeStatus.HIGH
    return RangeStatus.NORMAL


def get_reference_data(test_id: str) -> dict | None:
    return _ALL.get(test_id)
