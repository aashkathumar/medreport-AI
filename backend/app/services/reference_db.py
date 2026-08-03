"""
Reference Database Service.
Handles test alias resolution, normal range lookups by demographic (sex/age),
and status flagging for lab values.
"""
import re
from typing import Dict, Any, Optional
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


# In app/services/reference_db.py

# In app/services/reference_db.py

def flag_result(
    value: Any, 
    normal_range: Optional[Dict[str, float]], 
    llm_status: Optional[str] = None
) -> RangeStatus:
    # 1. Prefer LLM status if provided and valid
    if llm_status and llm_status in RangeStatus.__members__.values():
        return RangeStatus(llm_status)
    
    if value is None:
        return RangeStatus.UNKNOWN

    # 2. Numeric range comparison
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

    # 3. Qualitative fallback safety net
    val_str = str(value).strip().lower()
    if val_str in ["negative", "nil", "clear", "absent", "normal", "pale yellow", "straw"]:
        return RangeStatus.NORMAL
    elif val_str in ["positive", "1+", "2+", "3+", "4+", "reactive", "cloudy"]:
        return RangeStatus.HIGH

    return RangeStatus.UNKNOWN