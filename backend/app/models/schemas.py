from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class RangeStatus(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class TestResult(BaseModel):
    test_id: str
    raw_name: str
    value: float
    unit: str
    status: RangeStatus
    normal_range_min: Optional[float] = None
    normal_range_max: Optional[float] = None
    source: str = "NHS UK"


class ExplainedResult(BaseModel):
    test_id: str
    raw_name: str
    value: float
    unit: str
    status: RangeStatus
    normal_range_min: Optional[float] = None
    normal_range_max: Optional[float] = None
    source: str = "NHS UK"
    what_it_measures: str
    what_your_result_means: str
    lifestyle_suggestions: List[str]
    gp_question: str
    disclaimer: str


class UserProfile(BaseModel):
    user_id: str = "demo_user"
    age: int = 30
    sex: str = "unknown"
    diet_type: str = "omnivore"


class ReportResponse(BaseModel):
    report_id: str
    user_id: str
    explained_results: List[ExplainedResult]
    overall_summary: str
    top_gp_topics: List[str]
    top_lifestyle_change: str
    closing_message: str
    download_url: Optional[str] = None
    parse_method: str = "structured"
