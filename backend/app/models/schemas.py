from pydantic import BaseModel
from typing import Optional, List, Union
from enum import Enum


class RangeStatus(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class TestResult(BaseModel):
    test_id: str
    raw_name: str
    value: Union[str, float]
    unit: str
    status: RangeStatus
    panel_name: Optional[str] = None
    specimen: Optional[str] = None
    normal_range_min: Optional[float] = None
    normal_range_max: Optional[float] = None
    source: str = "NHS UK / NIH MedlinePlus"


class ExplainedResult(BaseModel):
    # ETHICS CONSTRAINT (Chris Clarke, project ethics reviewer): the
    # patient-facing output on this branch never compares a result against
    # a reference range or states what a specific value/status means
    # clinically -- that reads as clinical decision support, not viable for
    # an MSc-scope project. Accordingly `status`, `normal_range_min`,
    # `normal_range_max`, and `what_your_result_means` are deliberately NOT
    # patient-facing fields here (unlike feature/developv4.1, where they are
    # correct and intentional). `value`/`unit` are kept only so the patient
    # can see what was extracted from their report, not to support a
    # normal/high/low judgement about it.
    test_id: str
    raw_name: str
    value: Union[str, float]
    unit: str
    panel_name: Optional[str] = None
    specimen: Optional[str] = None
    source: str = "NHS UK / NIH MedlinePlus"
    source_urls: Optional[List[str]] = []
    what_it_measures: str
    lifestyle_suggestions: List[str]
    gp_question: Optional[str] = "What questions should I discuss with my GP regarding this test?"
    disclaimer: Optional[str] = (
        "This tool is for technical health literacy and GP consultation preparation only. "
        "It does not provide medical diagnosis, clinical evaluation, or treatment advice."
    )


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