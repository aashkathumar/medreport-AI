import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import TestResult, UserProfile, ExplainedResult
from app.services.pdf_parser import parse_report
from app.services.reference_db import get_normal_range, flag_result, get_reference_data
from app.services.llm_service import explain_test_result, generate_summary
from app.services.pdf_generator import generate_report_pdf
from app.db import save_report, get_user_reports

router = APIRouter()


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = "demo_user",
    age: int = 30,
    sex: str = "unknown",
    diet_type: str = "omnivore",
    parse_method: str = "auto",  # "auto" | "regex" | "llm" | "ocr"
):
    file_bytes = await file.read()
    parse_result = parse_report(file_bytes, method=parse_method)
    raw = parse_result["results"]

    if not raw:
        raise HTTPException(422, "Could not extract results. Please use manual entry.")

    profile = UserProfile(user_id=user_id, age=age, sex=sex, diet_type=diet_type)

    # Build TestResult objects
    test_results = []
    for r in raw:
        ref = get_reference_data(r["test_id"])
        if not ref:
            continue
        nr = get_normal_range(r["test_id"], sex, age)
        status = flag_result(r["value"], nr)
        test_results.append(TestResult(
            test_id=r["test_id"], raw_name=r["raw_name"],
            value=r["value"], unit=r["unit"], status=status,
            normal_range_min=nr["min"] if nr else None,
            normal_range_max=nr["max"] if nr else None,
        ))

    # Explain via LLM (RAG-grounded, see llm_service.py)
    explained = []
    for test in test_results:
        ref = get_reference_data(test.test_id)
        explained.append(explain_test_result(test, profile, ref))

    # Summary
    summary = generate_summary(explained, profile)

    # Save (local JSON file in dev, DynamoDB in production -- see app/db/__init__.py)
    report_data = {
        "explained_results": [r.model_dump() for r in explained],
        **summary,
        "parse_method": parse_result["method"],
    }
    report_id = save_report(user_id, report_data)

    return {
        "report_id": report_id,
        "parse_method": parse_result["method"],
        "explained_results": [r.model_dump() for r in explained],
        **summary,
    }


@router.post("/download-pdf")
async def download_pdf(report: dict):
    explained = [ExplainedResult(**r) for r in report.get("explained_results", [])]
    pdf_bytes = generate_report_pdf(
        explained_results=explained,
        overall_summary=report.get("overall_summary", ""),
        top_gp_topics=report.get("top_gp_topics", []),
        top_lifestyle_change=report.get("top_lifestyle_change", ""),
        closing_message=report.get("closing_message", ""),
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=medreport.pdf"},
    )


@router.get("/reports/{user_id}")
async def get_reports(user_id: str):
    return {"reports": get_user_reports(user_id)}


@router.get("/health")
async def health():
    return {"status": "ok"}
