import io
import concurrent.futures
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from app.models.schemas import TestResult, UserProfile, ExplainedResult, RangeStatus
from app.services.pdf_parser import parse_report
from app.services.reference_db import get_normal_range, flag_result, get_reference_data
from app.services.llm_service import explain_all_test_results_batched, generate_summary
from app.services.pdf_generator import generate_report_pdf
from app.services.llm_providers import PROVIDERS, get_all_openrouter_models, get_all_groq_models
from app.db import save_report, get_user_reports

router = APIRouter()

MAX_EXPLAIN_WORKERS = 4


def _default_ref(test: TestResult) -> dict:
    return {
        "source": "General Medical Reference",
        "plain_english": f"{test.raw_name} test.",
        "low_means": "Below standard range.",
        "high_means": "Above standard range.",
    }


@router.get("/llm/providers")
def get_available_providers():
    """Returns list of configured provider keys."""
    return {"providers": list(PROVIDERS.keys())}


@router.get("/llm/models/{provider}")
def get_provider_models(provider: str):
    """Fetches all live models available for a specified provider."""
    if provider == "openrouter":
        models = get_all_openrouter_models()
    elif provider == "groq_llama":
        models = get_all_groq_models()
    # Around line 35 in get_provider_models:
    elif provider == "gemini":
        models = ["gemini-1.5-flash"]
    elif provider == "mistral":
        models = ["mistral-small-latest"]
    else:
        raise HTTPException(404, detail=f"Provider '{provider}' not found.")

    return {"provider": provider, "total": len(models), "models": models}


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = "demo_user",
    age: int = 30,
    sex: str = "unknown",
    diet_type: str = "omnivore",
    parse_method: str = "auto",
    provider: str = Query(None, description="LLM provider name"),
    model_name: str = Query(None, description="Dynamic model name ID"),
):
    # 1. Read PDF bytes directly once
    file_bytes = await file.read()
    
    # 2. Pass file_bytes and provider down into parse_report
    parse_result = parse_report(file_bytes, method=parse_method, provider=provider)
    raw = parse_result["results"]

    if not raw:
        raise HTTPException(422, "Could not extract results. Please use manual entry.")

    profile = UserProfile(user_id=user_id, age=age, sex=sex, diet_type=diet_type)

    test_results = []
    for r in raw:
        ref = get_reference_data(r["test_id"])
        nr = get_normal_range(r["test_id"], sex, age) if ref else None
        
        # Pass the extracted LLM status as llm_status fallback
        status = flag_result(
            value=r["value"], 
            normal_range=nr, 
            llm_status=r.get("status")
        )

        test_results.append(TestResult(
            test_id=r["test_id"],
            raw_name=r["raw_name"],
            value=r["value"],
            unit=r.get("unit", ""),
            status=status,  # <--- Evaluated clinical status
            normal_range_min=nr["min"] if nr else None,
            normal_range_max=nr["max"] if nr else None,
        ))

 # 1 single batched LLM call for all tests
    explained = explain_all_test_results_batched(
        test_results, profile, provider=provider, model_name=model_name
    )

    summary = generate_summary(explained, profile, provider=provider, model_name=model_name)

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

@router.get("/health")
async def health():
    return {"status": "ok"}

# In app/api/routes.py

@router.post("/download-pdf")
async def download_pdf(data: dict):
    """
    Generates and returns a downloadable PDF binary for the frontend.
    """
    try:
        raw_explained = data.get("explained_results", [])
        
        # 1. Reconstruct Pydantic models from dicts cleanly
        explained_results = []
        for item in raw_explained:
            if isinstance(item, dict):
                # Ensure status is converted to a valid RangeStatus enum instance
                raw_status = item.get("status", "unknown")
                try:
                    status_enum = RangeStatus(raw_status)
                except ValueError:
                    status_enum = RangeStatus.UNKNOWN

                explained_results.append(ExplainedResult(
                    test_id=item.get("test_id", "UNKNOWN"),
                    raw_name=item.get("raw_name", "Test"),
                    value=item.get("value", "N/A"),
                    unit=item.get("unit", ""),
                    status=status_enum,
                    normal_range_min=item.get("normal_range_min"),
                    normal_range_max=item.get("normal_range_max"),
                    source=item.get("source", "NHS UK / Medical Consensus"),
                    what_it_measures=item.get("what_it_measures", ""),
                    what_your_result_means=item.get("what_your_result_means", ""),
                    lifestyle_suggestions=item.get("lifestyle_suggestions", []),
                    gp_question=item.get("gp_question", ""),
                    disclaimer=item.get("disclaimer", "")
                ))
            else:
                explained_results.append(item)

        # 2. Extract specific summary text fields
        overall_summary = data.get("overall_summary", "Your lab test results summary.")
        top_gp_topics = data.get("top_gp_topics", [])
        top_lifestyle_change = data.get("top_lifestyle_change", "Maintain a balanced lifestyle.")
        closing_message = data.get("closing_message", "Consult your GP for formal medical advice.")
        user_name = str(data.get("user_id", "Patient"))

        # 3. Call generate_report_pdf with exact positional arguments
        pdf_bytes = generate_report_pdf(
            explained_results, 
            overall_summary, 
            top_gp_topics, 
            top_lifestyle_change, 
            closing_message, 
            user_name
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=MedReport_AI_Summary.pdf"
            },
        )
    except Exception as e:
        print(f"❌ PDF Generation Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
