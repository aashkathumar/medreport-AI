from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from app.models.schemas import TestResult, UserProfile, ExplainedResult, RangeStatus
from app.services.pdf_parser import parse_report
from app.services.llm_service import explain_all_test_results_batched, generate_summary
from app.services.pdf_generator import generate_report_pdf
from app.config import settings
from app.services.llm_providers import (
    PROVIDERS,
    get_all_openrouter_models,
    get_all_groq_models,
    get_all_nvidia_models,
    get_all_gemini_models,
    provider_status,
)
from app.services.rag_service import index_health
from app.services.reference_db import resolve_test_id, canonicalize_test_name, finalize_result
import uuid

router = APIRouter()


@router.get("/llm/providers")
def get_available_providers():
    """Returns each provider with whether it has a credential configured and
    whether it can accept image (vision) payloads -- listing provider names
    alone hid the fact that a chain entry had no API key behind it."""
    return {"providers": provider_status()}


@router.get("/rag/health")
def get_rag_health():
    """Reports whether the FAISS index actually loaded. Retrieval failures
    used to be invisible: a missing index just meant every explanation
    quietly lost its grounding."""
    return index_health()


@router.get("/llm/models/{provider}")
def get_provider_models(provider: str):
    """Fetches all live models available for a specified provider."""
    if provider == "openrouter":
        models = get_all_openrouter_models()
    elif provider == "groq_llama":
        models = get_all_groq_models()
    elif provider == "nvidia":
        models = get_all_nvidia_models()
    elif provider == "gemini":
        # Was hardcoded to the retired "gemini-1.5-flash"; list what the key
        # can actually reach.
        models = get_all_gemini_models()
    elif provider == "mistral":
        models = [settings.mistral_model]
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

    # CHANGED: everything below this point is fully BLOCKING work - pdfplumber,
    # faiss, `requests`, and the LLM SDKs - and it used to run directly on the
    # event loop inside this `async def`. For the entire duration of a request
    # (20s on a small report, minutes on a large one) no other request could be
    # served at all: /health froze, and a second user appeared to hang. Handing
    # it to the threadpool keeps the loop free.
    return await run_in_threadpool(
        _process_upload, file_bytes, user_id, age, sex, diet_type,
        parse_method, provider, model_name,
    )


def _process_upload(
    file_bytes: bytes,
    user_id: str,
    age: int,
    sex: str,
    diet_type: str,
    parse_method: str,
    provider: str,
    model_name: str,
):
    # 2. Pass file_bytes, provider AND the patient demographics down into
    #    parse_report. CHANGED: sex/age were not forwarded, so
    #    finalize_result() resolved every sex-specific static range against
    #    the defaults ("unknown"/30) -- a female patient's haemoglobin was
    #    range-checked against the male range.
    parse_result = parse_report(
        file_bytes,
        method=parse_method,
        provider=provider,
        patient_sex=sex,
        patient_age=age,
    )
    raw = parse_result["results"]

    if not raw:
        raise HTTPException(422, "Could not extract results. Please use manual entry.")

    profile = UserProfile(user_id=user_id, age=age, sex=sex, diet_type=diet_type)

    # CHANGED: this loop used to re-derive status and ranges from the
    # 21-entry static REFERENCE_DB, discarding what parse_report() had
    # already computed. finalize_result() resolves each result against the
    # reference range PRINTED ON THIS REPORT first (the lab's own
    # method/instrument-specific range), falling back to the static table
    # only when the document printed nothing parseable -- that priority order
    # is the whole point of the ref_range work, and re-deriving here inverted
    # it. Any test outside the static table also lost its range entirely and
    # showed no normal range in the UI or the PDF.
    test_results = []
    for r in raw:
        test_results.append(TestResult(
            test_id=r["test_id"],
            raw_name=r["raw_name"],
            value=r["value"],
            unit=r.get("unit", ""),
            status=RangeStatus(r["status"]) if r.get("status") else RangeStatus.UNKNOWN,
            normal_range_min=r.get("normal_range_min"),
            normal_range_max=r.get("normal_range_max"),
            specimen=r.get("specimen"),
        ))

    # 1 single batched LLM call per EXPLAIN_BATCH_SIZE tests
    explained, degraded_test_ids, ungrounded_test_ids = explain_all_test_results_batched(
        test_results, profile, provider=provider, model_name=model_name
    )

    summary = generate_summary(explained, profile, provider=provider, model_name=model_name)
    summary_degraded = bool(summary.pop("summary_degraded", False))

    report_data = {
        "explained_results": [r.model_dump() for r in explained],
        **summary,
        "parse_method": parse_result["method"],
        "degraded_test_ids": degraded_test_ids,
        "ungrounded_test_ids": ungrounded_test_ids,
        "summary_degraded": summary_degraded,
    }
    # CHANGED: no longer persisted server-side. Storing full explained
    # reports indefinitely (the DynamoDB table has no TTL, unlike the S3
    # bucket's 1-day lifecycle rule) was a real gap between what the upload
    # page promises ("processed transiently, never stored raw") and what
    # actually happened -- and the read side of this (GET /reports/{user_id})
    # had no access control at all: any user_id could be typed in to view
    # another patient's history. Session-only avoids both: nothing exists
    # once the browser tab closes, and there's nothing cross-session to leak.
    report_id = str(uuid.uuid4())

    return {
        "report_id": report_id,
        "parse_method": parse_result["method"],
        "explained_results": [r.model_dump() for r in explained],
        "degraded_test_ids": degraded_test_ids,
        # Tests whose generated explanation failed source verification and was
        # replaced with a GP referral -- surfaced so the constraint is visible.
        "ungrounded_test_ids": ungrounded_test_ids,
        "summary_degraded": summary_degraded,
        **summary,
    }

@router.post("/explain-manual")
async def explain_manual(data: dict):
    """Explains manually-typed test values (the frontend's "Manual Entry"
    fallback, used when PDF extraction fails or a patient just wants to
    type one value in). Previously this had NO backend counterpart at
    all -- the frontend collected entries into session_state and had
    nothing to send them to. Mirrors /upload-pdf's pipeline from the point
    just after parsing: same test_id/range resolution, same batched
    explanation call, same response shape.
    """
    raw_results = data.get("results", [])
    if not raw_results:
        raise HTTPException(422, "No test values provided.")

    return await run_in_threadpool(
        _process_manual,
        raw_results,
        data.get("user_id", "demo_user"),
        data.get("age", 30),
        data.get("sex", "unknown"),
        data.get("diet_type", "omnivore"),
        data.get("provider"),
        data.get("model_name"),
    )


def _process_manual(raw_results, user_id, age, sex, diet_type, provider, model_name):
    profile = UserProfile(user_id=user_id, age=age, sex=sex, diet_type=diet_type)

    # Same resolution path pdf_parser.py uses for extracted rows, so a
    # manually-typed "Haemoglobin" gets the identical canonical id and
    # normal-range lookup a PDF-extracted "Haemoglobin" would -- no reason
    # for the two entry paths to disagree on what a test name means.
    test_results = []
    for r in raw_results:
        raw_name = (r.get("raw_name") or "").strip()
        if not raw_name:
            continue
        test_id = resolve_test_id(raw_name) or canonicalize_test_name(raw_name)
        finalized = finalize_result(
            {"test_id": test_id, "value": r.get("value"), "unit": r.get("unit", ""),
             "ref_range": None, "status": None},
            patient_sex=sex, patient_age=age,
        )
        test_results.append(TestResult(
            test_id=finalized["test_id"],
            raw_name=raw_name,
            value=finalized["value"],
            unit=finalized.get("unit", ""),
            status=RangeStatus(finalized["status"]) if finalized.get("status") else RangeStatus.UNKNOWN,
            normal_range_min=finalized.get("normal_range_min"),
            normal_range_max=finalized.get("normal_range_max"),
        ))

    if not test_results:
        raise HTTPException(422, "No valid test values provided.")

    explained, degraded_test_ids, ungrounded_test_ids = explain_all_test_results_batched(
        test_results, profile, provider=provider, model_name=model_name
    )
    summary = generate_summary(explained, profile, provider=provider, model_name=model_name)
    summary_degraded = bool(summary.pop("summary_degraded", False))

    return {
        "report_id": str(uuid.uuid4()),
        "parse_method": "manual",
        "explained_results": [r.model_dump() for r in explained],
        "degraded_test_ids": degraded_test_ids,
        "ungrounded_test_ids": ungrounded_test_ids,
        "summary_degraded": summary_degraded,
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
        # ETHICS CONSTRAINT (Chris Clarke): status/normal_range_min/max/
        # what_your_result_means no longer exist on ExplainedResult on this
        # branch (see schemas.py) -- dropped here to match.
        explained_results = []
        for item in raw_explained:
            if isinstance(item, dict):
                explained_results.append(ExplainedResult(
                    test_id=item.get("test_id", "UNKNOWN"),
                    raw_name=item.get("raw_name", "Test"),
                    value=item.get("value", "N/A"),
                    unit=item.get("unit", ""),
                    source=item.get("source", "NHS UK / Medical Consensus"),
                    source_urls=item.get("source_urls", []) or [],
                    what_it_measures=item.get("what_it_measures", ""),
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
        print(f"PDF Generation Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
