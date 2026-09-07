"""
End-to-end smoke test against a RUNNING backend.
Start the backend first: uvicorn app.main:app --reload --port 8000
Then, from backend/: python tests/smoke_test.py
"""
import os
import requests

API = "http://127.0.0.1:8000/api/v1"


def test_health():
    url = f"{API}/health"
    r = requests.get(url)
    
    # Print diagnostic information if health check does not return 200 OK
    print(f"Health check status: {r.status_code}")
    print(f"Health check response: {r.text}")
    
    assert r.status_code == 200, f"Expected 200 OK from health check, got {r.status_code}"
    print("Health check passed!")


def test_upload_with_sample_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), "sample_report.pdf")
    if not os.path.exists(pdf_path):
        print("No sample_report.pdf found, run generate_sample_pdf.py first. Skipping upload test.")
        return

    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{API}/upload-pdf",
            files={"file": ("sample.pdf", f, "application/pdf")},
            params={
                "user_id": "test_user", 
                "age": 35, 
                "sex": "female", 
                "diet_type": "vegetarian"
            },
            # The pipeline makes batched LLM calls for every test plus a
            # summary pass; a full report legitimately takes ~5-6 minutes.
            timeout=600,
        )

    print(f"Upload PDF Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Report ID: {data['report_id']}")
        print(f"Explained {len(data['explained_results'])} results")
    else:
        print(f"Response error detail: {r.text}")
    
    assert r.status_code == 200, f"Expected 200 OK from upload PDF endpoint, got {r.status_code}"


if __name__ == "__main__":
    test_health()
    test_upload_with_sample_pdf()