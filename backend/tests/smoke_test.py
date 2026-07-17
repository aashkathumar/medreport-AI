"""
End-to-end smoke test against a RUNNING backend.
Start the backend first: uvicorn app.main:app --reload --port 8000
Then, from backend/: python tests/smoke_test.py
"""
import os
import requests

API = "http://localhost:8000/api/v1"


def test_health():
    r = requests.get(f"{API}/health")
    assert r.status_code == 200
    print("Health check passed")


def test_upload_with_sample_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), "sample_report.pdf")
    if not os.path.exists(pdf_path):
        print("No sample_report.pdf found -- run generate_sample_pdf.py first. Skipping upload test.")
        return
    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{API}/upload-pdf",
            files={"file": ("sample.pdf", f, "application/pdf")},
            params={"user_id": "test_user", "age": 35, "sex": "female", "diet_type": "vegetarian"},
            timeout=60,
        )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Report ID: {data['report_id']}")
        print(f"Explained {len(data['explained_results'])} results")
    else:
        print(f"Response: {r.text}")


if __name__ == "__main__":
    test_health()
    test_upload_with_sample_pdf()
