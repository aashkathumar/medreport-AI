"""
Test script for verifying real/problematic PDF extraction and explanation.
"""
import os
import requests

API = "http://127.0.0.1:8000/api/v1"

def test_real_pdf():
    # Update this path if your file name is different
    pdf_path = os.path.join(os.path.dirname(__file__), "sterling-accuris-pathology-sample-report-unlocked.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        print("Please place your PDF in the tests/ directory!")
        return

    print(f"📄 Testing PDF: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{API}/upload-pdf",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            params={
                "user_id": "real_pdf_user",
                "age": 30,
                "sex": "female",
                "diet_type": "omnivore",
                "parse_method": "auto"  # Can also try "regex" or "llm" if auto falls back
            },
            timeout=180,
        )

    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS!")
        print(f"• Parse Method Used: {data.get('parse_method')}")
        print(f"• Report ID: {data.get('report_id')}")
        print(f"• Extracted & Explained Tests ({len(data.get('explained_results', []))} total):")
        for item in data.get("explained_results", []):
            print(f"  - {item.get('raw_name')}: {item.get('value')} {item.get('unit')} ({item.get('status')})")
        print(f"\nOverall Summary: {data.get('overall_summary')}")
    else:
        print("❌ FAILED!")
        print(f"Response Error: {response.text}")

if __name__ == "__main__":
    test_real_pdf()