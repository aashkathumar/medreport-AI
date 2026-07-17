import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.pdf_parser import _extract_from_text


def test_extraction_colon_format():
    text = "Haemoglobin: 14.2 g/dL\nWhite Blood Cell Count: 6.5 10^9/L"
    results = _extract_from_text(text)
    assert len(results) == 2
    assert results[0]["test_id"] == "HGB"
    assert results[0]["value"] == 14.2


def test_extraction_spaced_format():
    text = "Platelets      250    10^9/L"
    results = _extract_from_text(text)
    assert any(r["test_id"] == "PLT" for r in results)


if __name__ == "__main__":
    test_extraction_colon_format()
    test_extraction_spaced_format()
    print("All pdf_parser tests passed")
