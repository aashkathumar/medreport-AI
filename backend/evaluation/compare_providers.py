import csv
import sys
import time
import textstat
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import TestResult, UserProfile, RangeStatus
from app.services.reference_db import get_reference_data, get_normal_range, flag_result
from app.services.llm_service import explain_test_result

# A small fixed set of test cases -- same inputs across all providers,
# so any difference in output is attributable to the model, not the data.
TEST_CASES = [
    {"test_id": "HGB", "raw_name": "Haemoglobin", "value": 10.2, "unit": "g/dL"},
    {"test_id": "HBA1C", "raw_name": "HbA1c", "value": 58.0, "unit": "mmol/mol"},
    {"test_id": "FERRITIN", "raw_name": "Ferritin", "value": 9.0, "unit": "ng/mL"},
    {"test_id": "TSH", "raw_name": "TSH", "value": 5.8, "unit": "mIU/L"},
]

PROFILE = UserProfile(user_id="eval_user", age=35, sex="female", diet_type="vegetarian")
PROVIDERS_TO_TEST = ["gemini", "groq_llama", "mistral"]  # comment out any you don't have keys for


def run_comparison():
    rows = []
    for case in TEST_CASES:
        ref = get_reference_data(case["test_id"])
        nr = get_normal_range(case["test_id"], sex=PROFILE.sex)
        status = flag_result(case["value"], nr)
        test = TestResult(
            test_id=case["test_id"], raw_name=case["raw_name"],
            value=case["value"], unit=case["unit"], status=status,
            normal_range_min=nr["min"] if nr else None,
            normal_range_max=nr["max"] if nr else None,
        )

        for provider in PROVIDERS_TO_TEST:
            print(f"Running {case['raw_name']} through {provider}...")
            start = time.time()
            try:
                result = explain_test_result(test, PROFILE, ref, provider=provider)
                elapsed = round(time.time() - start, 2)
                rows.append({
                    "test": case["raw_name"],
                    "provider": provider,
                    "status": status.value,
                    "what_it_measures": result.what_it_measures,
                    "what_your_result_means": result.what_your_result_means,
                    "lifestyle_suggestions": " | ".join(result.lifestyle_suggestions),
                    "gp_question": result.gp_question,
                    "response_time_sec": elapsed,
                    "error": "",
                    "fk_grade_what_it_measures": textstat.flesch_kincaid_grade(result.what_it_measures),
                    "fk_grade_what_result_means": textstat.flesch_kincaid_grade(result.what_your_result_means),
                })
            except Exception as e:
                rows.append({
                    "test": case["raw_name"], "provider": provider, "status": status.value,
                    "what_it_measures": "", "what_your_result_means": "",
                    "lifestyle_suggestions": "", "gp_question": "",
                    "response_time_sec": "", "error": str(e),
                })

    out_path = Path(__file__).parent / "comparison_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone -- {len(rows)} rows written to {out_path}")


if __name__ == "__main__":
    run_comparison()