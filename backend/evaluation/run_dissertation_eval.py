"""One-off evaluation run for the dissertation's Chapter 4 (Results). Not
a permanent part of the application, run once by hand, with the same test
cases and scoring reused unchanged across branches for a direct
comparison.

Metrics: grounding_pass (whether the explanation passed the system's own
anti-hallucination verification without degrading to fallback, a genuine
existing behaviour rather than a metric invented for this evaluation),
completeness (required fields present, automatable rather than a
judgement call), and flesch_kincaid (readability grade of the combined
patient-facing text via the `textstat` package).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import textstat
from app.models.schemas import TestResult, UserProfile, RangeStatus
from app.services.llm_service import explain_all_test_results_batched

TEST_CASES = [
    # (category, raw_name, value, unit)
    ("Full blood count", "Haemoglobin", 14.2, "g/dL"),
    ("Full blood count", "White Blood Cell Count", 7.1, "10^9/L"),
    ("Full blood count", "Platelet Count", 260, "10^9/L"),
    ("Full blood count", "Haematocrit", 42.0, "%"),
    ("Full blood count", "Mean Corpuscular Volume", 89.0, "fL"),
    ("Lipid panel", "Total Cholesterol", 5.1, "mmol/L"),
    ("Lipid panel", "HDL Cholesterol", 1.4, "mmol/L"),
    ("Lipid panel", "LDL Cholesterol", 3.0, "mmol/L"),
    ("Lipid panel", "Triglycerides", 1.2, "mmol/L"),
    ("Lipid panel", "Total Cholesterol/HDL Ratio", 3.6, "ratio"),
    ("Urinalysis", "Urine pH", 6.0, "pH"),
    ("Urinalysis", "Urine Protein", "Negative", ""),
    ("Urinalysis", "Urine Glucose", "Negative", ""),
    ("Urinalysis", "Specific Gravity", 1.02, ""),
    ("Urinalysis", "Leukocyte Esterase", "Negative", ""),
]

PROFILE = UserProfile(user_id="dissertation_eval", age=45, sex="female", diet_type="omnivore")


def build_test_result(raw_name, value, unit):
    from app.services.reference_db import resolve_test_id, canonicalize_test_name, finalize_result
    test_id = resolve_test_id(raw_name) or canonicalize_test_name(raw_name)
    finalized = finalize_result(
        {"test_id": test_id, "value": value, "unit": unit, "ref_range": None, "status": None},
        patient_sex=PROFILE.sex, patient_age=PROFILE.age,
    )
    return TestResult(
        test_id=finalized["test_id"], raw_name=raw_name, value=finalized["value"],
        unit=finalized.get("unit", ""),
        status=RangeStatus(finalized["status"]) if finalized.get("status") else RangeStatus.UNKNOWN,
        normal_range_min=finalized.get("normal_range_min"),
        normal_range_max=finalized.get("normal_range_max"),
    )


def score_completeness(r):
    has_wim = bool(getattr(r, "what_it_measures", "") and r.what_it_measures.strip())
    has_lifestyle = len(getattr(r, "lifestyle_suggestions", []) or []) >= 2
    has_gp_q = bool(getattr(r, "gp_question", "") and r.gp_question.strip())
    return has_wim and has_lifestyle and has_gp_q


def patient_text(r):
    parts = [r.what_it_measures]
    parts += r.lifestyle_suggestions or []
    parts.append(r.gp_question)
    if hasattr(r, "what_your_result_means"):
        parts.append(r.what_your_result_means)
    return " ".join(p for p in parts if p)


def run():
    results_by_category = {}
    all_rows = []

    for category, raw_name, value, unit in TEST_CASES:
        tr = build_test_result(raw_name, value, unit)
        explained, degraded, ungrounded = explain_all_test_results_batched([tr], PROFILE)
        r = explained[0] if explained else None

        row = {
            "category": category, "raw_name": raw_name, "value": value, "unit": unit,
            "degraded": tr.test_id in degraded, "ungrounded": tr.test_id in ungrounded,
        }
        if r is not None:
            row["grounding_pass"] = not row["degraded"] and not row["ungrounded"]
            row["completeness"] = score_completeness(r)
            text = patient_text(r)
            row["fk_grade"] = round(textstat.flesch_kincaid_grade(text), 1) if text.strip() else None
        else:
            row["grounding_pass"] = False
            row["completeness"] = False
            row["fk_grade"] = None

        all_rows.append(row)
        results_by_category.setdefault(category, []).append(row)
        print(f"[{category}] {raw_name} = {value} {unit} -> "
              f"grounded={row['grounding_pass']} complete={row['completeness']} FK={row['fk_grade']}")

    print("\n=== SUMMARY ===")
    summary = {}
    for category, rows in results_by_category.items():
        n = len(rows)
        acc = sum(r["grounding_pass"] for r in rows) / n * 100
        comp = sum(r["completeness"] for r in rows) / n * 100
        fks = [r["fk_grade"] for r in rows if r["fk_grade"] is not None]
        fk = sum(fks) / len(fks) if fks else None
        summary[category] = {"n": n, "accuracy_pct": round(acc, 1), "completeness_pct": round(comp, 1),
                              "fk_grade": round(fk, 1) if fk is not None else None}
        print(f"{category}: n={n} accuracy={acc:.1f}% completeness={comp:.1f}% FK={fk}")

    n_all = len(all_rows)
    acc_all = sum(r["grounding_pass"] for r in all_rows) / n_all * 100
    comp_all = sum(r["completeness"] for r in all_rows) / n_all * 100
    fks_all = [r["fk_grade"] for r in all_rows if r["fk_grade"] is not None]
    fk_all = sum(fks_all) / len(fks_all) if fks_all else None
    summary["Overall"] = {"n": n_all, "accuracy_pct": round(acc_all, 1), "completeness_pct": round(comp_all, 1),
                           "fk_grade": round(fk_all, 1) if fk_all is not None else None}
    print(f"Overall: n={n_all} accuracy={acc_all:.1f}% completeness={comp_all:.1f}% FK={fk_all}")

    out = Path(__file__).resolve().parent / "eval_results.json"
    out.write_text(json.dumps({"rows": all_rows, "summary": summary}, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    run()
