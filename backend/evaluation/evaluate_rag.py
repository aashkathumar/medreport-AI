"""Evaluation harness for the constrained RAG pipeline, focused on
retrieval accuracy and readability against NHS UK / NIH MedlinePlus
source material rather than clinical interpretation.

Reports three things: retrieval accuracy (does a raw test name return
passages from the page that actually covers that analyte, via Precision@1/
Recall@k plus the rejection rate on out-of-corpus queries); groundedness
(what fraction of generated explanations pass post-generation source
verification, since ungrounded output is replaced by a GP referral in
production); and readability (Flesch-Kincaid grade and reading ease of
the generated text vs. the source passages it was built from).

Usage (from backend/):
    python evaluation/evaluate_rag.py              # retrieval only, no API calls
    python evaluation/evaluate_rag.py --generate   # also generates explanations
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import RangeStatus, TestResult, UserProfile
from app.services.rag_service import index_health, retrieve_context

# Raw test names as they actually appear on lab reports, paired with the
# canonical test_id whose source page should be retrieved.
RETRIEVAL_CASES = [
    ("Haemoglobin", "HGB"),
    ("Hb", "HGB"),
    ("Total Leucocyte Count", "WBC"),
    ("Platelet Count", "PLT"),
    ("Glycosylated Haemoglobin (HbA1c)", "HBA1C"),
    ("Fasting Blood Sugar", "GLUCOSE"),
    ("Serum Creatinine", "CREATININE"),
    ("Blood Urea Nitrogen", "UREA"),
    ("Uric Acid", "URIC_ACID"),
    ("Total Cholesterol", "CHOL"),
    ("Triglycerides", "TRIG"),
    ("HDL Cholesterol", "HDL"),
    ("SGPT (ALT)", "ALT"),
    ("SGOT (AST)", "AST"),
    ("Alkaline Phosphatase", "ALP"),
    ("Total Bilirubin", "BILIRUBIN"),
    ("Serum Albumin", "ALBUMIN"),
    ("TSH - Thyroid Stimulating Hormone", "TSH"),
    ("T3 - Triiodothyronine", "T3"),
    ("T4 - Thyroxine", "T4"),
    ("25(OH) Vitamin D", "VIT_D"),
    ("Vitamin B12", "B12"),
    ("Ferritin", "FERRITIN"),
    ("Serum Iron", "IRON"),
    ("C-Reactive Protein", "CRP"),
    ("ESR", "ESR"),
    ("Sodium (Na+)", "NA"),
    ("Potassium (K+)", "K"),
    ("Serum Calcium", "CA"),
    ("Urine Protein", "URINE_PROTEIN"),
    ("Urine Ketone", "URINE_KETONES"),
    ("Nitrite", "URINE_NITRITES"),
    ("PSA-Prostate Specific Antigen", "PSA"),
    ("MCV", "MCV"),
    ("Hematocrit", "HCT"),
]

# Must retrieve NOTHING, these are not covered by the NHS/NIH corpus, and
# the system is required to direct the user to their GP instead of inventing
# an explanation.
OUT_OF_CORPUS_CASES = [
    "Zorbleflax Index",
    "Quantum Meridian Factor",
    "Chakra Alignment Score",
]

K = 3


def evaluate_retrieval() -> dict:
    hits_at_1 = 0
    hits_at_k = 0
    misses = []

    for raw_name, expected_id in RETRIEVAL_CASES:
        passages = retrieve_context(raw_name, k=K, test_id=expected_id)
        ids_at_1 = set(passages[0].get("test_ids", [])) if passages else set()
        ids_at_k = {tid for p in passages for tid in p.get("test_ids", [])}

        if expected_id in ids_at_1:
            hits_at_1 += 1
        if expected_id in ids_at_k:
            hits_at_k += 1
        else:
            misses.append((raw_name, expected_id,
                           [p.get("topic") for p in passages] or ["<nothing>"]))

    rejected = sum(1 for name in OUT_OF_CORPUS_CASES if not retrieve_context(name, k=K))

    total = len(RETRIEVAL_CASES)
    return {
        "total": total,
        "precision_at_1": hits_at_1 / total,
        "recall_at_k": hits_at_k / total,
        "out_of_corpus_rejected": rejected,
        "out_of_corpus_total": len(OUT_OF_CORPUS_CASES),
        "misses": misses,
    }


def evaluate_generation() -> dict:
    """Generates explanations and measures groundedness + readability.
    Makes real API calls."""
    import textstat

    from app.services.llm_service import explain_all_test_results_batched

    profile = UserProfile(user_id="eval", age=45, sex="female", diet_type="omnivore")
    tests = [
        TestResult(test_id=tid, raw_name=name, value=1.0, unit="",
                   status=RangeStatus.NORMAL)
        for name, tid in RETRIEVAL_CASES[:12]
    ]

    explained, degraded, ungrounded = explain_all_test_results_batched(tests, profile)

    rows = []
    for r in explained:
        text = f"{r.what_it_measures} {r.what_your_result_means}"
        rows.append({
            "test_id": r.test_id,
            "raw_name": r.raw_name,
            "source": r.source,
            "source_urls": " | ".join(r.source_urls),
            "grounded": r.test_id not in ungrounded and r.test_id not in degraded,
            "fk_grade": round(textstat.flesch_kincaid_grade(text), 2),
            "reading_ease": round(textstat.flesch_reading_ease(text), 2),
            "what_it_measures": r.what_it_measures,
            "what_your_result_means": r.what_your_result_means,
        })

    # Readability of the NHS/NIH source text itself, for comparison.
    source_grades = []
    for name, tid in RETRIEVAL_CASES[:12]:
        for p in retrieve_context(name, k=1, test_id=tid):
            source_grades.append(textstat.flesch_kincaid_grade(p["text"]))

    out_path = Path(__file__).parent / "rag_evaluation_results.csv"
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    grounded = [r for r in rows if r["grounded"]]
    return {
        "generated": len(rows),
        "grounded": len(grounded),
        "ungrounded": len(ungrounded),
        "degraded": len(degraded),
        "mean_fk_grade": (sum(r["fk_grade"] for r in grounded) / len(grounded)) if grounded else 0,
        "mean_reading_ease": (sum(r["reading_ease"] for r in grounded) / len(grounded)) if grounded else 0,
        "mean_source_fk_grade": (sum(source_grades) / len(source_grades)) if source_grades else 0,
        "csv": out_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="also generate explanations (makes real API calls)")
    args = parser.parse_args()

    health = index_health()
    print("=" * 66)
    print("RAG INDEX")
    print("=" * 66)
    print(f"  available     : {health['available']}")
    print(f"  chunks        : {health['chunk_count']}")
    print(f"  embed model   : {health['embed_model']}")
    if not health["available"]:
        print(f"  ERROR: {health['error']}")
        return 1

    print()
    print("=" * 66)
    print("1. RETRIEVAL ACCURACY")
    print("=" * 66)
    r = evaluate_retrieval()
    print(f"  cases                 : {r['total']}")
    print(f"  Precision@1           : {r['precision_at_1']:.1%}")
    print(f"  Recall@{K}              : {r['recall_at_k']:.1%}")
    print(f"  out-of-corpus rejected: {r['out_of_corpus_rejected']}/{r['out_of_corpus_total']}"
          "   (must be all, else the retriever invents coverage)")
    if r["misses"]:
        print("\n  misses:")
        for name, expected, topics in r["misses"]:
            print(f"    {name} (wanted {expected}) -> {topics}")

    if args.generate:
        print()
        print("=" * 66)
        print("2. GROUNDEDNESS  +  3. READABILITY")
        print("=" * 66)
        g = evaluate_generation()
        print(f"  explanations generated : {g['generated']}")
        print(f"  passed source check    : {g['grounded']}")
        print(f"  rejected -> GP referral: {g['ungrounded']}")
        print(f"  provider failures      : {g['degraded']}")
        print(f"  mean FK grade (output) : {g['mean_fk_grade']:.1f}")
        print(f"  mean reading ease      : {g['mean_reading_ease']:.1f}")
        print(f"  mean FK grade (NHS/NIH source) : {g['mean_source_fk_grade']:.1f}")
        print(f"  per-test CSV -> {g['csv']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
