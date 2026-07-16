"""
Builds data/rag_chunks.json from the existing blood_tests.json / urine_tests.json.
Run this BEFORE build_rag_index.py.

Usage:
    cd data
    python build_rag_chunks.py
"""
import json
from pathlib import Path

_DIR = Path(__file__).parent


def build_chunks():
    chunks = []
    for filename in ["blood_tests.json", "urine_tests.json"]:
        data = json.loads((_DIR / filename).read_text())
        for test in data["tests"]:
            text = (
                f"{test['name']} ({test['id']}): {test['plain_english']} "
                f"If low: {test['low_means']} "
                f"If high: {test['high_means']} "
                f"Unit: {test['unit']}."
            )
            chunks.append({
                "test_id": test["id"],
                "text": text,
                "source": test.get("source", "NHS UK"),
            })
    (_DIR / "rag_chunks.json").write_text(json.dumps(chunks, indent=2))
    print(f"Built {len(chunks)} chunks -> data/rag_chunks.json")


if __name__ == "__main__":
    build_chunks()
