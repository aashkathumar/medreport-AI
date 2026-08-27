"""
Local, AWS-free storage backend used when APP_ENV=development.
Mirrors the interface of dynamo.py + s3.py so routes.py doesn't
need to know which backend is active -- see app/db/__init__.py.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

_STORE_DIR = Path(__file__).parent.parent.parent.parent / "local_data"
_STORE_DIR.mkdir(exist_ok=True)
_REPORTS_FILE = _STORE_DIR / "reports.json"
_PDF_DIR = _STORE_DIR / "pdfs"
_PDF_DIR.mkdir(exist_ok=True)


def _load():
    if _REPORTS_FILE.exists():
        return json.loads(_REPORTS_FILE.read_text())
    return []


def _save(data):
    _REPORTS_FILE.write_text(json.dumps(data, indent=2))


def save_report(user_id: str, report: dict) -> str:
    report_id = str(uuid.uuid4())
    data = _load()
    data.append({
        "user_id": user_id,
        "report_id": report_id,
        "timestamp": datetime.utcnow().isoformat(),
        "report_data": json.dumps(report),
    })
    _save(data)
    return report_id


def list_user_ids() -> list:
    """Distinct user_ids that have at least one saved report, most-recently-
    active first. Backs the frontend's ID picker, which replaced a free-text
    field patients had to retype (and could typo into someone else's ID)."""
    data = sorted(_load(), key=lambda x: x["timestamp"], reverse=True)
    seen = []
    for item in data:
        if item["user_id"] not in seen:
            seen.append(item["user_id"])
    return seen


def get_user_reports(user_id: str) -> list:
    data = _load()
    items = [d for d in data if d["user_id"] == user_id]
    for item in items:
        item["report_data"] = json.loads(item["report_data"])
    return sorted(items, key=lambda x: x["timestamp"], reverse=True)


def upload_report_pdf(pdf_bytes: bytes, user_id: str, report_id: str) -> str:
    path = _PDF_DIR / f"{user_id}_{report_id}.pdf"
    path.write_bytes(pdf_bytes)
    return f"file://{path.resolve()}"
