"""
AWS DynamoDB storage backend -- used when APP_ENV=production.
Requires real AWS credentials and a deployed DynamoDB table
(see template.yaml). Not used during local-only testing.
"""
import boto3
import json
import uuid
from datetime import datetime
from app.config import settings

_client = None


def _get_table():
    global _client
    if not _client:
        db = boto3.resource(
            "dynamodb",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        _client = db.Table(settings.dynamodb_table_name)
    return _client


def save_report(user_id: str, report: dict) -> str:
    report_id = str(uuid.uuid4())
    _get_table().put_item(Item={
        "user_id": user_id,
        "report_id": report_id,
        "timestamp": datetime.utcnow().isoformat(),
        "report_data": json.dumps(report),
    })
    return report_id


def get_user_reports(user_id: str) -> list:
    resp = _get_table().query(
        KeyConditionExpression="user_id = :uid",
        ExpressionAttributeValues={":uid": user_id},
    )
    items = resp.get("Items", [])
    for item in items:
        item["report_data"] = json.loads(item["report_data"])
    return sorted(items, key=lambda x: x["timestamp"], reverse=True)
