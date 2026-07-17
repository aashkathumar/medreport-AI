"""
AWS S3 storage backend -- used when APP_ENV=production.
Requires real AWS credentials and a deployed S3 bucket
(see template.yaml). Not used during local-only testing.
"""
import boto3
from app.config import settings

_s3 = None


def _get_s3():
    global _s3
    if not _s3:
        _s3 = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
    return _s3


def upload_report_pdf(pdf_bytes: bytes, user_id: str, report_id: str) -> str:
    key = f"reports/{user_id}/{report_id}.pdf"
    _get_s3().put_object(
        Bucket=settings.s3_bucket_name, Key=key,
        Body=pdf_bytes, ContentType="application/pdf",
    )
    return _get_s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket_name, "Key": key},
        ExpiresIn=3600,
    )
