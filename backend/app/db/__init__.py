"""
Storage backend switch.
APP_ENV=development (default, in .env) -> local JSON-file storage, no AWS needed.
APP_ENV=production                      -> real DynamoDB + S3 (requires AWS credentials).
"""
from app.config import settings

if settings.app_env == "development":
    from app.db.local_storage import save_report, get_user_reports, upload_report_pdf
else:
    from app.db.dynamo import save_report, get_user_reports
    from app.db.s3 import upload_report_pdf

__all__ = ["save_report", "get_user_reports", "upload_report_pdf"]
