from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    aws_region: str = "eu-west-2"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = "medreport-ai-bucket"
    dynamodb_table_name: str = "medreport-reports"
    app_env: str = "development"  # "development" = local storage, "production" = AWS

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
