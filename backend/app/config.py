from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    mistral_api_key: str = ""

    # Groq model ID -- set via .env, not hardcoded, since Groq deprecates
    # specific model IDs fairly often. Check the current live list with:
    #   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
    groq_model: str = "llama-3.1-8b-instant"

    # Which provider explain_test_result()/generate_summary() use by default.
    # One of: "claude", "gemini", "groq_llama", "mistral"
    default_llm_provider: str = "claude"

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
