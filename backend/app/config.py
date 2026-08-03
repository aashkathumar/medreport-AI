from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    groq_api_key: str = ""
    mistral_api_key: str = ""
    openrouter_api_key: str = "" 

    groq_model: str = "llama-3.1-8b-instant"
    # Option A (Recommended for Vision): Google Gemini 2.0 Flash via OpenRouter
    openrouter_model: str = "openrouter/free" 
    
    # Option B (Paid/High Performance): Standard paid Llama / Gemini
    # openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"

    # default_llm_provider: str = "openrouter"
    default_llm_provider: str = "groq_llama"  # Options: claude | gemini | groq_llama | groq_mistral | github_models

    aws_region: str = "eu-west-2"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = "medreport-ai-bucket"
    dynamodb_table_name: str = "medreport-reports"
    app_env: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()