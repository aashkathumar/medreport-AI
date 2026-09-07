from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Provider credentials ---------------------------------------------
    gemini_api_key: str = ""
    groq_api_key: str = ""
    mistral_api_key: str = ""
    openrouter_api_key: str = ""
    # Free key from https://build.nvidia.com (any model -> "Get API Key").
    nvidia_api_key: str = ""
    # Optional, provider_has_key() skips it in the chain until this is set.
    cohere_api_key: str = ""
    # Needs a token AND the account ID, the ID is part of the request URL.
    cloudflare_api_key: str = ""
    cloudflare_account_id: str = ""

    # --- Model selection ---------------------------------------------------
    # Every default below was verified live, the old ones were stale and
    # 404ing silently. groq_llama key name kept for backwards compatibility.
    groq_model: str = "openai/gpt-oss-120b"
    # Tracks Google's current flash model, keeps working as it gets renamed.
    gemini_model: str = "gemini-flash-latest"
    mistral_model: str = "mistral-small-latest"
    # Confirmed vision-capable. mistral-ocr-latest might do better on scans.
    mistral_vision_model: str = "mistral-small-latest"
    # Old default was pulled from OpenRouter; these two answer cleanly.
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_vision_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

    # NVIDIA NIM speaks the OpenAI wire format, hence its own vision setting.
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_vision_model: str = "meta/llama-3.2-90b-vision-instruct"

    # No default yet, pick one via live testing once the key is set.
    cohere_model: str = ""
    # Workers AI model ids are namespaced, e.g. "@cf/meta/llama-3.1-8b".
    cloudflare_model: str = ""

    # --- Fallback chains ---------------------------------------------------
    # Provider or provider:model, comma-separated in .env. Order is measured
    # (mistral-first beat groq-first on the 79-test report), not assumed.
    text_fallback_chain_raw: str = Field(
        default="mistral,gemini,groq_llama,nvidia,openrouter",
        validation_alias="TEXT_FALLBACK_CHAIN",
    )

    # Reserved for the Phase 2.5 retry only, so it's never drained by the
    # primary wave before a retry needs it. Gemini's small daily cap suits
    # this well: too little for a full report, plenty for a few stragglers.
    retry_reserved_chain_raw: str = Field(
        default="gemini:gemini-3.1-flash-lite,gemini:gemini-3.6-flash,gemini:gemini-3.5-flash",
        validation_alias="RETRY_RESERVED_CHAIN",
    )

    # Separate chain, groq_llama has no vision model on this account.
    vision_fallback_chain_raw: str = Field(
        default="gemini,mistral,nvidia,openrouter",
        validation_alias="VISION_FALLBACK_CHAIN",
    )

    # Pins one provider at the front of both chains when set.
    default_llm_provider: str = ""  # gemini | groq_llama | mistral | openrouter

    # --- LLM request tuning ------------------------------------------------
    # These existed but were never wired up, so one 429 used to demote a
    # whole request permanently. Now honoured properly by call_with_fallback.
    llm_max_retries: int = 2          # attempts per provider before moving on
    llm_retry_backoff_sec: float = 1.5

    # Applied to every provider; SDK defaults (600s) could hang a request for
    # ~30 minutes otherwise. Vision needs more time than text.
    llm_timeout_sec: float = 130.0
    llm_vision_timeout_sec: float = 180.0

    # Batches run concurrently, not one at a time, serial took minutes on a
    # 79-test report. 8 is the measured sweet spot before rate limits bite.
    llm_max_concurrency: int = 8

    # Bench a failing provider for this long instead of every later batch
    # rediscovering the same outage. Ignored if every provider is open.
    llm_circuit_cooldown_sec: float = 60.0

    # --- AWS ---------------------------------------------------------------
    # No storage settings, server-side persistence was removed after ethics
    # review; reports only live in the frontend's session now.
    aws_region: str = "eu-west-2"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    app_env: str = "development"

    @staticmethod
    def _split_chain(value: str) -> List[str]:
        return [part.strip() for part in (value or "").split(",") if part.strip()]

    @property
    def text_fallback_chain(self) -> List[str]:
        return self._split_chain(self.text_fallback_chain_raw)

    @property
    def vision_fallback_chain(self) -> List[str]:
        return self._split_chain(self.vision_fallback_chain_raw)

    @property
    def retry_reserved_chain(self) -> List[str]:
        return self._split_chain(self.retry_reserved_chain_raw)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
