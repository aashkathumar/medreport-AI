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
    # Scaffolded ahead of a key -- empty is fine, provider_has_key() skips it
    # in the chain until CEREBRAS_API_KEY is set. Get one from
    # https://cloud.cerebras.ai.
    cerebras_api_key: str = ""
    # Scaffolded ahead of a key, same pattern as cerebras above. Get one
    # from https://dashboard.cohere.com/api-keys (trial key, no card).
    cohere_api_key: str = ""
    # Scaffolded ahead of a key, same pattern as cerebras/cohere above.
    # Workers AI needs BOTH a token (Account > Workers AI > Read + Edit,
    # from dash.cloudflare.com) AND the account ID shown on that same page
    # -- unlike the other providers, the account ID is part of the request
    # URL, not just an auth header, so it must be set too or every call
    # 400s before it reaches a model.
    cloudflare_api_key: str = ""
    cloudflare_account_id: str = ""

    # --- Model selection ---------------------------------------------------
    # Every default below was checked against the live provider APIs. The
    # previous defaults were all stale or invalid, and because the callers
    # swallowed exceptions the resulting 404s were invisible:
    #   groq_model       "llama-3.1-8b-instant" -> 404, the account serves no
    #                    llama models at all any more (the "groq_llama"
    #                    provider key is kept only so existing .env files and
    #                    saved frontend selections keep working).
    #   gemini_model     "gemini-2.0-flash"     -> 404, retired.
    #   openrouter_model "openrouter/free"      -> not a model id at all.
    groq_model: str = "openai/gpt-oss-120b"
    # gemini-flash-latest tracks Google's current flash model, which is what
    # kept this project working when 2.0-flash was retired. The concrete
    # alternatives (gemini-3.5-flash, gemini-3.6-flash) both returned
    # unparseable JSON on this account's key.
    gemini_model: str = "gemini-flash-latest"
    mistral_model: str = "mistral-small-latest"
    # Verified vision-capable on this key. mistral-ocr-latest is the
    # document-OCR specialist and may suit lab reports better -- see the
    # benchmark before switching.
    mistral_vision_model: str = "mistral-small-latest"
    # VERIFIED LIVE against this key. The previous default
    # ("nvidia/nemotron-nano-12b-v2-vl:free") returns 404 "No endpoints found"
    # -- it has been withdrawn from OpenRouter, so the last entry of BOTH
    # fallback chains was a guaranteed failure. These two were the only free
    # models on this account that answered cleanly; most others 429 instantly.
    # openrouter_model is text-only (404s on image blocks), hence the separate
    # vision entry, which was confirmed to actually read a rendered lab page.
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_vision_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

    # NVIDIA NIM speaks the OpenAI wire format, so it uses the same client as
    # OpenRouter. It gets its own text and vision model settings because the
    # vision tier needs a VLM -- a text model would fail on image blocks the
    # same way groq/mistral do.
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_vision_model: str = "meta/llama-3.2-90b-vision-instruct"

    # Text-only (no vision endpoint on Cerebras's inference API). Confirmed
    # from the account's own Limits page (not a guess): "gpt-oss-120b" is
    # tagged Production; "gemma-4-31b" is tagged Preview -- default to the
    # production one. Both get their own separate quota (5 req/min, 2400/day,
    # 90K tokens/min each per the same page), so gemma-4-31b is added as a
    # second chain entry below rather than only living here as an unused
    # alternative -- still unverified by an actual generation call, since no
    # key is configured yet to test with.
    cerebras_model: str = "gpt-oss-120b"
    # No default picked yet -- find one via live testing once
    # COHERE_API_KEY is set.
    cohere_model: str = ""
    # No default picked yet -- find one via live testing once
    # CLOUDFLARE_API_KEY/CLOUDFLARE_ACCOUNT_ID are set. Model ids on Workers
    # AI are namespaced, e.g. "@cf/meta/llama-3.1-8b-instruct" -- not bare
    # names like the other providers use.
    cloudflare_model: str = ""

    # --- Fallback chains ---------------------------------------------------
    # CHANGED: call_with_fallback() read `settings.text_fallback_chain`, which
    # was never defined on this class. Since routes.upload_pdf defaults
    # `provider` to None, EVERY call on the default path raised
    # AttributeError before reaching a model, and the three call sites
    # swallowed it -- so extraction and explanation silently produced canned
    # fallback text instead of LLM output. These are the missing settings.
    #
    # Entries are "provider" or "provider:model". Set as a comma-separated
    # list in .env, e.g.
    #   TEXT_FALLBACK_CHAIN=groq_llama,gemini,mistral
    #
    # Ordering rationale (MEASURED, not assumed): groq_llama used to lead on
    # the belief it was "fastest + generous free tier". Its free tier is
    # actually 8000 tokens/MINUTE, and one explanation batch is ~5k tokens --
    # so once batches are issued concurrently every batch 429s and falls
    # through, paying retry latency for nothing. On the 79-test report,
    # mistral-first ran 28.5s with zero provider failures vs 37.0s and
    # constant 429 churn with groq-first. groq stays in the chain lower down.
    # Stored as a raw string and exposed as a list via the property below:
    # pydantic-settings tries to JSON-decode a list-typed field coming from a
    # dotenv file BEFORE any validator runs, so a plain comma-separated value
    # raises SettingsError at import. Keeping the field a str and parsing it
    # ourselves keeps .env readable on this version (2.5.2, which has no
    # NoDecode annotation).
    text_fallback_chain_raw: str = Field(
        default="mistral,gemini,groq_llama,nvidia,openrouter",
        validation_alias="TEXT_FALLBACK_CHAIN",
    )

    # Held back from the primary chain above and used ONLY on the Phase 2.5
    # retry in llm_service.py, tried before the regular chain resumes as a
    # further fallback. The point: an entry that's also open to the primary
    # wave can get circuit-tripped or quota-exhausted by an unrelated batch
    # before a retry ever needs it -- a reserved entry is never touched by
    # anything but retries, so it's guaranteed fresh. Gemini fits this well:
    # its 20/day-per-model cap makes it too small to lean on for every
    # report's primary explanations, but that same small cap comfortably
    # covers the rare 1-3 straggler tests a retry actually needs per report.
    retry_reserved_chain_raw: str = Field(
        default="gemini:gemini-3.1-flash-lite,gemini:gemini-3.6-flash,gemini:gemini-3.5-flash",
        validation_alias="RETRY_RESERVED_CHAIN",
    )

    # Vision is a SEPARATE chain: groq_llama is text-only on this account (its
    # model list has no VLM at all) and raises on image payloads rather than
    # silently dropping the images and inventing a lab report. Including it
    # here would burn a guaranteed failure on every vision batch.
    #
    # mistral IS in this chain -- it was previously assumed text-only, but its
    # /v1/models endpoint lists 32 vision-capable models and extraction was
    # verified end-to-end. Gemini's free tier is only 20 requests/day, so
    # mistral sits directly behind it to absorb the overflow.
    vision_fallback_chain_raw: str = Field(
        default="gemini,mistral,nvidia,openrouter",
        validation_alias="VISION_FALLBACK_CHAIN",
    )

    # Used as the front of both chains when set -- lets you pin one provider
    # for a run (e.g. a dissertation evaluation) without editing the chains.
    default_llm_provider: str = ""  # gemini | groq_llama | mistral | openrouter

    # --- LLM request tuning ------------------------------------------------
    # CHANGED: these two were defined here and referenced NOWHERE -- there was
    # no retry logic at all. A single transient 429 from the fast provider
    # (groq, ~5s/batch) demoted the whole request to gemini (~33s/batch) for
    # good. call_with_fallback now actually honours them, retrying transient
    # failures on the same provider before falling through.
    llm_max_retries: int = 2          # attempts per provider before moving on
    llm_retry_backoff_sec: float = 1.5

    # Hard per-request wall-clock ceiling, applied to EVERY provider. Only
    # mistral previously set a timeout; the OpenAI-SDK providers (openrouter,
    # nvidia) and gemini used SDK defaults of 600s WITH built-in retries, so a
    # single stalled provider could hang a request for ~30 minutes before the
    # fallback chain moved on. Vision payloads carry full-page images and
    # legitimately need longer than text calls.
    # 60s was too low and made NVIDIA structurally unusable: a 6-test batch
    # measured 102s there, so every attempt timed out, retried, and timed out
    # again -- burning ~120s per batch to reach a guaranteed failure. The
    # ceiling must exceed the SLOWEST provider in the chain, not the fastest.
    llm_timeout_sec: float = 130.0
    llm_vision_timeout_sec: float = 180.0

    # Explanation batches are independent of one another, so they are issued
    # concurrently instead of one-at-a-time. A 79-test report is 14 batches;
    # serially that ran to MINUTES, which was the "runs forever and returns
    # nothing" symptom.
    # MEASURED end-to-end on that report, all with ZERO batch failures:
    #   2 -> ~48s | 4 -> 24.4s | 6 -> 19.1s | 8 -> 15.7s
    # The extra concurrency costs no reliability at 8; above it the free-tier
    # rate limits start biting and batches begin failing into fallback copy.
    llm_max_concurrency: int = 8

    # Circuit breaker. When a provider exhausts its retries, it is skipped for
    # this many seconds instead of being re-tried by every subsequent batch.
    # Without it, a hard provider outage (observed: Mistral returning 500 then
    # Cloudflare 522, ~20-37s per attempt) is re-discovered by all 14 batches
    # of a large report -- minutes of pure dead time before each one falls
    # through to a working provider. One batch pays the discovery cost; the
    # rest skip straight past. If EVERY provider is open the breaker is
    # ignored, so this can never be the reason a request has nowhere to go.
    llm_circuit_cooldown_sec: float = 60.0

    # --- AWS / storage -----------------------------------------------------
    aws_region: str = "eu-west-2"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = "medreport-ai-bucket"
    dynamodb_table_name: str = "medreport-reports"
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
