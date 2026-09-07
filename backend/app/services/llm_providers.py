import base64
import json
import re
import requests
from typing import Union, List, Dict, Any, Optional
from app.config import settings


def _resolve_timeout(timeout: Optional[float] = None, has_images: bool = False) -> float:
    """Every provider call gets a hard wall-clock ceiling."""
    if timeout is not None:
        return timeout
    return settings.llm_vision_timeout_sec if has_images else settings.llm_timeout_sec


def _has_images(prompt: Union[str, list]) -> bool:
    return isinstance(prompt, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in prompt
    )


def _clean_json_string(text: Optional[str]) -> str:
    """Helper to remove markdown backticks (```json ... ```) or extract valid JSON object."""
    if not text or not str(text).strip():
        return "{}"
    cleaned = text.strip()
    # Remove markdown codeblocks
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Trim to the outermost {...} span. Used to miss trailing content after
    # a valid object and throw "Extra data"; a no-op on already-clean JSON.
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

    return cleaned or "{}"


def call_gemini(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 2500,
    model: str = None,
    timeout: float = None
) -> dict:
    """Calls Google Gemini API natively. Accepts text strings or vision content arrays."""
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)

    target_model = model or settings.gemini_model
    client = genai.GenerativeModel(target_model, system_instruction=system)

    contents = prompt
    if isinstance(prompt, list):
        parts = []
        for block in prompt:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    data_url = block.get("image_url", {}).get("url", "")
                    if data_url.startswith("data:"):
                        header, _, b64_data = data_url.partition(",")
                        mime = header.split(";")[0].replace("data:", "") or "image/png"
                        parts.append({"mime_type": mime, "data": base64.b64decode(b64_data)})
        contents = parts if parts else "Extract lab results"

    response = client.generate_content(
        contents,
        generation_config={
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        },
        request_options={"timeout": _resolve_timeout(timeout, _has_images(prompt))},
    )

    try:
        raw_text = response.text
    except Exception:
        # Catches safety blocks or empty candidate payloads
        raw_text = "{}"

    return json.loads(_clean_json_string(raw_text))


def call_groq_llama(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 4000,
    model: str = None,
    timeout: float = None
) -> dict:
    from groq import Groq
    client = Groq(
        api_key=settings.groq_api_key,
        timeout=_resolve_timeout(timeout),
        max_retries=0
    )
    target_model = model or settings.groq_model

    if isinstance(prompt, list) and any(b.get("type") == "image_url" for b in prompt if isinstance(b, dict)):
        raise ValueError("call_groq_llama is text-only and cannot process image content blocks.")

    user_content = prompt
    if isinstance(prompt, list):
        user_content = " ".join([b["text"] for b in prompt if isinstance(b, dict) and b.get("type") == "text"])

    msg = client.chat.completions.create(
        model=target_model,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    content = msg.choices[0].message.content or "{}"
    return json.loads(_clean_json_string(content))


def call_mistral(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 2500,
    model: str = None,
    timeout: float = None
) -> dict:
    has_images = _has_images(prompt)
    target_model = model or (
        settings.mistral_vision_model if has_images else settings.mistral_model
    )

    user_content = prompt
    if isinstance(prompt, list) and not has_images:
        user_content = " ".join([b["text"] for b in prompt if isinstance(b, dict) and b.get("type") == "text"])
    elif has_images:
        user_content = []
        for block in prompt:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = block.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url", "")
                user_content.append({"type": "image_url", "image_url": url})
            else:
                user_content.append(block)

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": target_model,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=_resolve_timeout(timeout, has_images),
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"] if data.get("choices") else "{}"
    return json.loads(_clean_json_string(content))


def call_cohere(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 2500,
    model: str = None,
    timeout: float = None
) -> dict:
    """Cohere v2 Chat API, text-only. Reply is nested in message.content as
    typed items, so this looks for the "text"-typed one, not content[0].
    response_format isn't supported by every model, so it's tried first and
    only dropped on that specific 400, not on any other error.
    """
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is not configured in .env!")

    if isinstance(prompt, list) and any(b.get("type") == "image_url" for b in prompt if isinstance(b, dict)):
        raise ValueError("call_cohere is text-only and cannot process image content blocks.")

    user_content = prompt
    if isinstance(prompt, list):
        user_content = " ".join([b["text"] for b in prompt if isinstance(b, dict) and b.get("type") == "text"])

    target_model = model or settings.cohere_model
    if not target_model:
        raise ValueError("No Cohere model configured, set 'model' or cohere_model.")

    headers = {
        "Authorization": f"Bearer {settings.cohere_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": target_model,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }
    resp = requests.post("https://api.cohere.com/v2/chat", headers=headers, json=body,
                          timeout=_resolve_timeout(timeout))
    if resp.status_code == 400 and "response_format is not supported" in resp.text:
        body.pop("response_format")
        resp = requests.post("https://api.cohere.com/v2/chat", headers=headers, json=body,
                              timeout=_resolve_timeout(timeout))
    resp.raise_for_status()
    data = resp.json()
    items = data.get("message", {}).get("content", []) or []
    text = next((it.get("text", "") for it in items if isinstance(it, dict) and it.get("type") == "text"), "")
    if not text and items and isinstance(items[0], dict):
        text = items[0].get("text", "")
    return json.loads(_clean_json_string(text or "{}"))


def call_cloudflare(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 2500,
    model: str = None,
    timeout: float = None
) -> dict:
    """Cloudflare Workers AI, text-only. Newer models return OpenAI-style
    choices[0].message.content; older ones have no "choices" key and put
    the answer in result.response instead, as a string or already-parsed
    dict, so this checks choices first and falls back to result.response.
    Reasoning models need max_tokens 2000-4000, 800 left several empty.
    """
    if not settings.cloudflare_api_key:
        raise ValueError("CLOUDFLARE_API_KEY is not configured in .env!")
    if not settings.cloudflare_account_id:
        raise ValueError("CLOUDFLARE_ACCOUNT_ID is not configured in .env!")

    if isinstance(prompt, list) and any(b.get("type") == "image_url" for b in prompt if isinstance(b, dict)):
        raise ValueError("call_cloudflare is text-only and cannot process image content blocks.")

    user_content = prompt
    if isinstance(prompt, list):
        user_content = " ".join([b["text"] for b in prompt if isinstance(b, dict) and b.get("type") == "text"])

    target_model = model or settings.cloudflare_model
    if not target_model:
        raise ValueError("No Cloudflare model configured, set 'model' or cloudflare_model.")

    url = f"https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}/ai/run/{target_model}"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.cloudflare_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=_resolve_timeout(timeout),
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", True):
        raise ValueError(f"Cloudflare Workers AI error: {data.get('errors')}")
    result = data.get("result", {})
    choices = result.get("choices", []) or []
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        return json.loads(_clean_json_string(content or "{}"))
    legacy_response = result.get("response")
    if isinstance(legacy_response, (dict, list)):
        return legacy_response
    return json.loads(_clean_json_string(legacy_response or "{}"))


def call_openrouter(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 2500,
    model: str = None,
    timeout: float = None
) -> dict:
    from openai import OpenAI

    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured in .env!")

    has_images = _has_images(prompt)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        timeout=_resolve_timeout(timeout, has_images),
        max_retries=0,
    )

    target_model = model or (
        settings.openrouter_vision_model if has_images else settings.openrouter_model
    )

    response = client.chat.completions.create(
        model=target_model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content if response.choices else "{}"
    return json.loads(_clean_json_string(content or "{}"))


def call_nvidia(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 2500,
    model: str = None,
    timeout: float = None
) -> dict:
    from openai import OpenAI

    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not configured in .env!")

    has_images = _has_images(prompt)
    target_model = model or (
        settings.nvidia_vision_model if has_images else settings.nvidia_model
    )

    client = OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
        timeout=_resolve_timeout(timeout, has_images),
        max_retries=0,
    )

    request = {
        "model": target_model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = client.chat.completions.create(
            **request, response_format={"type": "json_object"}
        )
    except Exception as e:
        if "response_format" not in str(e).lower():
            raise
        response = client.chat.completions.create(**request)

    content = response.choices[0].message.content if response.choices else "{}"
    return json.loads(_clean_json_string(content or "{}"))


def get_all_nvidia_models() -> list[str]:
    from openai import OpenAI
    if not settings.nvidia_api_key:
        return []
    client = OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)
    return [m.id for m in client.models.list().data]


def get_all_openrouter_models() -> list[str]:
    from openai import OpenAI
    if not settings.openrouter_api_key:
        return []
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
    models_page = client.models.list()
    return [m.id for m in models_page.data]


def get_all_gemini_models() -> list[str]:
    if not settings.gemini_api_key:
        return []
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    return [
        m.name.replace("models/", "")
        for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]


def get_all_groq_models() -> list[str]:
    from groq import Groq
    if not settings.groq_api_key:
        return []
    client = Groq(api_key=settings.groq_api_key)
    models_page = client.models.list()
    return [m.id for m in models_page.data]


PROVIDERS = {
    "gemini": call_gemini,
    "groq_llama": call_groq_llama,
    "mistral": call_mistral,
    "openrouter": call_openrouter,
    "nvidia": call_nvidia,
    "cohere": call_cohere,
    "cloudflare": call_cloudflare,
}

VISION_CAPABLE = {"gemini", "openrouter", "nvidia", "mistral"}  # cohere, cloudflare: text-only

_PROVIDER_KEY_ATTR = {
    "gemini": "gemini_api_key",
    "groq_llama": "groq_api_key",
    "mistral": "mistral_api_key",
    "openrouter": "openrouter_api_key",
    "nvidia": "nvidia_api_key",
    "cohere": "cohere_api_key",
    "cloudflare": "cloudflare_api_key",
}


def provider_has_key(provider: str) -> bool:
    return bool(getattr(settings, _PROVIDER_KEY_ATTR.get(provider, ""), ""))


def supports_vision(provider: str) -> bool:
    return provider in VISION_CAPABLE


def provider_status() -> list[dict]:
    return [
        {
            "provider": name,
            "configured": provider_has_key(name),
            "vision": supports_vision(name),
        }
        for name in PROVIDERS
    ]
