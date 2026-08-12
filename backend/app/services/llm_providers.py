import base64
import json
import re
import requests
from typing import Union, List, Dict, Any
from app.config import settings


def _clean_json_string(text: str) -> str:
    """Helper to remove markdown backticks (```json ... ```) if an LLM includes them."""
    if not text:
        return "{}"
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def call_gemini(system: str, prompt: Union[str, list], max_tokens: int = 2500, model: str = None) -> dict:
    """
    Calls Google Gemini API natively. Accepts text strings or vision content arrays.
    """
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    
    target_model = model or "gemini-2.0-flash"
    client = genai.GenerativeModel(target_model, system_instruction=system)

    # CHANGED: this used to flatten `prompt` down to ONLY its text blocks,
    # silently discarding every image_url block. Every "vision" extraction
    # call therefore sent Gemini nothing but an instruction like "extract
    # lab results from these images" - with no images attached - and
    # Gemini would still return confident, well-formed JSON by filling in
    # a plausible-looking lab report from training data instead of
    # erroring. That is the actual mechanism behind the wholesale
    # hallucination seen in production (see pdf_parser.py docstring: ~100+
    # fabricated tests for a 40-test report). Now builds proper multimodal
    # parts so Gemini genuinely sees the page images.
    contents = prompt
    if isinstance(prompt, list):
        parts = []
        for block in prompt:
            if block.get("type") == "text":
                parts.append(block["text"])
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
    )
    return json.loads(_clean_json_string(response.text))


def call_groq_llama(system: str, prompt: Union[str, list], max_tokens: int = 4000, model: str = None) -> dict:
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    target_model = model or settings.groq_model

    # CHANGED: this model is text-only. It used to silently flatten vision
    # payloads to just their text blocks (dropping every image), so it
    # "succeeded" on vision-extraction calls without ever seeing a page -
    # writing a plausible lab report from training data instead of failing
    # loudly. Now it refuses outright, so call_with_fallback moves on to a
    # provider that can actually see the images rather than masking the gap.
    if isinstance(prompt, list) and any(b.get("type") == "image_url" for b in prompt):
        raise ValueError("call_groq_llama is text-only and cannot process image content blocks.")

    user_content = prompt
    if isinstance(prompt, list):
        user_content = " ".join([b["text"] for b in prompt if b.get("type") == "text"])

    msg = client.chat.completions.create(
        model=target_model,
        max_tokens=max_tokens,
        temperature=0.0,  # Strict deterministic output prevents JSON key corruption
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    return json.loads(_clean_json_string(msg.choices[0].message.content))
    

def call_mistral(system: str, prompt: Union[str, list], max_tokens: int = 2500, model: str = None) -> dict:
    target_model = model or "mistral-small-latest"

    # CHANGED: same issue as call_groq_llama above - this text model used
    # to silently drop image blocks and hallucinate a confident answer with
    # no page content behind it. Fail loudly instead.
    if isinstance(prompt, list) and any(b.get("type") == "image_url" for b in prompt):
        raise ValueError("call_mistral is text-only and cannot process image content blocks.")

    user_content = prompt
    if isinstance(prompt, list):
        user_content = " ".join([b["text"] for b in prompt if b.get("type") == "text"])

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
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(_clean_json_string(content))


def call_openrouter(system: str, prompt: Union[str, list], max_tokens: int = 2500, model: str = None) -> dict:
    """
    Calls OpenRouter API using OpenAI SDK. Supports both text prompts AND multimodal vision payloads.
    """
    from openai import OpenAI
    
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured in .env!")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
    
    # Use specified model or fallback to openrouter_model
    # NOTE: this is the only provider that forwards image content blocks
    # as-is, so it's the fallback of last resort for vision extraction -
    # but only if `openrouter_model` (or the model passed in) is actually a
    # vision-capable model on OpenRouter. If it's a text-only model, the
    # same silent-drop-and-hallucinate failure mode as the other providers
    # can happen one level down at OpenRouter's own routing, invisibly to
    # this code. Worth confirming the configured model explicitly.
    target_model = model or getattr(settings, "openrouter_model", "openrouter/free")
    
    # OpenAI format natively accepts string OR list of content blocks (for vision data URLs)
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
    content = response.choices[0].message.content
    return json.loads(_clean_json_string(content))


def get_all_openrouter_models() -> list[str]:
    """Dynamically fetches all live model IDs on OpenRouter via single API key."""
    from openai import OpenAI
    if not settings.openrouter_api_key:
        return []
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
    models_page = client.models.list()
    return [m.id for m in models_page.data]


def get_all_groq_models() -> list[str]:
    """Dynamically fetches all live model IDs on Groq."""
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
}