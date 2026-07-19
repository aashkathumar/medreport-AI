"""
One function per LLM provider, all with the same signature, so
llm_service.py can call whichever one is configured without caring
which provider it actually is.

Each function takes (system, prompt, max_tokens) and returns a parsed
dict -- callers rely on the model returning valid JSON (the prompts in
llm_service.py explicitly instruct this).

Requires the relevant API key to be set in .env for whichever
provider(s) you actually use -- see config.py.

MODEL ID STABILITY NOTE:
Groq deprecates specific model IDs fairly often (e.g. llama-3.3-70b-versatile
and mistral-saba-24b were both retired mid-2026, shortly after being added
to an earlier version of this file). To reduce how often this file needs
editing:
  - The Groq model ID is read from GROQ_MODEL in .env (see config.py),
    not hardcoded, so a deprecation only needs an .env change.
  - Mistral is called directly via Mistral AI's own API (not through Groq),
    using their "-latest" alias (e.g. "mistral-small-latest"), which Mistral
    automatically repoints to their current model -- no manual updates needed.
Before relying on any Groq model ID, it's worth checking the live list:
    curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
"""
import json
import requests
from app.config import settings


def call_claude(system: str, prompt: str, max_tokens: int) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(msg.content[0].text)


def call_gemini(system: str, prompt: str, max_tokens: int, model: str = "gemini-2.5-flash") -> dict:
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    client = genai.GenerativeModel(model, system_instruction=system)
    response = client.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        },
    )
    return json.loads(response.text)


def call_groq_llama(system: str, prompt: str, max_tokens: int) -> dict:
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    msg = client.chat.completions.create(
        model=settings.groq_model,  # set GROQ_MODEL in .env -- see note above
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(msg.choices[0].message.content)


def call_mistral(system: str, prompt: str, max_tokens: int) -> dict:
    """
    Calls Mistral AI's own API directly (console.mistral.ai), NOT via Groq --
    Groq no longer reliably hosts a Mistral model. Uses the "mistral-small-latest"
    alias, which Mistral repoints to their current small model automatically.
    """
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "mistral-small-latest",
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


# Registry -- llm_service.py looks up the active provider here by name.
# Imports inside each function (rather than at module level) mean you
# only need the SDK installed for the provider(s) you actually use.
PROVIDERS = {
    "claude": call_claude,
    "gemini": call_gemini,
    "groq_llama": call_groq_llama,
    "mistral": call_mistral,
}
