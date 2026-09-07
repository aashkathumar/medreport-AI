import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.config import settings
from app.models.schemas import ExplainedResult, RangeStatus, TestResult, UserProfile
from app.services.llm_providers import (
    PROVIDERS,
    VISION_CAPABLE,
    provider_has_key,
    supports_vision,
)
from app.services.rag_service import retrieve_context
from app.services.reference_db import get_reference_data, grounding_test_id

# ETHICS CONSTRAINT: never ask the model to judge the patient's own reading
# against a range, that's the out-of-scope "clinical decision support" case.
SYSTEM = """You are an expert health literacy assistant helping patients understand what their laboratory tests are and how to support their health, based on NHS UK and NIH MedlinePlus standards.

For each test, write GENERAL, test-level guidance, never referencing the patient's own specific value, whether it is high/low/normal, or any reference range:
1. what_it_measures: Explain clearly what the biomarker does in the body.
2. lifestyle_suggestions: Provide 2-3 evidence-based dietary, hydration, or activity habits that generally support keeping this measurement in a healthy range, grounded in NHS/NIH guidance, phrased as good general practice for anyone, not as a response to this patient's result.
3. gp_question: Formulate a general, constructive question a patient could ask their GP about this test.

STRICT RULES:
- Do NOT mention, infer, or imply the patient's own value, or whether it is elevated, low, normal, or abnormal, anywhere in your answer. This tool never states what a specific result means, only what the test is and how to generally support that measurement.
- Base every explanation ONLY on the facts given in that test's own "Reference context" line. Do NOT pull in facts, causes, or figures that belong to a different test in this batch, even if they seem related.
- Do NOT introduce any medical fact, cause, condition, figure, or reference range that is not present in the reference context for that specific test. If the context does not cover something, omit it rather than inferring or guessing.
- Write in your own words - paraphrase the reference context, do not copy its sentences verbatim.
- Never diagnose, prescribe, or make clinical recommendations beyond general lifestyle guidance.

Respond ONLY with valid JSON."""

EXPLAIN_BATCH_SIZE = 6
RAG_PASSAGES_PER_TEST = 2
RAG_PASSAGE_CHARS = 550


def _build_grounding(test: TestResult) -> Dict[str, Any]:
    blocks: List[str] = []
    sources: List[Dict[str, str]] = []
    source_labels: List[str] = []

    # Used to skip the specimen redirect, so urine glucose matched blood
    # glucose's curated entry and got the wrong clinical meaning explained.
    resolved_id = grounding_test_id(test.test_id, specimen=getattr(test, "specimen", None))

    ref = get_reference_data(resolved_id)
    if ref:
        # ETHICS CONSTRAINT: skip low_means/high_means, only the general
        # description is used, regardless of this test's actual status.
        curated_text = ref.get("plain_english", "")
        if curated_text:
            label = ref.get("source", "NHS UK / NIH MedlinePlus")
            blocks.append(f"({label}) {curated_text}")
            source_labels.append(label)

    try:
        passages = retrieve_context(test.raw_name, k=RAG_PASSAGES_PER_TEST, test_id=resolved_id)
    except Exception:
        passages = []
    has_rag = False

    for passage in passages:
        text = (passage.get("text") or "").strip()
        if not text:
            continue
        label = passage.get("source", "NHS UK / NIH MedlinePlus")
        blocks.append(f"({label}) {text[:RAG_PASSAGE_CHARS]}")
        source_labels.append(label)
        has_rag = True
        url = passage.get("url")
        if url and not any(s.get("url") == url for s in sources):
            sources.append({"label": label, "url": url})

    # Used to default to a non-empty placeholder even with zero real
    # grounding, which invited the LLM to fabricate (e.g. "P2 Peak" as PSA).
    unique_labels = sorted(set(source_labels))
    return {
        "text": "\n".join(blocks),
        "sources": sources,
        "source_label": " / ".join(unique_labels) if unique_labels else "NHS UK / NIH MedlinePlus",
        "grounded": bool(blocks),
        # Whether a real RAG passage backs this, vs. only the curated JSON.
        "has_rag": has_rag,
    }


def _coerce_details(details: dict) -> dict:
    """Normalises one LLM explanation object into the shapes ExplainedResult
    expects, before pydantic ever sees it. Some models return
    lifestyle_suggestions as a string instead of a list, which used to crash
    the whole upload; this coerces what's recoverable instead."""
    if not isinstance(details, dict):
        return {}

    clean: Dict[str, Any] = {}

    for key in ("what_it_measures", "gp_question", "disclaimer"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            clean[key] = value.strip()

    suggestions = details.get("lifestyle_suggestions")
    if isinstance(suggestions, str):
        # A single string: split on newlines/bullets if it looks like a list,
        # otherwise keep it as a one-item list.
        parts = [p.strip(" -*\u2022\t") for p in suggestions.splitlines() if p.strip(" -*\u2022\t")]
        suggestions = parts if len(parts) > 1 else [suggestions.strip()]
    if isinstance(suggestions, list):
        flattened = [str(item).strip() for item in suggestions if str(item).strip()]
        if flattened:
            clean["lifestyle_suggestions"] = flattened

    return clean


# A standalone quantity, not adjacent to a letter/digit, so "T3", "B12", "CO2"
# read as names, not numbers, but "10^9/L" still splits into 10 and 9.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])(?<![A-Za-z]-)\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")

# Tokens too generic to prove topical relatedness either way.
_STOPWORDS = frozenset("""
about above after again against also your yours their there these those them
been being both cannot could does doing done down during each else even ever
every from further have having here hers herself himself into itself just
like made make many more most much must only other over same should some
such than that the then this through under until very what when where which
while will with within without would
""".split())


def _content_tokens(text: str) -> set:
    """Lowercased words of >=4 letters, de-pluralised, minus stopwords."""
    words = re.findall(r"[A-Za-z]{4,}", (text or "").lower())
    return {w[:-1] if w.endswith("s") else w for w in words} - _STOPWORDS


def _numbers_in(text: str) -> set:
    """Every standalone quantity in `text`, normalised to float."""
    out = set()
    for raw in _NUM_RE.findall(text or ""):
        try:
            out.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


# Generic lifestyle units ("150 minutes", "6-8 glasses") that used to get
# rejected as invented clinical figures, gutting a random ~5 tests per run
# non-reproducibly.
_LIFESTYLE_UNIT = (
    r"(?:minute|min|hour|hr|day|daily|week|weekly|month|glass|cup|litre|liter|"
    r"pint|portion|serving|piece|step|night|session|time|unit)"
)

# "150 minutes", "6-8 glasses", "2 to 3 portions", "5 a day", "7-8 hours".
_LIFESTYLE_QTY_RE = re.compile(
    rf"(\d[\d,]*(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d[\d,]*(?:\.\d+)?)\s*{_LIFESTYLE_UNIT}(?:es|s)?\b"
    rf"|(\d[\d,]*(?:\.\d+)?)\s*(?:a\s+|per\s+)?{_LIFESTYLE_UNIT}(?:es|s)?\b",
    re.IGNORECASE,
)


def _lifestyle_numbers_in(text: str) -> set:
    """Quantities in `text` that are attached to a generic lifestyle unit."""
    out = set()
    for match in _LIFESTYLE_QTY_RE.finditer(text or ""):
        for raw in match.groups():
            if not raw:
                continue
            try:
                out.add(float(raw.replace(",", "")))
            except ValueError:
                continue
    return out


def _verify_grounded(
    generated: str,
    context: str,
    allowed_numbers=None,
) -> Tuple[bool, str]:
    """Checks a generated explanation against its source context. Catches
    unsourced figures (invented thresholds/dosages) and topic drift (prose
    about a different test, e.g. "P2 Peak" explained as PSA). A rejection
    falls back to deterministic copy rather than risking misleading text.
    """
    generated = (generated or "").strip()
    if not generated:
        return False, "empty explanation"

    # --- 2. topic drift ---------------------------------------------------
    gen_tokens = _content_tokens(generated)
    ctx_tokens = _content_tokens(context)
    if ctx_tokens and gen_tokens and not (gen_tokens & ctx_tokens):
        return False, (
            "no subject-matter overlap with the reference context "
            "(explanation appears to describe a different test)"
        )

    # --- 1. unsourced figures --------------------------------------------
    allowed = set()
    for value in (allowed_numbers or ()):
        try:
            allowed.add(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            continue
    # A figure printed in the context is by definition sourced.
    allowed |= _numbers_in(context)
    # ...as is a generic lifestyle quantity. See _lifestyle_numbers_in.
    allowed |= _lifestyle_numbers_in(generated)

    unsourced = sorted(
        n for n in _numbers_in(generated)
        if not any(abs(n - a) < 1e-9 for a in allowed)
    )
    if unsourced:
        shown = ", ".join(f"{n:g}" for n in unsourced)
        return False, f"unsourced figure(s) not found in reference context: {shown}"

    return True, "ok"


def _references_other_test(gp_question: str, own_name: str, other_names: List[str]) -> Optional[str]:
    """Catches a gp_question naming a different test from the same batch.
    _verify_grounded missed this since it checks all fields as one blob;
    this checks gp_question alone against every other test's name.
    """
    own_tokens = _content_tokens(own_name)
    q_tokens = _content_tokens(gp_question)
    if not q_tokens:
        return None
    for other in other_names:
        distinguishing = _content_tokens(other) - own_tokens
        if distinguishing and distinguishing <= q_tokens:
            return other
    return None


# Key -> (unix time the circuit re-closes, reason). Module-level and
# lock-guarded because batches run concurrently in a thread pool and share it.
_CIRCUIT: Dict[str, Tuple[float, str]] = {}
_CIRCUIT_LOCK = threading.Lock()


def _is_provider_wide_failure(reason: str) -> bool:
    """Whether a failure reflects the PROVIDER/ACCOUNT rather than one
    model. A 429, a 5xx, or a connection-level failure genuinely affects
    every model on that provider; a plain timeout usually means one model
    was slow, not that the account or endpoint is down, so it should only
    cost that one model its cooldown, not its siblings'."""
    text = (reason or "").lower()
    return any(marker in text for marker in (
        "429", "rate limit", "500", "502", "503", "504",
        "connection", "nameresolutionerror", "max retries exceeded",
    ))


def _circuit_open(provider: str, entry: str) -> bool:
    with _CIRCUIT_LOCK:
        for key in (provider, entry):
            tripped = _CIRCUIT.get(key)
            if not tripped:
                continue
            if time.time() >= tripped[0]:
                del _CIRCUIT[key]      # cooled down; give it another chance
            else:
                return True
        return False


_RETRY_AFTER_RE = re.compile(r"(?:try again in|retry_delay|retry in)\D{0,20}?(\d+(?:\.\d+)?)", re.I)


def _cooldown_for(reason: str) -> float:
    """How long to bench a provider.

    Rate limits usually state when they will clear ("Please try again in
    7.755s"); honouring that is far better than a flat cooldown, which would
    bench the fastest provider in the chain for a full minute over a wait of a
    few seconds. Hard failures (5xx, connection errors) carry no such hint, so
    they get the configured cooldown.
    """
    match = _RETRY_AFTER_RE.search(reason or "")
    if match:
        try:
            # +1s of slack, and never bench longer than the flat cooldown.
            return min(float(match.group(1)) + 1.0, settings.llm_circuit_cooldown_sec)
        except ValueError:
            pass
    return settings.llm_circuit_cooldown_sec


def _trip_circuit(provider: str, entry: str, reason: str) -> None:
    key = provider if _is_provider_wide_failure(reason) else entry
    until = time.time() + _cooldown_for(reason)
    with _CIRCUIT_LOCK:
        _CIRCUIT[key] = (until, reason)
    print(f"Circuit opened for {key} for {until - time.time():.0f}s: {reason[:90]}")


def _reset_circuit(provider: str, entry: str) -> None:
    with _CIRCUIT_LOCK:
        _CIRCUIT.pop(provider, None)
        _CIRCUIT.pop(entry, None)


class LLMUnavailableError(RuntimeError):
    """No provider in the chain could serve this request. A distinct type
    so callers can tell this apart from a bug in our own code, previously
    both surfaced as a bare RuntimeError."""


def _build_chain(
    capability: str,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    chain_offset: int = 0,
    use_reserved: bool = False,
) -> List[str]:
    """Resolves the ordered provider chain for one request, split out of
    call_with_fallback so routing is testable on its own.

    BUG FOUND: every batch tried Mistral first, always, so under concurrency
    all simultaneous batches hit its rate limit at once while other
    providers sat idle. `chain_offset` rotates which provider each batch
    tries first, spreading load without any batch losing its fallback chain.

    use_reserved=True puts the retry-reserved chain first, since those
    entries were never touched by the primary wave and are guaranteed
    fresh; the regular chain still follows if the reserve also fails.
    """
    want_vision = capability == "vision"

    if provider:
        if want_vision and not supports_vision(provider):
            raise LLMUnavailableError(
                f"provider '{provider}' is text-only and cannot serve a vision "
                f"request; vision-capable providers: {sorted(VISION_CAPABLE)}"
            )
        if not provider_has_key(provider):
            raise LLMUnavailableError(f"provider '{provider}' has no API key configured")
        return [f"{provider}:{model_name}" if model_name else provider]

    chain = settings.vision_fallback_chain if want_vision else settings.text_fallback_chain
    if chain_offset and chain:
        offset = chain_offset % len(chain)
        chain = chain[offset:] + chain[:offset]

    reserved = settings.retry_reserved_chain if (use_reserved and not want_vision) else []
    ordered = reserved + chain

    usable, benched = [], []
    for entry in ordered:
        prov = entry.split(":", 1)[0]
        if prov not in PROVIDERS or not provider_has_key(prov):
            continue
        if want_vision and not supports_vision(prov):
            # Guaranteed failure, don't spend a network round-trip on it.
            continue
        if _circuit_open(prov, entry):
            benched.append(entry)       # recently failed; skip while cooling
            continue
        usable.append(entry)

    # Never let the breaker itself starve a request: if it has benched
    # everything, ignore it and try them all rather than failing outright.
    if not usable and benched:
        print("All providers have open circuits, ignoring breaker and retrying.")
        usable = benched

    if not usable:
        raise LLMUnavailableError(
            f"no configured, {capability}-capable provider in chain {chain}"
        )
    return usable


def _ungrounded_result(test: TestResult) -> ExplainedResult:
    """Honest placeholder for a test with no NHS/NIH grounding at all. No
    status/range fields exist on ExplainedResult, and this fallback text
    never references the patient's own reading, only that the test isn't
    covered, same as it would say for any patient."""
    return ExplainedResult(
        test_id=test.test_id,
        raw_name=test.raw_name,
        value=test.value,
        unit=(test.unit or "").strip(),
        source="Not covered by NHS UK / NIH MedlinePlus",
        source_urls=[],
        what_it_measures=f"NHS UK and NIH MedlinePlus reference material for {test.raw_name} was not available, so no explanation has been generated.",
        lifestyle_suggestions=["Discuss this test with your GP or healthcare provider.",
                               "Avoid making dietary or medication changes without professional advice."],
        gp_question=f"Could you explain what my {test.raw_name} test involves?",
        disclaimer="Educational information only. Consult your GP.",
    )


def _is_transient(exc: Exception) -> bool:
    """Worth retrying the same provider, vs. falling straight through. 429/
    5xx/timeouts are transient; a 404, 401, or our own "text-only provider
    got images" error will fail identically every time, so retrying those
    just adds latency to a guaranteed failure."""
    text = str(exc).lower()
    if any(marker in text for marker in ("404", "401", "403", "not found", "cannot process image")):
        return False
    return any(marker in text for marker in
               ("429", "rate limit", "500", "502", "503", "504", "timeout", "timed out",
                "overloaded", "connection", "temporarily"))


def call_with_fallback(
    system: str,
    prompt: Union[str, list],
    max_tokens: int = 2500,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    capability: Optional[str] = "text",
    timeout: Optional[float] = None,
    chain_offset: int = 0,
    use_reserved: bool = False,
) -> dict:
    """Tries each provider in the capability-appropriate chain, retrying
    transient failures on the same provider before moving on.

    Fixed three real bugs here: `capability` was accepted but ignored, so
    vision requests were routed down the text chain and burned a guaranteed
    failure on the first, text-only provider; `supports_vision` was imported
    but never called, so nothing actually stopped that; and the configured
    retry settings were defined but never read, so one transient 429 would
    permanently demote a request to a much slower provider. All three are
    honoured here now.
    """
    has_images = isinstance(prompt, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in prompt
    )
    want_vision = (capability == "vision") or has_images

    # Raises LLMUnavailableError if nothing in the chain can serve this.
    chain = _build_chain(
        "vision" if want_vision else "text", provider, model_name, chain_offset,
        use_reserved=use_reserved,
    )

    attempted: List[str] = []
    last_error: Optional[Exception] = None

    for entry in chain:
        prov, model = entry.split(":", 1) if ":" in entry else (entry, None)
        fn = PROVIDERS[prov]  # _build_chain guarantees this exists and is usable
        attempted.append(entry)
        for attempt in range(max(1, settings.llm_max_retries)):
            try:
                result = fn(system, prompt, max_tokens=max_tokens, model=model, timeout=timeout)
                _reset_circuit(prov, entry)     # healthy again
                return result
            except Exception as e:
                last_error = e
                is_last_attempt = attempt == max(1, settings.llm_max_retries) - 1
                if is_last_attempt or not _is_transient(e):
                    print(f"Provider {entry} failed: {e}")
                    _trip_circuit(prov, entry, str(e)[:120])
                    break
                backoff = settings.llm_retry_backoff_sec * (2 ** attempt)
                print(f"Provider {entry} transient failure (attempt {attempt + 1}), "
                      f"retrying in {backoff:.1f}s: {e}")
                time.sleep(backoff)

    raise LLMUnavailableError(
        f"All LLM providers failed for capability={'vision' if want_vision else 'text'}. "
        f"Attempted: {attempted or 'none (no provider had a usable key)'}. "
        f"Last error: {last_error}"
    )


def _patient_facing_text(details: Dict[str, Any]) -> str:
    """Every generated field a patient actually reads, as one blob.
    lifestyle_suggestions and gp_question are included deliberately, since
    they're the most actionable fields and an invented dosage there is
    exactly the case the verifier exists to catch."""
    parts = [
        details.get("what_it_measures", ""),
        " ".join(details.get("lifestyle_suggestions", []) or []),
        details.get("gp_question", ""),
    ]
    return " ".join(p for p in parts if p)


def _allowed_numbers_for(test: TestResult) -> set:
    """Figures an explanation may legitimately cite without appearing in the
    reference text: digits belonging to the unit string only. The patient's
    own reading and reference bounds are deliberately NOT allowed here, so
    if the model cites them anyway, the verifier below catches it rather
    than this list quietly permitting it. The unit still matters though:
    "10^9/L" parses as the two quantities 10 and 9, so without seeding
    them a correctly grounded platelet or WBC explanation would wrongly
    get rejected for citing "unsourced figures".
    """
    return _numbers_in(test.unit or "")


# Cache of VERIFIED explanations (written only after a result passes
# _verify_grounded + _references_other_test in Phase 3 below).
_WIM_CACHE_FILE = Path(__file__).resolve().parent.parent.parent.parent / "local_data" / "what_it_measures_cache.json"
_WIM_CACHE: Dict[str, str] = {}


def _wim_load() -> None:
    if not _WIM_CACHE_FILE.exists():
        return
    try:
        _WIM_CACHE.update(json.loads(_WIM_CACHE_FILE.read_text()))
    except Exception as e:
        print(f"Could not load what_it_measures cache from {_WIM_CACHE_FILE}: {e}")


def _wim_get(test_id: str) -> Optional[str]:
    return _WIM_CACHE.get(test_id)


_wim_load()


_CACHE_FILE = Path(__file__).resolve().parent.parent.parent.parent / "local_data" / "explanation_cache.json"
_EXPLANATION_CACHE: Dict[tuple, dict] = {}
_CACHE_LOCK = threading.Lock()


def _cache_load() -> None:
    """Reads the persisted cache into memory once, at import time. JSON has
    no tuple keys, so each entry is reassembled into one here."""
    if not _CACHE_FILE.exists():
        return
    try:
        entries = json.loads(_CACHE_FILE.read_text())
        for entry in entries:
            _EXPLANATION_CACHE[tuple(entry["key"])] = entry["value"]
    except Exception as e:
        # A corrupt/partial cache file should degrade to "cold cache", not
        # crash the whole service on startup.
        print(f"Could not load explanation cache from {_CACHE_FILE}: {e}")


def _cache_save() -> None:
    """Full rewrite on every new entry, the simplest way to avoid a
    corrupted file from a partial append at this project's small scale.

    BUG FOUND: on AWS Lambda, this path resolves inside the read-only
    container image, so a cache miss during a live request would raise on
    this write and fail the whole explanation call. Lambda never persisted
    across invocations anyway (each cold start gets a fresh filesystem), so
    a failed write here now degrades to in-memory-only caching for that
    container's lifetime instead of crashing the request. Local/GCP
    deployments have a writable filesystem and are unaffected."""
    try:
        _CACHE_FILE.parent.mkdir(exist_ok=True)
        entries = [{"key": list(k), "value": v} for k, v in _EXPLANATION_CACHE.items()]
        _CACHE_FILE.write_text(json.dumps(entries))
    except OSError as e:
        print(f"Could not persist explanation cache to {_CACHE_FILE} (read-only filesystem?): {e}")


_cache_load()


def _cache_key(test: TestResult) -> tuple:
    """Exact-match signature. Generation is deliberately general to the
    test, never referencing the patient's own reading or status (see the
    SYSTEM prompt / _allowed_numbers_for), so every patient with the same
    test_id and specimen gets identical, correctly grounded text; keying
    on just those two turns this into a per-test cache with a far higher
    hit rate than the old per-result design.

    specimen stays in the key because the same test_id can resolve to
    different reference material depending on specimen (urinary vs blood
    glucose both alias to GLUCOSE), so dropping it would let one specimen's
    cached text leak into the other's results.
    """
    return (test.test_id, getattr(test, "specimen", None))


def _cache_get(test: TestResult) -> Optional[dict]:
    with _CACHE_LOCK:
        cached = _EXPLANATION_CACHE.get(_cache_key(test))
    return dict(cached) if cached is not None else None


def _cache_put(test: TestResult, details: dict) -> None:
    with _CACHE_LOCK:
        _EXPLANATION_CACHE[_cache_key(test)] = dict(details)
        _cache_save()


def _explain_batch(
    llm_batch: List[Tuple[TestResult, Dict[str, Any]]],
    profile: UserProfile,
    provider: Optional[str],
    model_name: Optional[str],
    chain_offset: int = 0,
    use_reserved: bool = False,
) -> Dict[str, dict]:
    """Issues at most one LLM call for one batch and returns test_id ->
    explanation. Cache hits (see _cache_key) never reach the LLM at all,
    and whatever was resolved is returned even if the network call itself
    fails outright, which the caller fills in with fallback copy."""
    results: Dict[str, dict] = {}
    to_call: List[Tuple[TestResult, Dict[str, Any]]] = []
    for test, grounding in llm_batch:
        cached = _cache_get(test)
        if cached is not None:
            results[test.test_id] = cached
        else:
            to_call.append((test, grounding))

    if not to_call:
        return results

    # what_it_measures never depends on the patient's specific value, so any
    # test_id already in the precomputed cache (see _wim_get /
    # scripts/precompute_what_it_measures.py) doesn't need the LLM to write it
    precomputed: Dict[str, str] = {}
    tests_summary = []
    for test, grounding in to_call:
        # ETHICS CONSTRAINT: the patient's reading/status/range is genuinely
        # absent from the prompt, not just instructed against, so there's
        # nothing patient-specific for the model to reference by accident.
        wim = _wim_get(test.test_id)
        note = ""
        if wim:
            precomputed[test.test_id] = wim
            note = ('\n  NOTE: what_it_measures for this test is already known, '
                    'set it to "" in your response for this test_id, do not write it.')
        tests_summary.append(
            f"- Test ID: {test.test_id} | Name: {test.raw_name}\n"
            f"  Reference context: {grounding['text']}{note}"
        )

    joined = "\n".join(tests_summary)
    prompt = f"""Diet: {profile.diet_type}.

Tests to explain:
{joined}

Return valid JSON:
{{
  "explanations": [
    {{
      "test_id": "TEST_ID_HERE",
      "what_it_measures": "2-3 sentences explaining biomarker function.",
      "lifestyle_suggestions": ["General diet/lifestyle habit 1 that supports this measurement", "General habit 2"],
      "gp_question": "General question a patient could ask their doctor about this test.",
      "disclaimer": "Educational information only. Consult your GP."
    }}
  ]
}}"""

    try:
        data = call_with_fallback(
            system=SYSTEM, prompt=prompt, max_tokens=3000,
            provider=provider, model_name=model_name, capability="text",
            chain_offset=chain_offset, use_reserved=use_reserved,
        )
        raw = data.get("explanations", []) if isinstance(data, dict) else []
        for item in raw:
            if isinstance(item, dict):
                tid = item.get("test_id", "")
                if tid in precomputed:
                    # Force-set regardless of what the model returned for this
                    # field, it was told to leave it blank, but trusting that
                    # over just overwriting it would mean one non-compliant
                    item["what_it_measures"] = precomputed[tid]
                results[tid] = item
    except Exception as e:
        print(f"Explanation batch failed across all providers: {e}")
    return results  # cache hits survive even if the network call itself failed


def explain_all_test_results_batched(
    test_results: List[TestResult],
    profile: UserProfile,
    provider: str = None,
    model_name: str = None,
) -> Tuple[List[ExplainedResult], List[str], List[str]]:
    """Batches are issued concurrently rather than in a strict serial loop.
    A 79-test specimen report is 14 batches: ~70s serially on the fastest
    provider, and ~8 minutes once a rate limit pushed it onto a slower one,
    long enough to blow past the frontend's timeout and return nothing.
    Concurrency is bounded by settings.llm_max_concurrency so this doesn't
    just trade the latency for 429s. Output order matches input order
    regardless of completion order.
    """
    if not test_results:
        return [], [], []

    all_explained: List[ExplainedResult] = []
    degraded: List[str] = []
    ungrounded: List[str] = []

    # --- Phase 1: grounding + ungrounded skips (cheap, local, ~15ms/test) ---
    # `slots` keeps final output in input order: each entry is either a
    # finished ExplainedResult or a (batch_index, test, grounding) placeholder.
    slots: List[Any] = []
    batches: List[List[Tuple[TestResult, Dict[str, Any]]]] = []

    for batch_start in range(0, len(test_results), EXPLAIN_BATCH_SIZE):
        batch = test_results[batch_start:batch_start + EXPLAIN_BATCH_SIZE]
        llm_batch: List[Tuple[TestResult, Dict[str, Any]]] = []

        for test in batch:
            grounding = _build_grounding(test)
            if not grounding["grounded"]:
                # No real NHS/NIH material for this test - explaining it
                # anyway would mean the LLM inventing content (as happened
                # with "P2 Peak"/"P3 Peak" being explained as PSA).
                ungrounded.append(test.raw_name)
                slots.append(_ungrounded_result(test))
                continue
            llm_batch.append((test, grounding))
            slots.append((len(batches), test, grounding))

        if llm_batch:
            batches.append(llm_batch)

    if not batches:
        return [r for r in slots if isinstance(r, ExplainedResult)], degraded, ungrounded

    # --- Phase 2: fan the batch calls out concurrently ---------------------
    # chain_offset=i rotates which provider each batch tries FIRST (see
    # _build_chain) so the concurrent batches don't all queue behind the same
    workers = max(1, min(settings.llm_max_concurrency, len(batches)))
    exp_maps: List[Dict[str, dict]] = [{}] * len(batches)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_explain_batch, batch, profile, provider, model_name, i): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            exp_maps[futures[future]] = future.result()

    # --- Phase 2.5: retry whatever came back missing ------------------------
    # A batch call can fail outright (every provider unreachable at that
    # moment) or come back missing individual test_ids under the same
    missing: List[Tuple[TestResult, Dict[str, Any]]] = []
    for slot in slots:
        if isinstance(slot, ExplainedResult):
            continue
        batch_index, test, grounding = slot
        if test.test_id not in exp_maps[batch_index]:
            missing.append((test, grounding))

    retry_results: Dict[str, dict] = {}
    if missing:
        retry_batches = [missing[i:i + EXPLAIN_BATCH_SIZE] for i in range(0, len(missing), EXPLAIN_BATCH_SIZE)]
        retry_workers = max(1, min(settings.llm_max_concurrency, len(retry_batches)))
        with ThreadPoolExecutor(max_workers=retry_workers) as pool:
            futures = {
                pool.submit(
                    _explain_batch, batch, profile, provider, model_name, len(batches) + i,
                    use_reserved=True,
                ): batch
                for i, batch in enumerate(retry_batches)
            }
            for future in as_completed(futures):
                retry_results.update(future.result())

    # --- Phase 2.75: one more bounded retry for whatever is STILL missing ---
    # Phase 2.5 already retries against the reserve chain + primary chain, but
    # if a test is STILL unexplained after that, the only lever left is time
    still_missing = [(test, grounding) for test, grounding in missing if test.test_id not in retry_results]
    if still_missing:
        print(f"{len(still_missing)} test(s) still missing after the first retry, "
              f"waiting {settings.llm_circuit_cooldown_sec:.0f}s for circuit cooldowns "
              f"to clear before one final attempt.")
        time.sleep(settings.llm_circuit_cooldown_sec)
        final_batches = [still_missing[i:i + EXPLAIN_BATCH_SIZE] for i in range(0, len(still_missing), EXPLAIN_BATCH_SIZE)]
        final_workers = max(1, min(settings.llm_max_concurrency, len(final_batches)))
        with ThreadPoolExecutor(max_workers=final_workers) as pool:
            futures = {
                pool.submit(
                    _explain_batch, batch, profile, provider, model_name, len(batches) + i,
                ): batch
                for i, batch in enumerate(final_batches)
            }
            for future in as_completed(futures):
                retry_results.update(future.result())

    # --- Phase 3: reassemble in the original test order --------------------
    for slot in slots:
        if isinstance(slot, ExplainedResult):
            all_explained.append(slot)
            continue

        batch_index, test, grounding = slot
        raw_details = exp_maps[batch_index].get(test.test_id) or retry_results.get(test.test_id, {})
        details = _coerce_details(raw_details)

        # --- post-hoc grounding check ------------------------------------
        # The system prompt ASKS the model to stay inside the reference
        # context;
        if details:
            ok, reason = _verify_grounded(
                _patient_facing_text(details),
                grounding["text"],
                allowed_numbers=_allowed_numbers_for(test),
            )
            if not ok:
                # An ungrounded fact could be anywhere in the text (a wrong
                # threshold, an invented figure), can't trust any of it,
                # so the whole explanation is discarded.
                print(f"Rejected ungrounded explanation for {test.test_id}: {reason}")
                details = {}
            else:
                other_names = [t.raw_name for t, _ in batches[batch_index] if t.test_id != test.test_id]
                contaminated = _references_other_test(details.get("gp_question", ""), test.raw_name, other_names)
                if contaminated:
                    # BUG FOUND: this curated-JSON lookup used test.test_id
                    # directly, before any specimen redirect. "Urinary
                    # Glucose" resolves (via ALIAS_MAP) to the same test_id as
                    print(f"Replaced contaminated gp_question for {test.test_id}: "
                          f"referenced {contaminated} instead of {test.raw_name}")
                    details["gp_question"] = ExplainedResult.model_fields["gp_question"].default
                # Only ever cache a result that passed the grounding check --
                # the cache must never be able to serve an explanation that
                # would have been rejected.
                _cache_put(test, details)

        if not details:
            # The batch failed, or the model omitted this test_id from its
            # array.
            degraded.append(test.test_id)

        unit_display = (test.unit or "").strip()

        # BUG FOUND: source was cited even when the shown text was fallback
        # copy, not the verified explanation. Now only cited on a real pass.
        source_label = grounding["source_label"] if details else "Not independently verified"
        source_urls = [s["url"] for s in grounding["sources"]] if details else []

        # ETHICS CONSTRAINT: no status/range fields on this type at all, so
        # there's no "flag as unknown" step needed, there was never a
        # normal/high/low claim being made in the first place.
        all_explained.append(
            ExplainedResult(
                test_id=test.test_id,
                raw_name=test.raw_name,
                value=test.value,
                unit=unit_display,
                source=source_label,
                source_urls=source_urls,
                what_it_measures=details.get("what_it_measures", f"Evaluates what {test.raw_name} measures in the body."),
                lifestyle_suggestions=details.get("lifestyle_suggestions", ["Maintain balanced nutrition and regular physical activity."]),
                gp_question=details.get("gp_question", f"Could you explain what my {test.raw_name} test involves?"),
                disclaimer=details.get("disclaimer", "Educational information only. Consult your GP."),
            )
        )

    return all_explained, degraded, ungrounded


def generate_summary(
    explained_results: List[ExplainedResult],
    profile: UserProfile,
    provider: str = None,
    model_name: str = None,
) -> dict:
    """Rewritten under ethics review, since this used to build the summary
    around flagged "abnormal" results with each test's specific reading
    quoted, exactly the per-result interpretation now out of scope.

    Deliberately not an LLM call any more: once per-reading citation is
    off the table, there's nothing left to summarize that isn't already
    covered by each test's own verified fields, so this reuses that
    content deterministically instead of reintroducing fabrication risk
    for no benefit. No LLM call also means no rate-limit exposure here.
    """
    count = len(explained_results)
    plural = "test" if count == 1 else "tests"
    return {
        "summary_degraded": False,
        "overall_summary": (
            f"This report explains {count} {plural} from your results. Each explanation "
            "below describes what the test measures and general, NHS/NIH-grounded habits "
            "that support that area of health. It does not interpret your specific "
            "values, so please review your actual results with your GP."
        ),
        "top_gp_topics": [r.gp_question for r in explained_results[:3] if r.gp_question] or ["Routine follow-up"],
        "top_lifestyle_change": next(
            (s for r in explained_results for s in (r.lifestyle_suggestions or [])),
            "Focus on balanced nutrition, adequate hydration, and regular activity.",
        ),
        "closing_message": "Always discuss your specific results with your GP or healthcare provider.",
    }