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

SYSTEM = """You are an expert health literacy assistant helping patients understand laboratory test results based on NHS UK and NIH MedlinePlus standards.

For each test:
1. what_it_measures: Explain clearly what the biomarker does in the body.
2. what_your_result_means: Explain what the patient's specific value and status (normal, high, or low) indicate relative to reference ranges (e.g., elevated HbA1c reflects high blood sugar). Always conclude this section with a gentle reminder to review findings with their GP.
3. lifestyle_suggestions: Provide 2-3 evidence-based dietary, hydration, or activity adjustments grounded in NHS/NIH guidance.
4. gp_question: Formulate a constructive question for their doctor.

STRICT RULES:
- Base every explanation ONLY on the facts given in that test's own "Reference context" line. Do NOT pull in facts, causes, or figures that belong to a different test in this batch, even if they seem related.
- Do NOT introduce any medical fact, cause, condition, or figure that is not present in the reference context for that specific test. If the context does not cover something, omit it rather than inferring or guessing.
- Write in your own words - paraphrase the reference context, do not copy its sentences verbatim.
- Never diagnose, prescribe, or make clinical recommendations beyond general lifestyle guidance.

Respond ONLY with valid JSON."""

EXPLAIN_BATCH_SIZE = 6
RAG_PASSAGES_PER_TEST = 2
RAG_PASSAGE_CHARS = 550


def _build_grounding(test: TestResult) -> Dict[str, Any]:
    status = test.status.value if hasattr(test.status, "value") else str(test.status)
    blocks: List[str] = []
    sources: List[Dict[str, str]] = []
    source_labels: List[str] = []

    # BUG FOUND: this curated-JSON lookup used test.test_id directly, before
    # any specimen redirect. "Urinary Glucose" resolves (via ALIAS_MAP) to
    # the same test_id as blood glucose - "GLUCOSE" - and blood_tests.json
    # has a curated entry for it, so this branch matched FIRST and returned
    # "Blood glucose measures the amount of sugar in your blood..." as the
    # explanation for a urine dipstick result, where the clinical meaning is
    # entirely different (presence itself is abnormal, not a concentration
    # threshold). The RAG retrieval below already applied the specimen
    # redirect; this curated-lookup step - checked first and never
    # overridden - did not, so the fix had to close this path too.
    resolved_id = grounding_test_id(test.test_id, specimen=getattr(test, "specimen", None))

    ref = get_reference_data(resolved_id)
    if ref:
        curated = [ref.get("plain_english", "")]
        if status == "low" and ref.get("low_means"):
            curated.append(f"Low range context: {ref['low_means']}")
        elif status == "high" and ref.get("high_means"):
            curated.append(f"High range context: {ref['high_means']}")
        curated_text = " ".join(p for p in curated if p)
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

    # BUG FOUND: this used to default to a non-empty placeholder string
    # ("General NHS/NIH medical reference guidelines.") when nothing real was
    # found, which made every test - including ones with zero relevant
    # grounding (e.g. "P2 Peak", an HPLC electrophoresis peak with no
    # NHS/NIH coverage at all) - look grounded to the caller. The LLM was
    # then asked to explain it anyway with no real facts to draw on, and
    # produced a confident but entirely wrong explanation (borrowed from an
    # unrelated test - PSA - in the same batch). `grounded` now reports
    # honestly whether real reference material was found, so the caller can
    # skip the LLM call instead of inviting fabrication.
    unique_labels = sorted(set(source_labels))
    return {
        "text": "\n".join(blocks),
        "sources": sources,
        "source_label": " / ".join(unique_labels) if unique_labels else "NHS UK / NIH MedlinePlus",
        "grounded": bool(blocks),
        # Whether any RAG passage (as opposed to only the small curated JSON)
        # backs this test -- lets callers tell corpus-grounded tests apart
        # from ones resting solely on the 21-entry static table.
        "has_rag": has_rag,
    }


def _coerce_details(details: dict) -> dict:
    """Normalises one LLM explanation object into the shapes ExplainedResult
    requires, before pydantic ever sees it.

    BUG FOUND while benchmarking alternative OpenRouter models: the schema
    declares `lifestyle_suggestions: List[str]`, but models are only *asked*
    for a JSON array -- several free models (liquid/lfm-2.5-2.6b reproducibly)
    return a single STRING there instead. ExplainedResult() was constructed
    unguarded, so pydantic raised ValidationError, which nothing caught: the
    exception escaped explain_all_test_results_batched() and failed the WHOLE
    upload with a 500, discarding every other successfully explained test.
    One badly-shaped field from one model = no report at all.

    Rather than crash on a cosmetic shape difference, coerce what is
    recoverable and drop what is not, so a malformed field degrades to the
    deterministic fallback for that field alone.
    """
    if not isinstance(details, dict):
        return {}

    clean: Dict[str, Any] = {}

    for key in ("what_it_measures", "what_your_result_means", "gp_question", "disclaimer"):
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


# A standalone quantity: a run of digits (optionally comma-grouped, optionally
# decimal) NOT adjacent to a letter or digit. The boundary conditions are the
# whole point -- they keep analyte names out of the figure check. "T3", "T4",
# "B12", "P24" and "CO2" are NAMES, not numbers, and an earlier string-based
# check flagged them as invented figures and threw away correct explanations.
# Conversely "10^9/L" DOES read as the two quantities 10 and 9, because "^"
# and "/" are non-alphanumeric -- so callers must seed allowed_numbers with
# the unit's own digits (see explain_all_test_results_batched).
#
# The second lookbehind covers hyphenated NAMES: "omega-3", "omega-6",
# "COVID-19". A hyphen is non-alphanumeric, so the first lookbehind alone let
# "omega-3" read as the quantity 3, and correct dietary advice ("increase your
# intake of omega-3 fatty acids") was rejected as an unsourced figure. A digit
# after "letter-" is part of a name; after "digit-" it is a real range bound,
# so "6-8 glasses" and "10-15 minutes" still parse as quantities.
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


# Units that make a figure a GENERIC LIFESTYLE quantity rather than a clinical
# one: "150 minutes of activity a week", "6-8 glasses of water", "14 units of
# alcohol", "5 a day". These are public-health advice that applies to everyone,
# carry no per-patient clinical meaning, and are exactly what the SYSTEM prompt
# above ASKS the model to produce ("2-3 evidence-based dietary, hydration, or
# activity adjustments").
#
# BUG FOUND: the unsourced-figure check below treated these identically to an
# invented clinical threshold, so any explanation that took the prompt's
# instruction literally was rejected wholesale and replaced with content-free
# boilerplate ("Evaluates Cholesterol levels in the body."). Because the model
# phrases advice differently on each run, a DIFFERENT random ~5 tests were
# gutted every time -- confirmed by diffing two runs over the same PDF, where
# HbA1c/Urine Glucose recovered while Cholesterol/LDL-HDL Ratio/HBsAg broke.
# Non-reproducible output is disqualifying for an evaluated system.
#
# Clinical units are deliberately ABSENT from this list -- mg, g, dL, mmol,
# ng, IU, U/L and bare unit-less thresholds stay fully checked, so the
# dangerous class this guard was built for ("below 7.3 g/dL requires immediate
# transfusion", "325 mg ferrous sulphate three times daily") is still caught.
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
    """Checks one generated explanation against the reference context it was
    supposed to be built from. Returns (ok, reason).

    This is the post-hoc half of the anti-fabrication design: the system
    prompt ASKS the model to stay inside the context, and this verifies that
    it did. Two failure modes are caught, both seen in production:

      1. UNSOURCED FIGURES -- a threshold, cutoff or dosage that appears
         nowhere in the context ("below 7.3 g/dL requires immediate
         transfusion", "325 mg ferrous sulphate three times daily"). This is
         the dangerous class: specific, actionable, and invented. Checked
         exactly rather than statistically -- every standalone quantity must
         be traceable to the context, to the patient's own result/range, or
         to the unit string.

      2. TOPIC DRIFT -- prose about a different biomarker entirely, which is
         how "P2 Peak" came to be explained as PSA. Caught as a total absence
         of shared subject-matter vocabulary with the context.

    A coarse guard by design: it is meant to catch confident invention, not to
    grade prose. Anything it rejects falls back to deterministic copy, so a
    false positive costs wording, while a false negative could mislead a
    patient about their own health.
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
    """Catches a gp_question that names a DIFFERENT test from this batch.

    BUG FOUND (Sterling Accuris report, Nitrite): Nitrite's gp_question read
    "Could you explain the significance of the pus cells and epithelial cells
    in my urinalysis results?" -- correctly cited, but about two other tests
    later in the same batch, not Nitrite. _verify_grounded did not catch this
    because it checks the FOUR generated fields as one concatenated blob:
    Nitrite's other three fields were genuinely on-topic, so the blob as a
    whole shared plenty of vocabulary with the nitrite reference context, and
    "pus cells"/"epithelial cells" introduced no unsourced number either. A
    single contaminated field rode through on the correctness of the rest.

    This checks gp_question in isolation against every OTHER test's name in
    the batch: if a batch-mate's distinguishing words (its name minus
    whatever words it shares with the test actually being explained) all
    appear in the gp_question, the field is describing that other test.
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
#
# BUG FOUND: this used to be keyed on the bare provider name for EVERY kind
# of failure, so one bad/slow MODEL benched every sibling model on that
# provider for a full cooldown too. On a provider with many chain entries
# (Cloudflare: 15, Mistral: 6, Cohere: 11+3) that took a lot of healthy
# capacity offline over one flaky model -- the likely cause of an 85-test
# report taking 885s instead of the usual tens of seconds. Now a key can be
# EITHER a bare provider name (a provider-wide trip) or a full "provider:
# model" entry (a single-model trip) -- see _is_provider_wide_failure for
# which failures get which scope. _circuit_open checks both so a
# provider-wide trip still blocks every model under it, matching the
# original behaviour for the case the breaker was actually built around
# (Mistral's documented 500-then-522 outage).
_CIRCUIT: Dict[str, Tuple[float, str]] = {}
_CIRCUIT_LOCK = threading.Lock()


def _is_provider_wide_failure(reason: str) -> bool:
    """Whether a failure reflects the PROVIDER/ACCOUNT rather than one model.

    A 429 (shared account rate limit -- e.g. Groq's per-account TPM, or
    Cohere's shared 20/min trial pool across all its models), a 5xx
    (server-side outage), or a connection-level failure (DNS/refused -- the
    endpoint itself is unreachable, regardless of which model was asked
    for) genuinely affect every model on that provider, so those still
    bench the whole provider. A plain timeout is treated as model-specific:
    it usually means that one model was slow to respond, not that the
    account or endpoint is down, so it should only cost that one model its
    cooldown, not its siblings'.
    """
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
    """No provider in the chain could serve this request.

    A distinct type so callers can tell "every provider failed" apart from a
    bug in our own code -- previously both surfaced as a bare RuntimeError.
    """


def _build_chain(
    capability: str,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    chain_offset: int = 0,
    use_reserved: bool = False,
) -> List[str]:
    """Resolves the ordered provider chain for one request.

    Split out of call_with_fallback so the routing rules are testable on their
    own, and so an explicitly-requested provider that CANNOT serve the
    capability fails loudly here instead of silently returning an empty chain
    and looking like a generic "all providers failed".

    BUG FOUND: every batch used the SAME chain order (mistral first, always).
    Under concurrency this meant every one of the (up to 8) simultaneously-
    fired batches hit Mistral first, all at once -- the free-tier key's rate
    limit got hit by the concurrency itself, not by total request volume,
    while Groq/OpenRouter/Gemini sat idle until Mistral failed. `chain_offset`
    rotates which provider each batch tries FIRST (batch 0 starts at index 0,
    batch 1 at index 1, ...), spreading first-attempt load evenly across all
    configured providers while every batch still has the full chain as
    fallback -- no batch loses resilience, they just don't all queue behind
    the same provider at once.

    use_reserved=True (retry calls only) puts settings.retry_reserved_chain
    FIRST, ahead of the regular chain. Those entries are never touched by the
    primary wave, so they can't have been circuit-tripped or quota-exhausted
    by some unrelated batch before the retry gets to them -- they're
    guaranteed fresh. The regular chain still follows as a fallback if the
    reserve also fails, so a retry is never worse off than before this
    existed, only potentially better.
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
            # Guaranteed failure -- don't spend a network round-trip on it.
            continue
        if _circuit_open(prov, entry):
            benched.append(entry)       # recently failed; skip while cooling
            continue
        usable.append(entry)

    # Never let the breaker itself starve a request: if it has benched
    # everything, ignore it and try them all rather than failing outright.
    if not usable and benched:
        print("All providers have open circuits -- ignoring breaker and retrying.")
        usable = benched

    if not usable:
        raise LLMUnavailableError(
            f"no configured, {capability}-capable provider in chain {chain}"
        )
    return usable


def _ungrounded_result(test: TestResult) -> ExplainedResult:
    """Honest placeholder for a test with no NHS/NIH grounding at all.

    BUG FOUND (medreport_ai.pdf, "Amorphous Material"): this used to pass
    through test.status verbatim -- whatever RangeStatus the EXTRACTOR
    assigned before grounding was ever checked (e.g. a qualitative "Absent"
    value defaulting to NORMAL). That produced a green "Within normal
    range" badge sitting directly above text saying the test isn't covered
    by NHS/NIH at all and no explanation was generated -- a genuinely
    misleading pairing, even though the explanation text itself never
    fabricated anything. With no grounding whatsoever, there's no basis to
    claim normal/high/low, so this is always UNKNOWN here regardless of
    what the extractor guessed.
    """
    unit = (test.unit or "").strip()
    return ExplainedResult(
        test_id=test.test_id,
        raw_name=test.raw_name,
        value=test.value,
        unit=unit,
        status=RangeStatus.UNKNOWN,
        normal_range_min=getattr(test, "normal_range_min", None),
        normal_range_max=getattr(test, "normal_range_max", None),
        source="Not covered by NHS UK / NIH MedlinePlus",
        source_urls=[],
        what_it_measures=f"NHS UK and NIH MedlinePlus reference material for {test.raw_name} was not available, so no explanation has been generated.",
        what_your_result_means=f"Your result is {test.value} {unit}. This tool only reports what NHS UK and NIH MedlinePlus sources state, and they do not cover this test, so please discuss this result with your GP.",
        lifestyle_suggestions=["Discuss this specific result with your GP or healthcare provider.",
                               "Avoid making dietary or medication changes without professional advice."],
        gp_question=f"Could you explain what my {test.raw_name} result means?",
        disclaimer="Educational information only. Consult your GP.",
    )


def _is_transient(exc: Exception) -> bool:
    """Worth retrying the SAME provider, vs. falling straight through.

    429 / 5xx / timeouts are transient. A 404 (retired model), a 401 (bad key)
    or our own "text-only provider got images" ValueError will fail identically
    every time -- retrying those just adds latency to a guaranteed failure.
    """
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

    CHANGED (three real bugs):
      1. `capability` was accepted and then IGNORED -- this always read
         settings.text_fallback_chain, so VISION_FALLBACK_CHAIN was dead
         config that never took effect. Vision extraction was routed down the
         text chain, whose first entry (groq_llama) is text-only and raises on
         image payloads, burning a guaranteed-failed attempt on every single
         vision batch.
      2. `supports_vision` was imported and never called, so nothing actually
         stopped an image payload being handed to a text-only provider. Now a
         vision request skips text-only providers outright.
      3. settings.llm_max_retries / llm_retry_backoff_sec were defined in
         config and referenced nowhere -- there was no retry at all. One
         transient 429 from groq (~5s/batch) permanently demoted the request
         to gemini (~33s/batch). They are honoured here now.
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

    lifestyle_suggestions and gp_question are included deliberately: they are
    the most ACTIONABLE fields, and "take 325 mg ferrous sulphate three times
    daily" is exactly the invented-dosage case the verifier exists to stop.
    """
    parts = [
        details.get("what_it_measures", ""),
        details.get("what_your_result_means", ""),
        " ".join(details.get("lifestyle_suggestions", []) or []),
        details.get("gp_question", ""),
    ]
    return " ".join(p for p in parts if p)


def _allowed_numbers_for(test: TestResult) -> set:
    """Figures an explanation may legitimately cite without them appearing in
    the reference text: the patient's own result, their reference bounds, and
    any digits belonging to the unit string.

    The unit matters -- "10^9/L" parses as the two quantities 10 and 9, so
    without seeding them a perfectly grounded platelet or WBC explanation gets
    rejected for citing "unsourced figures".
    """
    allowed = set()
    for value in (test.value, getattr(test, "normal_range_min", None),
                  getattr(test, "normal_range_max", None)):
        try:
            allowed.add(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            continue
    allowed |= _numbers_in(test.unit or "")
    return allowed


# Cache of VERIFIED explanations (written only after a result passes
# _verify_grounded + _references_other_test in Phase 3 below). Keyed on the
# exact result signature, not just the test -- see _cache_key. Repeat
# uploads of the same PDF and different patients who happen to share a value
# both skip RAG retrieval AND the LLM entirely on a hit -- the single
# biggest lever on rate-limit exposure, since a cache hit makes zero network
# calls and so can never be rate-limited.
#
# PERSISTED TO DISK (2026-08-27): was process-lifetime only, so every
# `uvicorn --reload` restart during a config-tuning session (many, today)
# silently threw away everything accumulated. Backed by a local JSON file so
# it survives a restart -- holds no patient identity (see _cache_key: just a
# test/value/unit/status/specimen signature, the same for any patient with
# that same result), so this doesn't reopen the report-storage question from
# earlier -- there is no patient represented in this data at all.
# Precomputed `what_it_measures` cache -- keyed on canonical test_id ALONE,
# not the full (test_id, value, unit, status, specimen) signature _cache_key
# uses. Unlike what_your_result_means/lifestyle_suggestions/gp_question,
# what_it_measures never depends on the patient's specific value -- "what is
# hemoglobin" has one correct answer for every report that mentions HGB, so
# it's generated ONCE per test_id (see scripts/precompute_what_it_measures.py)
# and reused forever, instead of regenerated per report/value like the rest
# of the exact-match cache above. Built after the RAG corpus expansion
# (2026-08-29, 76 -> 408 test_ids) made this worth doing for a genuinely
# large, stable set of tests.
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
    no tuple keys, so each entry is stored as {"key": [...], "value": {...}}
    and reassembled into a tuple key here."""
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
    """Full rewrite on every new entry -- simplest way to avoid a corrupted
    file from a partial append, and the expected size (low thousands of
    entries at most, for a dissertation-scale project) makes that cheap."""
    _CACHE_FILE.parent.mkdir(exist_ok=True)
    entries = [{"key": list(k), "value": v} for k, v in _EXPLANATION_CACHE.items()]
    _CACHE_FILE.write_text(json.dumps(entries))


_cache_load()


def _cache_key(test: TestResult) -> tuple:
    """Exact-match signature. Deliberately includes the literal value, not
    just status -- what_your_result_means quotes the specific number, so a
    coarser (test_id, status) key could surface one patient's text quoting a
    different patient's figure. Two results only ever share a cache entry if
    every one of these fields is identical.
    """
    status_val = test.status.value if hasattr(test.status, "value") else str(test.status)
    return (
        test.test_id, str(test.value), (test.unit or "").strip(),
        status_val, getattr(test, "specimen", None),
    )


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
    """Issues at most ONE LLM call for one batch and returns test_id -> explanation.

    Tests with an exact-match cache hit (see _cache_key) never reach the LLM
    at all. If every test in the batch is a cache hit, no network call is
    made. Returns whatever was resolved (cache hits included) even if the
    network call itself fails entirely, which the caller treats as
    "degraded" for whatever is still missing and fills with deterministic
    fallback copy.
    """
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

    # what_it_measures never depends on the patient's specific value, so
    # any test_id already in the precomputed cache (see _wim_get /
    # scripts/precompute_what_it_measures.py) doesn't need the LLM to write
    # it at all -- the prompt tells it to skip that field for these tests
    # (smaller output, one fewer thing that can go wrong per test), and the
    # cached text is force-set onto the response afterward regardless of
    # what the model did or didn't produce for it.
    precomputed: Dict[str, str] = {}
    tests_summary = []
    for test, grounding in to_call:
        status_val = test.status.value if hasattr(test.status, "value") else str(test.status)
        min_val = getattr(test, "normal_range_min", None)
        max_val = getattr(test, "normal_range_max", None)
        range_str = f"{min_val}-{max_val}" if (min_val is not None or max_val is not None) else "standard"
        wim = _wim_get(test.test_id)
        note = ""
        if wim:
            precomputed[test.test_id] = wim
            note = ('\n  NOTE: what_it_measures for this test is already known -- '
                    'set it to "" in your response for this test_id, do not write it.')
        tests_summary.append(
            f"- Test ID: {test.test_id} | Name: {test.raw_name} | Value: {test.value} {test.unit} | "
            f"Range: {range_str} {test.unit} | Status: {status_val}\n"
            f"  Reference context: {grounding['text']}{note}"
        )

    joined = "\n".join(tests_summary)
    prompt = f"""Patient: {profile.age}yo {profile.sex}, {profile.diet_type} diet.

Tests to explain:
{joined}

Return valid JSON:
{{
  "explanations": [
    {{
      "test_id": "TEST_ID_HERE",
      "what_it_measures": "2-3 sentences explaining biomarker function.",
      "what_your_result_means": "2-3 sentences explaining this specific value, concluding with a reminder to review with GP.",
      "lifestyle_suggestions": ["Actionable diet/lifestyle advice 1", "Actionable advice 2"],
      "gp_question": "Focused question for doctor.",
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
                    # Force-set regardless of what the model returned for
                    # this field -- it was told to leave it blank, but
                    # trusting that over just overwriting it would mean one
                    # non-compliant model call away from silently losing the
                    # field, or a model writing something ungrounded here.
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
    """CHANGED: batches are now issued CONCURRENTLY instead of one at a time.

    They are completely independent of one another, but were run in a strict
    serial loop. A 79-test report (real Tier-1 output from a 19-page pathology
    PDF) is 14 batches: ~70s serially on the fastest provider, and ~8 MINUTES
    once a rate-limit pushed it onto a slower one -- long enough to look like
    an infinite hang and to blow past the frontend's timeout, so the user got
    no result at all. Concurrency is bounded by settings.llm_max_concurrency
    so we don't simply trade the latency for 429s.

    Ordering of the returned list is preserved regardless of completion order.
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
                # with "P2 Peak"/"P3 Peak" being explained as PSA). Skip the
                # LLM entirely and say so honestly instead.
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
    # provider's rate limit at once.
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
    # pressure. This retry fires AFTER every first-wave batch has finished,
    # so whatever concurrent load caused the failure has largely cleared --
    # it's a second attempt timed for when contention has eased, not a blind
    # immediate retry that would likely hit the same rate limit again. This
    # is what the 13/6/21-degraded pattern (same PDF, three uploads) was
    # missing: one attempt, no second chance.
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
    # Phase 2.5 already retries against the reserve chain + primary chain,
    # but if a test is STILL unexplained after that, the only lever left is
    # time -- a circuit-tripped provider only reopens after
    # settings.llm_circuit_cooldown_sec, so a batch retried immediately
    # after Phase 2.5 would hit the exact same still-open circuits it just
    # failed against. Rather than give up at that point and show "not
    # independently verified" while capacity that will be available again
    # in under a minute sits idle, this waits out the cooldown and tries
    # once more. Bounded to ONE extra pass, so a genuinely dead chain still
    # terminates rather than retrying forever.
    #
    # Reserve chain NOT reused here (use_reserved defaults to False) -- it
    # already had its one shot in Phase 2.5. Hammering it a second time
    # burns through its deliberately small, some-of-it-daily-capped entries
    # (Gemini, Cloudflare) for no real benefit over giving the primary
    # chain's 52 entries a second chance once their cooldowns have cleared.
    still_missing = [(test, grounding) for test, grounding in missing if test.test_id not in retry_results]
    if still_missing:
        print(f"{len(still_missing)} test(s) still missing after the first retry -- "
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
        # context; this verifies that it did, and throws the explanation away
        # if it did not. Rejected text is replaced with the same deterministic
        # copy used when a batch fails outright, so a patient never sees an
        # invented threshold, cutoff or dosage.
        if details:
            ok, reason = _verify_grounded(
                _patient_facing_text(details),
                grounding["text"],
                allowed_numbers=_allowed_numbers_for(test),
            )
            if not ok:
                # An ungrounded fact could be anywhere in the text (a wrong
                # threshold, an invented figure) -- can't trust any of it,
                # so the whole explanation is discarded.
                print(f"Rejected ungrounded explanation for {test.test_id}: {reason}")
                details = {}
            else:
                other_names = [t.raw_name for t, _ in batches[batch_index] if t.test_id != test.test_id]
                contaminated = _references_other_test(details.get("gp_question", ""), test.raw_name, other_names)
                if contaminated:
                    # BUG FOUND: this used to discard the WHOLE explanation
                    # over a bad gp_question alone, even though
                    # what_it_measures/what_your_result_means/
                    # lifestyle_suggestions had already passed the grounding
                    # check above and were perfectly fine -- one contaminated
                    # sentence demoted a good explanation to the generic
                    # "please discuss with your GP" fallback. Only
                    # gp_question is untrustworthy here (it names a sibling
                    # test from the same batch), so only it needs replacing --
                    # with the schema's own safe default question, kept in
                    # sync by reading it directly rather than duplicating the
                    # string.
                    print(f"Replaced contaminated gp_question for {test.test_id}: "
                          f"referenced {contaminated} instead of {test.raw_name}")
                    details["gp_question"] = ExplainedResult.model_fields["gp_question"].default
                # Only ever cache a result that passed the grounding check --
                # the cache must never be able to serve an explanation that
                # would have been rejected.
                _cache_put(test, details)

        if not details:
            # The batch failed, or the model omitted this test_id from its
            # array. Either way the copy below is canned, not generated --
            # record it so the UI can say so instead of passing deterministic
            # filler off as an explanation. `degraded` was previously
            # initialised, returned, and never once populated.
            degraded.append(test.test_id)

        unit_display = (test.unit or "").strip()
        result_str = f"{test.value} {unit_display}".strip()

        # BUG FOUND (Sterling Accuris report, Nitrite): source/source_urls
        # were always taken from `grounding`, regardless of whether the
        # DISPLAYED text actually came from it. When verification rejected
        # the generated explanation (above) or the model omitted this test
        # from its batch response, the real NHS/NIH page was still cited
        # next to generic fallback copy ("Evaluates Nitrite levels in the
        # body...") that has nothing to do with it - reading as sourced when
        # nothing shown was. Only attribute a source to text that actually
        # passed verification.
        source_label = grounding["source_label"] if details else "Not independently verified"
        source_urls = [s["url"] for s in grounding["sources"]] if details else []

        all_explained.append(
            ExplainedResult(
                test_id=test.test_id,
                raw_name=test.raw_name,
                value=test.value,
                unit=unit_display,
                # BUG FOUND (medreport_ai-2.pdf, ESR): same class of bug as
                # _ungrounded_result -- test.status is whatever the
                # EXTRACTOR guessed before this test's explanation was ever
                # verified, so a degraded/canned-fallback result (no real
                # explanation, "Not independently verified") could still
                # show a green "Within normal range" badge. No verified
                # basis to claim normal/high/low without real generated
                # (and grounding-checked) content, so this is UNKNOWN
                # whenever `details` is empty, same as the ungrounded case.
                status=test.status if details else RangeStatus.UNKNOWN,
                normal_range_min=getattr(test, "normal_range_min", None),
                normal_range_max=getattr(test, "normal_range_max", None),
                source=source_label,
                source_urls=source_urls,
                what_it_measures=details.get("what_it_measures", f"Evaluates {test.raw_name} levels in the body."),
                what_your_result_means=details.get("what_your_result_means", f"Your result is {result_str}. Please review this value with your GP."),
                lifestyle_suggestions=details.get("lifestyle_suggestions", ["Maintain balanced nutrition and regular physical activity."]),
                gp_question=details.get("gp_question", f"What does my {test.raw_name} result mean for my overall health?"),
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
    abnormal = [r for r in explained_results if r.status in (RangeStatus.HIGH, RangeStatus.LOW)]
    abnormal_line = ", ".join(f"{a.raw_name} ({a.value} {a.unit} - {a.status.value})" for a in abnormal) if abnormal else "All results normal"

    summary_prompt = f"""Patient: {profile.age}yo {profile.sex}.
Tests: {len(explained_results)}
Abnormal: {abnormal_line}

Return valid JSON:
{{
  "overall_summary": "A concise 3-sentence plain English summary of the main lab findings.",
  "top_gp_topics": ["Specific discussion point 1", "Specific discussion point 2"],
  "top_lifestyle_change": "One primary actionable lifestyle or dietary suggestion based on results.",
  "closing_message": "A supportive closing sentence reiterating clinical follow-up."
}}"""

    # BUG FOUND: with no constraint on this, the summary previously wrote
    # "low HDL (Hb A)" - HDL was actually normal, and "Hb A" is an unrelated
    # hemoglobin-electrophoresis test that happened to also be flagged low.
    # The two got merged into one false claim. Each item in "Abnormal" above
    # is already exactly "name (value unit - status)" - the model must not
    # combine, rename, or attach one test's name/status to another.
    # CHANGED: this prompt carried the anti-conflation rule but NOT the
    # anti-diagnosis rule that the per-test SYSTEM prompt has. The summary was
    # therefore free to write "hyperglycemia", "dyslipidemia" and "renal
    # function concerns" -- diagnostic language this tool must not produce,
    # and in the last case simply wrong (urea and BUN were LOW, which is not a
    # renal concern). Both rules now apply in both places.
    summary_system = (
        "You are a clinical report summarizer. Every abnormal test is listed "
        "individually in the \"Abnormal\" line as \"name (value unit - status)\". "
        "Refer to each test only by its own listed name and status - never combine "
        "two different tests into one claim, and never describe a test as high/low "
        "unless it is explicitly listed as abnormal. "
        "Never diagnose, name a disease or condition, or use clinical shorthand "
        "such as 'hyperglycemia', 'dyslipidemia', 'anaemia' or 'deficiency'. "
        "Describe only what the listed results show, in plain English, and direct "
        "the patient to their GP for interpretation. "
        "Do not state any number that does not appear in the data above. "
        "Return JSON."
    )

    # Every figure the summary is allowed to cite: the patient's own results
    # and reference bounds, plus the test count and their age (both stated in
    # the prompt). Anything else is invented.
    allowed = {float(len(explained_results))}
    try:
        allowed.add(float(profile.age))
    except (TypeError, ValueError):
        pass
    for r in explained_results:
        for value in (r.value, r.normal_range_min, r.normal_range_max):
            try:
                allowed.add(float(str(value).replace(",", "")))
            except (TypeError, ValueError):
                continue
        allowed |= _numbers_in(r.unit or "")

    try:
        res = call_with_fallback(system=summary_system, prompt=summary_prompt, max_tokens=1500,
                                 provider=provider, model_name=model_name, capability="text")
        if isinstance(res, dict) and res.get("overall_summary"):
            # CHANGED: the summary was the ONE patient-facing surface with no
            # grounding check at all -- and it is the first thing the patient
            # reads. Same rule as the per-test explanations: a figure that is
            # not in the data is not shown.
            blob = " ".join(str(res.get(k, "")) if not isinstance(res.get(k), list)
                            else " ".join(map(str, res.get(k) or []))
                            for k in ("overall_summary", "top_gp_topics",
                                      "top_lifestyle_change", "closing_message"))
            ok, reason = _verify_grounded(blob, abnormal_line, allowed_numbers=allowed)
            if not ok:
                print(f"Rejected ungrounded summary: {reason}")
                raise ValueError(reason)
            res["summary_degraded"] = False
            return res
    except Exception as e:
        print(f"Summary generation failed across all providers: {e}")

    # CHANGED: routes.py pops "summary_degraded" off this dict, but nothing
    # ever set it -- so a fully canned summary was reported to the UI as a
    # genuine one. Flagged honestly now.
    return {
        "summary_degraded": True,
        "overall_summary": f"Your report contains {len(explained_results)} tests. Several markers warrant discussion with your GP.",
        "top_gp_topics": [f"Review {r.raw_name} ({r.value} {r.unit})" for r in abnormal[:3]] if abnormal else ["Routine follow-up"],
        "top_lifestyle_change": "Focus on balanced nutrition, adequate hydration, and regular activity.",
        "closing_message": "Always discuss these results with your healthcare provider.",
    }