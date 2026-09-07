"""
Builds data/rag_chunks.json, the RAG corpus, by scraping NHS UK and
NIH MedlinePlus reference pages.

Usage (from repo root):
    python data/build_rag_chunks.py

Then rebuild the vector index:
    python scripts/build_rag_index.py

CHANGED (was silently producing a useless corpus):
  * Of the 4 URLs in the old TARGET_SOURCES, 3 returned 404. The failures
    were caught, printed, and then ignored - the script still exited 0 and
    still wrote rag_chunks.json. The resulting "RAG corpus" was 21 chunks
    scraped from a single generic NHS "Blood tests" landing page, every one
    of them tagged test_id="HGB" regardless of content. Retrieval could not
    have worked. Every URL in SOURCES below has been checked live, and the
    script now exits non-zero if any source fails, so a silent corpus
    regression can't happen again.
  * One chunk per <p> produced fragments too short to be useful grounding
    ("If a healthcare professional such as a GP..."). Extraction is now
    section-aware: it walks headings and emits one chunk per document
    section, so "What do the results mean?" arrives as a coherent passage.
  * A page can now ground several tests (a CBC page covers HGB/WBC/PLT/RBC/
    HCT/MCV), so chunks carry a list of test_ids rather than a single one.
"""
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

_DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.services.reference_db import canonicalize_test_name  # noqa: E402

USER_AGENT = (
    "MedReportAI-RAG/1.0 (academic dissertation project; "
    "reference-text retrieval; contact via repository)"
)
REQUEST_TIMEOUT = 25
POLITE_DELAY_SEC = 0.6  # be a good citizen; these are public health services

# Target chunk size. Sections longer than this are split on sentence
# boundaries; sections shorter than MIN_CHUNK_CHARS are dropped as boilerplate.
MAX_CHUNK_CHARS = 1100
MIN_CHUNK_CHARS = 120


# --------------------------------------------------------------------------
# Source map, every URL below returned HTTP 200 when this list was built.
MEDLINEPLUS = "NIH MedlinePlus"
NHS = "NHS UK"

_MP = "https://medlineplus.gov/lab-tests/{}/"

SOURCES = [
    # --- Full blood count ---------------------------------------------------
    {"ids": ["HGB", "WBC", "PLT", "RBC", "HCT", "MCV"], "src": MEDLINEPLUS,
     "url": _MP.format("complete-blood-count-cbc")},
    {"ids": ["HGB"], "src": MEDLINEPLUS, "url": _MP.format("hemoglobin-test")},
    {"ids": ["WBC"], "src": MEDLINEPLUS, "url": _MP.format("white-blood-count-wbc")},
    {"ids": ["PLT"], "src": MEDLINEPLUS, "url": _MP.format("platelet-tests")},
    {"ids": ["RBC"], "src": MEDLINEPLUS, "url": _MP.format("red-blood-cell-rbc-count")},
    {"ids": ["HCT"], "src": MEDLINEPLUS, "url": _MP.format("hematocrit-test")},
    {"ids": ["MCV"], "src": MEDLINEPLUS, "url": _MP.format("mcv-mean-corpuscular-volume")},
    {"ids": ["RETIC"], "src": MEDLINEPLUS, "url": _MP.format("reticulocyte-count")},
    {"ids": ["ESR"], "src": MEDLINEPLUS, "url": _MP.format("erythrocyte-sedimentation-rate-esr")},
    # The differential and red-cell indices.
    {"ids": ["NEUTROPHILS", "LYMPHOCYTES", "EOSINOPHILS", "MONOCYTES",
             "BASOPHILS", "WBC"], "src": MEDLINEPLUS,
     "url": _MP.format("blood-differential")},
    {"ids": ["RDW_CV", "RDW"], "src": MEDLINEPLUS,
     "url": _MP.format("rdw-red-cell-distribution-width")},
    {"ids": ["MPV"], "src": MEDLINEPLUS, "url": _MP.format("mpv-blood-test")},
    {"ids": ["MCH", "MCHC", "MCV"], "src": MEDLINEPLUS,
     "url": _MP.format("complete-blood-count-cbc")},

    # --- Iron studies / anaemia --------------------------------------------
    {"ids": ["FERRITIN"], "src": MEDLINEPLUS, "url": _MP.format("ferritin-blood-test")},
    {"ids": ["IRON", "FERRITIN"], "src": MEDLINEPLUS, "url": _MP.format("iron-tests")},
    {"ids": ["HGB", "FERRITIN", "IRON"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/iron-deficiency-anaemia/"},
    {"ids": ["B12", "FOLATE"], "src": MEDLINEPLUS, "url": _MP.format("vitamin-b-test")},
    {"ids": ["B12", "FOLATE"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/vitamin-b12-or-folate-deficiency-anaemia/"},

    # --- Diabetes / glucose -------------------------------------------------
    {"ids": ["HBA1C"], "src": MEDLINEPLUS, "url": _MP.format("hemoglobin-a1c-hba1c-test")},
    {"ids": ["GLUCOSE"], "src": MEDLINEPLUS, "url": _MP.format("blood-glucose-test")},
    {"ids": ["HBA1C", "GLUCOSE"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/type-2-diabetes/"},

    # --- Lipids -------------------------------------------------------------
    {"ids": ["CHOL", "LDL", "HDL", "TRIG"], "src": MEDLINEPLUS, "url": _MP.format("cholesterol-levels")},
    {"ids": ["TRIG"], "src": MEDLINEPLUS, "url": _MP.format("triglycerides-test")},
    {"ids": ["CHOL", "LDL", "HDL"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/high-cholesterol/cholesterol-levels/"},
    {"ids": ["CHOL", "LDL", "HDL", "TRIG"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/high-cholesterol/"},

    # --- Thyroid ------------------------------------------------------------
    {"ids": ["TSH"], "src": MEDLINEPLUS, "url": _MP.format("tsh-thyroid-stimulating-hormone-test")},
    {"ids": ["T4"], "src": MEDLINEPLUS, "url": _MP.format("thyroxine-t4-test")},
    {"ids": ["T3"], "src": MEDLINEPLUS, "url": _MP.format("triiodothyronine-t3-tests")},
    {"ids": ["TSH", "T4"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/underactive-thyroid-hypothyroidism/"},
    {"ids": ["TSH", "T3", "T4"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/overactive-thyroid-hyperthyroidism/"},

    # --- Kidney / electrolytes ---------------------------------------------
    {"ids": ["CREATININE"], "src": MEDLINEPLUS, "url": _MP.format("creatinine-test")},
    {"ids": ["UREA"], "src": MEDLINEPLUS, "url": _MP.format("bun-blood-urea-nitrogen")},
    {"ids": ["NA", "K", "CL", "CO2"], "src": MEDLINEPLUS, "url": _MP.format("electrolyte-panel")},
    {"ids": ["NA"], "src": MEDLINEPLUS, "url": _MP.format("sodium-blood-test")},
    {"ids": ["K"], "src": MEDLINEPLUS, "url": _MP.format("potassium-blood-test")},
    {"ids": ["CA"], "src": MEDLINEPLUS, "url": _MP.format("calcium-blood-test")},
    {"ids": ["MG"], "src": MEDLINEPLUS, "url": _MP.format("magnesium-blood-test")},
    {"ids": ["PHOS"], "src": MEDLINEPLUS, "url": _MP.format("phosphate-in-blood")},
    {"ids": ["URIC_ACID"], "src": MEDLINEPLUS, "url": _MP.format("uric-acid-test")},
    {"ids": ["URIC_ACID"], "src": NHS, "url": "https://www.nhs.uk/conditions/gout/"},
    {"ids": ["CREATININE", "UREA", "NA", "K", "CA", "GLUCOSE"], "src": MEDLINEPLUS,
     "url": _MP.format("basic-metabolic-panel-bmp")},
    {"ids": ["MICROALB"], "src": MEDLINEPLUS, "url": _MP.format("microalbumin-creatinine-ratio")},
    {"ids": ["EGFR", "GFR"], "src": MEDLINEPLUS,
     "url": _MP.format("glomerular-filtration-rate-gfr-test")},

    # --- Liver --------------------------------------------------------------
    {"ids": ["ALT", "AST", "ALP", "BILIRUBIN", "ALBUMIN"], "src": MEDLINEPLUS,
     "url": _MP.format("liver-function-tests")},
    {"ids": ["ALT"], "src": MEDLINEPLUS, "url": _MP.format("alt-blood-test")},
    {"ids": ["AST"], "src": MEDLINEPLUS, "url": _MP.format("ast-test")},
    {"ids": ["ALP"], "src": MEDLINEPLUS, "url": _MP.format("alkaline-phosphatase")},
    {"ids": ["BILIRUBIN"], "src": MEDLINEPLUS, "url": _MP.format("bilirubin-blood-test")},
    {"ids": ["ALBUMIN"], "src": MEDLINEPLUS, "url": _MP.format("albumin-blood-test")},
    {"ids": ["TOTAL_PROTEIN", "ALBUMIN", "GLOBULIN"], "src": MEDLINEPLUS,
     "url": _MP.format("total-protein-and-albumin-globulin-a-g-ratio")},
    {"ids": ["GLOBULIN"], "src": MEDLINEPLUS, "url": _MP.format("globulin-test")},
    {"ids": ["ALT", "AST", "ALP", "BILIRUBIN"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/liver-disease/"},

    # --- Inflammation / other ----------------------------------------------
    {"ids": ["CRP"], "src": MEDLINEPLUS, "url": _MP.format("c-reactive-protein-crp-test")},
    {"ids": ["VIT_D"], "src": MEDLINEPLUS, "url": _MP.format("vitamin-d-test")},
    {"ids": ["VIT_D"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/vitamins-and-minerals/vitamin-d/"},
    {"ids": ["PSA"], "src": MEDLINEPLUS, "url": _MP.format("prostate-specific-antigen-psa-test")},
    {"ids": ["HOMOCYSTEINE"], "src": MEDLINEPLUS, "url": _MP.format("homocysteine-test")},
    {"ids": ["IGE"], "src": MEDLINEPLUS, "url": _MP.format("allergy-blood-test")},
    {"ids": ["HIV"], "src": MEDLINEPLUS, "url": _MP.format("hiv-screening-test")},

    # --- Urinalysis ---------------------------------------------------------
    # BUG FOUND: URINE_COLOUR was previously patched directly into the built
    # rag_chunks.json (to fix "Colour" reporting "not covered" despite this
    # page's "Normal Results" section discussing urine colour explicitly), but
    {"ids": ["URINE_PROTEIN", "URINE_GLUCOSE", "URINE_PH", "URINE_KETONES",
             "URINE_NITRITES", "URINE_LEUKOCYTES", "URINE_COLOUR", "CLEARITY"],
     "src": MEDLINEPLUS, "url": "https://medlineplus.gov/ency/article/003579.htm"},
    {"ids": ["URINE_PROTEIN"], "src": MEDLINEPLUS, "url": _MP.format("protein-in-urine")},
    {"ids": ["URINE_GLUCOSE"], "src": MEDLINEPLUS, "url": _MP.format("glucose-in-urine-test")},
    {"ids": ["URINE_KETONES"], "src": MEDLINEPLUS, "url": _MP.format("ketones-in-urine")},
    {"ids": ["URINE_NITRITES", "URINE_LEUKOCYTES"], "src": MEDLINEPLUS,
     "url": _MP.format("nitrites-in-urine")},
    {"ids": ["URINE_LEUKOCYTES", "URINE_NITRITES"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/urinary-tract-infections-utis/"},
    {"ids": ["URINE_BILIRUBIN"], "src": MEDLINEPLUS, "url": _MP.format("bilirubin-in-urine")},
    {"ids": ["URINE_BLOOD"], "src": MEDLINEPLUS, "url": _MP.format("blood-in-urine")},
    {"ids": ["SPECIFIC_GRAVITY"], "src": MEDLINEPLUS,
     "url": "https://medlineplus.gov/ency/article/003587.htm"},
    {"ids": ["CASTS"], "src": MEDLINEPLUS,
     "url": "https://medlineplus.gov/ency/article/003586.htm"},
    # These three were previously reported "not covered" only because the URL
    # pattern was guessed wrong (tried .../urobilinogen-test/ etc, all 404).
    {"ids": ["UROBILINOGEN"], "src": MEDLINEPLUS,
     "url": "https://medlineplus.gov/lab-tests/urobilinogen-in-urine/"},
    {"ids": ["EPITHELIAL_CELLS"], "src": MEDLINEPLUS,
     "url": "https://medlineplus.gov/lab-tests/epithelial-cells-in-urine/"},
    {"ids": ["CRYSTALS"], "src": MEDLINEPLUS,
     "url": "https://medlineplus.gov/lab-tests/crystals-in-urine/"},

    # --- Blood type / infection screening -----------------------------------
    {"ids": ["ABO_TYPE", "RH_D_TYPE"], "src": NHS,
     "url": "https://www.nhs.uk/conditions/blood-groups/"},
    {"ids": ["HBSAG"], "src": NHS, "url": "https://www.nhs.uk/conditions/hepatitis-b/"},
    {"ids": ["HBSAG"], "src": MEDLINEPLUS, "url": "https://medlineplus.gov/hepatitisb.html"},
    {"ids": ["HIV"], "src": NHS, "url": "https://www.nhs.uk/conditions/hiv-and-aids/"},

    # --- General context ----------------------------------------------------
    {"ids": ["GENERAL"], "src": NHS, "url": "https://www.nhs.uk/tests-and-treatments/blood-tests/"},
]


# --------------------------------------------------------------------------
# Auto-discovery: MedlinePlus's own A-Z lab-tests index
# --------------------------------------------------------------------------
# CHANGED: the hand-curated SOURCES list above covers ~76 test_ids from ~60
_MEDLINEPLUS_INDEX = "https://medlineplus.gov/lab-tests/"


def discover_medlineplus_lab_tests(session: requests.Session) -> list[dict]:
    """Returns [{"url":..., "ids":[...], "src": MEDLINEPLUS}, ...] for every
    page linked from MedlinePlus's lab-tests A-Z index."""
    html = fetch(_MEDLINEPLUS_INDEX, session)
    soup = BeautifulSoup(html, "html.parser")
    root = _content_root(soup)
    discovered = []
    seen_urls = set()
    for a in root.find_all("a", href=True):
        href = a["href"]
        if not re.match(r"^https://medlineplus\.gov/lab-tests/[a-z0-9\-]+/?$", href):
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)
        title = clean_text(a.get_text(" "))
        if not title:
            continue
        discovered.append({"url": href, "ids": _derive_test_ids(title), "src": MEDLINEPLUS})
    return discovered


def _derive_test_ids(title: str) -> list[str]:
    """Best-effort canonical test_id(s) for an auto-discovered page.

    A scraped chunk is only reachable at explanation time if it's tagged
    with the SAME id a real report's raw field name resolves to via
    reference_db.canonicalize_test_name(), not just whatever the page's
    own title happens to be. canonicalize_test_name() is deliberately
    narrow (see its own docstring) and does NOT strip words like "Test" or
    "Levels", but a real lab report prints "Aldosterone", not "Aldosterone
    Test", so that stripping has to happen here, or every auto-discovered
    chunk would sit unused under a title-shaped id nothing ever matches.
    Also splits out any parenthetical abbreviation ("Adrenocorticotropic
    Hormone (ACTH)") as its OWN candidate id, since reports commonly print
    the abbreviation alone rather than the full name.
    """
    ids = []
    # Minimum 2 chars, a real medical abbreviation is never a single letter,
    # but a stray "(a)" (e.g.
    for paren in re.findall(r"\(([A-Za-z0-9\-/ ]{2,15})\)", title):
        cand = canonicalize_test_name(paren)
        if cand and cand != "UNKNOWN_TEST" and cand not in ids:
            ids.append(cand)
    base = re.sub(r"\([^)]*\)", " ", title)
    # Multi-word phrases first ("Blood Test" as a unit), or the bare "test"
    # rule alone leaves "Blood" behind (e.g.
    base = re.sub(
        r"\b(tumor\s*marker\s*test|blood\s*tests?|screening\s*tests?|"
        r"test|tests|screening|panel|levels?)\b",
        " ", base, flags=re.IGNORECASE,
    )
    cand = canonicalize_test_name(base)
    if cand and cand != "UNKNOWN_TEST" and cand not in ids:
        ids.append(cand)
    return ids or [canonicalize_test_name(title)]


# Section headings that carry no information about what a result means.
# Dropping them keeps the index focused on explanation-bearing text.
_SKIP_HEADING_RE = re.compile(
    r"(are there any risks|what happens during|how (is|do) .*(test|it) (work|done)|"
    r"do i need to do anything to prepare|will i need to do anything to prepare|"
    r"references|related (health )?topics|learn more|more information|"
    r"see also|sources|find a (test|clinic)|about this|patient handouts)",
    re.IGNORECASE,
)

# Boilerplate that survives content extraction on some pages.
_BOILERPLATE_RE = re.compile(
    r"(page last reviewed|next review due|urac|health on the net|"
    r"the information on this site should not be used|"
    r"a\.d\.a\.m\.|copyright|all rights reserved|cookies on the nhs|"
    r"medlineplus links to health information|"
    r"is the official website of|javascript)",
    re.IGNORECASE,
)

_STRIP_TAGS = ["script", "style", "nav", "aside", "footer", "header", "form",
               "noscript", "iframe", "button"]
_STRIP_SELECTORS = [
    ".mp-refs", ".mp-share", ".mp-related", ".usa-banner", ".breadcrumb",
    ".nhsuk-breadcrumb", ".nhsuk-contents-list", ".nhsuk-pagination",
    ".nhsuk-back-link", ".page-feedback", ".site-footer", "#mplus-links",
    ".goog-te-banner-frame",
]


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def _content_root(soup: BeautifulSoup):
    """Finds the main content container across the three page shapes used by
    MedlinePlus lab-tests, MedlinePlus encyclopedia, and NHS conditions."""
    for finder in (
        lambda: soup.find("div", class_="main"),          # MedlinePlus
        lambda: soup.find("main"),                          # NHS
        lambda: soup.find("article"),                       # fallback
        lambda: soup.find("body"),
    ):
        node = finder()
        if node is not None:
            return node
    return soup


def _split_long(text: str) -> list[str]:
    """Splits an over-long section on sentence boundaries, never mid-sentence."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, buf = [], ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) + 1 > MAX_CHUNK_CHARS:
            out.append(buf.strip())
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf.strip():
        out.append(buf.strip())
    return out


def extract_sections(html: bytes) -> tuple[str, list[tuple[str, str]]]:
    """Returns (page_title, [(heading, body_text), ...]).

    Walks the content root in document order, attributing every block of text
    to the most recent heading. This is what turns "What do the results mean?"
    into one coherent passage instead of a handful of orphaned sentences.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title:
        # "Creatinine Test: MedlinePlus Medical Test" -> "Creatinine Test"
        title = clean_text(soup.title.get_text()).split(":")[0].split(" - NHS")[0].strip()

    root = _content_root(soup)
    for tag in root.find_all(_STRIP_TAGS):
        tag.decompose()
    for selector in _STRIP_SELECTORS:
        for tag in root.select(selector):
            tag.decompose()

    sections: list[tuple[str, list[str]]] = []
    heading = title or "Overview"
    body: list[str] = []

    for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = clean_text(el.get_text(" "))
        if not text:
            continue
        if el.name in ("h1", "h2", "h3"):
            if body:
                sections.append((heading, body))
            heading = text
            body = []
        else:
            if _BOILERPLATE_RE.search(text):
                continue
            body.append(text)
    if body:
        sections.append((heading, body))

    out = []
    for head, lines in sections:
        if _SKIP_HEADING_RE.search(head):
            continue
        joined = " ".join(lines)
        if len(joined) < MIN_CHUNK_CHARS:
            continue
        out.append((head, joined))
    return title, out


def fetch(url: str, session: requests.Session) -> bytes:
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def build() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    chunks: list[dict] = []
    seen_texts: set[str] = set()
    failures: list[tuple[str, str]] = []
    chunk_id = 0

    print("Discovering MedlinePlus lab-tests index ...")
    try:
        discovered = discover_medlineplus_lab_tests(session)
        print(f"  found {len(discovered)} pages on the index")
    except Exception as e:
        print(f"  FAILED to fetch the index: {e}, continuing with the curated list only")
        discovered = []
    time.sleep(POLITE_DELAY_SEC)

    # Merge entries that point at the same URL, unioning their test_ids.
    merged: dict = {}
    for target in discovered + SOURCES:
        entry = merged.setdefault(
            target["url"], {"url": target["url"], "src": target["src"], "ids": []}
        )
        for test_id in target["ids"]:
            if test_id not in entry["ids"]:
                entry["ids"].append(test_id)
    sources = list(merged.values())

    for i, target in enumerate(sources, 1):
        url, ids, source = target["url"], target["ids"], target["src"]
        print(f"[{i:>2}/{len(sources)}] {', '.join(ids):<38} {url}")
        try:
            html = fetch(url, session)
        except Exception as e:
            print(f"          FAILED: {type(e).__name__}: {e}")
            failures.append((url, f"{type(e).__name__}: {e}"))
            continue

        title, sections = extract_sections(html)
        if not sections:
            print("          FAILED: no extractable content sections")
            failures.append((url, "no extractable content sections"))
            continue

        added = 0
        for heading, body in sections:
            for piece in _split_long(body):
                fingerprint = piece[:200].lower()
                if fingerprint in seen_texts:
                    continue  # same passage reached via two source pages
                seen_texts.add(fingerprint)
                chunks.append({
                    "id": chunk_id,
                    "test_ids": ids,
                    "topic": title or heading,
                    "section": heading,
                    "text": piece,
                    # What actually gets embedded: topic + section give the
                    # vector the context a bare passage lacks ("What do the
                    # results mean?" is meaningless without its topic).
                    "embed_text": f"{title} - {heading}: {piece}",
                    "source": source,
                    "url": url,
                })
                chunk_id += 1
                added += 1
        print(f"          {added} chunk(s) from {len(sections)} section(s)")
        time.sleep(POLITE_DELAY_SEC)

    _DATA_DIR.mkdir(exist_ok=True)
    chunks_path = _DATA_DIR / "rag_chunks.json"
    chunks_path.write_text(json.dumps(chunks, indent=2))

    covered = sorted({tid for c in chunks for tid in c["test_ids"]})
    print(f"\nWrote {len(chunks)} chunks to {chunks_path}")
    print(f"Covering {len(covered)} test IDs: {', '.join(covered)}")
    print("Next: python scripts/build_rag_index.py")

    if failures:
        # Exit non-zero so a broken corpus is loud. The old script printed
        # failures and exited 0, which is how 3 dead URLs went unnoticed.
        print(f"\n{len(failures)} source(s) FAILED:", file=sys.stderr)
        for url, err in failures:
            print(f"  {url}\n    {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(build())
