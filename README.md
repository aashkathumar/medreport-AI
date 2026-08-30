# MedReport AI

## Scope and constraints

This is a **technical software prototype**, evaluated using synthetic test
reports only.

- It is **not deployed to real patients** and does not provide independent
  medical advice.
- Explanations and lifestyle information are retrieved or summarised
  **exclusively from NHS UK and NIH MedlinePlus** via a constrained RAG
  pipeline. The corpus is restricted to those domains and the restriction is
  enforced at index-load time (`rag_service.ALLOWED_SOURCE_DOMAINS`), not
  merely requested in a prompt.
- Where information is unavailable from these sources, the system **directs
  the user to consult their GP** rather than generating an explanation.
  Generated text is verified against its retrieved source after generation;
  anything containing an unsourced figure, or drifting off-source, is replaced
  with a GP referral and reported in `ungrounded_test_ids`.
- Evaluation focuses on **retrieval accuracy and readability** against the
  NHS/NIH source material rather than clinical interpretation of results --
  see `backend/evaluation/evaluate_rag.py`.


Cloud-based AI system that explains blood test / urinalysis results in plain
English, grounded via RAG in NHS UK and NIH MedlinePlus reference data, with
personalised lifestyle suggestions. Built for the "Developing cloud-based AI
systems using AI-assisted technologies" dissertation project.

## Stack

- **Backend**: FastAPI (Python), deployed to AWS Lambda via Mangum + API Gateway
- **Frontend**: Streamlit
- **LLM**: pluggable providers with an automatic fallback chain -- Groq,
  Google Gemini, Mistral and OpenRouter. Text and vision use *separate*
  chains, because only Gemini and OpenRouter accept image payloads
  (see `TEXT_FALLBACK_CHAIN` / `VISION_FALLBACK_CHAIN` in `.env`).
- **RAG**: local FAISS vector index (cosine similarity) over NHS UK and NIH
  MedlinePlus reference text scraped by `data/build_rag_chunks.py`
  (sentence-transformers embeddings)
- **Storage**: local JSON files in dev / DynamoDB + S3 in production
- **IaC**: AWS SAM (`template.yaml`)

## Local setup (do this first)

### 1. Install backend dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env
```

and leave the AWS fields blank -- local mode needs no AWS account at all.

### 3. Build the RAG index (one-off, run from repo root)

```bash
cd data
python build_rag_chunks.py
cd ..
python scripts/build_rag_index.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90MB) from
Hugging Face automatically -- no manual download needed, just an internet
connection.

`build_rag_chunks.py` scrapes ~60 NHS UK / NIH MedlinePlus pages covering 51
test IDs. It **exits non-zero if any source fails**, so a dead URL can't
silently leave you with an empty corpus -- if it reports failures, fix or
remove those entries in `SOURCES` before building the index.

### 4. Sanity-check the RAG retrieval (optional but recommended)

```bash
python -c "
import sys; sys.path.insert(0, 'backend')
from app.services.rag_service import index_health, retrieve_context
print(index_health())
for r in retrieve_context('Serum Creatinine', test_id='CREATININE'):
    print(round(r['similarity'], 3), r['source'], '-', r['section'], '-', r['text'][:70])
"
```

`index_health()` must report `available: True`. The same check is exposed at
`GET /api/v1/rag/health` once the backend is running.

### 5. Run the unit tests (no API key needed)

```bash
cd backend
python tests/test_reference_db.py
python tests/test_pdf_parser.py
python tests/test_rag_and_fallback.py   # needs the RAG index from step 3
```

### 6. Run the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 7. Generate a sample PDF and smoke-test the full pipeline

In a new terminal:

```bash
cd backend
python tests/generate_sample_pdf.py
python tests/smoke_test.py
```

This calls the real LLM (Haiku by default) end-to-end -- confirm you see
`Explained N results` in the output.

### 8. Run the frontend

In another terminal:

```bash
cd frontend
pip install -r requirements.txt
streamlit run main.py
```

Open **http://localhost:8501**, upload `backend/tests/sample_report.pdf`
(or a real anonymised report), and click through Upload -> Results -> History.

## Cloud deployment (later step)

`sentence-transformers` is too large for a standard zip-based Lambda
deployment. Before running `sam deploy`, this project needs to move to a
container-image Lambda (Dockerfile-based) -- see the accompanying setup
notes for that step. Do not attempt `sam deploy` until local testing above
is fully working.

## Project structure

```
backend/
  app/
    main.py              FastAPI app + Lambda handler
    config.py             Settings (reads .env)
    api/routes.py          All HTTP endpoints
    models/schemas.py      Pydantic models
    services/
      reference_db.py      NHS/NIH structured reference lookups
      pdf_parser.py         Extracts test values from uploaded PDFs
      rag_service.py        FAISS-based semantic retrieval
      llm_service.py        Claude API calls (RAG-grounded prompts)
      pdf_generator.py      Builds the downloadable PDF report
    db/
      __init__.py           Switches between local_storage and dynamo/s3
      local_storage.py      Dev-mode storage (JSON files on disk)
      dynamo.py              Production DynamoDB storage
      s3.py                   Production S3 storage
  tests/                    Unit + smoke tests
  requirements.txt
  .env.example
  

frontend/
  app.py                    Streamlit entry point
  pages/                    Upload / Results / History pages
  requirements.txt

data/
  blood_tests.json          Structured reference data (blood)
  urine_tests.json          Structured reference data (urine)
  build_rag_chunks.py        Generates rag_chunks.json from the above

scripts/
  build_rag_index.py         Builds the FAISS index from rag_chunks.json

template.yaml                AWS SAM infrastructure-as-code
```
