# MedReport AI

Cloud-based AI system that explains blood test / urinalysis results in plain
English, grounded via RAG in NHS UK and NIH MedlinePlus reference data, with
personalised lifestyle suggestions. Built for the "Developing cloud-based AI
systems using AI-assisted technologies" dissertation project.

## Stack

- **Backend**: FastAPI (Python), deployed to AWS Lambda via Mangum + API Gateway
- **Frontend**: Streamlit
- **LLM**: Anthropic Claude API (Haiku for dev, Sonnet for evaluation)
- **RAG**: local FAISS vector index over NHS/NIH reference text (sentence-transformers embeddings)
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
cp .env.example .env
```

Edit `.env` and set your real `ANTHROPIC_API_KEY`. Leave `APP_ENV=development`
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

### 4. Sanity-check the RAG retrieval (optional but recommended)

```bash
python -c "
import sys; sys.path.insert(0, 'backend')
from app.services.rag_service import retrieve_context
for r in retrieve_context('haemoglobin low'):
    print(r['test_id'], '-', r['text'][:80])
"
```

### 5. Run the unit tests (no API key needed)

```bash
cd backend
python tests/test_reference_db.py
python tests/test_pdf_parser.py
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
streamlit run app.py
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
