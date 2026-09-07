# MedReport AI

Cloud-based AI system that explains blood test / urinalysis results in plain
English, grounded via RAG in NHS UK and NIH MedlinePlus reference data, with
personalised lifestyle suggestions.

## Scope and constraints

- Technical software prototype, evaluated using synthetic test reports only.
- Not deployed to real patients; not independent medical advice.
- Explanations are grounded exclusively in NHS UK and NIH MedlinePlus
  (`rag_service.ALLOWED_SOURCE_DOMAINS`).
- If a test isn't covered, the system directs the user to their GP instead
  of generating an explanation (`ungrounded_test_ids`).
- Evaluation: `backend/evaluation/evaluate_rag.py`.

## Stack

- **Backend**: FastAPI (Python) on AWS Lambda (container image, Mangum, Function URL)
- **Frontend**: Streamlit on a separate EC2 instance
- **LLM**: Mistral, Groq, NVIDIA, OpenRouter, Cohere, Cloudflare Workers AI (primary fallback chain) + Gemini (reserved retry chain)
- **RAG**: local FAISS index (cosine similarity) over NHS/NIH text, sentence-transformers embeddings
- **Storage**: none server-side; reports live only in the frontend session
- **IaC**: AWS SAM (`template.yaml`); `Dockerfile.gcp` is a Cloud Run comparison build, not live

## Local setup

### 1. Backend dependencies

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

Fill in at least one provider API key. Leave AWS fields blank for local mode.

### 3. Build the RAG index (from repo root)

```bash
cd data
python build_rag_chunks.py
cd ..
python scripts/build_rag_index.py
```

### 4. Check the RAG retrieval (optional)

```bash
python -c "
import sys; sys.path.insert(0, 'backend')
from app.services.rag_service import index_health, retrieve_context
print(index_health())
for r in retrieve_context('Serum Creatinine', test_id='CREATININE'):
    print(round(r['similarity'], 3), r['source'], '-', r['section'], '-', r['text'][:70])
"
```

Also exposed at `GET /api/v1/rag/health` once the backend is running.

### 5. Run the unit tests

```bash
cd backend
python tests/test_reference_db.py
python tests/test_pdf_parser.py
python tests/test_tier1_extraction.py
python tests/test_rag_and_fallback.py   # needs the RAG index from step 3
```

### 6. Run the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 7. Smoke-test the full pipeline

```bash
cd backend
python tests/generate_sample_pdf.py
python tests/smoke_test.py
```

### 8. Run the frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run main.py
```

Open http://localhost:8501, upload `backend/tests/sample_report.pdf`, click through Upload -> Results -> History.

## Cloud deployment (AWS)

**Live instance:**

| | |
|---|---|
| Application | http://18.132.96.4 |
| Backend API | `https://6mqiiylbn5lxs7sr62yw2hf4pe0gthzx.lambda-url.eu-west-2.on.aws/` |

### Prerequisites

```bash
brew install awscli aws-sam-cli bash
aws configure                           # region: eu-west-2, output: json
```

Docker Desktop must be running; `backend/.env` must have the provider API keys.

### Deploy

```bash
./deploy/push_to_ecr.sh
/opt/homebrew/bin/bash deploy/create_ssm_params.sh
./deploy/deploy_lambda.sh
```

### Frontend

```bash
aws ec2 create-key-pair --key-name medreport-key --region eu-west-2 \
  --query 'KeyMaterial' --output text > ~/medreport-key.pem
chmod 400 ~/medreport-key.pem

./deploy/launch_ec2.sh medreport-key
./deploy/provision_frontend.sh <public-ip> ~/medreport-key.pem
```

### Redeploying after a change

- Backend: `./deploy/push_to_ecr.sh` then `./deploy/deploy_lambda.sh`
- Frontend: `./deploy/provision_frontend.sh <ip> ~/medreport-key.pem`

### Operating the deployed app

```bash
# restart the frontend
ssh -i ~/medreport-key.pem ec2-user@<ip> 'sudo systemctl restart medreport-frontend.service'

# status and logs
ssh -i ~/medreport-key.pem ec2-user@<ip> \
  'sudo systemctl status medreport-frontend.service --no-pager && sudo tail -30 /var/log/medreport-frontend.log'

# health check (also warms the backend)
curl -s <function-url>api/v1/health
```

Cold start: ~56s after ~5-15 min idle (embedding model + FAISS index load). Warm: ~0.1s.

## Project structure

```
backend/
  app/
    main.py              FastAPI app + Lambda handler
    config.py             Settings (reads .env)
    api/routes.py          All HTTP endpoints
    models/schemas.py      Pydantic models
    services/
      reference_db.py      NHS/NIH structured reference lookups, alias resolution
      pdf_parser.py         Four-tier PDF/data extraction pipeline
      llm_extractor.py      Vision/text LLM extraction tiers used by pdf_parser
      llm_providers.py      Per-provider API call functions
      llm_service.py        Multi-provider fallback, RAG-grounded explanation engine
      rag_service.py        FAISS-based semantic retrieval
      pdf_generator.py      Builds the downloadable PDF report
  evaluation/
    evaluate_rag.py          Retrieval accuracy / groundedness / readability harness
    run_dissertation_eval.py Evaluation run behind the dissertation's Ch.4 table
  tests/                    Unit + smoke tests
  requirements.txt
  .env.example

frontend/
  main.py                   Streamlit entry point -- run THIS, not app.py
  app.py                    Home page (upload + profile)
  pages/
    2_results.py            Results page
    3_History.py            Session history page
  theme.py                  Styling and logo injection
  assets/                   Sidebar / icon logos
  requirements.txt

data/
  blood_tests.json          Structured reference data (blood)
  urine_tests.json          Structured reference data (urine)
  build_rag_chunks.py        Generates rag_chunks.json from the above

scripts/
  build_rag_index.py              Builds the FAISS index from rag_chunks.json
  generate_test_synonyms.py       LLM-generated test-name aliases
  precompute_what_it_measures.py  Precomputes what_it_measures
  precompute_full_explanations.py Precomputes all cached fields

deploy/
  push_to_ecr.sh             Build the Lambda image and push it to ECR
  create_ssm_params.sh       Load .env keys into SSM Parameter Store
  deploy_lambda.sh           Deploy/update the Lambda stack via SAM
  launch_ec2.sh              Launch the frontend EC2 instance
  provision_frontend.sh      Copy + start the frontend over SSH
  ec2_userdata.sh            First-boot script used by launch_ec2.sh

Dockerfile.aws               Lambda container image (AWS, live deployment)
Dockerfile.gcp               Cloud Run image (GCP comparison build, not deployed live)
template.yaml                AWS SAM infrastructure-as-code
```
