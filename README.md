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

- **Backend**: FastAPI (Python), deployed to AWS Lambda as a **container
  image** via Mangum, behind a **Lambda Function URL** -- not API Gateway,
  which caps requests at 29 seconds and would cut off longer report
  generations.
- **Frontend**: Streamlit, on a separate EC2 instance (it needs a persistent
  process, which Lambda's execution model does not provide).
- **LLM**: pluggable providers with an automatic fallback chain spanning
  eight providers -- Groq, Mistral, NVIDIA, OpenRouter, Google Gemini,
  Cohere, Cloudflare Workers AI, plus a Cerebras scaffold. Text and vision
  use *separate* chains, because only some providers accept image payloads
  (see `TEXT_FALLBACK_CHAIN` / `VISION_FALLBACK_CHAIN` in `.env`).
- **RAG**: local FAISS vector index (cosine similarity) over NHS UK and NIH
  MedlinePlus reference text scraped by `data/build_rag_chunks.py`
  (sentence-transformers embeddings)
- **Storage**: **none server-side.** Reports are held only in the frontend's
  Streamlit session and are gone when the session ends. This was a
  deliberate change made in response to ethics review of the project's
  original storage design, not an oversight. (`backend/app/db/` still
  contains the superseded DynamoDB/S3 modules; they are dead code, imported
  by nothing.)
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

Then edit `.env` and fill in at least one provider API key. Leave the AWS
fields blank -- local mode needs no AWS account at all.

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

This calls a real LLM end-to-end, using whichever provider is first in the
fallback chain that has a working key -- confirm you see
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

## Cloud deployment (AWS)

**Live instance:**

| | |
|---|---|
| Application (use this) | http://16.61.145.167 |
| Backend API | `https://6mqiiylbn5lxs7sr62yw2hf4pe0gthzx.lambda-url.eu-west-2.on.aws/` |

The backend is serverless, so the first request after ~15 minutes idle takes
about a minute while the container starts (see **Cold starts** below). If the
page seems to hang on your first upload, that is why -- it is not stuck.

The system is deployed as two independent services: the backend as a Lambda
container image behind a Function URL, the frontend as a Streamlit process
on an EC2 instance. `sentence-transformers` and `faiss-cpu` are far too
large for a zip-based Lambda package, which is why the backend is packaged
as a container image rather than deployed with a plain `sam deploy`.

### Prerequisites

```bash
brew install awscli aws-sam-cli bash   # bash 5.x: the scripts use
                                        # associative arrays, which the
                                        # bash 3.2 macOS ships cannot do
aws configure                           # region: eu-west-2, output: json
```

Docker Desktop must be running, and `backend/.env` must contain the
provider API keys (they are read from there, never committed).

### Deploy, in order

```bash
./deploy/push_to_ecr.sh            # build the image and push it to ECR
/opt/homebrew/bin/bash deploy/create_ssm_params.sh   # keys -> SSM Parameter Store
./deploy/deploy_lambda.sh          # create/update the Lambda stack
```

Each script hands off to the next through `deploy/.image_uri` and
`deploy/.function_url` (both gitignored), so no URI or URL is ever
copy-pasted by hand. `deploy_lambda.sh` prints the live Function URL when it
finishes.

### Then the frontend

```bash
aws ec2 create-key-pair --key-name medreport-key --region eu-west-2 \
  --query 'KeyMaterial' --output text > ~/medreport-key.pem
chmod 400 ~/medreport-key.pem

./deploy/launch_ec2.sh medreport-key                       # launch instance
./deploy/provision_frontend.sh <public-ip> ~/medreport-key.pem   # install + start
```

`provision_frontend.sh` copies the frontend over SSH rather than having the
instance clone it, because **this repository is private** and a bare EC2
instance holds no GitHub credentials. Passing a token or deploy key through
user-data would be worse: user-data is readable from instance metadata and
is echoed into the console log.

It also wires `MEDREPORT_API_URL` to the deployed Function URL and installs
a systemd unit, so the app restarts on failure and survives a reboot.

### Redeploying after a change

- **Backend change**: `./deploy/push_to_ecr.sh` then `./deploy/deploy_lambda.sh`.
- **Frontend change**: `./deploy/provision_frontend.sh <ip> ~/medreport-key.pem`
  (deploying the backend does not touch the frontend).

### Operating the deployed app

```bash
# restart the frontend
ssh -i ~/medreport-key.pem ec2-user@<ip> 'sudo systemctl restart medreport-frontend.service'

# check status and logs
ssh -i ~/medreport-key.pem ec2-user@<ip> \
  'sudo systemctl status medreport-frontend.service --no-pager && sudo tail -30 /var/log/medreport-frontend.log'

# check the backend is alive (also warms it -- see below)
curl -s <function-url>api/v1/health
```

**Cold starts.** The first backend request after ~5-15 minutes idle takes
roughly 56 seconds, because a 3.42GB image has to load the embedding model
and FAISS index. Warm requests are ~0.1s. Hit `/api/v1/health` before any
demonstration so the first real upload does not absorb that delay. The
frontend has no cold start -- only the page's first call to the backend does.

### Deployment gotchas worth knowing

These all cost real debugging time on the first live deploy, and none of
them can surface in local testing:

- **Lambda rejects OCI image manifests.** Current Docker builds emit an OCI
  index with provenance attestations by default; Lambda requires Docker
  Manifest V2 Schema 2. `push_to_ecr.sh` therefore builds with
  `--provenance=false --sbom=false --output type=image,oci-mediatypes=false`.
- **Architecture must match.** Building on Apple Silicon produces an arm64
  image, while Lambda defaults to x86_64, so `template.yaml` declares
  `Architectures: [arm64]` explicitly.
- **`ssm-secure` does not work in Lambda environment variables.**
  CloudFormation supports that dynamic reference in some resources but not
  this one, so the parameters are plain `String` and resolved with
  `{{resolve:ssm:...}}`.
- **Filename casing matters on Linux.** macOS is case-insensitive, so
  `App.py` and `app.py` resolve identically there and diverge on EC2.
- **macOS `mktemp` does not randomise** a template when a suffix follows the
  `X`s, so the deploy scripts use `$$` instead.

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
      llm_service.py        Multi-provider LLM calls (RAG-grounded prompts)
      pdf_generator.py      Builds the downloadable PDF report
    db/                     DEAD CODE -- superseded by the ethics-driven
                            removal of server-side storage; imported by
                            nothing. Kept only as a record of the original
                            design.
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
  build_rag_index.py         Builds the FAISS index from rag_chunks.json

deploy/
  push_to_ecr.sh             Build the Lambda image and push it to ECR
  create_ssm_params.sh       Load .env keys into SSM Parameter Store
  deploy_lambda.sh           Deploy/update the Lambda stack via SAM
  launch_ec2.sh              Launch the frontend EC2 instance
  provision_frontend.sh      Copy + start the frontend over SSH
  ec2_userdata.sh            First-boot script used by launch_ec2.sh

Dockerfile.aws               Lambda container image (AWS)
Dockerfile.gcp               Cloud Run image (GCP comparison build)
template.yaml                AWS SAM infrastructure-as-code
```
