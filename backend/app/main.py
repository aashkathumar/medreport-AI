import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from app.api.routes import router
from app.services.rag_service import index_health

app = FastAPI(title="MedReport AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def _warm_rag_index() -> None:
    """Loads the FAISS index + SentenceTransformer in the background at boot.

    Otherwise the FIRST upload of the process pays ~1.7s of model/index load
    on top of everything else, on the request path. Runs in a daemon thread so
    a slow (or failing) load never blocks the server from accepting traffic --
    index_health() already records the failure reason for /rag/health.
    """
    def _warm() -> None:
        try:
            health = index_health()
            print(f"RAG warm-up: available={health['available']} "
                  f"chunks={health['chunk_count']} error={health['error']}")
        except Exception as e:  # never take the server down over a warm-up
            print(f"RAG warm-up failed: {e}")

    threading.Thread(target=_warm, name="rag-warmup", daemon=True).start()

# Lambda handler, Mangum wraps FastAPI for AWS Lambda.
# Not used when running locally with `uvicorn app.main:app`.
handler = Mangum(app, lifespan="off")
