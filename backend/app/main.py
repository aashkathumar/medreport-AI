from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from app.api.routes import router

app = FastAPI(title="MedReport AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Lambda handler -- Mangum wraps FastAPI for AWS Lambda.
# Not used when running locally with `uvicorn app.main:app`.
handler = Mangum(app, lifespan="off")
