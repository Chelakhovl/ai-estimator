from __future__ import annotations

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.preview import router as preview_router

app = FastAPI(
    title="Combit Estimator AI Service",
    version="0.1.0",
    description="AI preview service for Combit Estimator Power Platform integration.",
)
app.include_router(health_router)
app.include_router(preview_router)
