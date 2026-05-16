from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.document_analysis import router as document_analysis_router
from app.api.health import router as health_router
from app.api.normalize import router as normalize_router
from app.api.preview import router as preview_router
from app.api.project_intake import router as project_intake_router
from app.api.scope_extract import router as scope_extract_router
from app.api.templates import router as templates_router
from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    is_local = settings.app_env.lower() in {"local", "test"}

    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set — all AI endpoints will run in mock/degraded mode."
        )
    if not settings.openai_model:
        logger.warning(
            "OPENAI_MODEL is not set — preview endpoint will use mock mode."
        )
    if not settings.service_api_key:
        if is_local:
            logger.warning(
                "SERVICE_API_KEY is not set — API authentication is disabled for all routes."
            )
        else:
            raise RuntimeError(
                "SERVICE_API_KEY must be set in non-local environments. "
                "Set APP_ENV=local to run without authentication."
            )
    yield


app = FastAPI(
    title="Combit Estimator AI Service",
    version="0.1.0",
    description="AI preview service for Combit Estimator Power Platform integration.",
    lifespan=lifespan,
)
app.include_router(document_analysis_router)
app.include_router(health_router)
app.include_router(normalize_router)
app.include_router(preview_router)
app.include_router(project_intake_router)
app.include_router(scope_extract_router)
app.include_router(templates_router)
