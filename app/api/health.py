from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", service="combit-estimator-ai-service", mode=settings.preview_mode)
