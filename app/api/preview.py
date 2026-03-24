from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import PreviewRequest, PreviewResponse
from app.security import require_api_key
from app.services.matcher import generate_preview

router = APIRouter(prefix="/v1/estimate", tags=["estimate-preview"])


@router.post("/preview", response_model=PreviewResponse)
def estimate_preview(
    payload: PreviewRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> PreviewResponse:
    return generate_preview(payload)
