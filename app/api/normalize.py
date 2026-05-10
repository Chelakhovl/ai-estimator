from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import PromptNormalizationRequest, PromptNormalizationResponse
from app.security import require_api_key
from app.services.intake_normalizer import normalize_intake_prompt

router = APIRouter(prefix="/v1/estimate", tags=["estimate-normalize"])


@router.post("/normalize", response_model=PromptNormalizationResponse)
def normalize_prompt_intake(
    payload: PromptNormalizationRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> PromptNormalizationResponse:
    return normalize_intake_prompt(payload)
