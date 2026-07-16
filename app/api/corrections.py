from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import BatchCorrectionsRequest, BatchCorrectionsResponse
from app.security import require_api_key
from app.services.corrections_parser import parse_batch_corrections

router = APIRouter(prefix="/v1/corrections", tags=["batch-corrections"])


@router.post("/parse", response_model=BatchCorrectionsResponse)
def parse_corrections(
    payload: BatchCorrectionsRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> BatchCorrectionsResponse:
    return parse_batch_corrections(payload)
