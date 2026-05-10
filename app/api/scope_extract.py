from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import ScopeExtractionRequest, ScopeExtractionResponse
from app.security import require_api_key
from app.services.scope_extractor import extract_scope

router = APIRouter(prefix="/v1/estimate", tags=["scope-extraction"])


@router.post("/extract-scope", response_model=ScopeExtractionResponse)
def extract_structured_scope(
    payload: ScopeExtractionRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> ScopeExtractionResponse:
    return extract_scope(payload.prompt)
