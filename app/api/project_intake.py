from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import (
    IntakeChatRequest,
    IntakeChatResponse,
    IntakeFinalizeRequest,
    IntakeFinalizeResponse,
)
from app.security import require_api_key
from app.services.project_intake_service import chat_project_intake, finalize_project_intake

router = APIRouter(prefix="/v1/project-intake", tags=["project-intake"])


@router.post("/chat", response_model=IntakeChatResponse)
def project_intake_chat(
    payload: IntakeChatRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> IntakeChatResponse:
    return chat_project_intake(payload)


@router.post("/finalize", response_model=IntakeFinalizeResponse)
def project_intake_finalize(
    payload: IntakeFinalizeRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> IntakeFinalizeResponse:
    return finalize_project_intake(payload)
