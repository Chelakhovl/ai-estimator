from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import CustomWorkChatRequest, CustomWorkChatResponse
from app.security import require_api_key
from app.services.custom_work_service import chat_custom_work

router = APIRouter(prefix="/v1/estimate", tags=["estimate-custom-work"])


@router.post("/custom-work/chat", response_model=CustomWorkChatResponse)
def custom_work_chat(
    payload: CustomWorkChatRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> CustomWorkChatResponse:
    return chat_custom_work(payload)
