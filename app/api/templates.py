from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import PromptTemplatesResponse
from app.security import require_api_key
from app.services.prompt_templates import get_prompt_templates

router = APIRouter(prefix="/v1/estimate", tags=["estimate-templates"])


@router.get("/prompt-templates", response_model=PromptTemplatesResponse)
def prompt_templates(
    _authorized: Annotated[None, Depends(require_api_key)],
) -> PromptTemplatesResponse:
    return PromptTemplatesResponse(templates=get_prompt_templates())
