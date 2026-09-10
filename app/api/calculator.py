from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas import (
    CalculatorExplainRequest,
    CalculatorExplainResponse,
    CalculatorParseRequest,
    CalculatorParseResponse,
)
from app.security import require_api_key
from app.services.calculator_ai import explain_estimate, parse_project

router = APIRouter(prefix="/v1/calculator", tags=["public-calculator"])


@router.post("/parse-project", response_model=CalculatorParseResponse)
def calculator_parse_project(
    payload: CalculatorParseRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> CalculatorParseResponse:
    """Free-text project description -> the wizard's own fields (client reviews the pre-fill)."""
    return parse_project(payload.text)


@router.post("/explain-estimate", response_model=CalculatorExplainResponse)
def calculator_explain_estimate(
    payload: CalculatorExplainRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> CalculatorExplainResponse:
    """Category-level explanation of the ballpark figure. No £-per-item, no work breakdown."""
    return explain_estimate(payload)
