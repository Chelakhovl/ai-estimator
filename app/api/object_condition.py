from __future__ import annotations

from fastapi import APIRouter, Depends

from app.security import require_api_key
from app.services.object_condition_service import (
    ObjectConditionRequest,
    ObjectConditionResponse,
    ObjectConditionService,
)

router = APIRouter(prefix="/v1/object-condition", tags=["object-condition"])

_service = ObjectConditionService()


@router.post("/assess", response_model=ObjectConditionResponse, dependencies=[Depends(require_api_key)])
def assess_object_condition(request: ObjectConditionRequest) -> ObjectConditionResponse:
    """Assesses the visible condition of one furniture/fixture item from a camera frame captured
    mid-scan (RefurbwayRoomScan, iOS) — called once per newly detected object, in the background,
    never blocking the scan itself. Falls back to mode: "mock" (condition: "unknown") when the LLM
    isn't configured or the call fails, same convention as every other AI endpoint in this service."""
    return _service.assess(request)
