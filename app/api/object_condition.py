from __future__ import annotations

from fastapi import APIRouter, Depends

from app.security import require_api_key
from app.services.object_condition_service import (
    ObjectConditionBatchRequest,
    ObjectConditionBatchResponse,
    ObjectConditionRequest,
    ObjectConditionResponse,
    ObjectConditionService,
)

router = APIRouter(prefix="/v1/object-condition", tags=["object-condition"])

_service = ObjectConditionService()


@router.post("/assess", response_model=ObjectConditionResponse, dependencies=[Depends(require_api_key)])
def assess_object_condition(request: ObjectConditionRequest) -> ObjectConditionResponse:
    """Assesses the visible condition of a single furniture/fixture item. Prefer `/assess-batch`
    when assessing several objects from the same scan — this single-item path internally calls the
    same batch logic with one item, so it costs the same per-call as before, just without the
    cross-object savings batching gives you. Falls back to mode: "mock" when the LLM isn't configured
    or the call fails, same convention as every other AI endpoint in this service."""
    return _service.assess(request)


@router.post(
    "/assess-batch", response_model=ObjectConditionBatchResponse, dependencies=[Depends(require_api_key)]
)
def assess_object_condition_batch(request: ObjectConditionBatchRequest) -> ObjectConditionBatchResponse:
    """Assesses several furniture/fixture items from the same scan in ONE vision call — the preferred
    path for RefurbwayRoomScan (iOS), which batches objects recognised within a short window during
    scanning rather than firing one request per object. Also deduplicates items whose category and
    dimensions closely match (e.g. two near-identical dining chairs) so visually-similar items share
    one assessment instead of paying for a vision call each. Capped at 12 items per request."""
    return _service.assess_batch(request)
