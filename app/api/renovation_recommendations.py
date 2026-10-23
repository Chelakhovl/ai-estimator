from __future__ import annotations

from fastapi import APIRouter, Depends

from app.security import require_api_key
from app.services.renovation_recommendations_service import (
    RenovationRecommendationsRequest,
    RenovationRecommendationsResponse,
    RenovationRecommendationsService,
)

router = APIRouter(prefix="/v1/renovation-recommendations", tags=["renovation-recommendations"])

_service = RenovationRecommendationsService()


@router.post(
    "/suggest", response_model=RenovationRecommendationsResponse, dependencies=[Depends(require_api_key)]
)
def suggest_renovation_recommendations(
    request: RenovationRecommendationsRequest,
) -> RenovationRecommendationsResponse:
    """Given a scanned room's objects (with optional condition assessments) and its existing scope
    items, suggests prioritised renovation recommendations. Falls back to a mock (no-recommendation)
    response if the LLM isn't configured or fails, same convention as every other AI endpoint in this
    service."""
    return _service.suggest(request)
