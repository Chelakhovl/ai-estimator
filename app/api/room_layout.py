from __future__ import annotations

from fastapi import APIRouter, Depends

from app.security import require_api_key
from app.services.room_layout_service import (
    RoomLayoutRequest,
    RoomLayoutResponse,
    RoomLayoutService,
)

router = APIRouter(prefix="/v1/room-layout", tags=["room-layout"])

_service = RoomLayoutService()


@router.post("/suggest", response_model=RoomLayoutResponse, dependencies=[Depends(require_api_key)])
def suggest_room_layout(request: RoomLayoutRequest) -> RoomLayoutResponse:
    """Given a scanned room's fixed geometry and its current furniture/fixtures, suggest alternative
    placements for those same objects. Never invents or removes objects — only repositions what's
    already there. Falls back to a mock (no-suggestion) response if the LLM isn't configured or fails,
    same convention as every other AI endpoint in this service."""
    return _service.suggest(request)
