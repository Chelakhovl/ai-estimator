from __future__ import annotations

from fastapi import APIRouter, Depends

from app.security import require_api_key
from app.services.room_redesign_service import (
    RoomRedesignRequest,
    RoomRedesignResponse,
    RoomRedesignService,
)

router = APIRouter(prefix="/v1/room-redesign", tags=["room-redesign"])

_service = RoomRedesignService()


@router.post("/generate", response_model=RoomRedesignResponse, dependencies=[Depends(require_api_key)])
def generate_room_redesign(request: RoomRedesignRequest) -> RoomRedesignResponse:
    """Given a photo of a scanned room and a style, generates a redesigned version of that same photo
    (same walls/windows/perspective, new furniture/decor/colours). Unlike every other AI endpoint in
    this service, this one generates an image rather than structured text — see
    app/services/room_redesign_service.py for why. Falls back to a mock (no-image) response if the LLM
    isn't configured or fails, same convention as every other AI endpoint in this service."""
    return _service.generate(request)
