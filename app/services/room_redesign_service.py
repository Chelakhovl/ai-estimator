from __future__ import annotations

import base64
import io
import logging
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.config import settings

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Request/response shapes ──────────────────────────────────────────────────
#
# Unlike room_layout_service.py / renovation_recommendations_service.py, this service generates an
# IMAGE, not structured text — it calls `client.images.edit` (gpt-image-1), not
# `client.responses.parse`. There is no object-id filtering step here because the output isn't JSON
# with ids to validate; the only thing to check is that a usable image actually came back.

_STYLE_PROMPTS: dict[str, str] = {
    "scandinavian": (
        "Redesign this room in a Scandinavian interior style: light wood tones, white and soft "
        "neutral walls, minimal clutter, cozy textiles, plenty of natural light."
    ),
    "modern": (
        "Redesign this room in a modern interior style: clean lines, a neutral palette with one bold "
        "accent colour, sleek furniture, minimal ornamentation."
    ),
    "traditional": (
        "Redesign this room in a traditional interior style: warm wood tones, classic furniture "
        "shapes, rich fabrics, a timeless and homely feel."
    ),
    "minimalist": (
        "Redesign this room in a minimalist interior style: very few items, a monochrome or near-"
        "monochrome palette, open floor space, nothing decorative beyond what's functional."
    ),
}

_SHARED_INSTRUCTIONS = (
    " Keep the room's actual structure unchanged — the same walls, windows, doors and camera "
    "perspective as the original photo. Only change the furniture, decor, colours and finishes."
)


class RoomRedesignRequest(BaseModel):
    room_name: str = ""
    style: str
    image_base64: str = Field(repr=False)  # JPEG of the room, repr=False keeps it out of error/log lines


class RoomRedesignResponse(BaseModel):
    image_base64: str | None = None
    mode: str  # "llm" | "mock"
    warnings: list[str] = Field(default_factory=list)


class RoomRedesignError(RuntimeError):
    pass


class RoomRedesignService:
    def __init__(self) -> None:
        # gpt-image-1 is OpenAI's current image generation/edit model — separate from
        # settings.openai_model (used for text/structured-output calls elsewhere in this service),
        # since it's a different capability, not a "stronger" or "weaker" version of the same thing.
        self.model = "gpt-image-1"
        self.enabled = bool(settings.openai_api_key)
        self.client: OpenAI | None = None
        if self.enabled:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=3,
            )

    def is_enabled(self) -> bool:
        return self.enabled

    def generate(self, request: RoomRedesignRequest) -> RoomRedesignResponse:
        if request.style not in _STYLE_PROMPTS:
            # Validated here, not just left to the client — a client bug or a tampered request
            # shouldn't be able to send an arbitrary string into the image-edit prompt.
            return RoomRedesignResponse(
                image_base64=None,
                mode="mock",
                warnings=[f"Unknown style '{request.style}'."],
            )

        if not self.enabled or self.client is None:
            return self._mock_response()

        try:
            return self._llm_response(request)
        except Exception:
            logger.exception("Room redesign image generation failed — falling back to mock response.")
            response = self._mock_response()
            response.warnings.append("AI redesign service was unavailable; try again later.")
            return response

    def _llm_response(self, request: RoomRedesignRequest) -> RoomRedesignResponse:
        assert self.client is not None
        started_at = perf_counter()

        prompt = _STYLE_PROMPTS[request.style] + _SHARED_INSTRUCTIONS
        image_bytes = base64.b64decode(request.image_base64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = "room.jpg"  # the SDK reads this to infer the upload's content type

        response = self.client.images.edit(
            model=self.model,
            image=image_file,
            prompt=prompt,
            size="1024x1024",
            quality="medium",
        )

        data = response.data or []
        if not data or not data[0].b64_json:
            raise RoomRedesignError("Model returned no image data.")

        logger.info(
            "Room redesign generated in %dms (style=%s)",
            int((perf_counter() - started_at) * 1000),
            request.style,
        )

        return RoomRedesignResponse(image_base64=data[0].b64_json, mode="llm")

    def _mock_response(self) -> RoomRedesignResponse:
        """Deterministic fallback when the LLM path is unavailable/disabled — same mock-mode convention
        as the rest of this codebase: stays usable end-to-end without an API key, just without a real
        generated image."""
        return RoomRedesignResponse(
            image_base64=None,
            mode="mock",
            warnings=["AI room redesign isn't configured on this server yet."],
        )
