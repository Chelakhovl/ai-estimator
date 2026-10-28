from __future__ import annotations

import asyncio
import base64
import io
import logging
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from app.config import settings

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Request/response shapes ──────────────────────────────────────────────────
#
# Unlike room_layout_service.py / renovation_recommendations_service.py, this service generates an
# IMAGE, not structured text — it calls `client.images.edit` (gpt-image-2.5-sunburst), not
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
    # 1-3 JPEGs of the room (different angles from a single scan), repr=False keeps them out of
    # error/log lines.
    images_base64: list[str] = Field(repr=False)
    # Same length as images_base64, one slot per image. Each entry is a PNG the same pixel dimensions
    # as the matching image, opaque over every wall/door/window/opening and transparent elsewhere (see
    # the iOS client's RoomPhotoCapture.structureMaskPNGData) — OpenAI's own images.edit convention is
    # the inverse of what "mask" usually implies: the TRANSPARENT regions are what gets edited, the
    # opaque ones are protected. An entry is None when the client couldn't build a mask for that frame
    # (older app version, or that frame's camera pose put no structural surface in view) — _llm_response
    # falls back to prompt-only geometry preservation for that one frame, same behaviour this had before
    # per-frame masks existed.
    masks_base64: list[str | None] = Field(default_factory=list, repr=False)

    @model_validator(mode="after")
    def _pad_masks_to_match_images(self) -> "RoomRedesignRequest":
        # The iOS client always sends masks_base64 at the same length as images_base64 (see
        # RoomRedesignClient.swift), but pad defensively rather than raising — an older/misbehaving
        # client sending a short or absent masks_base64 should just lose mask protection on the
        # trailing frames, not fail the whole request.
        if len(self.masks_base64) < len(self.images_base64):
            self.masks_base64 = self.masks_base64 + [None] * (
                len(self.images_base64) - len(self.masks_base64)
            )
        return self


class RoomRedesignResponse(BaseModel):
    images_base64: list[str] = Field(default_factory=list)
    mode: str  # "llm" | "mock"
    warnings: list[str] = Field(default_factory=list)


class RoomRedesignError(RuntimeError):
    pass


class RoomRedesignService:
    def __init__(self) -> None:
        # gpt-image-2.5-sunburst is OpenAI's current image generation/edit model — separate from
        # settings.openai_model (used for text/structured-output calls elsewhere in this service),
        # since it's a different capability, not a "stronger" or "weaker" version of the same thing.
        # gpt-image-1 (the original choice here) was shut down by OpenAI on 2026-10-23; every call
        # since then failed with "openai.BadRequestError: ... The model 'gpt-image-1' does not exist"
        # and silently fell back to the mock response via the except-block below, so the feature looked
        # "available" (200 OK, no visible error) but never actually generated a real redesign.
        self.model = "gpt-image-2.5-sunburst"
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
                images_base64=[],
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

        # Each frame's images.edit call is an independent, blocking HTTP request — fan them out with
        # asyncio.to_thread/gather rather than looping sequentially, since a 3-frame scan would
        # otherwise take 3x as long for no reason. asyncio.run() is used (rather than making this
        # method itself async) because the FastAPI route above calls generate()/_llm_response()
        # synchronously; nothing else in this codebase has an async call chain yet, so this keeps the
        # concurrency contained to this one method instead of threading "async" through the route.
        results = asyncio.run(self._edit_all_frames(request, prompt))

        images_base64: list[str] = []
        warnings: list[str] = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.exception(
                    "Room redesign frame %d/%d failed (style=%s); skipping it.",
                    index + 1,
                    len(results),
                    request.style,
                    exc_info=result,
                )
                continue
            images_base64.append(result)

        if not images_base64:
            raise RoomRedesignError("Model returned no image data for any frame.")

        logger.info(
            "Room redesign generated %d/%d frame(s) in %dms (style=%s)",
            len(images_base64),
            len(results),
            int((perf_counter() - started_at) * 1000),
            request.style,
        )

        return RoomRedesignResponse(images_base64=images_base64, mode="llm", warnings=warnings)

    async def _edit_all_frames(
        self, request: RoomRedesignRequest, prompt: str
    ) -> list[str | BaseException]:
        """Runs one images.edit call per (image, mask) pair in parallel via worker threads — the
        OpenAI SDK's images.edit is a blocking call, so asyncio.to_thread is what lets these overlap.
        return_exceptions=True so one frame's failure doesn't cancel the others still in flight."""
        tasks = [
            asyncio.to_thread(self._edit_one_frame, prompt, image_b64, mask_b64)
            for image_b64, mask_b64 in zip(request.images_base64, request.masks_base64)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _edit_one_frame(self, prompt: str, image_base64: str, mask_base64: str | None) -> str:
        assert self.client is not None

        image_bytes = base64.b64decode(image_base64)
        image_file = io.BytesIO(image_bytes)
        image_file.name = "room.jpg"  # the SDK reads this to infer the upload's content type

        edit_kwargs: dict = {
            "model": self.model,
            "image": image_file,
            "prompt": prompt,
            # "auto" (not a fixed "1024x1024") — a room photo from the iOS client's camera is a
            # non-square 4:3-ish portrait, and forcing a square output would conflict with the mask,
            # which OpenAI requires to be the EXACT same pixel dimensions as the input image.
            "size": "auto",
            "quality": "medium",
        }
        # With a mask, "keep the structure unchanged" becomes an API-enforced constraint on the opaque
        # pixels rather than only a prompt instruction the model can (and does, in practice) partially
        # ignore — walls drifting, a window moving, the perspective subtly shifting. Without one, this
        # call is unchanged from before masks existed.
        if mask_base64:
            mask_bytes = base64.b64decode(mask_base64)
            mask_file = io.BytesIO(mask_bytes)
            mask_file.name = "mask.png"
            edit_kwargs["mask"] = mask_file

        response = self.client.images.edit(**edit_kwargs)

        data = response.data or []
        if not data or not data[0].b64_json:
            raise RoomRedesignError("Model returned no image data.")

        return data[0].b64_json

    def _mock_response(self) -> RoomRedesignResponse:
        """Deterministic fallback when the LLM path is unavailable/disabled — same mock-mode convention
        as the rest of this codebase: stays usable end-to-end without an API key, just without a real
        generated image."""
        return RoomRedesignResponse(
            images_base64=[],
            mode="mock",
            warnings=["AI room redesign isn't configured on this server yet."],
        )
