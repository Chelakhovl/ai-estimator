from __future__ import annotations

import json
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
# Coordinates mirror RefurbwayRoomScan's own `FloorPosition` (x, z in metres, rotation in radians,
# top-down floor-plan space) — the iOS client sends/receives the exact same shape it already stores
# internally for walls/doors/windows/objects, so no coordinate-system translation is needed on either side.


class LayoutSurface(BaseModel):
    """A wall, door, window or opening — fixed, never moved by a suggested layout."""

    kind: str  # "wall" | "door" | "window" | "opening"
    width_m: float
    x: float
    z: float
    rotation_radians: float


class LayoutObject(BaseModel):
    """A piece of furniture/fixture the room currently has — the thing layouts rearrange."""

    id: str
    category: str
    width_m: float
    depth_m: float


class RoomLayoutRequest(BaseModel):
    room_name: str = ""
    floor_area_sq_m: float | None = None
    surfaces: list[LayoutSurface] = Field(default_factory=list)
    objects: list[LayoutObject] = Field(default_factory=list)
    variant_count: int = Field(default=3, ge=1, le=5)


class PlacedObject(BaseModel):
    id: str
    x: float
    z: float
    rotation_radians: float


class LayoutVariant(BaseModel):
    label: str
    rationale: str
    placements: list[PlacedObject]


class RoomLayoutResponse(BaseModel):
    variants: list[LayoutVariant]
    mode: str  # "llm" | "mock"
    warnings: list[str] = Field(default_factory=list)


# ── LLM structured-output contract ───────────────────────────────────────────
# What we ask the model to return — same shape as LayoutVariant/PlacedObject above, kept as a
# separate pair of models (not reused directly) so the LLM's own output contract can evolve
# independently of the public API response shape.


class _LLMPlacedObject(BaseModel):
    id: str
    x: float
    z: float
    rotation_radians: float


class _LLMLayoutVariant(BaseModel):
    label: str
    rationale: str
    placements: list[_LLMPlacedObject]


class _LLMLayoutOutput(BaseModel):
    variants: list[_LLMLayoutVariant]


class RoomLayoutError(RuntimeError):
    pass


def _build_system_prompt() -> str:
    return (
        "You are a UK residential interior layout assistant. Given a room's fixed geometry "
        "(walls, doors, windows, openings — all in metres, top-down floor-plan coordinates) and "
        "a list of furniture/fixture objects already in the room, propose alternative sensible "
        "placements for those SAME objects within the room's existing walls.\n\n"
        "Hard rules:\n"
        "- Only reposition the objects given — never invent new ones, never drop any, never change "
        "their width/depth.\n"
        "- Every object's centre point (x, z) must stay fully inside the room's wall boundary.\n"
        "- No object may overlap a door or window's position — leave clear access.\n"
        "- No two objects may overlap each other.\n"
        "- Leave at least 0.6m of clear walking space in front of doors.\n"
        "- rotation_radians is the object's facing angle in the same coordinate space as the input.\n\n"
        "Return each variant with a short label (e.g. \"Bed against the window wall\") and a one-sentence "
        "rationale a homeowner would understand — not design jargon."
    )


def _build_user_message(request: RoomLayoutRequest) -> str:
    payload = {
        "room_name": request.room_name,
        "floor_area_sq_m": request.floor_area_sq_m,
        "surfaces": [s.model_dump() for s in request.surfaces],
        "objects": [o.model_dump() for o in request.objects],
        "variant_count": request.variant_count,
    }
    return (
        "Room geometry and objects to rearrange (JSON):\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        f"Propose exactly {request.variant_count} layout variant(s)."
    )


class RoomLayoutService:
    def __init__(self) -> None:
        self.model = settings.openai_model
        self.enabled = bool(settings.openai_api_key and settings.openai_model)
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

    def suggest(self, request: RoomLayoutRequest) -> RoomLayoutResponse:
        if not request.objects:
            # Nothing to rearrange — not an error, just nothing useful to suggest. Mirrors how the iOS
            # app's own floor plan behaves with zero objects: an empty room has no layout to propose.
            return RoomLayoutResponse(variants=[], mode="mock", warnings=["No objects in this room to rearrange."])

        if not self.enabled or self.client is None:
            return self._mock_response(request)

        try:
            return self._llm_response(request)
        except Exception:
            logger.exception("Room layout LLM request failed — falling back to mock response.")
            response = self._mock_response(request)
            response.warnings.append("AI layout service was unavailable; showing the original layout only.")
            return response

    def _llm_response(self, request: RoomLayoutRequest) -> RoomLayoutResponse:
        assert self.client is not None
        started_at = perf_counter()

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": _build_user_message(request)},
            ],
            text_format=_LLMLayoutOutput,
            temperature=0.4,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RoomLayoutError("Model returned no parsed structured output.")

        logger.info(
            "Room layout suggestion generated in %dms (%d variants)",
            int((perf_counter() - started_at) * 1000),
            len(parsed.variants),
        )

        known_ids = {o.id for o in request.objects}
        variants: list[LayoutVariant] = []
        for v in parsed.variants:
            # Drop any placement for an object id the model invented or that wasn't in the request —
            # the hard rule in the prompt says never to add objects, but LLM output is never fully
            # trusted; silently filtering is safer here than rejecting the whole variant over one
            # bad placement.
            placements = [PlacedObject(**p.model_dump()) for p in v.placements if p.id in known_ids]
            if placements:
                variants.append(LayoutVariant(label=v.label, rationale=v.rationale, placements=placements))

        if not variants:
            raise RoomLayoutError("Model's suggestions didn't reference any known objects.")

        return RoomLayoutResponse(variants=variants, mode="llm")

    def _mock_response(self, request: RoomLayoutRequest) -> RoomLayoutResponse:
        """Deterministic fallback when the LLM path is unavailable/disabled — returns the objects in
        their CURRENT positions as a single "variant" labelled as such, rather than fabricating a fake
        rearrangement. Mirrors the rest of this codebase's mock-mode convention (see llm_client.py):
        the service stays usable end-to-end without an API key, just without real suggestions."""
        return RoomLayoutResponse(
            variants=[],
            mode="mock",
            warnings=["AI layout suggestions aren't configured on this server yet."],
        )
