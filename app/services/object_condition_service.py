from __future__ import annotations

import json
import logging
from time import perf_counter

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# ── Request/response shapes ──────────────────────────────────────────────────


class ObjectConditionRequest(BaseModel):
    """One furniture/fixture RoomPlan just detected mid-scan, plus a camera frame grabbed at the
    moment of detection (see RefurbwayRoomScan's ARSession.currentFrame read — not a delegate hook,
    a plain property read, so this never touches the AR session RoomPlan itself owns)."""

    category: str  # RoomPlan's own category, e.g. "Stove", "Bathtub" — see DetectedObject.category
    width_m: float
    height_m: float
    depth_m: float
    room_name: str = ""
    image_base64: str  # JPEG, captured at the moment this object was recognised


class ObjectConditionResponse(BaseModel):
    category: str  # echoed back so the iOS client can match this response to the right object
    condition: str  # "good" | "worn" | "needs_replacement" | "unknown"
    condition_notes: str = ""
    renovation_flags: list[str] = Field(default_factory=list)
    mode: str  # "llm" | "mock"


_JSON_SCHEMA_HINT = """\
Return ONLY valid JSON, no markdown fences:
{
  "condition": "<one of: good | worn | needs_replacement | unknown>",
  "condition_notes": "<one short sentence a homeowner would understand, visible wear/damage if any>",
  "renovation_flags": ["<short practical note, e.g. 'check for leaks around base' — 0 to 3 items>"]
}
"""


def _build_prompt(request: ObjectConditionRequest) -> str:
    return (
        "You are assessing a single piece of furniture or a fixture in a UK home, photographed during "
        f"a room scan. RoomPlan identified it as: \"{request.category}\" "
        f"(roughly {request.width_m:.1f}m × {request.height_m:.1f}m × {request.depth_m:.1f}m)"
        + (f" in the {request.room_name}" if request.room_name else "")
        + ".\n\n"
        "Look at the photo and judge its visible condition for someone planning a renovation — not "
        "a design critique, just practical wear/damage/age signs actually visible in the image. If the "
        "photo is too blurry, too dark, or the object isn't clearly visible, say condition \"unknown\" "
        "rather than guessing.\n\n" + _JSON_SCHEMA_HINT
    )


class ObjectConditionService:
    def __init__(self) -> None:
        self.enabled = bool(settings.openai_api_key)
        self.model = settings.openai_intake_fast_model or "gpt-4o-mini"

    def is_enabled(self) -> bool:
        return self.enabled

    def assess(self, request: ObjectConditionRequest) -> ObjectConditionResponse:
        if not self.enabled:
            return self._mock_response(request)
        try:
            return self._llm_response(request)
        except Exception:
            logger.exception(
                "Object condition assessment failed for category=%s — falling back to mock.",
                request.category,
            )
            return self._mock_response(request)

    def _llm_response(self, request: ObjectConditionRequest) -> ObjectConditionResponse:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=3,
        )

        started_at = perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(request)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request.image_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        logger.info(
            "Object condition assessed: category=%s model=%s latency=%dms",
            request.category,
            self.model,
            latency_ms,
        )

        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {}

        condition = data.get("condition", "unknown")
        if condition not in {"good", "worn", "needs_replacement", "unknown"}:
            condition = "unknown"

        flags_raw = data.get("renovation_flags", [])
        flags = [str(f).strip() for f in flags_raw if isinstance(f, (str, int, float))][:3]
        flags = [f for f in flags if f]

        return ObjectConditionResponse(
            category=request.category,
            condition=condition,
            condition_notes=str(data.get("condition_notes", "")).strip(),
            renovation_flags=flags,
            mode="llm",
        )

    def _mock_response(self, request: ObjectConditionRequest) -> ObjectConditionResponse:
        return ObjectConditionResponse(
            category=request.category,
            condition="unknown",
            condition_notes="AI condition assessment isn't configured on this server yet.",
            renovation_flags=[],
            mode="mock",
        )
