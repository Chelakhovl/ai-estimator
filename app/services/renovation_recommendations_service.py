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
# Mirrors RefurbwayRoomScan's ScannedRoom/DetectedObject shape — same convention as
# room_layout_service.py, no coordinate-system or vocabulary translation needed on the iOS side.


class RecommendationObject(BaseModel):
    """A piece of furniture/fixture already assessed by the object-condition endpoint."""

    id: str
    category: str
    condition: str | None = None  # "good" | "worn" | "needs_replacement" | None (not yet assessed)
    condition_notes: str | None = None


class RenovationRecommendationsRequest(BaseModel):
    room_name: str = ""
    floor_area_sq_m: float | None = None
    objects: list[RecommendationObject] = Field(default_factory=list)
    existing_scope_items: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    title: str
    rationale: str
    priority: str  # "high" | "medium" | "low"
    related_object_id: str | None = None


class RenovationRecommendationsResponse(BaseModel):
    recommendations: list[Recommendation]
    mode: str  # "llm" | "mock"
    warnings: list[str] = Field(default_factory=list)


# ── LLM structured-output contract ───────────────────────────────────────────


class _LLMRecommendation(BaseModel):
    title: str
    rationale: str
    priority: str
    related_object_id: str | None = None


class _LLMRecommendationsOutput(BaseModel):
    recommendations: list[_LLMRecommendation]


class RenovationRecommendationsError(RuntimeError):
    pass

_VALID_PRIORITIES = {"high", "medium", "low"}


def _build_system_prompt() -> str:
    return (
        "You are a UK residential renovation advisor. Given a scanned room's furniture/fixture "
        "objects (each with an optional condition assessment) and the room's existing renovation "
        "scope items, propose prioritised renovation recommendations a homeowner would find useful.\n\n"
        "Hard rules:\n"
        "- priority must be exactly one of: \"high\", \"medium\", \"low\".\n"
        "- Objects already flagged condition=\"needs_replacement\" are the strongest signal — "
        "recommendations tied to them should usually be \"high\" priority.\n"
        "- Objects flagged condition=\"worn\" are a weaker signal — usually \"medium\" or \"low\".\n"
        "- Don't repeat a recommendation that duplicates an existing scope item already listed.\n"
        "- related_object_id must be an id from the objects list, or omitted for room-wide/cosmetic "
        "recommendations that aren't tied to one specific item.\n"
        "- Write for a homeowner, not a contractor — plain language, no trade jargon.\n"
        "- Keep each rationale to one sentence."
    )


def _build_user_message(request: RenovationRecommendationsRequest) -> str:
    payload = {
        "room_name": request.room_name,
        "floor_area_sq_m": request.floor_area_sq_m,
        "objects": [o.model_dump() for o in request.objects],
        "existing_scope_items": request.existing_scope_items,
    }
    return (
        "Room and object data (JSON):\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Propose renovation recommendations, prioritised by what matters most first."
    )


class RenovationRecommendationsService:
    def __init__(self) -> None:
        # Unlike room-layout's simple spatial repositioning, this is an analytical/advisory task closer
        # to the main estimation pipeline — uses the stronger reserved model, not the cheaper fast one.
        self.model = settings.openai_model
        self.enabled = bool(settings.openai_api_key and self.model)
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

    def suggest(self, request: RenovationRecommendationsRequest) -> RenovationRecommendationsResponse:
        if not request.objects:
            return RenovationRecommendationsResponse(
                recommendations=[],
                mode="mock",
                warnings=["No objects in this room to base recommendations on."],
            )

        if not self.enabled or self.client is None:
            return self._mock_response()

        try:
            return self._llm_response(request)
        except Exception:
            logger.exception("Renovation recommendations LLM request failed — falling back to mock response.")
            response = self._mock_response()
            response.warnings.append("AI recommendations service was unavailable; try again later.")
            return response

    def _llm_response(self, request: RenovationRecommendationsRequest) -> RenovationRecommendationsResponse:
        assert self.client is not None
        started_at = perf_counter()

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": _build_user_message(request)},
            ],
            text_format=_LLMRecommendationsOutput,
            temperature=0.4,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RenovationRecommendationsError("Model returned no parsed structured output.")

        logger.info(
            "Renovation recommendations generated in %dms (%d items)",
            int((perf_counter() - started_at) * 1000),
            len(parsed.recommendations),
        )

        known_ids = {o.id for o in request.objects}
        recommendations: list[Recommendation] = []
        for r in parsed.recommendations:
            # Same defensive filtering convention as room_layout_service.py: never trust LLM output
            # fully. An invalid priority or a related_object_id the model invented gets corrected/dropped
            # rather than rejecting the whole recommendation.
            priority = r.priority if r.priority in _VALID_PRIORITIES else "medium"
            related_id = r.related_object_id if r.related_object_id in known_ids else None
            recommendations.append(
                Recommendation(
                    title=r.title,
                    rationale=r.rationale,
                    priority=priority,
                    related_object_id=related_id,
                )
            )

        return RenovationRecommendationsResponse(recommendations=recommendations, mode="llm")

    def _mock_response(self) -> RenovationRecommendationsResponse:
        """Deterministic fallback when the LLM path is unavailable/disabled — same mock-mode convention
        as room_layout_service.py and the rest of this codebase: stays usable end-to-end without an API
        key, just without real recommendations."""
        return RenovationRecommendationsResponse(
            recommendations=[],
            mode="mock",
            warnings=["AI renovation recommendations aren't configured on this server yet."],
        )
