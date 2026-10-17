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


class ObjectConditionItem(BaseModel):
    """One object within a batch request — same fields as `ObjectConditionRequest` minus `room_name`,
    which is shared across the whole batch (every object in it comes from the same scan)."""

    id: str  # client-assigned, echoed back unchanged so the caller can match responses to requests
    category: str
    width_m: float
    height_m: float
    depth_m: float
    image_base64: str


class ObjectConditionBatchRequest(BaseModel):
    room_name: str = ""
    items: list[ObjectConditionItem] = Field(default_factory=list, max_length=12)


class ObjectConditionBatchResult(BaseModel):
    id: str
    category: str
    condition: str
    condition_notes: str = ""
    renovation_flags: list[str] = Field(default_factory=list)


class ObjectConditionBatchResponse(BaseModel):
    results: list[ObjectConditionBatchResult]
    mode: str  # "llm" | "mock"


# Stable across requests by construction — the prompt NEVER interpolates request-specific values
# (category/dimensions move into a per-item block instead, see `_build_batch_user_content`). This is
# what makes OpenAI's automatic prompt caching apply to it: the cached-token discount only kicks in
# when a long-enough prefix (this system prompt) is byte-for-byte identical across calls. Interpolating
# even one field here would silently break caching for every request, so new instructions belong in the
# fixed rules below, not in a dynamic prefix.
_SYSTEM_PROMPT = """\
You are assessing furniture and fixtures in UK homes from photos taken during a room scan, for \
someone planning a renovation — not a design critique, just practical visible condition.

You will be given one or more items. For each item you're told RoomPlan's own category label and \
approximate dimensions, followed immediately by a photo of that specific item.

For EACH item, judge its condition from the photo alone:
- "good": no notable wear or damage visible
- "worn": visible but minor wear — scuffs, fading, dated style, nothing urgent
- "needs_replacement": visible damage, heavy wear, or clear signs it should be replaced during a \
renovation
- "unknown": the photo is too blurry, too dark, or the item isn't clearly visible — never guess \
when you can't actually see it

Also note 0-3 short, practical renovation_flags per item where relevant (e.g. "check for leaks \
around base") — genuinely useful observations only, not filler.

Return ONLY valid JSON, no markdown fences, in this exact shape:
{
  "results": [
    {"item_index": 0, "condition": "<good|worn|needs_replacement|unknown>", "condition_notes": \
"<one short homeowner-friendly sentence>", "renovation_flags": ["<short note>", ...]},
    ...
  ]
}
item_index must match the 0-based position of the item in the order you were given them, and there \
must be exactly one result per item, in any order.
"""


def _item_label(index: int, item: ObjectConditionItem, room_name: str) -> str:
    return (
        f"Item {index}: \"{item.category}\" "
        f"(roughly {item.width_m:.1f}m × {item.height_m:.1f}m × {item.depth_m:.1f}m)"
        + (f" in the {room_name}" if room_name else "")
        + ". Photo of item {index} follows.".format(index=index)
    )


def _build_batch_user_content(request: ObjectConditionBatchRequest) -> list[dict]:
    content: list[dict] = []
    for index, item in enumerate(request.items):
        content.append({"type": "text", "text": _item_label(index, item, request.room_name)})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{item.image_base64}",
                    "detail": "low",
                },
            }
        )
    return content


class ObjectConditionService:
    def __init__(self) -> None:
        self.enabled = bool(settings.openai_api_key)
        self.model = settings.openai_intake_fast_model or "gpt-4o-mini"
        # Dedup within a single batch: identical category + near-identical dimensions is treated as
        # "probably the same kind of item" (e.g. two scanned dining chairs) — assessed once, the result
        # copied to every matching item, rather than paying for N near-identical vision calls. Keyed by
        # a coarse rounding of dimensions (10cm buckets) so "0.61m" and "0.60m" chairs still dedup.
        self._dedup_rounding_m = 0.1

    def is_enabled(self) -> bool:
        return self.enabled

    def assess(self, request: ObjectConditionRequest) -> ObjectConditionResponse:
        """Single-item path — kept for callers that only ever have one object to assess at a time.
        Prefer `assess_batch` when assessing several objects from the same scan: it's one network
        round-trip and one vision call instead of N, and still benefits from the same prompt-caching
        system prompt (see `_SYSTEM_PROMPT`)."""
        batch_response = self.assess_batch(
            ObjectConditionBatchRequest(
                room_name=request.room_name,
                items=[
                    ObjectConditionItem(
                        id="single",
                        category=request.category,
                        width_m=request.width_m,
                        height_m=request.height_m,
                        depth_m=request.depth_m,
                        image_base64=request.image_base64,
                    )
                ],
            )
        )
        if not batch_response.results:
            return self._mock_response(request)
        result = batch_response.results[0]
        return ObjectConditionResponse(
            category=result.category,
            condition=result.condition,
            condition_notes=result.condition_notes,
            renovation_flags=result.renovation_flags,
            mode=batch_response.mode,
        )

    def assess_batch(self, request: ObjectConditionBatchRequest) -> ObjectConditionBatchResponse:
        if not request.items:
            return ObjectConditionBatchResponse(results=[], mode="mock")
        if not self.enabled:
            return self._mock_batch_response(request)
        try:
            return self._llm_batch_response(request)
        except Exception:
            logger.exception(
                "Object condition batch assessment failed (%d items) — falling back to mock.",
                len(request.items),
            )
            return self._mock_batch_response(request)

    def _llm_batch_response(self, request: ObjectConditionBatchRequest) -> ObjectConditionBatchResponse:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=3,
        )

        # Dedup pass: group items whose (category, rounded dimensions) match — only the first of each
        # group is actually sent to the model, the rest are filled in from its result afterwards.
        groups: dict[tuple[str, int, int, int], list[int]] = {}
        for index, item in enumerate(request.items):
            key = (
                item.category,
                round(item.width_m / self._dedup_rounding_m),
                round(item.height_m / self._dedup_rounding_m),
                round(item.depth_m / self._dedup_rounding_m),
            )
            groups.setdefault(key, []).append(index)

        representative_indices = [indices[0] for indices in groups.values()]
        items_to_send = [request.items[i] for i in representative_indices]
        trimmed_request = ObjectConditionBatchRequest(room_name=request.room_name, items=items_to_send)

        started_at = perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_batch_user_content(trimmed_request)},
            ],
            max_tokens=200 * len(items_to_send) + 100,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        cached_tokens = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = getattr(details, "cached_tokens", 0) if details else 0
        logger.info(
            "Object condition batch assessed: %d items (%d sent after dedup) model=%s latency=%dms cached_tokens=%d",
            len(request.items),
            len(items_to_send),
            self.model,
            latency_ms,
            cached_tokens,
        )

        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {}

        raw_results = data.get("results", [])
        results_by_item_index: dict[int, dict] = {}
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            item_index = raw.get("item_index")
            if isinstance(item_index, int):
                results_by_item_index[item_index] = raw

        # Expand the deduped results back out to every original item, including the ones that were
        # skipped as duplicates — each gets a copy of its group representative's result.
        final_results: list[ObjectConditionBatchResult] = []
        for group_key, indices in groups.items():
            representative_position = representative_indices.index(indices[0])
            raw = results_by_item_index.get(representative_position, {})
            condition = raw.get("condition", "unknown")
            if condition not in {"good", "worn", "needs_replacement", "unknown"}:
                condition = "unknown"
            notes = str(raw.get("condition_notes", "")).strip()
            flags_raw = raw.get("renovation_flags", [])
            flags = [str(f).strip() for f in flags_raw if isinstance(f, (str, int, float))][:3]
            flags = [f for f in flags if f]

            for original_index in indices:
                final_results.append(
                    ObjectConditionBatchResult(
                        id=request.items[original_index].id,
                        category=request.items[original_index].category,
                        condition=condition,
                        condition_notes=notes,
                        renovation_flags=flags,
                    )
                )

        return ObjectConditionBatchResponse(results=final_results, mode="llm")

    def _mock_response(self, request: ObjectConditionRequest) -> ObjectConditionResponse:
        return ObjectConditionResponse(
            category=request.category,
            condition="unknown",
            condition_notes="AI condition assessment isn't configured on this server yet.",
            renovation_flags=[],
            mode="mock",
        )

    def _mock_batch_response(self, request: ObjectConditionBatchRequest) -> ObjectConditionBatchResponse:
        return ObjectConditionBatchResponse(
            results=[
                ObjectConditionBatchResult(
                    id=item.id,
                    category=item.category,
                    condition="unknown",
                    condition_notes="AI condition assessment isn't configured on this server yet.",
                    renovation_flags=[],
                )
                for item in request.items
            ],
            mode="mock",
        )
