from __future__ import annotations

import logging
import re
from time import perf_counter

from app.config import settings
from app.schemas import (
    BatchCorrectionsRequest,
    BatchCorrectionsResponse,
    CorrectionPatch,
    ItemContext,
    LLMBatchCorrectionsOutput,
    LLMCorrectionPatch,
)
from app.services.retry import call_with_request_retries

logger = logging.getLogger(__name__)

_FIELD_DISPLAY = {
    "area": "Area",
    "quantity": "Quantity",
    "work_name_override": "Item name",
    "client_title_override": "Client title",
    "client_description_override": "Client description",
    "internal_note": "Internal note",
    "display_mode": "Display mode",
    "profit": "Profit %",
}

_DISPLAY_MODE_ALIASES = {
    "excluded": "excluded",
    "exclude": "excluded",
    "excl": "excluded",
    "by arrangement": "by_arrangement",
    "by_arrangement": "by_arrangement",
    "b.a.": "by_arrangement",
    "ba": "by_arrangement",
    "pc sum": "pc_sum",
    "pc": "pc_sum",
    "pc_sum": "pc_sum",
    "provisional": "provisional_sum",
    "provisional sum": "provisional_sum",
    "provisional_sum": "provisional_sum",
    "prov": "provisional_sum",
    "included": "included",
    "include": "included",
}


def _resolve_llm_patches(
    llm_patches: list[LLMCorrectionPatch], items: list[ItemContext]
) -> tuple[list[CorrectionPatch], list[str]]:
    """Resolve each patch's real database item_id from its display_index, since the
    model is never shown the real id (only the '#' column) and so cannot return it
    reliably. An out-of-range display_index is dropped to unresolved rather than
    ever reaching the bulk-update API with an unverified id."""
    item_by_index = {item.display_index: item for item in items}
    resolved: list[CorrectionPatch] = []
    unresolved: list[str] = []

    for patch in llm_patches:
        item = item_by_index.get(patch.display_index)
        if item is None:
            unresolved.append(
                f"{patch.ref} — {patch.field}: {patch.new_value} (could not resolve item)"
            )
            continue
        resolved.append(
            CorrectionPatch(
                item_id=item.id,
                ref=patch.ref,
                name=patch.name,
                field=patch.field,
                old_value=patch.old_value,
                new_value=patch.new_value,
            )
        )

    return resolved, unresolved


def _validate_and_normalize_patches(
    patches: list[CorrectionPatch], unresolved: list[str]
) -> tuple[list[CorrectionPatch], list[str]]:
    """Guard against the model returning a field name outside the documented
    set, or a display_mode value that isn't one of the five canonical enum
    values — either would otherwise flow straight into the bulk-update API."""
    valid_patches: list[CorrectionPatch] = []
    extra_unresolved: list[str] = []

    for patch in patches:
        if patch.field not in _FIELD_DISPLAY:
            extra_unresolved.append(
                f"{patch.ref} — unrecognised field '{patch.field}': {patch.new_value}"
            )
            continue

        if patch.field == "display_mode":
            normalized = _DISPLAY_MODE_ALIASES.get(patch.new_value.strip().lower())
            if normalized is None:
                extra_unresolved.append(
                    f"{patch.ref} — unrecognised display mode '{patch.new_value}'"
                )
                continue
            patch.new_value = normalized

        valid_patches.append(patch)

    return valid_patches, [*unresolved, *extra_unresolved]


def _build_system_prompt() -> str:
    return """You are an AI assistant for a construction estimating platform.
Your job is to parse a free-form list of corrections to quote line items and return structured patch operations.

The user will provide:
1. A numbered table of current quote items (index, ref code, name, area, quantity)
2. A block of correction text (usually copied from a client/contractor email or batch instruction)

Your task:
- For each correction clause, identify which item it refers to (by ref code, index number, or partial name match)
- Identify which field is being changed: area, quantity, work_name_override, client_title_override, client_description_override, internal_note, display_mode, profit
- Extract the new value as a clean string (no units embedded — area values keep their unit text e.g. "94m²", quantities are numeric strings e.g. "45.00")
- For display_mode, normalise to: included | excluded | by_arrangement | pc_sum | provisional_sum
- For profit, extract the numeric % value as a string e.g. "12.50"
- Put the old value from the item context table into old_value
- If a clause is ambiguous or cannot be mapped to any item, put the raw text in unresolved[]
- Do NOT invent items that are not in the context table
- Do NOT make assumptions when the ref or name does not match — unresolved is better than wrong

Return a JSON object matching this schema:
{
  "patches": [
    {
      "display_index": <int, the "#" column value from the item table>,
      "ref": "<string>",
      "name": "<current item name>",
      "field": "<field name>",
      "old_value": "<current value>",
      "new_value": "<new value>"
    }
  ],
  "unresolved": ["<clause text that could not be matched>"]
}

One correction clause can produce multiple patches (e.g. "rename and change area" → two patch objects for the same item).
"""


def _build_user_message(request: BatchCorrectionsRequest) -> str:
    lines = ["## Current quote items\n"]
    lines.append("| # | Ref | Name | Area | Qty |")
    lines.append("|---|-----|------|------|-----|")
    for item in request.items:
        lines.append(
            f"| {item.display_index} | {item.ref} | {item.name} | {item.area or '—'} | {item.quantity} |"
        )
    lines.append("\n## Correction instructions\n")
    lines.append(request.text.strip())
    return "\n".join(lines)


def _mock_parse(request: BatchCorrectionsRequest) -> BatchCorrectionsResponse:
    """Regex-based mock that handles the most common correction patterns without an LLM."""
    patches: list[CorrectionPatch] = []
    unresolved: list[str] = []

    item_by_ref = {item.ref: item for item in request.items}
    item_by_index = {item.display_index: item for item in request.items}

    # Split on line breaks and semicolons to get individual clauses
    raw_clauses = re.split(r"[;\n]+", request.text)
    clauses = [c.strip() for c in raw_clauses if c.strip()]

    for clause in clauses:
        item: ItemContext | None = None

        # Try to find item by ref code (e.g. "1.1.4")
        ref_match = re.search(r"\b(\d+\.\d+\.\d+)\b", clause)
        if ref_match:
            item = item_by_ref.get(ref_match.group(1))

        # Try by index number ("item 3", "#3", "позиция 3")
        if item is None:
            idx_match = re.search(r"(?:item|#|позиция)\s*(\d+)", clause, re.IGNORECASE)
            if idx_match:
                item = item_by_index.get(int(idx_match.group(1)))

        if item is None:
            unresolved.append(clause)
            continue

        matched = False

        # Area: "change area to 94m²", "area = 94m²", "площадь 94m²"
        area_match = re.search(
            r"(?:area|площадь)[^\d]*(\d+(?:\.\d+)?\s*m²?)", clause, re.IGNORECASE
        )
        if area_match:
            patches.append(
                CorrectionPatch(
                    item_id=item.id,
                    ref=item.ref,
                    name=item.name,
                    field="area",
                    old_value=item.area or "",
                    new_value=area_match.group(1).strip(),
                )
            )
            matched = True

        # Quantity: "change qty to 45", "quantity = 45.5"
        qty_match = re.search(
            r"(?:qty|quantity|количество)[^\d]*(\d+(?:\.\d+)?)", clause, re.IGNORECASE
        )
        if qty_match:
            patches.append(
                CorrectionPatch(
                    item_id=item.id,
                    ref=item.ref,
                    name=item.name,
                    field="quantity",
                    old_value=item.quantity,
                    new_value=f"{float(qty_match.group(1)):.2f}",
                )
            )
            matched = True

        # Profit: "profit to 15%", "margin = 12.5%"
        profit_match = re.search(
            r"(?:profit|margin)[^\d]*(\d+(?:\.\d+)?)\s*%?", clause, re.IGNORECASE
        )
        if profit_match:
            patches.append(
                CorrectionPatch(
                    item_id=item.id,
                    ref=item.ref,
                    name=item.name,
                    field="profit",
                    old_value="",
                    new_value=f"{float(profit_match.group(1)):.2f}",
                )
            )
            matched = True

        # Display mode: "mark as excluded", "set to by arrangement", "display as pc sum"
        alias_pattern = "|".join(
            re.escape(alias)
            for alias in sorted(_DISPLAY_MODE_ALIASES, key=len, reverse=True)
        )
        mode_match = re.search(
            rf"(?:mark as|set to|display as|display_mode)\s*[:\-]?\s*({alias_pattern})\b",
            clause,
            re.IGNORECASE,
        )
        if mode_match:
            patches.append(
                CorrectionPatch(
                    item_id=item.id,
                    ref=item.ref,
                    name=item.name,
                    field="display_mode",
                    old_value="",
                    new_value=_DISPLAY_MODE_ALIASES[mode_match.group(1).lower()],
                )
            )
            matched = True

        # Internal note: "note: check with client", "internal note - confirm spec"
        note_match = re.search(
            r"(?:internal\s+note|note)\s*[:\-]\s*(.+)", clause, re.IGNORECASE
        )
        if note_match:
            patches.append(
                CorrectionPatch(
                    item_id=item.id,
                    ref=item.ref,
                    name=item.name,
                    field="internal_note",
                    old_value="",
                    new_value=note_match.group(1).strip(),
                )
            )
            matched = True

        if not matched:
            unresolved.append(clause)

    patches, unresolved = _validate_and_normalize_patches(patches, unresolved)
    return BatchCorrectionsResponse(
        patches=patches, unresolved=unresolved, service_mode="mock"
    )


def _llm_parse(request: BatchCorrectionsRequest) -> BatchCorrectionsResponse:
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=3,
    )
    model = settings.openai_model

    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _build_user_message(request)},
    ]

    def _call_llm() -> LLMBatchCorrectionsOutput | None:
        # Try responses.parse (newer SDK)
        responses_api = getattr(client, "responses", None)
        if callable(getattr(responses_api, "parse", None)):
            response = client.responses.parse(
                model=model,
                input=messages,
                text_format=LLMBatchCorrectionsOutput,
                temperature=0,
            )
            return getattr(response, "output_parsed", None)
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=LLMBatchCorrectionsOutput,
            temperature=0,
        )
        return response.choices[0].message.parsed if response.choices else None

    started_at = perf_counter()
    try:
        parsed = call_with_request_retries(
            _call_llm, label="batch_corrections LLM parse"
        )

        if parsed is None:
            raise ValueError("Model returned no structured output.")

        resolved_patches, resolve_unresolved = _resolve_llm_patches(
            parsed.patches, request.items
        )
        valid_patches, unresolved = _validate_and_normalize_patches(
            resolved_patches, [*parsed.unresolved, *resolve_unresolved]
        )
        logger.info(
            "batch_corrections LLM parse completed in %sms, %d patches, %d unresolved",
            int((perf_counter() - started_at) * 1000),
            len(valid_patches),
            len(unresolved),
        )
        return BatchCorrectionsResponse(
            patches=valid_patches,
            unresolved=unresolved,
            service_mode="real",
        )

    except Exception as exc:
        logger.warning("LLM corrections parse failed (%s) — falling back to mock", exc)
        result = _mock_parse(request)
        result.service_mode = "mock-fallback"
        return result


def parse_batch_corrections(
    request: BatchCorrectionsRequest,
) -> BatchCorrectionsResponse:
    if not (settings.openai_api_key and settings.openai_model):
        return _mock_parse(request)
    return _llm_parse(request)
