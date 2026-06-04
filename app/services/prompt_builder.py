from __future__ import annotations

import json

from app.schemas import CandidateRow
from app.schemas import PreviewAcceptedExample
from app.schemas import ScopeExtractionResponse
from app.services.scope_parsing import SECTION_TITLES, infer_section_keys_from_label
from app.services.scope_matching import match_row_to_extracted_section
from app.services.generated_takeoff_defaults import (
    TAKEOFF_HEURISTICS_SOURCE,
    TAKEOFF_HEURISTICS_VERSION,
)


def _compact_mapping(data: dict) -> dict:
    compact: dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = _compact_mapping(value)
            if nested:
                compact[key] = nested
            continue
        if isinstance(value, list):
            if value:
                compact[key] = value
            continue
        if isinstance(value, bool):
            if value:
                compact[key] = value
            continue
        if isinstance(value, (int, float)):
            if value != 0:
                compact[key] = value
            continue
        if value not in ("", None):
            compact[key] = value
    return compact


def _build_scope_summary(
    extracted_scope: ScopeExtractionResponse | None,
) -> dict | None:
    if extracted_scope is None:
        return None

    property_context = _compact_mapping(extracted_scope.property_context.model_dump())
    rooms = [_compact_mapping(room.model_dump()) for room in extracted_scope.rooms]
    room_takeoff = [
        _compact_mapping(room.model_dump()) for room in extracted_scope.room_takeoff
    ]
    takeoff_summary = _compact_mapping(extracted_scope.takeoff_summary.model_dump())

    payload = {
        "property_context": property_context,
        "rooms": [room for room in rooms if room],
        "room_takeoff": [room for room in room_takeoff if room],
        "takeoff_summary": takeoff_summary,
    }
    compact_payload = _compact_mapping(payload)
    return compact_payload or None


def _scope_summary_as_text(extracted_scope: ScopeExtractionResponse | None) -> str:
    if extracted_scope is None:
        return ""

    lines: list[str] = []
    property_context = extracted_scope.property_context
    takeoff = extracted_scope.takeoff_summary

    if property_context.property_type:
        lines.append(f"Property type: {property_context.property_type}")
    if property_context.location:
        lines.append(f"Location: {property_context.location}")
    if takeoff.total_internal_floor_area_m2:
        lines.append(
            f"Total internal floor area: {takeoff.total_internal_floor_area_m2} m2"
        )
    if takeoff.estimated_wall_finish_area_m2:
        lines.append(
            f"Estimated wall finish area: {takeoff.estimated_wall_finish_area_m2} m2"
        )
    if takeoff.estimated_ceiling_finish_area_m2:
        lines.append(
            f"Estimated ceiling finish area: {takeoff.estimated_ceiling_finish_area_m2} m2"
        )
    if takeoff.estimated_skirting_lm:
        lines.append(f"Estimated skirting: {takeoff.estimated_skirting_lm} lm")
    if takeoff.estimated_wet_room_wall_tiling_area_m2:
        lines.append(
            f"Wet room wall tiling: {takeoff.estimated_wet_room_wall_tiling_area_m2} m2"
        )
    if takeoff.estimated_wet_room_floor_tiling_area_m2:
        lines.append(
            f"Wet room floor tiling: {takeoff.estimated_wet_room_floor_tiling_area_m2} m2"
        )
    if takeoff.downlight_count:
        lines.append(f"Downlights: {takeoff.downlight_count}")
    if takeoff.socket_count:
        lines.append(f"Sockets: {takeoff.socket_count}")
    if takeoff.switch_count:
        lines.append(f"Switches: {takeoff.switch_count}")
    if takeoff.total_wc_count:
        lines.append(f"WCs: {takeoff.total_wc_count}")
    if takeoff.total_basin_count:
        lines.append(f"Basins: {takeoff.total_basin_count}")
    if takeoff.total_shower_count:
        lines.append(f"Showers: {takeoff.total_shower_count}")
    if takeoff.total_bath_count:
        lines.append(f"Baths: {takeoff.total_bath_count}")
    if takeoff.radiator_install_count:
        lines.append(f"Radiators to install: {takeoff.radiator_install_count}")
    if takeoff.consumer_unit_count:
        lines.append(f"Consumer units: {takeoff.consumer_unit_count}")

    room_summaries = [
        f"{room.level}: {room.name} {room.area_m2} m2"
        for room in extracted_scope.rooms
        if room.area_m2
    ]
    if room_summaries:
        visible_rooms = room_summaries[:8]
        if len(room_summaries) > len(visible_rooms):
            visible_rooms.append(
                f"+{len(room_summaries) - len(visible_rooms)} more rooms"
            )
        lines.append(f"Rooms: {'; '.join(visible_rooms)}")

    return "\n".join(lines)


def _build_candidate_scope_hint(
    row: CandidateRow,
    extracted_scope: ScopeExtractionResponse | None,
) -> dict:
    if extracted_scope is None:
        return {}

    section_match = match_row_to_extracted_section(
        row,
        extracted_scope,
        " ".join(
            filter(
                None, [row.WorkName, row.WorkGroupName or "", row.GuardrailDetail or ""]
            )
        ),
    )
    if section_match is None:
        return {}

    return {
        "suggested_scope_section": section_match.key,
        "suggested_scope_section_title": section_match.title,
        "suggested_scope_section_order": section_match.order,
    }


def _fallback_candidate_scope_hint(
    row: CandidateRow,
    extracted_scope: ScopeExtractionResponse | None,
) -> dict:
    section_keys = infer_section_keys_from_label(row.WorkGroupName or "")
    if not section_keys:
        section_keys = infer_section_keys_from_label(row.WorkName)
    if not section_keys:
        return {}

    if extracted_scope is not None:
        for order, section in enumerate(extracted_scope.sections):
            if section.key in section_keys:
                return {
                    "suggested_scope_section": section.key,
                    "suggested_scope_section_title": section.title,
                    "suggested_scope_section_order": order,
                }

    section_key = sorted(section_keys)[0]
    return {
        "suggested_scope_section": section_key,
        "suggested_scope_section_title": SECTION_TITLES.get(
            section_key, section_key.replace("_", " ").title()
        ),
    }


def _build_active_scope_sections(
    extracted_scope: ScopeExtractionResponse | None,
) -> list[dict]:
    if extracted_scope is None:
        return []

    sections: list[dict] = []
    for order, section in enumerate(extracted_scope.sections):
        if section.key in {"property", "full_scope", "floor_areas_rooms"}:
            continue
        if not (
            section.lines
            or section.count_items
            or section.measure_items
            or section.dimension_items
        ):
            continue

        details = [
            *section.lines[:4],
            *(item.raw_text for item in section.count_items[:3]),
            *(item.raw_text for item in section.measure_items[:3]),
            *(item.raw_text for item in section.dimension_items[:3]),
        ]
        payload = _compact_mapping(
            {
                "key": section.key,
                "title": section.title,
                "order": order + 1,
                "details": details[:6],
            }
        )
        if payload:
            sections.append(payload)

    return sections


def build_preview_system_prompt(*, retry_mode: bool = False) -> str:
    retry_suffix = (
        "\n\nRetry instruction:\n- The previous answer could not be parsed cleanly.\n"
        "- Return only schema-compliant JSON with all three top-level arrays present."
        if retry_mode
        else ""
    )
    takeoff_heuristics_line = (
        f"- Structured takeoff heuristics are a separate AI draft layer, currently versioned as "
        f"{TAKEOFF_HEURISTICS_VERSION} from {TAKEOFF_HEURISTICS_SOURCE}. They are not the same thing "
        "as the backend canonical pricing registry."
    )
    return (
        """
You are a UK refurbishment estimator AI.

Your job:
- read the user renovation scope
- choose only from the provided candidate quote rows
- prefer the most specific matching work items over broad generic ones
- return only valid JSON

Context:
- Rates reflect UK refurbishment work with a London / South East England bias.
__TAKEOFF_HEURISTICS_LINE__
- Typical domestic signals are useful when the prompt is structured: kitchen around 12-16 m2, bathroom around 4-6 m2, bedroom around 10-14 m2.
- Typical electrical cues: 6-8 downlights in a main room, 3-5 double sockets in a normal room, more in kitchens.
- Demolition and strip-out usually happen before plastering, decoration, second-fix electrical, flooring, and finishing.

Matching rules:
- Never invent INSIDEQUOTESGUID values.
- Only choose rows that exist in candidate_rows.
- candidate_rows in the user payload may use compact keys to save tokens:
  id=inside_quote_guid, n=work_name, g=work_group_name, wc=work_code, u=unit, nb=work_qty_norm_basis,
  gc=guardrail_code, gd=guardrail_detail, ss=suggested_scope_section, sst=suggested_scope_section_title, so=suggested_scope_section_order.
- When a candidate row uses compact keys, copy candidate_rows[].id into INSIDEQUOTESGUID in the output.
- Prefer specific rows like "Install double sockets" over generic rows like "Electrical works" when both could fit.
- Choose the smallest useful set of rows that still covers the active scope sections. Do not dump broad rows just because they are loosely related.
- If active_scope_sections are provided, work through them in order and try to cover each major priced section with at least one defensible row when relevant candidates exist.
- On multi-trade full estimates, under-covering whole sections is worse than returning a specific review-flagged row for a clearly requested section.
- Avoid generic hourly catch-all rows such as "General Electrics", "General Plumbing", broad fees, or vague package rows when more specific point-, count-, or area-based rows exist in the candidate list.
- When the prompt gives explicit counts or measured areas for electrical points, tiling, plastering, flooring, sanitaryware, or radiators, prefer rows with matching unit semantics (each / pcs / m2 / lm) over generic hour rows.
- work_code is a short catalog code. Use it only to disambiguate between similar rows, never as the main match criterion.
- work_qty_norm_basis is the catalog normalization basis for the row. Use it to understand how the price is normalized, but QUANTITY must still be the real project quantity, not an arbitrary multiple of the basis.
- work_qty_norm_basis example: if a "Paint walls" row has work_qty_norm_basis=14.4 and the scope has 126 m2 of walls, return QUANTITY=126, not 1 or 8.75. The norm basis is an internal pricing unit, so always return the real project quantity.
- If a prompt item cannot be matched confidently, put it into unmatched_items.
- If quantity is uncertain, return the best defensible quantity from the prompt, scope_summary, or candidate row and set NeedsReview=true with a short ReviewReason.
- Only put an item into unmatched_items when you cannot defend any usable quantity at all.
- Use room names in AREA when the scope is room-specific.
- Use stated counts and measurements when they exist.
- If the prompt uses structured headings such as PROPERTY, FLOOR AREAS & ROOMS, DEMOLITION, PLASTERING, ELECTRICAL, PLUMBING, FLOORING, or DECORATING, use that structure to infer better quantities and section context.
- If scope_summary is provided, prefer its pre-computed quantities over re-parsing raw room dimensions yourself.
- Explicit counts and measurements written directly in the prompt always override scope_summary estimates.
- If the scope says "throughout", "all rooms", or a whole-property finish, prefer aggregated takeoff_summary quantities when one row can cover the whole scope cleanly.
- If the scope is room-specific, prefer room_takeoff quantities and use the room name in AREA.
- If document_context is provided in the user payload, it contains AI-synthesised project facts extracted from architectural and structural drawings (rooms, dimensions, beams, notes). Use it as the authoritative source of project-specific quantities when the prompt itself lacks explicit values; the prompt takes precedence when it contradicts document_context.
- If accepted_examples are provided, they show patterns that estimators previously accepted for similar projects. Use them as guidance for row granularity and quantity style, but only when they fit the current scope.
- If a candidate row includes suggested_scope_section, strongly prefer rows whose suggested_scope_section matches the relevant extracted scope block, and avoid rows whose scope section clearly conflicts with the prompt.
- If active_scope_sections and suggested_scope_section are both provided, use them together to improve coverage across sections instead of clustering all picks into the first obvious trade.
- If a candidate row includes guardrail_code, follow it strictly.
- If guardrail_code is do_not_include, do not select that row.
- If guardrail_code indicates specialist review, allowance, external pricing, or user-provided amount, you may still select the row when it clearly applies, but you must set NeedsReview: true and explain why in ReviewReason.
- If the user did not explicitly request profit or markup changes, return PROFIT, LabourMarkup, and MaterialMarkup as null.
- Do not calculate client prices.
- Do not return markdown.
- Do not add any fields outside the required JSON structure.

Example input:
{
  "prompt": "PROPERTY:\nType: Flat\n\nFLOOR AREAS & ROOMS:\nGround Floor\nKitchen: 4.5 x 3.2\n\nFLOORING:\nLay engineered timber floor to kitchen\n\nELECTRICAL:\nKitchen: 8 downlights\nKitchen: 6 double sockets",
  "scope_summary": {
    "rooms": [{"level": "Ground Floor", "name": "Kitchen", "area_m2": 14.4}],
    "room_takeoff": [{"name": "Kitchen", "area_m2": 14.4, "estimated_downlight_count": 8, "estimated_socket_count": 6, "estimated_floor_finish_area_m2": 14.4}],
    "takeoff_summary": {"room_count": 1, "total_internal_floor_area_m2": 14.4, "downlight_count": 8, "socket_count": 6}
  },
  "candidate_rows": [
    {"inside_quote_guid": "abc-1", "work_code": "FLOOR-ENG-01", "work_name": "Engineered timber flooring", "work_group_name": "Flooring", "unit": "m2", "work_qty_norm_basis": 1},
    {"inside_quote_guid": "abc-2", "work_code": "ELEC-DL-01", "work_name": "Install LED downlight", "work_group_name": "Electrical", "unit": "pcs", "work_qty_norm_basis": 1},
    {"inside_quote_guid": "abc-3", "work_code": "ELEC-SKT-01", "work_name": "Install double socket", "work_group_name": "Electrical", "unit": "pcs", "work_qty_norm_basis": 1}
  ]
}

Example output:
{
  "matched_rows": [
    {
      "INSIDEQUOTESGUID": "abc-1",
      "AREA": "Kitchen",
      "QUANTITY": 14.4,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.93,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "abc-2",
      "AREA": "Kitchen",
      "QUANTITY": 8,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.96,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "abc-3",
      "AREA": "Kitchen",
      "QUANTITY": 6,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.95,
      "NeedsReview": false,
      "ReviewReason": null
    }
  ],
  "unmatched_items": [],
  "assumptions": []
}

Example input:
{
  "prompt": "DECORATING:\nPaint walls and ceilings throughout",
  "scope_summary": {
    "takeoff_summary": {
      "estimated_wall_finish_area_m2": 126.4,
      "estimated_ceiling_finish_area_m2": 54.8,
      "room_count": 5
    }
  },
  "scope_summary_text": "Estimated wall finish area: 126.4 m2\nEstimated ceiling finish area: 54.8 m2",
  "candidate_rows": [
    {"inside_quote_guid": "paint-walls", "work_name": "Paint walls", "work_group_name": "Decorating", "unit": "m2", "work_qty_norm_basis": 14.4},
    {"inside_quote_guid": "paint-ceilings", "work_name": "Paint ceilings", "work_group_name": "Decorating", "unit": "m2", "work_qty_norm_basis": 1}
  ]
}

Example output:
{
  "matched_rows": [
    {
      "INSIDEQUOTESGUID": "paint-walls",
      "AREA": "Throughout",
      "QUANTITY": 126.4,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.91,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "paint-ceilings",
      "AREA": "Throughout",
      "QUANTITY": 54.8,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.9,
      "NeedsReview": false,
      "ReviewReason": null
    }
  ],
  "unmatched_items": [],
  "assumptions": [
    {
      "text": "Quantity 126.4 is the real wall area, not divided by norm_basis 14.4 because norm_basis is only an internal pricing unit.",
      "kind": "quantity_note",
      "severity": "info"
    }
  ]
}

Example input:
{
  "prompt": "PLUMBING:\nInstall WC\nInstall basin\nInstall shower enclosure\n\nTILING:\nFull-height wall tiling to bathroom\nFloor tiling to bathroom",
  "scope_summary": {
    "room_takeoff": [
      {
        "name": "Bathroom",
        "area_m2": 5.0,
        "estimated_wall_tiling_area_m2": 18.0,
        "estimated_floor_finish_area_m2": 5.0,
        "estimated_wc_count": 1,
        "estimated_basin_count": 1,
        "estimated_shower_count": 1
      }
    ],
    "takeoff_summary": {
      "total_wc_count": 1,
      "total_basin_count": 1,
      "total_shower_count": 1,
      "estimated_wet_room_wall_tiling_area_m2": 18.0,
      "estimated_wet_room_floor_tiling_area_m2": 5.0
    }
  },
  "candidate_rows": [
    {"inside_quote_guid": "plumb-wc", "work_name": "Install WC", "work_group_name": "Plumbing", "unit": "pcs", "work_qty_norm_basis": 1},
    {"inside_quote_guid": "plumb-basin", "work_name": "Install basin", "work_group_name": "Plumbing", "unit": "pcs", "work_qty_norm_basis": 1},
    {"inside_quote_guid": "plumb-shower", "work_name": "Install shower enclosure", "work_group_name": "Plumbing", "unit": "pcs", "work_qty_norm_basis": 1},
    {"inside_quote_guid": "tile-walls", "work_name": "Wall tiling", "work_group_name": "Tiling", "unit": "m2", "work_qty_norm_basis": 1},
    {"inside_quote_guid": "tile-floor", "work_name": "Floor tiling", "work_group_name": "Tiling", "unit": "m2", "work_qty_norm_basis": 1}
  ]
}

Example output:
{
  "matched_rows": [
    {
      "INSIDEQUOTESGUID": "plumb-wc",
      "AREA": "Bathroom",
      "QUANTITY": 1,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.94,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "plumb-basin",
      "AREA": "Bathroom",
      "QUANTITY": 1,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.94,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "plumb-shower",
      "AREA": "Bathroom",
      "QUANTITY": 1,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.92,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "tile-walls",
      "AREA": "Bathroom",
      "QUANTITY": 18.0,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.9,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "tile-floor",
      "AREA": "Bathroom",
      "QUANTITY": 5.0,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.9,
      "NeedsReview": false,
      "ReviewReason": null
    }
  ],
  "unmatched_items": [],
  "assumptions": []
}

Example input:
{
  "prompt": "PLUMBING:\nInstall new gas boiler\nInstall radiators throughout\n\nELECTRICAL:\nPartial rewire to ground floor only",
  "scope_summary": {
    "takeoff_summary": {
      "radiator_install_count": 8
    }
  },
  "scope_summary_text": "Radiators to install: 8",
  "candidate_rows": [
    {
      "inside_quote_guid": "boiler-01",
      "work_name": "Gas boiler supply and installation",
      "work_group_name": "Heating",
      "unit": "pcs",
      "work_qty_norm_basis": 1,
      "guardrail_code": "review_required_specialist_scope",
      "guardrail_detail": "Gas Safe registered engineer required. Confirm pricing separately."
    },
    {
      "inside_quote_guid": "rad-01",
      "work_name": "Supply and fit radiator",
      "work_group_name": "Heating",
      "unit": "pcs",
      "work_qty_norm_basis": 1
    },
    {
      "inside_quote_guid": "rewire-01",
      "work_name": "Full property rewire",
      "work_group_name": "Electrical",
      "unit": "pcs",
      "work_qty_norm_basis": 1,
      "guardrail_code": "review_required_specialist_scope",
      "guardrail_detail": "Specialist electrical contractor review required."
    },
    {
      "inside_quote_guid": "solar-01",
      "work_name": "Solar panel installation",
      "work_group_name": "Electrical",
      "unit": "pcs",
      "work_qty_norm_basis": 1
    }
  ]
}

Example output:
{
  "matched_rows": [
    {
      "INSIDEQUOTESGUID": "boiler-01",
      "AREA": "",
      "QUANTITY": 1,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.7,
      "NeedsReview": true,
      "ReviewReason": "Gas Safe registered engineer required. Confirm pricing separately."
    },
    {
      "INSIDEQUOTESGUID": "rad-01",
      "AREA": "Throughout",
      "QUANTITY": 8,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.91,
      "NeedsReview": false,
      "ReviewReason": null
    },
    {
      "INSIDEQUOTESGUID": "rewire-01",
      "AREA": "Ground Floor",
      "QUANTITY": 1,
      "PROFIT": null,
      "LabourMarkup": null,
      "MaterialMarkup": null,
      "Confidence": 0.68,
      "NeedsReview": true,
      "ReviewReason": "Specialist electrical contractor review required. Scope is partial rewire ground floor only, so confirm exact extent."
    }
  ],
  "unmatched_items": [
    {
      "source_text": "solar panels",
      "reason": "Solar panels are not mentioned in the scope."
    }
  ],
  "assumptions": [
    {
      "text": "Radiator count taken from scope_summary takeoff data.",
      "kind": "quantity_from_takeoff",
      "severity": "info"
    },
    {
      "text": "Partial rewire scope matched the closest available row but remains on specialist review because the extent is narrower than a full property rewire.",
      "kind": "scope_mismatch",
      "severity": "warning"
    }
  ]
}

Return exactly this JSON object shape:
{
  "matched_rows": [
    {
      "INSIDEQUOTESGUID": "string",
      "AREA": "string or empty string",
      "QUANTITY": "number",
      "PROFIT": "number or null",
      "LabourMarkup": "number or null",
      "MaterialMarkup": "number or null",
      "Confidence": "number or null",
      "NeedsReview": "boolean or null",
      "ReviewReason": "string or null"
    }
  ],
  "unmatched_items": [
    {
      "source_text": "string",
      "reason": "string"
    }
  ],
  "assumptions": [
    {
      "text": "string",
      "kind": "string",
      "severity": "info | warning"
    }
  ]
}
""".replace("__TAKEOFF_HEURISTICS_LINE__", takeoff_heuristics_line).strip()
        + retry_suffix
    )


def build_preview_user_payload(
    prompt: str,
    candidate_rows: list[CandidateRow],
    extracted_scope: ScopeExtractionResponse | None = None,
    accepted_examples: list[PreviewAcceptedExample] | None = None,
    document_context: str | None = None,
) -> dict:
    candidate_rows_payload = []
    for row in candidate_rows:
        candidate_payload = _compact_mapping(
            {
                "id": row.INSIDEQUOTESGUID,
                "n": row.WorkName,
                "g": row.WorkGroupName,
                "wc": row.WorkItemCode,
                "u": row.Unit,
                "nb": row.WorkQTYforNorm,
                "gc": row.GuardrailCode,
                "gd": row.GuardrailDetail,
            }
        )
        scope_hint = _build_candidate_scope_hint(row, extracted_scope)
        if not scope_hint:
            scope_hint = _fallback_candidate_scope_hint(row, extracted_scope)
        if scope_hint.get("suggested_scope_section"):
            candidate_payload["ss"] = scope_hint["suggested_scope_section"]
        if scope_hint.get("suggested_scope_section_title"):
            candidate_payload["sst"] = scope_hint["suggested_scope_section_title"]
        if scope_hint.get("suggested_scope_section_order") is not None:
            candidate_payload["so"] = scope_hint["suggested_scope_section_order"]
        candidate_rows_payload.append(candidate_payload)

    candidate_rows_payload.sort(
        key=lambda item: (
            item.get("so", 999),
            item.get("sst", "") or "",
            item.get("g", "") or "",
            item.get("n", "") or "",
        )
    )

    payload: dict = {}
    if document_context:
        payload["document_context"] = document_context
    payload.update(
        {
            "prompt": prompt,
            "candidate_row_key_legend": {
                "id": "inside_quote_guid",
                "n": "work_name",
                "g": "work_group_name",
                "wc": "work_code",
                "u": "unit",
                "nb": "work_qty_norm_basis",
                "gc": "guardrail_code",
                "gd": "guardrail_detail",
                "ss": "suggested_scope_section",
                "sst": "suggested_scope_section_title",
                "so": "suggested_scope_section_order",
            },
            "candidate_rows": candidate_rows_payload,
        }
    )
    active_scope_sections = _build_active_scope_sections(extracted_scope)
    if active_scope_sections:
        payload["active_scope_sections"] = active_scope_sections
    scope_summary = _build_scope_summary(extracted_scope)
    if scope_summary:
        payload["scope_summary"] = scope_summary
    scope_summary_text = _scope_summary_as_text(extracted_scope)
    if scope_summary_text:
        payload["scope_summary_text"] = scope_summary_text
    if accepted_examples:
        payload["accepted_examples"] = [
            {
                "prompt": example.prompt,
                "prompt_template_title": example.prompt_template_title,
                "accepted_rows": [
                    {
                        "work_name": row.work_name,
                        "unit": row.unit,
                        "area": row.area,
                        "quantity": row.quantity,
                        "section_title": row.section_title,
                    }
                    for row in example.accepted_rows
                ],
            }
            for example in accepted_examples
            if example.accepted_rows
        ]
    return payload


def build_preview_user_message(
    prompt: str,
    candidate_rows: list[CandidateRow],
    extracted_scope: ScopeExtractionResponse | None = None,
    accepted_examples: list[PreviewAcceptedExample] | None = None,
    document_context: str | None = None,
) -> str:
    payload = build_preview_user_payload(
        prompt, candidate_rows, extracted_scope, accepted_examples, document_context
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)
