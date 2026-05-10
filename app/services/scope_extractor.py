from __future__ import annotations

import re

from app.schemas import (
    ExtractedCountItem,
    ExtractedDimensionItem,
    ExtractedMeasureItem,
    ExtractedPropertyContext,
    ExtractedRoom,
    ExtractedSection,
    PreviewAssumption,
    ScopeExtractionResponse,
)
from app.services.scope_parsing import (
    ROOM_LEVELS,
    ROOM_LEVELS_LOWER,
    SECTION_TITLES,
    clean_line,
    infer_default_room_level,
    normalize_heading,
    prepare_prompt_lines,
    split_project_scope_line,
)
from app.services.scope_takeoff import (
    TAKEOFF_HEURISTICS_SOURCE,
    TAKEOFF_HEURISTICS_VERSION,
    build_room_takeoff_with_overrides,
    build_takeoff_summary,
)


def _round(value: float) -> float:
    return round(value, 2)


def _clean_line(line: str) -> str:
    return clean_line(line)


def _prepare_prompt_lines(prompt: str) -> list[str]:
    return prepare_prompt_lines(prompt)


def _split_project_scope_line(line: str) -> list[str]:
    return split_project_scope_line(line)


def _normalize_heading(line: str) -> str:
    return normalize_heading(line)


def _section_key_for_line(line: str) -> str | None:
    normalized = _normalize_heading(line)
    for key in SECTION_TITLES:
        if normalized == key:
            return key
    return None


def _parse_room(line: str, level: str | None) -> ExtractedRoom | None:
    if not level:
        return None
    match = re.match(
        r"^(?P<name>.+?):\s*(?:approx\.?\s*)?(?P<width>\d+(?:\.\d+)?)\s*(?:m)?\s*[×x]\s*(?P<length>\d+(?:\.\d+)?)\s*(?:m)?$",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None
    width = float(match.group("width"))
    length = float(match.group("length"))
    return ExtractedRoom(
        level=level,
        name=match.group("name").strip(),
        width_m=width,
        length_m=length,
        area_m2=_round(width * length),
    )


def _infer_default_room_level(prompt: str) -> str | None:
    return infer_default_room_level(prompt)


def _resolve_item_name(raw_name: str | None, context: str | None, fallback: str) -> str:
    if raw_name:
        return raw_name.strip()
    if context:
        return context.strip()
    return fallback.strip()


def _parse_dimension_item(line: str, section_key: str, context: str | None) -> ExtractedDimensionItem | None:
    count = 1.0
    working = line
    plain_dims_match = re.match(
        r"^(?P<dims>\d+(?:\.\d+)?\s*m?\s*[×x]\s*\d+(?:\.\d+)?\s*m?(?:\s*[×x]\s*\d+(?:\.\d+)?\s*m?)?)$",
        line,
        re.IGNORECASE,
    )
    if plain_dims_match:
        if not context:
            return None
        name = context
        working = plain_dims_match.group("dims")
    else:
        counted_box = re.match(
            r"^(?P<count>\d+(?:\.\d+)?)\s*[×x]\s*\((?P<dims>\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?(?:\s*[×x]\s*\d+(?:\.\d+)?)?)\s*m?\)$",
            line,
            re.IGNORECASE,
        )
        if counted_box:
            count = float(counted_box.group("count"))
            working = counted_box.group("dims")
            name = _resolve_item_name(None, context, line)
        else:
            named_match = re.match(
                r"^(?P<name>[A-Za-z][A-Za-z0-9 /+&()'-]*?)(?:\:|\()?\s*(?P<dims>\d+(?:\.\d+)?\s*m?\s*[×x]\s*\d+(?:\.\d+)?\s*m?(?:\s*[×x]\s*\d+(?:\.\d+)?\s*m?)?)(?:\))?$",
                line,
                re.IGNORECASE,
            )
            if not named_match:
                return None
            name = _resolve_item_name(named_match.group("name"), context, line)
            working = named_match.group("dims")
            count_prefix = re.match(r"^(?P<count>\d+(?:\.\d+)?)\s*[×x]\s*(?P<rest>.+)$", name, re.IGNORECASE)
            if count_prefix and "(" in line:
                count = float(count_prefix.group("count"))
                name = count_prefix.group("rest").strip()

    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", working)]
    if len(numbers) < 2:
        return None

    length_m = numbers[0]
    width_m = numbers[1]
    height_m = numbers[2] if len(numbers) >= 3 else None
    area_m2 = _round(count * length_m * width_m)
    volume_m3 = _round(area_m2 * height_m) if height_m else None

    return ExtractedDimensionItem(
        section_key=section_key,
        name=name,
        raw_text=line,
        count=count,
        length_m=length_m,
        width_m=width_m,
        height_m=height_m,
        area_m2=area_m2,
        volume_m3=volume_m3,
    )


def _parse_count_item(line: str, section_key: str, context: str | None) -> ExtractedCountItem | None:
    patterns = (
        re.match(r"^(?P<count>\d+(?:\.\d+)?)\s*[×x]\s*(?P<name>.+)$", line, re.IGNORECASE),
        re.match(r"^Remove\s+(?P<count>\d+(?:\.\d+)?)\s+(?P<name>.+)$", line, re.IGNORECASE),
        re.match(r"^(?P<name>.+?)\((?P<count>\d+(?:\.\d+)?)\s*total\)$", line, re.IGNORECASE),
        re.match(r"^(?P<name>.+?)\((?P<count>\d+(?:\.\d+)?)\)$", line, re.IGNORECASE),
        re.match(r"^(?P<count>\d+(?:\.\d+)?)\s*\((?P<name>[^)]+)\)$", line, re.IGNORECASE),
        re.match(r"^(?P<count>\d+(?:\.\d+)?)\s+(?P<name>[A-Za-z][A-Za-z +/&-]+)$", line, re.IGNORECASE),
        re.match(r"^(?P<name>Total posts|Junctions|Radiators|Sockets|Switches|Extractor fans)\s*:\s*(?P<count>\d+(?:\.\d+)?)$", line, re.IGNORECASE),
    )
    for match in patterns:
        if not match:
            continue
        raw_name = match.groupdict().get("name")
        if raw_name and raw_name.lower() in ROOM_LEVELS_LOWER and context:
            name = context
        else:
            name = _resolve_item_name(raw_name, context, line)
        raw_name = name.lower()
        if any(token in raw_name for token in ("m2", "m3", " m ", "£")):
            continue
        return ExtractedCountItem(
            section_key=section_key,
            name=name,
            quantity=float(match.group("count")),
            raw_text=line,
        )
    if line.lower() in {"tree + roots", "tree", "fence", "fences"}:
        return ExtractedCountItem(
            section_key=section_key,
            name=line,
            quantity=1.0,
            raw_text=line,
        )
    return None


def _parse_measure_item(line: str, section_key: str, context: str | None) -> ExtractedMeasureItem | None:
    colon_match = re.match(
        r"^(?P<label>[A-Za-z /&()+-]+):\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>m³|m3|m2|m²|m|mm|lm|l)?$",
        line,
        re.IGNORECASE,
    )
    if colon_match:
        label = colon_match.group("label").strip()
        unit = (colon_match.group("unit") or "").lower().replace("m³", "m3").replace("m²", "m2") or None
        name = f"{context} {label}".strip() if context else label
        return ExtractedMeasureItem(
            section_key=section_key,
            name=name,
            value=float(colon_match.group("value")),
            unit=unit,
            raw_text=line,
        )

    inline_match = re.match(
        r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>m³|m3|m2|m²|m|mm|lm|l)\s*(?P<label>.+)$",
        line,
        re.IGNORECASE,
    )
    if inline_match:
        return ExtractedMeasureItem(
            section_key=section_key,
            name=_resolve_item_name(inline_match.group("label"), context, line),
            value=float(inline_match.group("value")),
            unit=inline_match.group("unit").lower().replace("m³", "m3").replace("m²", "m2"),
            raw_text=line,
        )

    dash_match = re.match(
        r"^(?P<label>.+?)\s*-\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>m³|m3|m2|m²|m|mm|lm|l)$",
        line,
        re.IGNORECASE,
    )
    if dash_match:
        return ExtractedMeasureItem(
            section_key=section_key,
            name=_resolve_item_name(dash_match.group("label"), context, line),
            value=float(dash_match.group("value")),
            unit=dash_match.group("unit").lower().replace("m³", "m3").replace("m²", "m2"),
            raw_text=line,
        )

    return None


def extract_scope(prompt: str) -> ScopeExtractionResponse:
    lines = _prepare_prompt_lines(prompt)

    property_context = ExtractedPropertyContext()
    rooms: list[ExtractedRoom] = []
    sections: dict[str, ExtractedSection] = {}
    assumptions: list[PreviewAssumption] = []

    current_section_key: str | None = None
    current_room_level: str | None = None
    current_context: str | None = None
    default_room_level = _infer_default_room_level(prompt)

    for line in lines:
        section_key = _section_key_for_line(line)
        if section_key and current_context and not line.isupper():
            section_key = None
        if section_key:
            current_section_key = section_key
            current_context = None
            if section_key == "floor_areas_rooms":
                current_room_level = None
            sections.setdefault(
                section_key,
                ExtractedSection(key=section_key, title=SECTION_TITLES[section_key]),
            )
            continue

        if current_section_key == "floor_areas_rooms" and line in ROOM_LEVELS:
            current_room_level = line
            current_context = line
            continue

        if current_section_key == "property":
            if line.lower().startswith("type:"):
                property_context.property_type = line.split(":", 1)[1].strip()
                continue
            if line.lower().startswith("location:"):
                property_context.location = line.split(":", 1)[1].strip()
                continue

        if current_section_key == "full_scope" and ":" not in line:
            property_context.project_scopes.extend(_split_project_scope_line(line))
            continue

        if current_section_key == "floor_areas_rooms":
            room = _parse_room(line, current_room_level or default_room_level)
            if room:
                rooms.append(room)
                continue

        if not current_section_key:
            continue

        section = sections.setdefault(
            current_section_key,
            ExtractedSection(key=current_section_key, title=SECTION_TITLES[current_section_key]),
        )

        if line.endswith(":") and not re.match(r"^(Type|Location):", line, re.IGNORECASE):
            current_context = line[:-1].strip()
            section.lines.append(line)
            continue

        section.lines.append(line)

        dimension_item = _parse_dimension_item(line, current_section_key, current_context)
        if dimension_item:
            section.dimension_items.append(dimension_item)

        count_item = _parse_count_item(line, current_section_key, current_context)
        if count_item:
            section.count_items.append(count_item)

        measure_item = _parse_measure_item(line, current_section_key, current_context)
        if measure_item:
            section.measure_items.append(measure_item)

    if not rooms:
        assumptions.append(
            PreviewAssumption(
                text="No room schedule was parsed from the prompt. Floor and room takeoff values will stay incomplete until room dimensions are provided.",
                kind="room_schedule_missing",
                severity="warning",
            )
        )

    if any(scope.lower() in {"rear extension", "loft conversion", "full house refurbishment", "full house refurb"} for scope in property_context.project_scopes):
        assumptions.append(
            PreviewAssumption(
                text="This prompt contains a large multi-scope project. Use the extracted takeoff as a draft baseline, then review structural, MEP, and allowance items before final pricing.",
                kind="large_scope_review",
                severity="info",
            )
        )

    room_takeoff = build_room_takeoff_with_overrides(rooms, list(sections.values()))
    takeoff_summary = build_takeoff_summary(
        rooms,
        list(sections.values()),
        project_scopes=property_context.project_scopes,
        room_takeoff=room_takeoff,
    )

    if rooms:
        assumptions.append(
            PreviewAssumption(
                text=(
                    "Derived wall, ceiling, skirting, boarding, insulation, and loft/roof takeoff uses AI-side room-height and build-up heuristics "
                    f"({TAKEOFF_HEURISTICS_SOURCE} {TAKEOFF_HEURISTICS_VERSION}). Treat these values as a strong draft baseline, then adjust on review where the layout is unusual."
                ),
                kind="derived_finish_takeoff",
                severity="info",
            )
        )

    if takeoff_summary.door_set_count and takeoff_summary.door_supply_allowance_total:
        assumptions.append(
            PreviewAssumption(
                text="Door supply allowance was calculated from counted door sets, not individual leaves. Review double pocket door pricing if the allowance should be per leaf instead.",
                kind="door_allowance_method",
                severity="warning",
            )
        )

    if takeoff_summary.client_supply_sanitaryware:
        assumptions.append(
            PreviewAssumption(
                text="The prompt indicates sanitaryware is client supplied. Keep AI draft rows focused on install labour and review any sanitary supply/material rows before apply.",
                kind="client_supply_sanitaryware",
                severity="warning",
            )
        )

    if takeoff_summary.client_supply_appliances:
        assumptions.append(
            PreviewAssumption(
                text="The prompt indicates appliances are client supplied. Keep AI draft rows focused on connection / install labour and review appliance supply rows before apply.",
                kind="client_supply_appliances",
                severity="warning",
            )
        )

    return ScopeExtractionResponse(
        property_context=property_context,
        rooms=rooms,
        room_takeoff=room_takeoff,
        sections=list(sections.values()),
        takeoff_summary=takeoff_summary,
        assumptions=assumptions,
        error_text="",
    )
