from __future__ import annotations

import re

from app.schemas import (
    ClarificationQuestion,
    ClarificationQuestionOption,
    PromptNormalizationRequest,
    PromptNormalizationResponse,
    PromptBlocker,
)
from app.services.scope_extractor import extract_scope
from app.services.scope_parsing import (
    ROOM_LEVEL_ORDER,
    SECTION_HEADING_ALIASES,
    SECTION_TITLES,
    clean_line,
    infer_default_room_level,
    normalize_heading,
    prepare_prompt_lines,
    split_project_scope_line,
)

_NORMALIZED_SECTION_ORDER = [
    "property",
    "full_scope",
    "floor_areas_rooms",
    "demolition",
    "groundworks",
    "steelworks",
    "structure",
    "carpentry",
    "ceilings_and_insulation",
    "doors",
    "joinery",
    "plastering",
    "tiling",
    "plumbing",
    "heating",
    "electrical",
    "decorating",
    "flooring",
    "windows",
    "woodwork_painting",
    "preliminaries",
    "commercial_settings",
]
_NORMALIZE_ONLY_SECTION_TITLES = {
    "preliminaries": "Preliminaries",
    "commercial_settings": "Commercial Settings",
}
_ROOM_LEVEL_LOOKUP = {level.lower(): level for level in ROOM_LEVEL_ORDER}
_LOOSE_HEADING_ALIASES = {
    **{section_key: section_key for section_key in SECTION_TITLES},
    **SECTION_HEADING_ALIASES,
    "details": "property",
    "walls": "plastering",
    "bathroom": "bathroom_context",
    "kitchen": "kitchen_context",
}
_START_HEADING_ALIASES = {
    key: value
    for key, value in _LOOSE_HEADING_ALIASES.items()
    if key not in {"details", "walls"}
}
_START_HEADING_PATTERNS = sorted(
    _START_HEADING_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
)
_KEYWORD_SECTION_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("preliminaries", ("wait", "load")),
    ("commercial_settings", ("materials", "markup")),
    ("commercial_settings", ("labour", "%")),
    ("commercial_settings", ("profit", "%")),
    ("demolition", ("strip",)),
    ("demolition", ("remove",)),
    ("demolition", ("demolition",)),
    ("groundworks", ("foundation",)),
    ("groundworks", ("trench",)),
    ("groundworks", ("pad",)),
    ("groundworks", ("soil",)),
    ("groundworks", ("patio",)),
    ("steelworks", ("steel",)),
    ("steelworks", ("beam",)),
    ("steelworks", ("ukb",)),
    ("steelworks", ("uc",)),
    ("steelworks", ("column",)),
    ("steelworks", ("post",)),
    ("steelworks", ("junction",)),
    ("structure", ("roof",)),
    ("structure", ("brick",)),
    ("structure", ("block",)),
    ("structure", ("dormer",)),
    ("structure", ("slab",)),
    ("structure", ("cavity",)),
    ("carpentry", ("stud",)),
    ("carpentry", ("chipboard",)),
    ("carpentry", ("joist",)),
    ("carpentry", ("acoustilay",)),
    ("ceilings_and_insulation", ("ceiling",)),
    ("ceilings_and_insulation", ("plasterboard",)),
    ("ceilings_and_insulation", ("insulation",)),
    ("ceilings_and_insulation", ("rockwool",)),
    ("ceilings_and_insulation", ("pir",)),
    ("tiling", ("tile",)),
    ("tiling", ("tiling",)),
    ("tiling", ("tanking",)),
    ("doors", ("door",)),
    ("doors", ("eclisse",)),
    ("joinery", ("stair",)),
    ("joinery", ("skirting",)),
    ("joinery", ("architrave",)),
    ("joinery", ("window board",)),
    ("plastering", ("skim",)),
    ("plastering", ("plaster",)),
    ("plastering", ("make good",)),
    ("plumbing", ("basin",)),
    ("plumbing", ("wc",)),
    ("plumbing", ("toilet",)),
    ("plumbing", ("bath",)),
    ("plumbing", ("shower",)),
    ("plumbing", ("sanitary",)),
    ("plumbing", ("radiator",)),
    ("plumbing", ("pipework",)),
    ("plumbing", ("underfloor heating",)),
    ("plumbing", ("ufh",)),
    ("heating", ("boiler",)),
    ("heating", ("megaflo",)),
    ("heating", ("cylinder",)),
    ("heating", ("gas",)),
    ("electrical", ("rewire",)),
    ("electrical", ("consumer", "unit")),
    ("electrical", ("downlight",)),
    ("electrical", ("spotlight",)),
    ("electrical", ("socket",)),
    ("electrical", ("switch",)),
    ("electrical", ("extractor",)),
    ("electrical", ("appliance", "connection")),
    ("electrical", ("thermostat",)),
    ("decorating", ("paint",)),
    ("decorating", ("mist",)),
    ("decorating", ("coat",)),
    ("decorating", ("dulux",)),
    ("flooring", ("flooring",)),
    ("flooring", ("engineered",)),
    ("flooring", ("laminate",)),
    ("flooring", ("vinyl",)),
    ("flooring", ("carpet",)),
    ("flooring", ("floor", "level")),
    ("windows", ("velux",)),
    ("windows", ("window",)),
    ("windows", ("upvc",)),
    ("woodwork_painting", ("woodwork",)),
    ("woodwork_painting", ("architrave", "paint")),
    ("woodwork_painting", ("skirting", "paint")),
]
_BATHROOM_FIXTURE_GROUPS: dict[str, tuple[str, ...]] = {
    "wc": ("wc", "toilet"),
    "basin": ("basin",),
    "bath": ("bath",),
    "shower": ("shower",),
}
_SUPPLY_MARKERS = (
    "client supply",
    "client supplied",
    "by others",
    "contractor supply",
    "supply allowance",
    "supply and fit",
    "supply & fit",
)
_STEEL_AMBIGUITY_MARKERS = (
    "multiple",
    "sizes provided",
    "counts given",
    "total provided",
    "provided",
)
_WASTE_ACCESS_AMBIGUITY_MARKERS = (
    "no skip access",
    "restricted access",
    "through the house",
    "through house",
    "long carry",
    "stair access",
    "stairs only",
    "bag out",
    "rubble bags",
    "waste bags",
)
_WASTE_ACCESS_EXPLICIT_MARKERS = (
    "wait and load",
    "wait & load",
    "skip on driveway",
    "skip in driveway",
    "grab lorry",
    "market allowance",
)
_CLARIFICATION_PLACEHOLDER_VALUES = {
    "tbc",
    "unknown",
    "unsure",
    "n/a",
    "na",
    "same as prompt",
}
_ROOM_CAPTURE_CONTEXTS = {None, "property", "full_scope", "floor_areas_rooms"}
_NON_ROOM_DIMENSION_TOKENS = ("dormer",)


def _normalize_heading(line: str) -> str:
    return normalize_heading(line)


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip().rstrip(":")
    if len(stripped) < 3 or len(stripped) > 48:
        return False
    if _room_line_match(stripped):
        return False
    if any(character.isdigit() for character in stripped):
        return False
    if stripped.isupper():
        return True
    return bool(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9 &/()+-]{2,}", stripped)
    ) and line.strip().endswith(":")


def _is_room_level(line: str) -> bool:
    return line.lower().rstrip(":") in _ROOM_LEVEL_LOOKUP


def _canonical_room_level(line: str) -> str | None:
    return _ROOM_LEVEL_LOOKUP.get(line.lower().rstrip(":"))


def _infer_default_room_level(prompt: str) -> str | None:
    return infer_default_room_level(prompt)


def _split_inline_scope_clauses(
    line: str, current_heading_hint: str | None
) -> list[str]:
    lowered = line.lower()
    if current_heading_hint not in {None, "full_scope"}:
        return [line]
    if lowered.startswith(("type:", "property:", "location:")):
        return [line]
    if "," not in line and ";" not in line:
        return [line]
    split_lines = [part.strip() for part in re.split(r"[;,]", line) if part.strip()]
    if len(split_lines) <= 1:
        return [line]
    return split_lines


def _append_room_clause_if_possible(
    clause: str,
    *,
    current_heading_hint: str | None,
    current_room_level: str | None,
    default_room_level: str | None,
    rooms_by_level: dict[str, list[str]],
    unassigned_room_lines: list[str],
) -> bool:
    room_match = _room_line_match(clause)
    if not room_match:
        return False
    room_name = room_match.group("name").strip()
    if current_heading_hint not in _ROOM_CAPTURE_CONTEXTS:
        return False
    if any(token in room_name.lower() for token in _NON_ROOM_DIMENSION_TOKENS):
        return False

    normalized_room_line = (
        f"{room_name}: {room_match.group('width')} x {room_match.group('length')}"
    )
    level = current_room_level or default_room_level
    if level:
        rooms_by_level.setdefault(level, []).append(normalized_room_line)
    else:
        unassigned_room_lines.append(normalized_room_line)
    return True


def _contains_keyword_phrase(text: str, keyword: str) -> bool:
    normalized_keyword = keyword.strip().lower()
    if not normalized_keyword:
        return False
    if re.search(r"[^a-z0-9\s]", normalized_keyword):
        return normalized_keyword in text
    pattern = (
        r"\b"
        + r"\W+".join(re.escape(part) for part in normalized_keyword.split())
        + r"\b"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def _heading_key_pattern(heading_key: str) -> str:
    parts = [re.escape(part) for part in heading_key.split("_") if part]
    return r"[\W_]*".join(parts)


def _extract_heading_at_start(
    line: str, current_heading_hint: str | None
) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or re.match(
        r"^(type|location|property)\s*:", stripped, re.IGNORECASE
    ):
        return None
    for heading_alias, heading_key in _START_HEADING_PATTERNS:
        pattern = re.compile(
            rf"^[^A-Za-z0-9]*(?:{_heading_key_pattern(heading_alias)})(?:\s*\((?P<paren>[^)]*)\))?(?:\s*[:—–-]\s*(?P<rest>.*))?$",
            re.IGNORECASE,
        )
        match = pattern.match(stripped)
        if not match:
            continue
        if heading_key in {"bathroom_context", "kitchen_context"}:
            context_heading_pattern = re.compile(
                rf"^[^A-Za-z0-9]*(?:{_heading_key_pattern(heading_alias)})(?:\s*\(|\s*:)",
                re.IGNORECASE,
            )
            if not context_heading_pattern.match(stripped):
                continue
            heading_lead = stripped.split(":", 1)[0].strip()
            looks_like_context_section_heading = bool(
                re.match(r"^[A-Z][A-Z0-9 &/()'-]*$", heading_lead)
                or re.match(
                    rf"^[^A-Za-z0-9]*(?:{_heading_key_pattern(heading_alias)})\s*\(",
                    stripped,
                    re.IGNORECASE,
                )
            )
            if (
                current_heading_hint in _NORMALIZED_SECTION_ORDER
                and current_heading_hint not in {"full_scope", "floor_areas_rooms"}
            ):
                if not looks_like_context_section_heading:
                    continue
        remainder = (match.group("rest") or "").strip()
        prefix_note = (match.group("paren") or "").strip()
        if prefix_note:
            remainder = f"{prefix_note} {remainder}".strip()
        return heading_key, remainder
    return None


def _extract_heading_at_end(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    for heading_alias, heading_key in _START_HEADING_PATTERNS:
        pattern = re.compile(
            rf"^(?P<before>.+?)\s+[^A-Za-z0-9]*\s*(?P<heading>{_heading_key_pattern(heading_alias)})$",
            re.IGNORECASE,
        )
        match = pattern.match(stripped)
        if not match:
            continue
        heading_text = (match.group("heading") or "").strip()
        if heading_text.upper() != heading_text:
            continue
        before = (match.group("before") or "").strip()
        if before:
            return heading_key, before
    return None


def _route_line_to_section(line: str, current_heading_hint: str | None) -> str | None:
    lowered = line.lower()
    if lowered.startswith("type:") or lowered.startswith("location:"):
        return "property"
    if current_heading_hint == "property":
        return "property"
    if current_heading_hint == "floor_areas_rooms":
        return "floor_areas_rooms"
    if current_heading_hint == "bathroom_context":
        for section_key, keywords in _KEYWORD_SECTION_RULES:
            if all(_contains_keyword_phrase(lowered, keyword) for keyword in keywords):
                return section_key
        return "full_scope"
    if current_heading_hint == "kitchen_context":
        for section_key, keywords in _KEYWORD_SECTION_RULES:
            if all(_contains_keyword_phrase(lowered, keyword) for keyword in keywords):
                return section_key
        if any(
            _contains_keyword_phrase(lowered, keyword)
            for keyword in ("kitchen", "unit", "units", "cabinet", "worktop")
        ):
            return "joinery"
        return "full_scope"
    if current_heading_hint == "full_scope" and ":" not in line:
        return "full_scope"
    if current_heading_hint in _NORMALIZED_SECTION_ORDER:
        return current_heading_hint
    for section_key, keywords in _KEYWORD_SECTION_RULES:
        if all(_contains_keyword_phrase(lowered, keyword) for keyword in keywords):
            return section_key
    if re.search(
        r"\b(refurb|refurbishment|renovation|conversion|extension|fit out|fit-out)\b",
        lowered,
    ):
        return "full_scope"
    return None


def _format_answer_value(value: str | float | bool | list[str]) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value).strip()


def _merge_prompt_with_answers(prompt: str, answers: list) -> str:
    if not answers:
        return prompt
    lines = [prompt.strip(), "", "CLARIFICATIONS"]
    for answer in answers:
        formatted = _format_answer_value(answer.value)
        if not formatted:
            continue
        lines.append(f"{answer.question_id}: {formatted}")
    return "\n".join(line for line in lines if line is not None).strip()


def _has_answered_clarification(prompt: str, question_id: str) -> bool:
    match = re.search(
        rf"^\s*{re.escape(question_id)}\s*:\s*(.+)$",
        prompt,
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return False
    answer = match.group(1).strip()
    if not answer:
        return False
    return answer.lower() not in _CLARIFICATION_PLACEHOLDER_VALUES


def _room_line_match(line: str) -> re.Match[str] | None:
    return re.match(
        r"^(?P<name>.+?):\s*(?:approx\.?\s*)?(?P<width>\d+(?:\.\d+)?)\s*(?:m)?\s*[×x]\s*(?P<length>\d+(?:\.\d+)?)\s*(?:m)?(?:\s*[=,].*)?$",
        line,
        re.IGNORECASE,
    )


def _collect_structured_sections(
    prompt: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str], dict[str, str]]:
    lines = prepare_prompt_lines(prompt)
    sections = {key: [] for key in _NORMALIZED_SECTION_ORDER}
    rooms_by_level = {level: [] for level in ROOM_LEVEL_ORDER}
    property_values: dict[str, str] = {}
    current_heading_hint: str | None = None
    current_room_level = _infer_default_room_level(prompt)
    default_room_level = current_room_level
    room_schedule_active = False

    for raw_line in lines:
        line = clean_line(raw_line)
        if not line:
            continue

        if room_schedule_active and _is_room_level(line):
            current_room_level = _canonical_room_level(line)
            current_heading_hint = "floor_areas_rooms"
            continue

        if room_schedule_active:
            room_match = _room_line_match(line)
            if room_match:
                level = current_room_level or default_room_level
                if level:
                    rooms_by_level.setdefault(level, []).append(
                        f"{room_match.group('name').strip()}: {room_match.group('width')} x {room_match.group('length')}"
                    )
                    continue

        start_heading = _extract_heading_at_start(line, current_heading_hint)
        if start_heading:
            heading_key, remainder = start_heading
            current_heading_hint = heading_key
            room_schedule_active = heading_key == "floor_areas_rooms"
            if room_schedule_active:
                current_room_level = default_room_level
            if not remainder:
                continue
            for clause in _split_inline_scope_clauses(remainder, current_heading_hint):
                if _append_room_clause_if_possible(
                    clause,
                    current_heading_hint=current_heading_hint,
                    current_room_level=current_room_level,
                    default_room_level=default_room_level,
                    rooms_by_level=rooms_by_level,
                    unassigned_room_lines=sections["floor_areas_rooms"],
                ):
                    continue
                routed_section = _route_line_to_section(clause, current_heading_hint)
                if routed_section == "property":
                    if clause.lower().startswith("type:"):
                        property_values["Type"] = clause.split(":", 1)[1].strip()
                    elif clause.lower().startswith("property:"):
                        property_values["Type"] = clause.split(":", 1)[1].strip()
                    elif clause.lower().startswith("location:"):
                        property_values["Location"] = clause.split(":", 1)[1].strip()
                    else:
                        sections["property"].append(clause)
                    continue
                if routed_section == "full_scope":
                    sections["full_scope"].extend(split_project_scope_line(clause))
                    continue
                if routed_section == "floor_areas_rooms":
                    room_match = _room_line_match(clause)
                    if room_match:
                        level = current_room_level or default_room_level
                        if level:
                            rooms_by_level.setdefault(level, []).append(
                                f"{room_match.group('name').strip()}: {room_match.group('width')} x {room_match.group('length')}"
                            )
                            continue
                    sections["floor_areas_rooms"].append(clause)
                    continue
                if routed_section:
                    sections[routed_section].append(clause)
            continue

        end_heading = _extract_heading_at_end(line)
        if end_heading:
            heading_key, retained_line = end_heading
            if retained_line:
                if _append_room_clause_if_possible(
                    retained_line,
                    current_heading_hint=current_heading_hint,
                    current_room_level=current_room_level,
                    default_room_level=default_room_level,
                    rooms_by_level=rooms_by_level,
                    unassigned_room_lines=sections["floor_areas_rooms"],
                ):
                    current_heading_hint = heading_key
                    room_schedule_active = heading_key == "floor_areas_rooms"
                    if room_schedule_active:
                        current_room_level = default_room_level
                    continue
                routed_section = _route_line_to_section(
                    retained_line, current_heading_hint
                )
                if routed_section == "full_scope":
                    sections["full_scope"].extend(
                        split_project_scope_line(retained_line)
                    )
                elif routed_section == "floor_areas_rooms":
                    room_match = _room_line_match(retained_line)
                    if room_match:
                        level = current_room_level or default_room_level
                        if level:
                            rooms_by_level.setdefault(level, []).append(
                                f"{room_match.group('name').strip()}: {room_match.group('width')} x {room_match.group('length')}"
                            )
                    else:
                        sections["floor_areas_rooms"].append(retained_line)
                elif routed_section:
                    sections[routed_section].append(retained_line)
            current_heading_hint = heading_key
            room_schedule_active = heading_key == "floor_areas_rooms"
            if room_schedule_active:
                current_room_level = default_room_level
            continue

        normalized_heading = _normalize_heading(line)
        heading_key = _LOOSE_HEADING_ALIASES.get(normalized_heading)
        if heading_key:
            current_heading_hint = heading_key
            room_schedule_active = heading_key == "floor_areas_rooms"
            if heading_key == "floor_areas_rooms":
                current_room_level = default_room_level
            continue
        if _looks_like_heading(line):
            current_heading_hint = None
            room_schedule_active = False
            continue

        for clause in _split_inline_scope_clauses(line, current_heading_hint):
            if _append_room_clause_if_possible(
                clause,
                current_heading_hint=current_heading_hint,
                current_room_level=current_room_level,
                default_room_level=default_room_level,
                rooms_by_level=rooms_by_level,
                unassigned_room_lines=sections["floor_areas_rooms"],
            ):
                continue
            routed_section = _route_line_to_section(clause, current_heading_hint)
            if routed_section == "property":
                if clause.lower().startswith("type:"):
                    property_values["Type"] = clause.split(":", 1)[1].strip()
                elif clause.lower().startswith("property:"):
                    property_values["Type"] = clause.split(":", 1)[1].strip()
                elif clause.lower().startswith("location:"):
                    property_values["Location"] = clause.split(":", 1)[1].strip()
                else:
                    sections["property"].append(clause)
                continue

            if routed_section == "full_scope":
                sections["full_scope"].extend(split_project_scope_line(clause))
                continue

            if routed_section == "floor_areas_rooms":
                room_match = _room_line_match(clause)
                if room_match:
                    level = current_room_level or default_room_level
                    if level:
                        rooms_by_level.setdefault(level, []).append(
                            f"{room_match.group('name').strip()}: {room_match.group('width')} x {room_match.group('length')}"
                        )
                        continue
                sections["floor_areas_rooms"].append(clause)
                continue

            if routed_section:
                sections[routed_section].append(clause)

    return sections, rooms_by_level, lines, property_values


def _build_normalized_prompt_markdown(
    sections: dict[str, list[str]],
    rooms_by_level: dict[str, list[str]],
    property_values: dict[str, str],
) -> str:
    blocks: list[str] = []

    property_block: list[str] = []
    if property_values.get("Type"):
        property_block.append(f"Type: {property_values['Type']}")
    if property_values.get("Location"):
        property_block.append(f"Location: {property_values['Location']}")
    property_block.extend(
        line for line in sections["property"] if line not in property_block
    )
    if property_block:
        blocks.append("PROPERTY:\n" + "\n".join(property_block))

    full_scope_lines = [line for line in sections["full_scope"] if line]
    if full_scope_lines:
        blocks.append("FULL SCOPE:\n" + "\n".join(full_scope_lines))

    room_block_lines: list[str] = []
    for level in ROOM_LEVEL_ORDER:
        room_lines = rooms_by_level.get(level) or []
        if not room_lines:
            continue
        room_block_lines.append(level)
        room_block_lines.extend(room_lines)
    room_block_lines.extend(
        line for line in sections["floor_areas_rooms"] if line not in room_block_lines
    )
    if room_block_lines:
        blocks.append("FLOOR AREAS & ROOMS:\n" + "\n".join(room_block_lines))

    for section_key in _NORMALIZED_SECTION_ORDER:
        if section_key in {"property", "full_scope", "floor_areas_rooms"}:
            continue
        lines = sections.get(section_key) or []
        if not lines:
            continue
        title = (
            SECTION_TITLES.get(section_key)
            or _NORMALIZE_ONLY_SECTION_TITLES.get(section_key)
            or section_key.replace("_", " ").title()
        )
        blocks.append(f"{title.upper()}:\n" + "\n".join(lines))

    return "\n\n".join(blocks).strip()


def _bathroom_fixture_question(
    prompt: str, normalized_scope
) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    if not any(
        keyword in normalized
        for keyword in ("bathroom", "ensuite", "shower room", "wc", "sanitary")
    ):
        return None
    if _has_answered_clarification(prompt, "bathroom_fixture_scope"):
        return None
    fixture_hits = {
        fixture_name
        for fixture_name, keywords in _BATHROOM_FIXTURE_GROUPS.items()
        if any(keyword in normalized for keyword in keywords)
    }
    if len(fixture_hits) >= 2:
        return None
    if normalized_scope.takeoff_summary.total_sanitary_set_count > 0:
        return None
    return ClarificationQuestion(
        id="bathroom_fixture_scope",
        label="Which bathroom fixtures are in scope?",
        reason="Bathroom fit-out was mentioned without a complete sanitary fixture set, which can materially change plumbing, tiling, and install rows.",
        answer_type="multi_select",
        options=[
            ClarificationQuestionOption(value="wc", label="WC"),
            ClarificationQuestionOption(value="basin", label="Basin"),
            ClarificationQuestionOption(value="bath", label="Bath"),
            ClarificationQuestionOption(value="shower", label="Shower"),
        ],
        applies_to_section="Plumbing",
    )


def _room_level_question(prompt: str, normalized_scope) -> ClarificationQuestion | None:
    if normalized_scope.rooms:
        return None
    if not any(_room_line_match(line) for line in prepare_prompt_lines(prompt)):
        return None
    if _has_answered_clarification(prompt, "room_level"):
        return None
    if _infer_default_room_level(prompt):
        return None
    return ClarificationQuestion(
        id="room_level",
        label="Which level do these room dimensions belong to?",
        reason="Room dimensions were detected, but the prompt does not say whether they sit on the ground, first, second, basement, or loft level.",
        answer_type="single_select",
        options=[
            ClarificationQuestionOption(value="Ground Floor", label="Ground Floor"),
            ClarificationQuestionOption(value="First Floor", label="First Floor"),
            ClarificationQuestionOption(value="Second Floor", label="Second Floor"),
            ClarificationQuestionOption(value="Basement", label="Basement"),
            ClarificationQuestionOption(value="Loft", label="Loft"),
        ],
        applies_to_section="Floor Areas & Rooms",
    )


def _steel_question(prompt: str) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    if not any(
        keyword in normalized
        for keyword in ("steel", "beam", "ukb", "uc", "column", "post")
    ):
        return None
    if _has_answered_clarification(prompt, "steel_scope_detail"):
        return None
    if not any(marker in normalized for marker in _STEEL_AMBIGUITY_MARKERS):
        return None
    return ClarificationQuestion(
        id="steel_scope_detail",
        label="What steel sizes, counts, or exact beam list should be priced?",
        reason="The prompt references steelwork with placeholder wording like 'multiple' or 'sizes provided', so the estimator rows may stay incomplete without the exact schedule.",
        answer_type="short_text",
        applies_to_section="Steelworks",
    )


def _supply_question(prompt: str) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    if not any(
        keyword in normalized
        for keyword in (
            "sanitaryware",
            "appliance",
            "appliances",
            "kitchen",
            "kitchen units",
        )
    ):
        return None
    if _has_answered_clarification(prompt, "supply_split"):
        return None
    if any(marker in normalized for marker in _SUPPLY_MARKERS) or re.search(
        r"\b(supply|supplied|supply only)\b", normalized
    ):
        return None
    if not (
        "sanitaryware" in normalized
        or re.search(
            r"(kitchen|appliance|kitchen units?).{0,40}(install|fit|fitting|supply|supplied|units|refurb)",
            normalized,
        )
        or re.search(
            r"(install|fit|fitting|supply|supplied|units|refurb).{0,40}(kitchen|appliance|kitchen units?)",
            normalized,
        )
    ):
        return None
    return ClarificationQuestion(
        id="supply_split",
        label="Are kitchens, sanitaryware, or appliances client supplied or contractor supplied?",
        reason="Supply responsibility is not explicit, so the draft may mix install-only rows with supply-and-fit rows.",
        answer_type="single_select",
        options=[
            ClarificationQuestionOption(value="client_supply", label="Client supplied"),
            ClarificationQuestionOption(
                value="contractor_supply", label="Contractor supplied"
            ),
            ClarificationQuestionOption(
                value="mixed_or_unsure", label="Mixed / unsure"
            ),
        ],
        applies_to_section="Commercial Settings",
    )


def _tiling_question(prompt: str, normalized_scope) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    if "tile" not in normalized and "tiling" not in normalized:
        return None
    if _has_answered_clarification(prompt, "tiling_area_scope"):
        return None
    if normalized_scope.takeoff_summary.estimated_wet_room_wall_tiling_area_m2 > 0:
        return None
    if normalized_scope.takeoff_summary.estimated_wet_room_floor_tiling_area_m2 > 0:
        return None
    if re.search(r"\d+(?:\.\d+)?\s*(?:m2|m²)", normalized, re.IGNORECASE):
        return None
    return ClarificationQuestion(
        id="tiling_area_scope",
        label="What tiling area or which rooms should be tiled?",
        reason="Tiling was requested without a usable room geometry or measured area, so the quantity will otherwise stay generic.",
        answer_type="short_text",
        applies_to_section="Tiling",
    )


def _scaffold_question(prompt: str) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    if not any(
        keyword in normalized
        for keyword in (
            "scaffold",
            "scaffolding",
            "temporary roof",
            "rubbish chute",
            "hoist",
        )
    ):
        return None
    if _has_answered_clarification(prompt, "scaffold_allowance_scope"):
        return None
    return ClarificationQuestion(
        id="scaffold_allowance_scope",
        label="Is scaffolding or access plant required, and should it be carried as allowance or subcontract quote?",
        reason="Scaffold and related access costs must not be auto-priced from internal rates, so the draft needs an explicit allowance / subcontract direction first.",
        answer_type="short_text",
        applies_to_section="Preliminaries / Access",
    )


def _permit_cdm_question(prompt: str) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    keywords = (
        "cdm",
        "principal contractor",
        "skip permit",
        "road permit",
        "parking suspension",
        "parking permit",
        "highway permit",
    )
    if not any(keyword in normalized for keyword in keywords):
        return None
    if _has_answered_clarification(prompt, "permit_cdm_scope"):
        return None
    return ClarificationQuestion(
        id="permit_cdm_scope",
        label="Which permit or CDM costs actually apply, and who is carrying them?",
        reason="Permit / CDM wording was detected, but these costs should not be guessed from standard rates or included until ownership and inclusion are confirmed.",
        answer_type="short_text",
        applies_to_section="Commercial Settings",
    )


def _allowance_question(prompt: str) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    keywords = (
        "client allowance",
        "architect allowance",
        "provisional sum",
        "pc sum",
        "prime cost",
        "supply allowance",
    )
    if not any(keyword in normalized for keyword in keywords):
        return None
    if _has_answered_clarification(prompt, "allowance_scope_detail"):
        return None
    return ClarificationQuestion(
        id="allowance_scope_detail",
        label="Should the allowance be carried as-is, replaced, or excluded?",
        reason="Allowance wording was detected. The draft should not silently turn that into ordinary internal pricing without an explicit estimator decision.",
        answer_type="short_text",
        applies_to_section="Commercial Settings",
    )


def _ceiling_height_question(
    prompt: str, normalized_scope
) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    has_wall_work = any(
        kw in normalized
        for kw in (
            "plaster",
            "skim",
            "emulsion",
            "paint wall",
            "paint ceil",
            "decor",
            "tiling",
            "tile",
        )
    )
    has_rooms = bool(normalized_scope.rooms)
    if not has_wall_work or not has_rooms:
        return None
    already_given = bool(
        re.search(
            r"ceiling\s*height|floor.to.ceiling|storey\s*height|\d+\.?\d*\s*m\s+(?:ceiling|height)",
            normalized,
            re.IGNORECASE,
        )
    )
    if already_given:
        return None
    if _has_answered_clarification(prompt, "ceiling_height"):
        return None
    return ClarificationQuestion(
        id="ceiling_height",
        label="What is the floor-to-ceiling height in the main areas?",
        reason="Wall area calculations for plastering, painting, and tiling depend on ceiling height. Without it the system assumes 2.4m, which can under-price a Victorian terrace by 15–25%.",
        answer_type="single_select",
        options=[
            ClarificationQuestionOption(
                value="2.4m", label="2.4m — standard modern build"
            ),
            ClarificationQuestionOption(
                value="2.55m", label="2.55m — mid-century or newer terrace"
            ),
            ClarificationQuestionOption(
                value="2.75m", label="2.75m — older / pre-war property"
            ),
            ClarificationQuestionOption(
                value="3.0m", label="3.0m — Victorian / period high ceilings"
            ),
            ClarificationQuestionOption(value="varies", label="Varies floor by floor"),
        ],
        applies_to_section="Plastering / Decorating",
        required=False,
    )


def _door_count_question(prompt: str) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    has_woodwork = any(
        kw in normalized for kw in ("gloss", "woodwork", "skirting", "architrave")
    )
    if not has_woodwork:
        return None
    has_explicit_count = bool(
        re.search(r"\b\d+\s*(?:no\.?\s*)?(?:internal\s*)?doors?\b", normalized)
        or re.search(r"\bdoors?\s*[:=]\s*\d+", normalized)
    )
    if has_explicit_count:
        return None
    if _has_answered_clarification(prompt, "door_count"):
        return None
    return ClarificationQuestion(
        id="door_count",
        label="How many internal doors need gloss / woodwork painting?",
        reason="Woodwork painting was requested without a door count. Doors are priced per unit and the total can vary significantly.",
        answer_type="number",
        applies_to_section="Woodwork Painting",
        required=False,
    )


def _waste_access_question(prompt: str) -> ClarificationQuestion | None:
    normalized = prompt.lower()
    if not any(marker in normalized for marker in _WASTE_ACCESS_AMBIGUITY_MARKERS):
        return None
    if any(marker in normalized for marker in _WASTE_ACCESS_EXPLICIT_MARKERS):
        return None
    if _has_answered_clarification(prompt, "waste_access_scope"):
        return None
    return ClarificationQuestion(
        id="waste_access_scope",
        label="How should waste be handled for this restricted-access scope?",
        reason="Waste access constraints were detected, but the prompt does not say whether the job should use skip access, wait & load, rubble bags, or a market allowance path.",
        answer_type="single_select",
        options=[
            ClarificationQuestionOption(
                value="skip_on_driveway", label="Skip / normal skip access"
            ),
            ClarificationQuestionOption(value="wait_load", label="Wait & load"),
            ClarificationQuestionOption(
                value="rubble_bags", label="Rubble bags / bag-out"
            ),
            ClarificationQuestionOption(
                value="market_allowance", label="Carry as market allowance"
            ),
        ],
        applies_to_section="Preliminaries / Waste",
    )


def _build_questions(prompt: str, normalized_scope) -> list[ClarificationQuestion]:
    questions = [
        _room_level_question(prompt, normalized_scope),
        _ceiling_height_question(prompt, normalized_scope),
        _bathroom_fixture_question(prompt, normalized_scope),
        _steel_question(prompt),
        _supply_question(prompt),
        _door_count_question(prompt),
        _scaffold_question(prompt),
        _permit_cdm_question(prompt),
        _allowance_question(prompt),
        _waste_access_question(prompt),
        _tiling_question(prompt, normalized_scope),
    ]
    deduped: list[ClarificationQuestion] = []
    seen_ids: set[str] = set()
    for question in questions:
        if question is None or question.id in seen_ids:
            continue
        seen_ids.add(question.id)
        deduped.append(question)
    return deduped[:9]


def _build_missing_items(questions: list[ClarificationQuestion]) -> list[str]:
    return [question.reason for question in questions]


def _build_blockers(questions: list[ClarificationQuestion]) -> list[PromptBlocker]:
    return [
        PromptBlocker(
            code=question.id,
            title=question.label,
            message=question.reason,
            severity="warning" if question.required else "info",
            scope_label=question.applies_to_section,
            recommended_action="Answer this before AI builds the draft so the estimate can proceed on a reliable scope basis.",
        )
        for question in questions
    ]


def _build_warnings(normalized_scope) -> list[str]:
    warnings: list[str] = []
    for assumption in normalized_scope.assumptions:
        if assumption.severity != "warning":
            continue
        text = (assumption.text or "").strip()
        if text and text not in warnings:
            warnings.append(text)
    return warnings[:6]


def _normalization_confidence(
    normalized_scope, questions: list[ClarificationQuestion]
) -> float:
    recognized_sections = sum(
        1 for section in normalized_scope.sections if section.lines
    )
    confidence = 0.45
    if normalized_scope.property_context.property_type:
        confidence += 0.08
    if normalized_scope.property_context.location:
        confidence += 0.05
    if normalized_scope.rooms:
        confidence += 0.14
    confidence += min(recognized_sections, 6) * 0.04
    confidence -= len(questions) * 0.08
    if normalized_scope.error_text:
        confidence -= 0.15
    return max(0.15, min(0.98, round(confidence, 2)))


def normalize_intake_prompt(
    request: PromptNormalizationRequest,
) -> PromptNormalizationResponse:
    prompt_with_answers = _merge_prompt_with_answers(
        request.prompt, request.clarification_answers
    )
    sections, rooms_by_level, _lines, property_values = _collect_structured_sections(
        prompt_with_answers
    )
    normalized_prompt_markdown = _build_normalized_prompt_markdown(
        sections, rooms_by_level, property_values
    )

    if not normalized_prompt_markdown:
        return PromptNormalizationResponse(
            status="unsupported",
            confidence=0.18,
            normalized_prompt_markdown="",
            normalized_scope=None,
            questions=[],
            blockers=[],
            warnings=[],
            missing_items=[
                "The prompt could not be organized into a usable property, room, or trade scope structure."
            ],
            error_text="We could not safely normalize this prompt into a structured scope.",
        )

    normalized_scope = extract_scope(normalized_prompt_markdown)
    questions = _build_questions(prompt_with_answers, normalized_scope)
    blockers = _build_blockers(questions)
    confidence = _normalization_confidence(normalized_scope, questions)
    status = "needs_clarification" if questions else "ready"
    warnings = _build_warnings(normalized_scope)
    missing_items = _build_missing_items(questions)

    if not normalized_scope.sections and not normalized_scope.rooms:
        status = "unsupported"
        confidence = min(confidence, 0.22)
        missing_items = [
            "The prompt did not produce any rooms or actionable trade sections after normalization."
        ]

    return PromptNormalizationResponse(
        status=status,
        confidence=confidence,
        normalized_prompt_markdown=normalized_prompt_markdown,
        normalized_scope=normalized_scope,
        questions=questions,
        blockers=blockers,
        warnings=warnings,
        missing_items=missing_items,
        error_text=""
        if status != "unsupported"
        else "We could not safely normalize this prompt into a reliable structured scope.",
    )
