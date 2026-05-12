from __future__ import annotations

import re


SECTION_TITLES = {
    "property": "Property",
    "full_scope": "Full Scope",
    "floor_areas_rooms": "Floor Areas & Rooms",
    "demolition": "Demolition",
    "groundworks": "Groundworks",
    "steelworks": "Steelworks",
    "structure": "Structure",
    "carpentry": "Carpentry",
    "ceilings_and_insulation": "Ceilings & Insulation",
    "doors": "Doors",
    "joinery": "Joinery",
    "plastering": "Plastering",
    "tiling": "Tiling",
    "plumbing": "Plumbing",
    "heating": "Heating",
    "electrical": "Electrical",
    "decorating": "Decorating",
    "flooring": "Flooring",
    "windows": "Windows",
    "woodwork_painting": "Woodwork Painting",
    "preliminaries": "Preliminaries",
    "commercial_settings": "Commercial Settings",
}

SECTION_HEADING_ALIASES = {
    "project_details": "property",
    "details": "property",
    "scope": "full_scope",
    "project_scope": "full_scope",
    "full_scope": "full_scope",
    "floor_areas_and_rooms": "floor_areas_rooms",
    "floor_areas": "floor_areas_rooms",
    "areas": "floor_areas_rooms",
    "rooms": "floor_areas_rooms",
    "room_schedule": "floor_areas_rooms",
    "strip_out": "demolition",
    "stripout": "demolition",
    "stripping": "demolition",
    "demolition_stripping": "demolition",
    "structural_works": "structure",
    "structural": "structure",
    "roofing": "structure",
    "builder_s_work": "structure",
    "builders_work": "structure",
    "floor_build_up": "flooring",
    "floor_build_up_full_flat_excl_bathroom": "flooring",
    "carpentry_joinery": "joinery",
    "carpentry_and_joinery": "joinery",
    "partitions": "carpentry",
    "ceilings": "ceilings_and_insulation",
    "ceilings_insulation": "ceilings_and_insulation",
    "insulation": "ceilings_and_insulation",
    "heating_plumbing": "plumbing",
    "heating_and_plumbing": "plumbing",
    "electrics": "electrical",
    "electrical_works": "electrical",
    "decorations": "decorating",
    "painting": "decorating",
    "paintwork": "decorating",
    "ground_works": "groundworks",
    "external_works": "groundworks",
    "site_costs": "preliminaries",
    "site_setup": "preliminaries",
    "commercial": "commercial_settings",
}

ROOM_LEVEL_ORDER = ["Basement", "Lower Ground Floor", "Ground Floor", "First Floor", "Second Floor", "Loft"]
ROOM_LEVELS = set(ROOM_LEVEL_ORDER)
ROOM_LEVELS_LOWER = {level.lower() for level in ROOM_LEVELS}
_INLINE_SECTION_TITLES = tuple(sorted(SECTION_TITLES.values(), key=len, reverse=True))
_COMMON_PROJECT_SCOPE_PHRASES = tuple(
    sorted(
        {
            "Bathroom renovation",
            "Full flat refresh",
            "Full house refurb",
            "Full house refurbishment",
            "Full property refresh",
            "Ground floor refurbishment",
            "Kitchen refurbishment",
            "Loft conversion",
            "Rear extension",
            "Whole house refurb",
            "Whole house renovation",
        },
        key=len,
        reverse=True,
    )
)
_INLINE_ROOM_SEPARATOR_PATTERN = re.compile(
    r"(?:(?<=[0-9\)])\s+(?=[A-Za-z][A-Za-z0-9 /+&()'-]*?:\s*\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?)|,\s*(?=[A-Za-z][A-Za-z0-9 /+&()'-]*?:\s*\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?))",
    re.IGNORECASE,
)


def clean_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    stripped = re.sub(r"^[^A-Za-z0-9£]+", "", stripped)
    stripped = stripped.replace("–", "-")
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def normalize_heading(line: str) -> str:
    normalized = line.lower().strip().rstrip(":")
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return SECTION_HEADING_ALIASES.get(normalized, normalized)


def infer_section_keys_from_label(label: str) -> set[str]:
    keys: set[str] = set()
    if not label:
        return keys

    normalized = normalize_heading(label)
    if normalized in SECTION_TITLES:
        keys.add(normalized)

    for part in re.split(r"\s*(?:/|,|\+|&|\band\b)\s*", label, flags=re.IGNORECASE):
        normalized_part = normalize_heading(part)
        if normalized_part in SECTION_TITLES:
            keys.add(normalized_part)

    return keys


def prepare_prompt_lines(prompt: str) -> list[str]:
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.count("\n") >= 3:
        lines = [clean_line(line) for line in normalized.splitlines()]
        return [line for line in lines if line]

    def _replace_inline_section_title(match: re.Match[str]) -> str:
        title = match.group("title")
        has_colon = bool(match.group("colon"))
        if not has_colon and not title.isupper():
            return match.group(0)
        return f"\n{title}\n"

    for title in _INLINE_SECTION_TITLES:
        normalized = re.sub(
            rf"\s*(?P<title>{re.escape(title)})(?P<colon>\s*:)?\s*",
            _replace_inline_section_title,
            normalized,
            flags=re.IGNORECASE,
        )

    for level in sorted(ROOM_LEVELS, key=len, reverse=True):
        normalized = re.sub(
            rf"(?<![A-Za-z0-9])(?P<level>{re.escape(level)})(?=(?:\s+[A-Za-z][A-Za-z0-9 /+&()'-]*?:\s*\d)|(?:\s*\n)|$)\s*:?\s*",
            lambda match: f"\n{match.group('level')}\n",
            normalized,
            flags=re.IGNORECASE,
        )

    normalized = re.sub(r"\s+(Type|Location)\s*:\s*", lambda match: f"\n{match.group(1).title()}: ", normalized, flags=re.IGNORECASE)
    normalized = _INLINE_ROOM_SEPARATOR_PATTERN.sub("\n", normalized)

    lines = [clean_line(line) for line in normalized.splitlines()]
    return [line for line in lines if line]


def split_project_scope_line(line: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for phrase in _COMMON_PROJECT_SCOPE_PHRASES:
        match = re.search(rf"\b{re.escape(phrase)}\b", line, re.IGNORECASE)
        if match:
            matches.append((match.start(), match.group(0)))

    if len(matches) <= 1:
        return [line.strip()]

    matches.sort(key=lambda item: item[0])
    return [phrase.strip() for _, phrase in matches]


def infer_default_room_level(prompt: str) -> str | None:
    normalized = prompt.lower()
    level_patterns = (
        ("lower ground floor flat", "Lower Ground Floor"),
        ("basement flat", "Basement"),
        ("ground floor flat", "Ground Floor"),
        ("first floor flat", "First Floor"),
        ("second floor flat", "Second Floor"),
        ("loft flat", "Loft"),
    )
    for needle, level in level_patterns:
        if needle in normalized:
            return level
    if "flat" in normalized or "converted" in normalized:
        return "First Floor"
    house_keywords = ("house", "terraced", "detached", "semi-detached", "semi detached", "bungalow", "cottage", "property", "maisonette")
    if any(keyword in normalized for keyword in house_keywords):
        return "Ground Floor"
    return None
