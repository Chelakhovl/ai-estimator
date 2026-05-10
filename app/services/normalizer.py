from __future__ import annotations

import re

_ROOM_WORDS = {
    "kitchen",
    "bathroom",
    "hallway",
    "living room",
    "bedroom",
    "corridor",
    "dining room",
    "utility room",
    "garage",
    "loft",
}

_UNIT_ALIASES = {
    "m2": "m2",
    "sqm": "m2",
    "sq m": "m2",
    "m3": "m3",
    "m": "m",
    "lm": "m",
    "linear m": "m",
    "linear metre": "m",
    "linear meter": "m",
    "pcs": "pcs",
    "pc": "pcs",
    "ea": "pcs",
}

_TOKEN_ALIASES = {
    "scaffolding": "scaffold",
    "scaffold": "scaffold",
    "electrical": "electric",
    "electrics": "electric",
    "plumbing": "plumb",
    "decorations": "decorate",
    "decorating": "decorate",
    "painting": "paint",
    "refurbishment": "refurb",
    "renovation": "refurb",
    "renovations": "refurb",
    "conversion": "convert",
    "conversions": "convert",
    "partitions": "partition",
    "plants": "plant",
    "foundations": "foundation",
    "walls": "wall",
    "ceilings": "ceiling",
    "floors": "floor",
    "cabinets": "cabinet",
    "sockets": "socket",
    "switches": "switch",
    "downlights": "downlight",
    "lights": "light",
    "radiators": "radiator",
    "fans": "fan",
    "extractors": "extractor",
    "spotlights": "spotlight",
    "spotlight": "spotlight",
    "appliances": "appliance",
    "appliance": "appliance",
    "connections": "connect",
    "connection": "connect",
    "connected": "connect",
    "connect": "connect",
    "reconnect": "connect",
    "circuits": "circuit",
    "rewiring": "rewire",
    "rewires": "rewire",
    "hardwired": "hardwire",
    "hardwiring": "hardwire",
    "wiring": "wire",
    "ducted": "duct",
    "ducting": "duct",
    "tiles": "tile",
    "tiling": "tile",
    "doors": "door",
    "frames": "frame",
    "stairs": "stair",
    "staircase": "stair",
    "rooflights": "rooflight",
    "skylights": "skylight",
    "cheeks": "cheek",
    "parapets": "parapet",
    "trims": "trim",
    "upstands": "upstand",
    "fascias": "fascia",
    "soffits": "soffit",
    "verges": "verge",
    "copings": "coping",
    "fd30": "fire",
    "fd60": "fire",
    "fd90": "fire",
    "skirtings": "skirting",
    "architraves": "architrave",
    "boarding": "board",
    "boards": "board",
    "insulated": "insulation",
    "insulations": "insulation",
    "rafters": "rafter",
    "windows": "window",
    "throughout": "all",
}

_PHRASE_ALIASES = {
    "all rooms": "throughout",
    "full strip": "strip remove",
    "heating plant": "boiler",
    "supply & fit": "install",
    "supply and fit": "install",
    "linear metre": "lm",
    "linear meter": "lm",
    "linear metres": "lm",
    "linear meters": "lm",
}

_TOKEN_EQUIVALENCE_GROUPS = (
    {"remove", "removal", "demolish", "demolition", "strip", "rip"},
    {"partition", "stud", "studwork"},
    {"paint", "decorate"},
    {"plaster", "skim"},
    {"render", "plaster", "skim"},
    {"tile", "tiling"},
    {"socket", "outlet"},
    {"light", "lighting", "downlight", "spotlight"},
    {"extractor", "fan", "ventilation"},
    {"cabinet", "cupboard", "furniture"},
    {"install", "fit", "fix", "supply"},
    {"floor", "flooring", "screed"},
    {"ceiling", "soffit"},
    {"boiler", "heat", "heating", "radiator", "plant", "combi"},
    {"bathroom", "wet", "ensuite", "wc"},
    {"worktop", "countertop", "counter"},
    {"external", "outside", "exterior"},
)

_TOKEN_EXPANSIONS: dict[str, set[str]] = {}
for group in _TOKEN_EQUIVALENCE_GROUPS:
    for token in group:
        _TOKEN_EXPANSIONS[token] = group - {token}

_ACTION_PREFIXES = (
    "strip out",
    "install",
    "fit",
    "replace",
    "paint",
    "tile",
    "reconnect",
    "build",
    "skim",
    "excavate",
    "remove",
)

_SCOPE_DESCRIPTOR_WORDS = {
    "kitchen",
    "bathroom",
    "loft",
    "extension",
    "refurb",
    "convert",
    "house",
    "flat",
    "property",
    "rear",
    "side",
}

_NON_DESCRIPTOR_HINT_WORDS = {
    "plumb",
    "electric",
    "tile",
    "paint",
    "fit",
    "install",
    "remove",
    "skim",
    "floor",
    "ceiling",
    "wall",
    "first",
    "second",
    "fix",
    "allowance",
    "scaffold",
    "permit",
    "cdm",
    "temporary",
    "roof",
    "basin",
    "sink",
    "toilet",
    "wc",
    "bath",
    "shower",
    "vanity",
    "skirting",
    "architrave",
    "stair",
    "window",
    "board",
    "door",
    "storage",
    "cupboard",
    "wardrobe",
    "extractor",
    "fan",
    "duct",
}



def normalize_prompt(prompt: str) -> str:
    prompt = prompt.replace("\n", ", ")
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt.strip()


def _apply_phrase_aliases(value: str) -> str:
    normalized = value.lower()
    for phrase, replacement in sorted(_PHRASE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"\b{re.escape(phrase)}\b", replacement, normalized)
    return normalized



def split_prompt_segments(prompt: str) -> list[str]:
    parts = re.split(r"[,;]", prompt)
    segments: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        conjunction_parts = re.split(r"\band\b", part, flags=re.IGNORECASE)
        shared_quantity = len(re.findall(r"\d+(?:\.\d+)?\s*(?:m2|sqm|sq m|m3|m|pcs|pc|ea)\b", part, re.IGNORECASE)) == 1
        if (
            len(conjunction_parts) > 1
            and not shared_quantity
            and all(len(tokenize(chunk.strip())) >= 2 for chunk in conjunction_parts if chunk.strip())
        ):
            first_chunk = conjunction_parts[0].strip()
            action_prefix = next((prefix for prefix in _ACTION_PREFIXES if first_chunk.lower().startswith(prefix)), "")
            for index, chunk in enumerate(conjunction_parts):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if index > 0 and action_prefix and not any(chunk.lower().startswith(prefix) for prefix in _ACTION_PREFIXES):
                    chunk = f"{action_prefix} {chunk}"
                segments.append(chunk)
        else:
            segments.append(part)
    return segments or [prompt.strip()]



def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    value = unit.strip().lower()
    return _UNIT_ALIASES.get(value, value)



def parse_quantity_and_unit(segment: str) -> tuple[float | None, str | None]:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(m2|sqm|sq m|m3|lm|linear m|linear metre|linear meter|m|pcs|pc|ea)\b",
        segment,
        re.IGNORECASE,
    )
    if not match:
        return None, None

    quantity = float(match.group(1))
    unit = normalize_unit(match.group(2))
    return quantity, unit



def detect_area(segment: str) -> str:
    lower = segment.lower()
    for room in sorted(_ROOM_WORDS, key=len, reverse=True):
        if room in lower:
            return room.title()
    return ""



def tokenize_terms(value: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", _apply_phrase_aliases(value))
    return [_TOKEN_ALIASES.get(token, token) for token in tokens]


def tokenize(value: str) -> set[str]:
    return set(tokenize_terms(value))


def expand_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(_TOKEN_EXPANSIONS.get(token, set()))
    return expanded


def is_scope_descriptor_segment(segment: str) -> bool:
    tokens = tokenize(segment)
    if not tokens:
        return False
    if any(prefix in segment.lower() for prefix in _ACTION_PREFIXES):
        return False
    if tokens & _NON_DESCRIPTOR_HINT_WORDS:
        return False
    if re.search(r"\d", segment):
        return False
    return bool(tokens & _SCOPE_DESCRIPTOR_WORDS)
