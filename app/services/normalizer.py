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
    "pcs": "pcs",
    "pc": "pcs",
    "ea": "pcs",
}



def normalize_prompt(prompt: str) -> str:
    prompt = prompt.replace("\n", ", ")
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt.strip()



def split_prompt_segments(prompt: str) -> list[str]:
    parts = re.split(r"[,;]", prompt)
    segments = [part.strip() for part in parts if part.strip()]
    return segments or [prompt.strip()]



def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    value = unit.strip().lower()
    return _UNIT_ALIASES.get(value, value)



def parse_quantity_and_unit(segment: str) -> tuple[float | None, str | None]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(m2|sqm|sq m|m3|m|pcs|pc|ea)\b", segment, re.IGNORECASE)
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



def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))
