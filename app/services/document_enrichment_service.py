from __future__ import annotations

import json
import logging
import re
from time import perf_counter

from app.config import settings

logger = logging.getLogger(__name__)

# ── Shared JSON output schema ──────────────────────────────────────────────────

_JSON_SCHEMA = """\

Return ONLY valid JSON, no markdown fences:
{
  "page_type": "<one of: floor_plan | structural_drawing | structural_calc | section_elevation | schedule | other>",
  "description": "<complete transcription — see instructions above>",
  "rooms": [
    {"name": "Kitchen", "floor_level": "Ground Floor", "width_m": 3.5, "length_m": 4.2},
    {"name": "WC", "floor_level": "Ground Floor", "width_m": 1.8, "length_m": null}
  ],
  "dimensions": [
    {"element": "Overall building width", "width_m": 8.5, "length_m": null}
  ],
  "work_items": [
    {"trade": "structural", "description": "Remove load-bearing wall between kitchen and dining"}
  ],
  "structural_notes": ["300mm cavity wall, internal leaf 100mm dense concrete block min 7.3N/mm²"],
  "uncertainties": ["Dimension on north wall partially obscured"]
}
"""

# ── Floor plan ─────────────────────────────────────────────────────────────────

_FLOOR_PLAN_PROMPT = """\
You are analysing a floor plan from a UK residential architectural or structural drawing package.

A floor plan shows the building layout from above. You will see:
- Room outlines (walls as thick lines or hatched areas)
- Room labels inside outlines (e.g. "Kitchen", "Bedroom 1", "WC", "Ensuite")
- Floor level title (e.g. "Ground Floor Plan", "First Floor Plan", "Loft Plan")
- Dimension lines: thin lines with arrowheads or diagonal tick marks at each end,
  a number above or below the line. Numbers in millimetres (4+ digits, e.g. 3500)
  or metres (e.g. 3.5 or 3.5m). Convert mm → m by dividing by 1000.
- Overall floor dimensions along the perimeter
- New/existing/demolished wall indicators from the legend
- Door and window openings

Your job — extract every piece of information visible:

1. page_type: "floor_plan"

2. description: Complete transcription. State the floor level. For every room: name and
   its exact dimensions read from the nearest dimension lines. Then all overall dimensions.
   Then any notes, annotations, work callouts.

3. rooms: Every room/space label visible.
   For EACH room, find the dimension lines along its walls:
   - Dimension lines run outside the room boundary, parallel to a wall
   - A chain of dimension strings often breaks a wall into segments — sum them for total room size
   - Horizontal dimension line → width_m
   - Vertical dimension line → length_m
   - Convert mm → m: 3500 → 3.5, 4200 → 4.2, 1800 → 1.8
   - If only one dimension visible → width_m, leave length_m null
   Include rooms with no visible dimensions (leave width_m and length_m null).

4. dimensions: Overall/external dimensions NOT belonging to a specific room (total width,
   total depth, extension setback, garden boundary distance, etc.)

5. work_items: New walls, demolished walls, new openings, extensions explicitly marked.
   Use trades: structural | carpentry | finishes | general

6. structural_notes: Any structural or material specs visible (wall types, floor build-up, etc.)

7. uncertainties: Any room label or dimension that is unclear or partially hidden
""" + _JSON_SCHEMA

# ── Structural drawing ─────────────────────────────────────────────────────────

_STRUCTURAL_DRAWING_PROMPT = """\
You are analysing a structural engineering drawing from a UK residential project.

Structural drawings contain:
- General arrangement plan showing positions of beams, columns, foundations
- Member schedules (tables) — the most important data source:
    Steel Members: Ref | Description | Level | Section (e.g. UC 254x254x107) | Grade (e.g. S275)
    Timber Members: Ref | Member | Level | Section (e.g. 63x200 @400mm c/c) | Class (e.g. C24)
    Lintels: Ref | Type (e.g. Catnic_CG90/100, Naylor_ER2) | Dimensions (e.g. 136x1950mm) | Material
    Concrete/Foundations: Ref | Dimensions (e.g. 600x2300mm) | Reinforcement (e.g. H12@175mm c/c btm) | Grade (C30)
- Key/Legend — symbols for wall types, beam types, etc.
- General notes — material specs, fixing requirements, construction sequence
- Connection details — padstones, base plates, bolted connections

Your job — transcribe the schedules completely. Every row matters.

1. page_type: "structural_drawing"

2. description: Transcribe ALL schedule data row by row. For steel: every beam (B1, B2...)
   with level/section/grade; every column (C1, C2...) with level/section/grade. For timber:
   every joist/rafter/beam with section and class. For lintels: every lintel with type,
   dimensions, material. For foundations: every foundation with dimensions, reinforcement, grade.
   Include "Omitted" rows. Then all general notes (exact text). Then key/legend entries.
   Then any dimension strings visible on the plan itself.

3. rooms: Any space labels visible on the arrangement plan (may be few). Include dimensions
   if dimension lines are shown adjacent to the space.

4. dimensions: Overall structural dimensions on the plan (span distances, grid lines, setbacks).

5. work_items: Construction tasks explicitly noted (e.g. "Install beam B1 at high level",
   "Pour 150mm concrete slab", "Prop existing structure before demolition").
   Trades: structural | general

6. structural_notes: All general notes verbatim. Connection details and fixing specifications.
   Padstone dimensions. Bolt sizes and spacings.

7. uncertainties: Schedule rows that are illegible, split across pages, or partially visible
""" + _JSON_SCHEMA

# ── Structural calculations ────────────────────────────────────────────────────

_STRUCTURAL_CALC_PROMPT = """\
You are analysing a structural engineering calculation page from a UK residential project.

Structural calculations contain:
- Loading analysis: dead loads (DL), imposed/live loads (IL), wind loads — in kN/m², kN/m, kN
- Member design: bending moment capacity, shear capacity, deflection check
- Foundation design: bearing pressure, pad/strip dimensions
- Section headings identifying which element is being checked (e.g. "Beam B1", "Foundation F2")
- Code references (BS EN 1990, BS EN 1991, etc.)

Your job — extract all numerical values and their engineering context.

1. page_type: "structural_calc"

2. description: For each section/element being calculated: its reference (e.g. "Beam B1"),
   the loads applied (exact values with units), the section selected, and the check result
   (e.g. "PASS — utilisation 0.72"). Include all load values verbatim.

3. rooms: Leave empty []

4. dimensions: Member sizes that are the RESULT of calculations (e.g. "Use 300x300mm pad foundation",
   "Adopt UC 203x203x46")

5. work_items: Leave empty []

6. structural_notes: All load values, capacity figures, design assumptions, and pass/fail results.
   Format: "Beam B1: DL=5.2kN/m, IL=3.0kN/m, design moment=24.5kNm, capacity=31.2kNm PASS"

7. uncertainties: Anything illegible or cut off
""" + _JSON_SCHEMA

# ── Section / elevation ────────────────────────────────────────────────────────

_SECTION_ELEVATION_PROMPT = """\
You are analysing a section or elevation drawing from a UK residential architectural package.

Section drawings (vertical cut through the building) show:
- Floor-to-floor heights and overall building height
- Level datums (FFL = Finished Floor Level, SSL = Structural Slab Level, GL = Ground Level)
- Wall construction layers from outside to inside (e.g. brick / cavity / insulation / blockwork)
- Roof construction and pitch angle
- Joist and beam positions in cross-section with their sizes
- Staircase geometry (rise, going, handrail height)

Elevation drawings (external face view) show:
- Window and door positions with reference codes (W1, D1, etc.)
- Window and door sizes if dimension lines are shown
- External wall material (brick, render, timber cladding, etc.)
- Overall building height and storey heights
- Eaves and ridge heights

Your job — extract every height, level, and dimension visible.

1. page_type: "section_elevation"

2. description: State whether this is a section or elevation (and which: A-A, B-B, North, South, etc.).
   List all level datums with values (e.g. FFL Ground = 0.000, FFL First = 2.950). All floor-to-floor
   heights. Wall and roof build-up layers. Any beam/joist sizes visible in section. Window and door
   references and sizes if shown.

3. rooms: Any room labels visible in the section cut. Include ceiling height if shown.

4. dimensions: All vertical and horizontal dimensions — floor heights, room heights, roof pitch
   dimensions, window/door sizes, wall thicknesses.

5. work_items: Alterations explicitly shown (new roof, altered floor level, new opening, etc.)

6. structural_notes: Structural sizes visible in section (joist depths, beam sizes, wall thickness,
   insulation thickness, floor build-up specification).

7. uncertainties: Any level or dimension that is unclear
""" + _JSON_SCHEMA

# ── Schedule (door / window / finish / room data) ──────────────────────────────

_SCHEDULE_PROMPT = """\
You are analysing a schedule drawing from a UK residential architectural package.
Schedules include: door schedules, window schedules, finish schedules, room data sheets.

A schedule is a table where rows = individual elements (each door, window, or room)
and columns = their properties (reference, dimensions, type, material, location, etc.).

Your job — transcribe the schedule table completely.

1. page_type: "schedule"

2. description: State the schedule type (e.g. "Door Schedule"). Then transcribe EVERY row:
   reference code, all dimensions, type/material, room location, any special notes.
   Example: "D1: 900x2100mm, solid core oak, Hallway, FD30 fire door;
             D2: 762x2100mm, hollow core, Bedroom 1; ..."

3. rooms: All room names referenced as locations in the schedule.
   For finish schedules: include the room with its floor/wall/ceiling finishes as structural_notes.

4. dimensions: All element sizes found (door/window widths and heights, etc.)

5. work_items: Leave empty []

6. structural_notes: Any specification notes (fire rating, acoustic rating, ironmongery, finish spec)

7. uncertainties: Any illegible cells or references
""" + _JSON_SCHEMA

# ── Generic fallback ───────────────────────────────────────────────────────────

_OTHER_PROMPT = """\
You are analysing a page from a UK residential construction drawing package.

Identify what type of page this is and extract all visible information.

1. page_type: classify as one of: floor_plan | structural_drawing | structural_calc |
   section_elevation | schedule | other

2. description: Complete transcription of all text, numbers, tables, and annotations on the page.
   Include everything — do not summarise.

3. rooms: Any space/room labels visible with any dimensions shown

4. dimensions: Any measurements visible with their associated elements

5. work_items: Any construction tasks explicitly noted

6. structural_notes: Any structural or material specifications

7. uncertainties: Anything unclear, illegible, or ambiguous
""" + _JSON_SCHEMA

# ── Prompt selection ───────────────────────────────────────────────────────────

_PROMPTS: dict[str, str] = {
    "floor_plan": _FLOOR_PLAN_PROMPT,
    "structural_drawing": _STRUCTURAL_DRAWING_PROMPT,
    "structural_calc": _STRUCTURAL_CALC_PROMPT,
    "section_elevation": _SECTION_ELEVATION_PROMPT,
    "schedule": _SCHEDULE_PROMPT,
}


def _select_prompt(page_type: str) -> str:
    return _PROMPTS.get(page_type, _OTHER_PROMPT)


# ── JSON parser ────────────────────────────────────────────────────────────────

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE | re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_json(content: str) -> dict:
    # 1. Raw parse — model returned clean JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences with regex (safe against nested backticks in content)
    stripped = _FENCE_OPEN.sub("", content.strip(), count=1)
    stripped = _FENCE_CLOSE.sub("", stripped.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Extract the outermost {...} block (handles prose + JSON mixed responses)
    m = _JSON_BLOCK.search(content)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM JSON response: %r", content[:300])
    return {}


# ── Main entry point ───────────────────────────────────────────────────────────

def enrich_page_with_vision(
    page_image_base64: str,
    page_number: int,
    source_file: str,
    page_type: str,
    existing_rooms: list[dict],
    existing_notes: list[str],
) -> dict:
    """
    Sends a page image to GPT-4o vision with a page-type-specific prompt.
    Returns enriched facts dict. Gracefully degrades on any failure.
    """
    if not settings.openai_api_key:
        return _fallback("OpenAI not configured.")

    model = settings.openai_intake_fast_model or "gpt-4o-mini"

    prompt_text = _select_prompt(page_type)

    context_lines: list[str] = []
    if existing_rooms:
        room_names = ", ".join(r.get("name", "") for r in existing_rooms if r.get("name"))
        if room_names:
            context_lines.append(f"Previously found rooms (partial): {room_names}")
    if existing_notes:
        context_lines.append(f"Previously found notes (partial): {'; '.join(existing_notes[:5])}")
    if context_lines:
        prompt_text += "\n\nContext from earlier programmatic extraction:\n" + "\n".join(context_lines)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=3,
        )

        started_at = perf_counter()
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{page_image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )

        latency_ms = int((perf_counter() - started_at) * 1000)
        logger.info(
            "Vision enrichment p%s of '%s' (type=%s) — model=%s latency=%dms",
            page_number, source_file, page_type, model, latency_ms,
        )

        content = response.choices[0].message.content or "{}"
        data = _parse_llm_json(content)
        data["model_name"] = model
        data["used_vision"] = True
        data["fallback_reason"] = None
        data.setdefault("description", "")
        return data

    except Exception as exc:
        logger.warning(
            "Vision enrichment failed for p%s of '%s': %s", page_number, source_file, exc
        )
        return _fallback(str(exc))


def _fallback(reason: str) -> dict:
    return {
        "page_type": "other",
        "description": "",
        "rooms": [],
        "dimensions": [],
        "work_items": [],
        "structural_notes": [],
        "uncertainties": [],
        "model_name": "",
        "used_vision": False,
        "fallback_reason": reason,
    }
