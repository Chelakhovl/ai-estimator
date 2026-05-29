from __future__ import annotations

from app.schemas import ProjectBriefSection, ProjectBriefV1

_PLUMBING_KW = frozenset(
    [
        "plumb",
        "water supply",
        "soil",
        "waste",
        "basin",
        "bath",
        "shower",
        "wc",
        "toilet",
        "radiator",
        "boiler",
        "cylinder",
        "ufh",
        "underfloor heat",
        "gas",
        "heating",
        "sink",
        "tap",
        "sanitar",
        "powerflush",
        "pump",
        "cistern",
        "valve",
        "hot water",
        "cold water",
    ]
)
_ELECTRICAL_KW = frozenset(
    [
        "electr",
        "socket",
        "switch",
        " light",
        "downlight",
        "consumer unit",
        "fuse board",
        "alarm",
        "extractor fan",
        "rewire",
        "cable",
        "pendant",
        "spot",
        "power point",
        "tv point",
        "data point",
        "circuit",
        "ev charge",
        "doorbell",
    ]
)
_PLASTERING_KW = frozenset(["plaster", "skim", "render", "dot and dab"])
_TILING_KW = frozenset(["til", "grout", "mosaic", "porcelain", "ceramic"])
_DECORATING_KW = frozenset(
    ["paint", "decorat", "decor", "emulsion", "gloss", "mist coat", "redecorate"]
)
_FLOORING_KW = frozenset(
    [
        "floor",
        "screed",
        "carpet",
        "laminate",
        "hardwood",
        "timber board",
        "lvt",
        "vinyl",
        "parquet",
        "engineered",
        "subfloor",
    ]
)


def _section_lines(section: ProjectBriefSection) -> list[str]:
    lines: list[str] = []
    if section.summary.strip():
        lines.append(section.summary.strip())
    for fact in section.facts:
        text = fact.text.strip()
        if text:
            lines.append(text)
    return lines


def _matches_any(text: str, keywords: frozenset[str]) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _split_mep(lines: list[str]) -> tuple[list[str], list[str]]:
    plumbing: list[str] = []
    electrical: list[str] = []
    unmatched: list[str] = []
    for line in lines:
        is_plumbing = _matches_any(line, _PLUMBING_KW)
        is_electrical = _matches_any(line, _ELECTRICAL_KW)
        if is_plumbing:
            plumbing.append(line)
        if is_electrical:
            electrical.append(line)
        if not is_plumbing and not is_electrical:
            unmatched.append(line)
    # Unmatched MEP lines go to plumbing (more common in residential scopes)
    return plumbing + unmatched, electrical


def _split_finishes(
    lines: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    plastering: list[str] = []
    tiling: list[str] = []
    decorating: list[str] = []
    flooring: list[str] = []
    unmatched: list[str] = []
    for line in lines:
        matched = False
        if _matches_any(line, _PLASTERING_KW):
            plastering.append(line)
            matched = True
        if _matches_any(line, _TILING_KW):
            tiling.append(line)
            matched = True
        if _matches_any(line, _DECORATING_KW):
            decorating.append(line)
            matched = True
        if _matches_any(line, _FLOORING_KW):
            flooring.append(line)
            matched = True
        if not matched:
            unmatched.append(line)
    # Unmatched finishes lines go to plastering as the most generic finishes bucket
    return plastering + unmatched, tiling, decorating, flooring


def render_project_brief_to_structured_scope(project_brief: ProjectBriefV1) -> str:
    overview_lines = _section_lines(project_brief.project_overview)
    property_type = overview_lines[0] if overview_lines else "Renovation project"
    location = overview_lines[1] if len(overview_lines) > 1 else "TBC"

    mep_lines = _section_lines(project_brief.mep)
    plumbing_lines, electrical_lines = _split_mep(mep_lines)

    finishes_lines = _section_lines(project_brief.finishes)
    plastering_lines, tiling_lines, decorating_lines, flooring_lines = _split_finishes(
        finishes_lines
    )

    sections: list[tuple[str, list[str]]] = [
        ("PROPERTY", [f"Type: {property_type}", f"Location: {location}"]),
        ("FLOOR AREAS & ROOMS", _section_lines(project_brief.areas_dimensions)),
        ("DEMOLITION", _section_lines(project_brief.internal_build)),
        ("STRUCTURE", _section_lines(project_brief.structural)),
        ("PLUMBING", plumbing_lines),
        ("ELECTRICAL", electrical_lines),
        ("PLASTERING", plastering_lines),
        ("TILING", tiling_lines),
        ("DECORATING", decorating_lines),
        ("FLOORING", flooring_lines),
        ("PRELIMINARIES", _section_lines(project_brief.site_conditions)),
        ("FULL SCOPE", _section_lines(project_brief.proposed_layout_scope)),
    ]

    rendered_blocks: list[str] = []
    for title, lines in sections:
        deduped: list[str] = []
        seen: set[str] = set()
        for line in lines:
            normalized = " ".join(line.split()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        if not deduped:
            continue
        rendered_blocks.append(f"{title}:\n" + "\n".join(deduped))

    return "\n\n".join(rendered_blocks).strip()
