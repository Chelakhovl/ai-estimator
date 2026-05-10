from __future__ import annotations

from dataclasses import dataclass
import re

from app.schemas import CandidateRow, ExtractedRoomTakeoff, ExtractedSection, PreviewAssumption, ScopeExtractionResponse
from app.services.normalizer import detect_area, normalize_unit, tokenize
from app.services.scope_parsing import infer_section_keys_from_label


@dataclass(frozen=True)
class TakeoffSuggestion:
    quantity: float | None = None
    area: str = ""
    needs_review: bool = False
    review_reason: str | None = None
    assumption: PreviewAssumption | None = None


@dataclass(frozen=True)
class SectionAssemblyMatch:
    key: str
    title: str
    order: int


_NON_ACTION_SECTION_KEYS = {"property", "full_scope", "floor_areas_rooms"}
_SECTION_HINT_TOKENS: dict[str, set[str]] = {
    "demolition": {"strip", "remove", "demolition", "demo", "break", "clear"},
    "groundworks": {"foundation", "ground", "concrete", "excavate", "soil", "patio", "driveway", "trench"},
    "steelworks": {"steel", "beam", "column", "post", "junction", "connection"},
    "structure": {"structure", "slab", "roof", "wall", "brick", "block", "dormer", "skylight", "cheek", "parapet", "upstand", "trim", "fascia", "soffit", "coping"},
    "carpentry": {"carpentry", "stud", "timber", "chipboard", "joist", "frame"},
    "ceilings_and_insulation": {"ceiling", "plasterboard", "board", "insulation", "pir", "acoustic", "rafter"},
    "doors": {"door", "frame", "pocket"},
    "joinery": {"joinery", "stair", "storage", "cupboard"},
    "plastering": {"plaster", "skim", "render"},
    "tiling": {"tile", "grout", "splashback"},
    "plumbing": {"plumb", "radiator", "basin", "toilet", "bath", "shower", "ufh"},
    "heating": {"boiler", "megaflo", "cylinder", "heating", "gas"},
    "electrical": {"electric", "socket", "switch", "downlight", "spotlight", "extractor", "consumer", "rewire", "hardwire", "duct", "outlet", "fitting", "thermostat", "appliance", "oven", "hob"},
    "decorating": {"decorate", "paint", "mist", "coat"},
    "flooring": {"floor", "laminate", "engineered", "timber", "carpet", "vinyl", "wood"},
    "windows": {"window", "velux", "glazing"},
    "woodwork_painting": {"skirting", "architrave", "stair", "woodwork", "door"},
    "preliminaries": {"prelim", "protection", "toilet", "welfare", "management", "delivery", "site", "wait", "load"},
}


def build_scope_matching_context(extracted_scope: ScopeExtractionResponse) -> str:
    parts: list[str] = []
    property_context = extracted_scope.property_context
    if property_context.property_type:
        parts.append(property_context.property_type)
    if property_context.location:
        parts.append(property_context.location)
    parts.extend(property_context.project_scopes)

    for section in extracted_scope.sections:
        if section.lines:
            parts.append(section.title)
        parts.extend(item.name for item in section.count_items[:8])
        parts.extend(item.name for item in section.measure_items[:8])
        parts.extend(item.name for item in section.dimension_items[:8])

    return ", ".join(part for part in parts if part)


def is_extracted_context_segment(segment: str, extracted_scope: ScopeExtractionResponse) -> bool:
    normalized_segment = segment.strip().lower().rstrip(":")
    if not normalized_segment:
        return True

    if normalized_segment in {"ground floor", "first floor", "loft", "allowance", "radiators"}:
        return True

    if normalized_segment in {section.title.lower() for section in extracted_scope.sections}:
        return True

    if normalized_segment in {scope.lower() for scope in extracted_scope.property_context.project_scopes}:
        return True

    for room in extracted_scope.rooms:
        room_name = room.name.lower()
        if normalized_segment.startswith(f"{room_name}:"):
            suffix = normalized_segment.split(":", 1)[1].strip()
            is_dimension_suffix = bool(
                re.match(
                    r"^\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?(?:\s*[×x]\s*\d+(?:\.\d+)?)?$",
                    suffix,
                    re.IGNORECASE,
                )
            )
            if not suffix or is_dimension_suffix:
                return True
        if room_name in normalized_segment and bool(
            re.search(r"\d+(?:\.\d+)?\s*[×x]\s*\d+(?:\.\d+)?", normalized_segment, re.IGNORECASE)
        ):
            return True

    return False


def _token_set(row: CandidateRow, segment: str) -> set[str]:
    return tokenize(
        " ".join(
            filter(
                None,
                [
                    row.WorkName,
                    row.WorkItemCode or "",
                    row.WorkGroupName or "",
                    segment,
                ],
            )
        )
    )


def _section_token_set(section: ExtractedSection) -> set[str]:
    parts = [section.title]
    parts.extend(section.lines[:20])
    parts.extend(item.name for item in section.count_items[:8])
    parts.extend(item.name for item in section.measure_items[:8])
    parts.extend(item.name for item in section.dimension_items[:8])
    return tokenize(" ".join(part for part in parts if part))


def _build_assumption(work_name: str, quantity: float, reason: str, severity: str = "info") -> PreviewAssumption:
    return PreviewAssumption(
        text=f"Quantity for '{work_name}' was derived from the structured scope takeoff: {quantity:g}. {reason}",
        kind="takeoff_quantity",
        severity=severity,
    )


def _match_room_takeoffs_for_segment(
    segment: str,
    extracted_scope: ScopeExtractionResponse,
) -> list[ExtractedRoomTakeoff]:
    segment_tokens = tokenize(segment)
    if not segment_tokens:
        return []

    matched: list[ExtractedRoomTakeoff] = []
    for room in extracted_scope.room_takeoff:
        room_tokens = tokenize(room.name)
        if not room_tokens:
            continue
        shared = segment_tokens & room_tokens
        if not shared:
            continue
        non_numeric_shared = {token for token in shared if not token.isdigit()}
        if non_numeric_shared:
            matched.append(room)

    return matched


def _build_room_based_assumption(
    work_name: str,
    quantity: float,
    rooms: list[ExtractedRoomTakeoff],
    reason: str,
    *,
    severity: str = "warning",
) -> PreviewAssumption:
    room_names = ", ".join(room.name for room in rooms[:4])
    if len(rooms) > 4:
        room_names = f"{room_names}, +{len(rooms) - 4} more"
    return _build_assumption(
        work_name,
        quantity,
        f"{reason} Matched rooms: {room_names}.",
        severity=severity,
    )


def _wet_room_matches(rooms: list[ExtractedRoomTakeoff]) -> list[ExtractedRoomTakeoff]:
    return [room for room in rooms if room.room_type in {"bathroom", "ensuite", "wc"}]


def _plumbing_room_matches(rooms: list[ExtractedRoomTakeoff]) -> list[ExtractedRoomTakeoff]:
    return [room for room in rooms if room.room_type in {"bathroom", "ensuite", "wc", "kitchen", "utility"}]


def _sum_room_metric(rooms: list[ExtractedRoomTakeoff], field_name: str) -> float:
    return float(sum(getattr(room, field_name, 0.0) for room in rooms))


def _sum_summary_fixture_metric(summary, field_name: str) -> float:
    return float(getattr(summary, field_name, 0.0) or 0.0)


def _room_area_label(rooms: list[ExtractedRoomTakeoff]) -> str:
    if not rooms:
        return ""
    unique_names = []
    for room in rooms:
        if room.name not in unique_names:
            unique_names.append(room.name)
    if len(unique_names) == 1:
        return unique_names[0]

    unique_levels = []
    for room in rooms:
        level = room.level.title()
        if level not in unique_levels:
            unique_levels.append(level)
    if len(unique_levels) == 1:
        return unique_levels[0]

    return "Multiple rooms"


def _has_extracted_scope_line(
    extracted_scope: ScopeExtractionResponse,
    *,
    section_keys: tuple[str, ...],
    keywords: tuple[str, ...],
) -> bool:
    for section in extracted_scope.sections:
        if section.key not in section_keys:
            continue
        for line in section.lines:
            normalized = line.lower()
            if all(keyword in normalized for keyword in keywords):
                return True
    return False


def match_row_to_extracted_section(
    row: CandidateRow,
    extracted_scope: ScopeExtractionResponse,
    segment: str,
) -> SectionAssemblyMatch | None:
    row_tokens = _token_set(row, segment)
    segment_tokens = tokenize(segment)
    candidate_section_keys: set[str] = set()
    for value in (row.WorkGroupName or "", row.WorkName, row.WorkItemCode or ""):
        candidate_section_keys.update(infer_section_keys_from_label(value))
    best: tuple[float, SectionAssemblyMatch] | None = None

    for order, section in enumerate(extracted_scope.sections):
        if section.key in _NON_ACTION_SECTION_KEYS:
            continue
        section_tokens = _section_token_set(section)
        if not section_tokens:
            continue
        score = float(len(row_tokens & section_tokens) * 2 + len(segment_tokens & section_tokens))
        hint_tokens = _SECTION_HINT_TOKENS.get(section.key, set())
        if hint_tokens and row_tokens & hint_tokens:
            score += 2
        if hint_tokens and segment_tokens & hint_tokens:
            score += 1
        if candidate_section_keys:
            if section.key in candidate_section_keys:
                score += 5
            else:
                score -= 0.5
        if score <= 0:
            continue
        match = SectionAssemblyMatch(key=section.key, title=section.title, order=order)
        if best is None or score > best[0]:
            best = (score, match)

    return best[1] if best else None


def suggest_quantity_from_takeoff(
    row: CandidateRow,
    extracted_scope: ScopeExtractionResponse,
    segment: str,
) -> TakeoffSuggestion:
    summary = extracted_scope.takeoff_summary
    unit = normalize_unit(row.Unit)
    tokens = _token_set(row, segment)
    matched_rooms = _match_room_takeoffs_for_segment(segment, extracted_scope)
    matched_wet_rooms = _wet_room_matches(matched_rooms)
    matched_plumbing_rooms = _plumbing_room_matches(matched_rooms)

    if (
        unit == "pcs"
        and summary.client_supply_sanitaryware
        and "supply" in tokens
        and any(token in tokens for token in {"sanitary", "sanitaryware", "wc", "toilet", "basin", "bath", "shower", "sink", "vanity"})
    ):
        quantity = (
            _sum_room_metric(matched_wet_rooms, "estimated_sanitary_set_count")
            if matched_wet_rooms
            else _sum_summary_fixture_metric(summary, "total_sanitary_set_count")
        )
        if quantity <= 0:
            quantity = 1.0
        return TakeoffSuggestion(
            quantity=quantity,
            needs_review=True,
            review_reason="The prompt marks sanitaryware as client supplied. Review this supply row carefully and keep only labour/install lines if supply is excluded from the quote.",
            assumption=_build_assumption(
                row.WorkName,
                quantity,
                "The structured prompt indicates client-supplied sanitaryware, so this supply-related row was held for manual pricing review.",
                severity="warning",
            ),
        )

    if (
        unit == "pcs"
        and summary.client_supply_appliances
        and "supply" in tokens
        and any(
            token in tokens
            for token in {"appliance", "appliances", "oven", "hob", "extractor", "dishwasher", "fridge", "freezer", "washing", "dryer"}
        )
    ):
        return TakeoffSuggestion(
            quantity=1.0,
            needs_review=True,
            review_reason="The prompt marks appliances as client supplied. Review this supply row and keep only connection / install labour if appliance supply is excluded from the quote.",
            assumption=_build_assumption(
                row.WorkName,
                1.0,
                "The structured prompt indicates client-supplied appliances, so this supply-related row was held for manual pricing review.",
                severity="warning",
            ),
        )

    if unit == "m2" and not matched_rooms:
        if any(token in tokens for token in {"engineered", "laminate", "vinyl", "timber", "wood"}) and summary.estimated_hard_floor_scope_area_m2:
            quantity = float(summary.estimated_hard_floor_scope_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Floor finish quantity was derived from the explicit flooring section, using only the rooms and levels that were marked for hard-floor finishes. Confirm exclusions, transitions, and any rooms finished by others.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used structured hard-floor scope area from the extracted flooring section.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"carpet", "underlay"}) and summary.estimated_carpet_scope_area_m2:
            quantity = float(summary.estimated_carpet_scope_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Carpet or underlay quantity was derived from the explicit flooring section, using only the rooms and levels marked for carpet scope. Confirm areas by others and any rooms finished in another material.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used structured carpet scope area from the extracted flooring section.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"slab", "screed", "concrete"}) and summary.estimated_extension_slab_area_m2:
            quantity = float(summary.estimated_extension_slab_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Extension slab quantity was derived from the extracted rear extension footprint. Confirm whether this row should cover the slab only or the full floor build-up including insulation and screed.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived rear extension slab area from the structured prompt.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"flat", "roof", "warm", "grp"}) and summary.estimated_extension_flat_roof_area_m2:
            quantity = float(summary.estimated_extension_flat_roof_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Flat roof quantity was derived from the extracted extension roof baseline. Confirm overhangs, parapets, and any areas priced separately as rooflights or trims.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived rear extension flat-roof area from the structured prompt.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"brick", "block", "cavity", "external", "wall"}) and any(
            token in tokens for token in {"extension", "rear"}
        ) and summary.estimated_extension_external_wall_area_m2:
            quantity = float(summary.estimated_extension_external_wall_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Extension wall quantity was derived from the extracted extension footprint and assumes three exposed sides with standard height and opening deductions. Confirm actual façade geometry and openings.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived rear extension external wall area from the structured prompt.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"roof", "dormer", "loft", "rafter"}) and any(
            token in tokens for token in {"remove", "strip", "demo", "demolition"}
        ) and summary.estimated_roof_removal_area_m2:
            quantity = float(summary.estimated_roof_removal_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Roof removal quantity was derived from explicit roof demolition areas plus any percentage-based loft roof removal notes. Confirm which roof zones are actually stripped back.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived roof-removal area from the structured prompt.",
                    severity="warning",
                ),
            )
        if "dormer" in tokens and summary.estimated_dormer_build_area_m2:
            quantity = float(summary.estimated_dormer_build_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Dormer quantity was derived from the extracted dormer footprint. Confirm whether this row should price front face only, full build area, or a separate cheek/cladding item.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived dormer build area from the structured prompt.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"roof", "dormer", "loft", "rafter"}) and any(
            token in tokens for token in {"plasterboard", "board", "insulation", "pir"}
        ) and summary.estimated_loft_roof_finish_area_m2:
            quantity = float(summary.estimated_loft_roof_finish_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Loft roof lining quantity was derived from loft floor area plus dormer footprint. Confirm sloping ceiling geometry, knee walls, and dormer cheek treatment before pricing.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived loft / dormer roof finish area from the structured prompt.",
                    severity="warning",
                ),
            )

    if matched_rooms and unit == "m2":
        if "tile" in tokens and "wall" in tokens and matched_wet_rooms:
            quantity = float(sum(room.estimated_wall_tiling_area_m2 for room in matched_wet_rooms))
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Wall tiling quantity was derived from matched wet-room geometry. Confirm full-height tiling, niches, and areas intentionally left un-tiled.",
                assumption=_build_room_based_assumption(
                    row.WorkName,
                    quantity,
                    matched_wet_rooms,
                    "Used room-specific wet-room wall tiling takeoff from the structured prompt.",
                ),
            )
        if "tile" in tokens and "floor" in tokens and matched_wet_rooms:
            quantity = float(sum(room.estimated_floor_finish_area_m2 for room in matched_wet_rooms))
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Floor tiling quantity was derived from matched wet-room floor areas. Confirm shower trays, built-in units, and areas finished in another material.",
                assumption=_build_room_based_assumption(
                    row.WorkName,
                    quantity,
                    matched_wet_rooms,
                    "Used room-specific wet-room floor tiling takeoff from the structured prompt.",
                ),
            )
        if ("paint" in tokens or "decorate" in tokens) and "ceiling" in tokens:
            quantity = float(sum(room.estimated_ceiling_finish_area_m2 for room in matched_rooms))
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Ceiling quantity was derived from the matched room schedule, not the whole property. Confirm exclusions and partial-room finishes.",
                assumption=_build_room_based_assumption(
                    row.WorkName,
                    quantity,
                    matched_rooms,
                    "Used room-specific ceiling takeoff from the structured prompt.",
                ),
            )
        if any(token in tokens for token in {"paint", "decorate", "skim", "plaster", "render"}) and "wall" in tokens:
            quantity = float(sum(room.estimated_wall_finish_area_m2 for room in matched_rooms))
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Wall quantity was derived from the matched room schedule, not the whole property. Confirm openings, full-height tiling, and irregular layouts.",
                assumption=_build_room_based_assumption(
                    row.WorkName,
                    quantity,
                    matched_rooms,
                    "Used room-specific wall finish takeoff from the structured prompt.",
                ),
            )
        if any(token in tokens for token in {"plasterboard", "board"}) and "ceiling" in tokens:
            quantity = float(sum(room.estimated_ceiling_finish_area_m2 for room in matched_rooms))
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Ceiling boarding quantity was derived from the matched room schedule. Confirm voids, slopes, and dormer reductions.",
                assumption=_build_room_based_assumption(
                    row.WorkName,
                    quantity,
                    matched_rooms,
                    "Used room-specific ceiling boarding takeoff from the structured prompt.",
                ),
            )
        if any(token in tokens for token in {"floor", "laminate", "vinyl", "timber", "engineered", "carpet"}):
            quantity = float(sum(room.area_m2 for room in matched_rooms))
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Floor finish quantity was derived from the matched room schedule. Confirm which rooms actually receive this finish.",
                assumption=_build_room_based_assumption(
                    row.WorkName,
                    quantity,
                    matched_rooms,
                    "Used room-specific floor area from the structured prompt.",
                ),
            )

    if unit == "pcs" and not matched_rooms:
        if "electric" in tokens and any(token in tokens for token in {"second", "fix"}) and summary.estimated_electrical_second_fix_points:
            quantity = float(summary.estimated_electrical_second_fix_points)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Electrical second-fix quantity was derived from the whole-house room schedule. Confirm whether this line should stay as a bundled second-fix item or split into sockets, switches, and lighting rows.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived whole-property electrical second-fix allowances from the structured prompt.",
                    severity="warning",
                ),
            )
        if "plumb" in tokens and any(token in tokens for token in {"second", "fix"}) and summary.estimated_plumbing_second_fix_points:
            quantity = float(summary.estimated_plumbing_second_fix_points)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Plumbing second-fix quantity was derived from the whole-property room schedule. Confirm whether this should stay as a bundled second-fix line or be split by fixture type.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived whole-property plumbing second-fix allowances from the structured prompt.",
                    severity="warning",
                ),
            )

    if matched_rooms and unit == "pcs":
        if any(token in tokens for token in {"downlight", "spotlight"}):
            quantity = _sum_room_metric(matched_rooms, "estimated_downlight_count")
            if quantity <= 0:
                quantity = _sum_room_metric(matched_rooms, "estimated_lighting_point_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Downlight quantity was derived from matched room lighting allowances and any explicit spotlight notes. Confirm feature pendants, strip lighting, and fittings that should stay outside the downlight count.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific downlight counts from the structured prompt.",
                    ),
                )
        if "socket" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_socket_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Socket count was derived from the matched room schedule using room-type second-fix heuristics. Confirm kitchen appliance runs and any client-specific layouts.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific socket allowances from the structured prompt.",
                    ),
                )
        if "switch" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_switch_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Switch count was derived from the matched room schedule using room-type second-fix heuristics. Confirm two-way switching and grouped switching arrangements.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific switch allowances from the structured prompt.",
                    ),
                )
        if any(token in tokens for token in {"downlight", "spotlight", "light"}) and "wall" not in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_lighting_point_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Lighting points were derived from matched room areas using second-fix heuristics. Confirm feature lighting, pendants, and any rooms finished with another fitting type.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific lighting point allowances from the structured prompt.",
                    ),
                )
        if ("extractor" in tokens or "fan" in tokens) and "duct" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_ducted_extractor_count")
            if quantity <= 0:
                quantity = _sum_room_metric(matched_rooms, "estimated_extractor_fan_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Ducted extractor quantity was derived from matched room ventilation notes and wet-room / kitchen baselines. Confirm duct runs, wall caps, and where passive vents replace a powered fan.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific ducted extractor counts from the structured prompt.",
                    ),
                )
        if "extractor" in tokens or ("fan" in tokens and not any(token in tokens for token in {"ceiling", "wall"})):
            quantity = _sum_room_metric(matched_rooms, "estimated_extractor_fan_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Extractor fan count was derived from matched wet rooms, utility, and kitchen areas. Confirm where ducting or passive vents replace a powered fan.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific extractor fan allowances from the structured prompt.",
                    ),
                )
        if "appliance" in tokens and any(token in tokens for token in {"hardwire", "wire"}):
            quantity = _sum_room_metric(matched_rooms, "estimated_hardwired_appliance_count")
            if quantity <= 0:
                quantity = _sum_room_metric(matched_rooms, "estimated_appliance_connection_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Hardwired appliance quantity was derived from matched kitchen / utility electrical notes. Confirm which appliances are hardwired versus standard plug-in connections and whether each point should be split out separately.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific hardwired appliance counts from the structured prompt.",
                    ),
                )
        if "electric" in tokens and any(token in tokens for token in {"second", "fix", "fit", "off"}) and not any(
            token in tokens for token in {"socket", "switch", "downlight", "spotlight", "light", "extractor", "fan", "consumer", "board", "rewire"}
        ):
            quantity = _sum_room_metric(matched_rooms, "estimated_electrical_second_fix_points")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Electrical second-fix quantity was derived from matched room socket, switch, lighting, and extractor allowances. Confirm whether this row should be room-based, per point, or split into separate second-fix items.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific electrical second-fix allowances from the structured prompt.",
                    ),
                )
        if any(token in tokens for token in {"cistern", "concealed"}) or (
            "wc" in tokens and "wall" in tokens and "hung" in tokens
        ) or ("toilet" in tokens and "wall" in tokens and "hung" in tokens):
            quantity = _sum_room_metric(matched_rooms, "estimated_concealed_cistern_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Concealed cistern quantity was derived from matched wet-room overrides. Confirm whether wall-hung WC frames, cisterns, and pans are split into separate rows.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific concealed cistern counts from the structured prompt.",
                    ),
                )
        if any(token in tokens for token in {"toilet", "wc"}) and not any(token in tokens for token in {"frame", "door"}):
            quantity = _sum_room_metric(matched_rooms, "estimated_wc_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="WC quantity was derived from matched bathroom / ensuite / WC rooms. Confirm if this row should cover pans only or a full WC set with concealed frames and extras.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific WC fixture counts from the structured prompt.",
                    ),
                )
        if "basin" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_basin_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Basin quantity was derived from matched bathroom / ensuite / WC rooms. Confirm double basins, vanity units, and cloakroom-specific exclusions.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific basin fixture counts from the structured prompt.",
                    ),
                )
        if "vanity" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_vanity_unit_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Vanity quantity was derived from matched bathroom / ensuite rooms. Confirm whether this row should cover vanity carcass only or a bundled vanity+bowl set.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific vanity unit counts from the structured prompt.",
                    ),
                )
        if "sink" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_sink_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Sink quantity was derived from matched kitchen / utility rooms. Confirm double-bowl sinks, prep sinks, and whether appliances are priced separately.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific sink fixture counts from the structured prompt.",
                    ),
                )
        if "appliance" in tokens and any(token in tokens for token in {"connect", "connection", "connections", "reconnect"}):
            quantity = _sum_room_metric(matched_rooms, "estimated_appliance_connection_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Appliance connections were derived from matched kitchen / utility room baselines and any explicit appliance-connection notes. Confirm which appliances are in scope and whether hardwiring or plumbing tails are priced separately.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific appliance connection counts from the structured prompt.",
                    ),
                )
        if "bath" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_bath_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Bath quantity was derived from matched bathroom rooms. Confirm whether shower trays or wet-room sets should be priced separately instead.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific bath fixture counts from the structured prompt.",
                    ),
                )
        if "shower" in tokens:
            quantity = _sum_room_metric(matched_rooms, "estimated_shower_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Shower quantity was derived from matched bathroom / ensuite rooms. Confirm tray, enclosure, valve, and wet-room areas that should be split into separate rows.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_rooms,
                        "Used room-specific shower fixture counts from the structured prompt.",
                    ),
                )
        if any(token in tokens for token in {"sanitary", "suite"}) and not any(
            token in tokens for token in {"wc", "toilet", "basin", "sink", "shower", "bath", "radiator", "first", "second", "fix"}
        ):
            quantity = _sum_room_metric(matched_wet_rooms, "estimated_sanitary_set_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Sanitary install quantity was derived as one sanitary set per matched wet room. Review whether this row should stay bundled or be split by fixture type and room.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_wet_rooms,
                        "Used room-specific sanitary-set counts from the structured prompt.",
                    ),
                )
        if any(token in tokens for token in {"plumb", "first", "second"}) and not any(
            token in tokens for token in {"wc", "toilet", "basin", "sink", "shower", "bath", "radiator", "boiler", "cylinder", "sanitary", "suite"}
        ):
            quantity = float(sum(room.estimated_plumbing_second_fix_points for room in matched_plumbing_rooms))
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Plumbing quantity was derived from matched room plumbing points. Review whether this row should be per fixture, per room, or split between first-fix and second-fix labour.",
                    assumption=_build_room_based_assumption(
                        row.WorkName,
                        quantity,
                        matched_plumbing_rooms,
                        "Used room-specific plumbing points from the structured prompt.",
                    ),
                )
    if unit == "pcs":
        if any(token in tokens for token in {"stair", "staircase"}) and summary.staircase_count > 0:
            quantity = _sum_summary_fixture_metric(summary, "staircase_count")
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Staircase count was derived from explicit joinery scope in the structured prompt. Confirm whether this row should cover the full staircase, balustrade, or only a joinery sub-package.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used extracted staircase counts from the structured prompt.",
                    severity="warning",
                ),
            )
        if "window" in tokens and "board" in tokens and summary.window_board_count > 0:
            quantity = _sum_summary_fixture_metric(summary, "window_board_count")
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Window board count was derived from explicit woodwork / joinery scope and extracted window openings. Confirm whether this row should be per board set, per opening, or split by material type.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used extracted window board counts from the structured prompt.",
                    severity="warning",
                ),
            )
        if "architrave" in tokens and summary.architrave_set_count > 0 and _has_extracted_scope_line(
            extracted_scope,
            section_keys=("woodwork_painting", "joinery"),
            keywords=("architrave",),
        ):
            quantity = _sum_summary_fixture_metric(summary, "architrave_set_count")
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Architrave set count was derived from explicit woodwork scope and counted door sets. Confirm whether this row should be per opening set or broken into linear metres.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used extracted architrave set counts from the structured prompt.",
                    severity="warning",
                ),
            )
        if "pocket" in tokens and "door" in tokens:
            quantity = 0.0
            reason = "Used extracted pocket door set counts from the structured prompt."
            review_reason = "Pocket door quantity was derived from explicit doors scope. Confirm whether this row is per set, per leaf, or should split ironmongery, cassette, and fire-rating extras."
            if "fire" in tokens and summary.fire_rated_pocket_door_set_count > 0:
                quantity = _sum_summary_fixture_metric(summary, "fire_rated_pocket_door_set_count")
                reason = "Used extracted fire-rated pocket door set counts from the structured prompt."
                review_reason = "Fire-rated pocket door quantity was derived from explicit doors scope. Confirm per-set versus per-leaf pricing and whether ironmongery or fire upgrades sit in separate rows."
            elif "double" in tokens and summary.double_pocket_door_set_count > 0:
                quantity = _sum_summary_fixture_metric(summary, "double_pocket_door_set_count")
                reason = "Used extracted double pocket door set counts from the structured prompt."
                review_reason = "Double pocket door quantity was derived from explicit doors scope. Confirm per-set versus per-leaf pricing and whether both leaves or ironmongery are priced separately."
            elif summary.pocket_door_set_count > 0:
                quantity = _sum_summary_fixture_metric(summary, "pocket_door_set_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason=review_reason,
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        reason,
                        severity="warning",
                    ),
                )
        if "door" in tokens and "fire" in tokens and summary.fire_rated_door_set_count > 0:
            quantity = _sum_summary_fixture_metric(summary, "fire_rated_door_set_count")
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Fire-rated door quantity was derived from the structured doors scope. Confirm whether this row is per door set, whether pocket doors are separate, and whether closer or ironmongery packs sit elsewhere.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used extracted fire-rated door set counts from the structured prompt.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"storage", "cupboard", "wardrobe"}) and any(
            token in tokens for token in {"door", "set"}
        ) and summary.storage_door_set_count > 0:
            quantity = _sum_summary_fixture_metric(summary, "storage_door_set_count")
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Storage door-set quantity was derived from explicit joinery storage notes. Confirm whether this row covers doors only or a full fitted-storage package.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used extracted storage door-set counts from the structured prompt.",
                    severity="warning",
                ),
            )
        if ("consumer" in tokens and "unit" in tokens) or ("distribution" in tokens and "board" in tokens):
            quantity = _sum_summary_fixture_metric(summary, "consumer_unit_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Consumer unit count was derived from the structured electrical scope. Confirm whether this is a single replacement board, a split-load upgrade, or part of a wider rewire package.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used extracted consumer unit counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if any(token in tokens for token in {"downlight", "spotlight"}):
            quantity = _sum_summary_fixture_metric(summary, "downlight_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Downlight quantity was derived from the structured electrical scope. Confirm feature pendants, strip lighting, and any fittings that should sit outside the downlight count.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used extracted downlight counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if ("extractor" in tokens or "fan" in tokens) and "duct" in tokens:
            quantity = _sum_summary_fixture_metric(summary, "ducted_extractor_count")
            if quantity <= 0:
                quantity = summary.extractor_fan_count
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Ducted extractor quantity was derived from the structured prompt. Confirm duct runs, wall caps, and where a passive vent or recirculating unit should not be counted as a ducted fan.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used extracted ducted-extractor counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if "appliance" in tokens and any(token in tokens for token in {"hardwire", "wire"}):
            quantity = _sum_summary_fixture_metric(summary, "hardwired_appliance_count")
            if quantity <= 0:
                quantity = _sum_summary_fixture_metric(summary, "total_appliance_connection_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Hardwired appliance quantity was derived from the structured electrical scope. Confirm which appliances are hardwired versus standard plug-in connections and whether each point should be split into dedicated rows.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used extracted hardwired appliance counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if any(token in tokens for token in {"cistern", "concealed"}) or (
            "wc" in tokens and "wall" in tokens and "hung" in tokens
        ) or ("toilet" in tokens and "wall" in tokens and "hung" in tokens):
            quantity = _sum_summary_fixture_metric(summary, "total_concealed_cistern_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Concealed cistern quantity was derived from whole-property wet-room overrides. Confirm whether frames, cisterns, and WC pans are split into separate supply/install rows.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property concealed cistern counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if any(token in tokens for token in {"toilet", "wc"}) and not any(token in tokens for token in {"frame", "door"}):
            quantity = _sum_summary_fixture_metric(summary, "total_wc_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="WC quantity was derived from whole-property room fixture baselines. Confirm whether all wet rooms really take a WC pan and whether concealed cisterns or frames are separate.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property WC fixture counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if "basin" in tokens:
            quantity = _sum_summary_fixture_metric(summary, "total_basin_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Basin quantity was derived from whole-property room fixture baselines. Confirm cloakrooms, double basins, and vanity exclusions.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property basin counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if "vanity" in tokens:
            quantity = _sum_summary_fixture_metric(summary, "total_vanity_unit_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Vanity quantity was derived from whole-property room baselines. Confirm double-vanity rooms, cloakroom exclusions, and whether bowls or worktops are priced separately.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property vanity unit counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if "sink" in tokens:
            quantity = _sum_summary_fixture_metric(summary, "total_sink_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Sink quantity was derived from whole-property kitchen and utility baselines. Confirm prep sinks, utility tubs, and appliance connections priced separately.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property sink counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if "appliance" in tokens and any(token in tokens for token in {"connect", "connection", "connections", "reconnect"}):
            quantity = _sum_summary_fixture_metric(summary, "total_appliance_connection_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Appliance connections were derived from whole-property kitchen / utility baselines and any explicit appliance connection notes. Confirm which appliances are in scope and whether they should be split by trade.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property appliance connection counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if "bath" in tokens:
            quantity = _sum_summary_fixture_metric(summary, "total_bath_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Bath quantity was derived from whole-property bathroom baselines. Confirm if any bathrooms are shower-only or if freestanding baths need separate handling.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property bath counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if "shower" in tokens:
            quantity = _sum_summary_fixture_metric(summary, "total_shower_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Shower quantity was derived from whole-property ensuite / bathroom baselines. Confirm whether trays, screens, valves, and wet-room sets should be priced separately.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property shower counts from the structured prompt.",
                        severity="warning",
                    ),
                )
        if any(token in tokens for token in {"sanitary", "suite", "bathroom"}) and not any(
            token in tokens for token in {"wc", "toilet", "basin", "sink", "shower", "bath", "radiator", "plumb", "first", "second", "fix"}
        ):
            quantity = _sum_summary_fixture_metric(summary, "total_sanitary_set_count")
            if quantity > 0:
                return TakeoffSuggestion(
                    quantity=quantity,
                    needs_review=True,
                    review_reason="Sanitary install quantity was derived as one sanitary set per wet room across the whole property. Review whether this quote should stay bundled or split by fixture type and room.",
                    assumption=_build_assumption(
                        row.WorkName,
                        quantity,
                        "Used whole-property sanitary-set counts from the structured prompt.",
                        severity="warning",
                    ),
                )

    exact_count_cases: list[tuple[bool, float, str]] = [
        ("socket" in tokens, summary.socket_count, "Used the extracted electrical count from the structured prompt."),
        ("switch" in tokens, summary.switch_count, "Used the extracted electrical count from the structured prompt."),
        (("extractor" in tokens or "fan" in tokens), summary.extractor_fan_count, "Used the extracted extractor fan count from the structured prompt."),
        (("velux" in tokens), summary.velux_count, "Used the extracted roof window count from the structured prompt."),
        (("window" in tokens and "velux" not in tokens), summary.upvc_window_count, "Used the extracted uPVC window count from the structured prompt."),
        (("radiator" in tokens and ("remove" in tokens or "strip" in tokens or "demolition" in tokens)), summary.radiator_removal_count, "Used the extracted radiator removal count from the structured prompt."),
        (("radiator" in tokens), summary.radiator_install_count, "Used the extracted radiator install count from the structured prompt."),
        (("door" in tokens), summary.door_set_count, "Used the counted door sets from the structured prompt."),
        (("junction" in tokens or "connection" in tokens), summary.steel_junction_count, "Used the extracted steel junction count from the structured prompt."),
        (("post" in tokens or "column" in tokens), summary.steel_post_count, "Used the extracted steel post count from the structured prompt."),
        (("tree" in tokens), summary.tree_count, "Used the extracted tree count from the structured prompt."),
        (("fence" in tokens), summary.fence_count, "Used the extracted fence count from the structured prompt."),
    ]
    for condition, quantity, reason in exact_count_cases:
        if condition and quantity and unit == "pcs":
            needs_review = "door" in tokens
            return TakeoffSuggestion(
                quantity=float(quantity),
                area=detect_area(segment),
                needs_review=needs_review,
                review_reason="Structured takeoff count was used. Review if this should be per leaf or per set." if needs_review else None,
                assumption=_build_assumption(row.WorkName, float(quantity), reason, severity="warning" if needs_review else "info"),
            )

    if unit == "m3":
        if "soil" in tokens and summary.soil_volume_m3:
            quantity = float(summary.soil_volume_m3)
            return TakeoffSuggestion(
                quantity=quantity,
                assumption=_build_assumption(row.WorkName, quantity, "Used the extracted soil volume from the structured prompt."),
            )
        if "trench" in tokens and summary.trench_foundation_volume_m3:
            quantity = float(summary.trench_foundation_volume_m3)
            return TakeoffSuggestion(
                quantity=quantity,
                assumption=_build_assumption(row.WorkName, quantity, "Used the extracted trench foundation concrete volume."),
            )
        if any(token in tokens for token in {"foundation", "concrete", "excavate", "excavation", "slab"}) and summary.total_foundation_concrete_m3:
            quantity = float(summary.total_foundation_concrete_m3)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Structured takeoff provided foundation volume, but confirm whether this row should use total, trench, or internal foundation volume only.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used the extracted overall foundation takeoff as a draft baseline.",
                    severity="warning",
                ),
            )

    if unit == "m2":
        if ("paint" in tokens or "decorate" in tokens) and "ceiling" in tokens and summary.estimated_ceiling_finish_area_m2:
            quantity = float(summary.estimated_ceiling_finish_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Ceiling finish quantity was derived from room floor areas. Confirm exclusions such as bathrooms or areas finished by others.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived ceiling finish area from the room schedule.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"paint", "decorate", "skim", "plaster", "render"}) and "wall" in tokens and summary.estimated_wall_finish_area_m2:
            quantity = float(summary.estimated_wall_finish_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Wall finish quantity was derived from room dimensions using standard height and opening deductions. Review where layouts are irregular or heavily tiled.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived wall finish area from the room schedule.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"plasterboard", "board"}) and "ceiling" in tokens and summary.estimated_ceiling_finish_area_m2:
            quantity = float(summary.estimated_ceiling_finish_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Ceiling boarding quantity was derived from room floor areas. Review for voids, dormers, and partial ceiling rebuilds.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived ceiling boarding area from the room schedule.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"plasterboard", "board"}) and not any(
            token in tokens for token in {"partition", "stud", "wall"}
        ) and summary.estimated_boarding_coverage_m2:
            quantity = float(summary.estimated_boarding_coverage_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Boarding quantity was derived from ceiling boarding and roof-board coverage. Review for partial rebuilds, dormers, and partition-only work.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived boarding coverage from the structured scope takeoff.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"insulation", "pir"}) and any(
            token in tokens for token in {"roof", "rafter", "loft", "dormer"}
        ) and summary.roof_area_m2:
            quantity = float(summary.roof_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Roof insulation quantity was derived from the extracted roof area. Review dormers, slopes, and warm-roof build-up exclusions.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used extracted roof area as a roof-insulation baseline.",
                    severity="warning",
                ),
            )
        if any(token in tokens for token in {"insulation", "pir", "acoustic"}) and summary.estimated_insulation_coverage_m2:
            quantity = float(summary.estimated_insulation_coverage_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Insulation coverage was derived from the extracted floor and roof build-up notes. Review which levels and build-ups actually take this item.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used derived insulation coverage from the structured scope takeoff.",
                    severity="warning",
                ),
            )
        if "roof" in tokens and summary.roof_area_m2:
            quantity = float(summary.roof_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                assumption=_build_assumption(row.WorkName, quantity, "Used the extracted roof area from the structured prompt."),
            )
        if "patio" in tokens and summary.patio_area_m2:
            quantity = float(summary.patio_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                assumption=_build_assumption(row.WorkName, quantity, "Used the extracted patio area from the structured prompt."),
            )
        if any(token in tokens for token in {"floor", "laminate", "vinyl", "timber", "engineered"}) and summary.total_internal_floor_area_m2:
            quantity = float(summary.total_internal_floor_area_m2)
            return TakeoffSuggestion(
                quantity=quantity,
                needs_review=True,
                review_reason="Structured takeoff used total internal floor area as a draft flooring quantity. Confirm which levels actually receive this finish.",
                assumption=_build_assumption(
                    row.WorkName,
                    quantity,
                    "Used total internal floor area as a flooring baseline.",
                    severity="warning",
                ),
            )

    if unit == "m" and "skirting" in tokens and summary.estimated_skirting_lm:
        if matched_rooms:
            quantity = float(sum(room.estimated_skirting_lm for room in matched_rooms))
            return TakeoffSuggestion(
                quantity=quantity,
                area=_room_area_label(matched_rooms),
                needs_review=True,
                review_reason="Skirting length was derived from the matched room schedule. Confirm fitted joinery, tile upstands, and doorway deductions.",
                assumption=_build_room_based_assumption(
                    row.WorkName,
                    quantity,
                    matched_rooms,
                    "Used room-specific skirting length from the structured prompt.",
                ),
            )
        quantity = float(summary.estimated_skirting_lm)
        return TakeoffSuggestion(
            quantity=quantity,
            needs_review=True,
            review_reason="Skirting length was derived from room perimeters with a standard door deduction. Review where fitted furniture or tile upstands change the run length.",
            assumption=_build_assumption(
                row.WorkName,
                quantity,
                "Used derived skirting length from the room schedule.",
                severity="warning",
            ),
        )

    if unit == "m" and "architrave" in tokens and summary.estimated_architrave_lm and _has_extracted_scope_line(
        extracted_scope,
        section_keys=("woodwork_painting", "joinery"),
        keywords=("architrave",),
    ):
        quantity = float(summary.estimated_architrave_lm)
        return TakeoffSuggestion(
            quantity=quantity,
            needs_review=True,
            review_reason="Architrave length was derived from counted door sets using a standard per-opening allowance. Review double doors, pocket frames, and openings without architraves.",
            assumption=_build_assumption(
                row.WorkName,
                quantity,
                "Used derived architrave length from counted door sets in the structured prompt.",
                severity="warning",
            ),
        )

    if unit == "m" and "window" in tokens and "board" in tokens and summary.estimated_window_board_lm:
        quantity = float(summary.estimated_window_board_lm)
        return TakeoffSuggestion(
            quantity=quantity,
            needs_review=True,
            review_reason="Window board length was derived from extracted window opening widths. Review returns, bay geometry, and whether roof windows or boards by others should be excluded.",
            assumption=_build_assumption(
                row.WorkName,
                quantity,
                "Used derived window board length from extracted window openings in the structured prompt.",
                severity="warning",
            ),
        )

    return TakeoffSuggestion()
