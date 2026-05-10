from __future__ import annotations

import re

from app.schemas import ExtractedRoom, ExtractedRoomTakeoff, ExtractedSection, StructuredTakeoffSummary
from app.services.generated_takeoff_defaults import (
    ARCHITRAVE_LM_PER_DOOR_SET as _ARCHITRAVE_LM_PER_DOOR_SET,
    LEVEL_WALL_HEIGHTS_M as _LEVEL_WALL_HEIGHTS_M,
    OPENING_DEDUCTION_FACTOR as _OPENING_DEDUCTION_FACTOR,
    SKIRTING_DEDUCTION_FACTOR as _SKIRTING_DEDUCTION_FACTOR,
    TAKEOFF_HEURISTICS_SOURCE,
    TAKEOFF_HEURISTICS_VERSION,
    WET_ROOM_TILING_FACTOR as _WET_ROOM_TILING_FACTOR,
)
_LEVEL_AREA_LOOKUP = {
    "basement": "basement_area_m2",
    "lower ground floor": "basement_area_m2",
    "ground floor": "ground_floor_area_m2",
    "first floor": "first_floor_area_m2",
    "second floor": "second_floor_area_m2",
    "loft": "loft_area_m2",
}
_ROOM_TYPE_KEYWORDS = {
    "ensuite": "ensuite",
    "en suite": "ensuite",
    "shower room": "shower_room",
    "bathroom": "bathroom",
    "wc": "wc",
    "cloakroom": "wc",
    "utility": "utility",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "study": "study",
    "living": "living",
    "dining": "living",
    "hallway": "circulation",
    "hall": "circulation",
    "landing": "circulation",
    "store": "store",
}


def _round(value: float) -> float:
    return round(value, 2)


def _sum_room_area(rooms: list[ExtractedRoom], level: str) -> float:
    return _round(sum(room.area_m2 for room in rooms if room.level.lower() == level.lower()))


def _sum_estimated_wall_finish_area(rooms: list[ExtractedRoom]) -> float:
    total = 0.0
    for room in rooms:
        height = _LEVEL_WALL_HEIGHTS_M.get(room.level.lower(), 2.4)
        perimeter = 2 * (room.width_m + room.length_m)
        total += perimeter * height * _OPENING_DEDUCTION_FACTOR
    return _round(total)


def _estimate_room_wall_finish_area(room: ExtractedRoom) -> float:
    height = _LEVEL_WALL_HEIGHTS_M.get(room.level.lower(), 2.4)
    perimeter = 2 * (room.width_m + room.length_m)
    return _round(perimeter * height * _OPENING_DEDUCTION_FACTOR)


def _sum_estimated_ceiling_area(rooms: list[ExtractedRoom]) -> float:
    return _round(sum(room.area_m2 for room in rooms))


def _sum_estimated_skirting_lm(rooms: list[ExtractedRoom]) -> float:
    total = 0.0
    for room in rooms:
        perimeter = 2 * (room.width_m + room.length_m)
        total += perimeter * _SKIRTING_DEDUCTION_FACTOR
    return _round(total)


def _estimate_room_skirting_lm(room: ExtractedRoom) -> float:
    perimeter = 2 * (room.width_m + room.length_m)
    return _round(perimeter * _SKIRTING_DEDUCTION_FACTOR)


def _non_wet_rooms_for_level(room_takeoff: list[ExtractedRoomTakeoff], level: str | None = None) -> list[ExtractedRoomTakeoff]:
    rooms = [
        room
        for room in room_takeoff
        if room.room_type not in {"bathroom", "ensuite", "shower_room", "wc"}
    ]
    if not level:
        return rooms
    return [room for room in rooms if room.level.lower() == level.lower()]


def _classify_room_type(room_name: str) -> str:
    normalized = room_name.lower()
    for keyword, room_type in _ROOM_TYPE_KEYWORDS.items():
        if keyword in normalized:
            return room_type
    return "general"


def _estimate_room_wall_tiling_area(room: ExtractedRoom, room_type: str) -> float:
    if room_type not in {"bathroom", "ensuite", "shower_room", "wc"}:
        return 0.0
    return _round(_estimate_room_wall_finish_area(room) * _WET_ROOM_TILING_FACTOR)


def _estimate_room_plumbing_points(room_type: str) -> float:
    if room_type in {"bathroom", "ensuite", "shower_room"}:
        return 3.0
    if room_type == "wc":
        return 2.0
    if room_type == "utility":
        return 1.0
    if room_type == "kitchen":
        return 2.0
    return 0.0


def _estimate_room_fixture_counts(room_type: str) -> dict[str, float]:
    if room_type == "bathroom":
        return {
            "estimated_wc_count": 1.0,
            "estimated_basin_count": 1.0,
            "estimated_bath_count": 1.0,
            "estimated_shower_count": 0.0,
            "estimated_sink_count": 0.0,
            "estimated_sanitary_set_count": 1.0,
            "estimated_vanity_unit_count": 0.0,
            "estimated_concealed_cistern_count": 0.0,
            "estimated_appliance_connection_count": 0.0,
        }
    if room_type == "ensuite":
        return {
            "estimated_wc_count": 1.0,
            "estimated_basin_count": 1.0,
            "estimated_bath_count": 0.0,
            "estimated_shower_count": 1.0,
            "estimated_sink_count": 0.0,
            "estimated_sanitary_set_count": 1.0,
            "estimated_vanity_unit_count": 0.0,
            "estimated_concealed_cistern_count": 0.0,
            "estimated_appliance_connection_count": 0.0,
        }
    if room_type == "shower_room":
        return {
            "estimated_wc_count": 1.0,
            "estimated_basin_count": 1.0,
            "estimated_bath_count": 0.0,
            "estimated_shower_count": 1.0,
            "estimated_sink_count": 0.0,
            "estimated_sanitary_set_count": 1.0,
            "estimated_vanity_unit_count": 0.0,
            "estimated_concealed_cistern_count": 0.0,
            "estimated_appliance_connection_count": 0.0,
        }
    if room_type == "wc":
        return {
            "estimated_wc_count": 1.0,
            "estimated_basin_count": 1.0,
            "estimated_bath_count": 0.0,
            "estimated_shower_count": 0.0,
            "estimated_sink_count": 0.0,
            "estimated_sanitary_set_count": 1.0,
            "estimated_vanity_unit_count": 0.0,
            "estimated_concealed_cistern_count": 0.0,
            "estimated_appliance_connection_count": 0.0,
        }
    if room_type in {"kitchen", "utility"}:
        return {
            "estimated_wc_count": 0.0,
            "estimated_basin_count": 0.0,
            "estimated_bath_count": 0.0,
            "estimated_shower_count": 0.0,
            "estimated_sink_count": 1.0,
            "estimated_sanitary_set_count": 0.0,
            "estimated_vanity_unit_count": 0.0,
            "estimated_concealed_cistern_count": 0.0,
            "estimated_appliance_connection_count": 4.0 if room_type == "kitchen" else 2.0,
        }
    return {
        "estimated_wc_count": 0.0,
        "estimated_basin_count": 0.0,
        "estimated_bath_count": 0.0,
        "estimated_shower_count": 0.0,
        "estimated_sink_count": 0.0,
        "estimated_sanitary_set_count": 0.0,
        "estimated_vanity_unit_count": 0.0,
        "estimated_concealed_cistern_count": 0.0,
        "estimated_appliance_connection_count": 0.0,
    }


def _estimate_room_socket_count(room: ExtractedRoom, room_type: str) -> float:
    area = room.area_m2
    if room_type == "kitchen":
        return float(max(6, round(area / 3)))
    if room_type == "utility":
        return float(max(3, round(area / 3.5)))
    if room_type == "living":
        return float(max(6, round(area / 4)))
    if room_type == "bedroom":
        return float(max(4, round(area / 4)))
    if room_type == "study":
        return float(max(4, round(area / 3.5)))
    if room_type == "circulation":
        return float(max(1, round(area / 6)))
    if room_type == "store":
        return 1.0
    if room_type in {"bathroom", "ensuite", "shower_room", "wc"}:
        return 0.0
    return float(max(2, round(area / 5)))


def _estimate_room_switch_count(room: ExtractedRoom, room_type: str) -> float:
    area = room.area_m2
    if room_type in {"bathroom", "ensuite", "shower_room", "wc", "utility", "bedroom", "study", "store"}:
        return 1.0
    if room_type == "kitchen":
        return float(max(2, round(area / 12)))
    if room_type == "living":
        return float(max(2, round(area / 14)))
    if room_type == "circulation":
        return float(max(1, round(area / 10)))
    return float(max(1, round(area / 12)))


def _estimate_room_lighting_points(room: ExtractedRoom, room_type: str) -> float:
    area = room.area_m2
    if room_type in {"bathroom", "ensuite", "shower_room", "wc"}:
        return float(max(2, round(area / 2)))
    if room_type in {"kitchen", "living"}:
        return float(max(4, round(area / 4)))
    if room_type in {"bedroom", "study"}:
        return float(max(4, round(area / 4.5)))
    if room_type == "utility":
        return float(max(2, round(area / 4)))
    if room_type == "circulation":
        return float(max(2, round(area / 5)))
    if room_type == "store":
        return 1.0
    return float(max(2, round(area / 5)))


def _estimate_room_extractor_fan_count(room_type: str) -> float:
    if room_type in {"bathroom", "ensuite", "shower_room", "wc", "kitchen", "utility"}:
        return 1.0
    return 0.0


def _match_room_takeoff_for_line(
    room_takeoff: list[ExtractedRoomTakeoff],
    line: str,
) -> list[ExtractedRoomTakeoff]:
    normalized = line.lower()
    matched_by_name = [room for room in room_takeoff if room.name.lower() in normalized]
    if matched_by_name:
        return matched_by_name

    room_type_keywords = {
        "bathroom": {"bathroom"},
        "ensuite": {"ensuite", "en suite"},
        "shower_room": {"shower room"},
        "wc": {"wc", "cloakroom"},
        "kitchen": {"kitchen"},
        "utility": {"utility"},
    }
    matched: list[ExtractedRoomTakeoff] = []
    for room_type, keywords in room_type_keywords.items():
        if any(keyword in normalized for keyword in keywords):
            matched.extend(room for room in room_takeoff if room.room_type == room_type)

    return matched


def _estimate_flooring_scope_areas(
    sections: list[ExtractedSection],
    room_takeoff: list[ExtractedRoomTakeoff],
) -> tuple[float, float]:
    hard_floor_rooms: dict[tuple[str, str], ExtractedRoomTakeoff] = {}
    carpet_rooms: dict[tuple[str, str], ExtractedRoomTakeoff] = {}
    active_level: str | None = None

    for line in _iter_section_lines(sections, "flooring"):
        normalized = line.lower().strip()
        if not normalized:
            continue
        heading = normalized.rstrip(":")
        if heading in _LEVEL_AREA_LOOKUP:
            active_level = heading
            continue
        if normalized.endswith(":"):
            continue
        if any(keyword in normalized for keyword in ("by others", "excluded", "exclude", "by owner", "not in scope")):
            continue

        floor_scope: str | None = None
        if any(keyword in normalized for keyword in ("carpet", "underlay")):
            floor_scope = "carpet"
        elif any(keyword in normalized for keyword in ("engineered", "laminate", "vinyl", "lvt", "timber", "wood")) or (
            "floor" in normalized and "carpet" not in normalized
        ):
            floor_scope = "hard"
        if not floor_scope:
            continue

        matched_rooms = _match_room_takeoff_for_line(room_takeoff, normalized)
        if not matched_rooms:
            if any(keyword in normalized for keyword in ("throughout", "all", "full house", "full property")):
                matched_rooms = _non_wet_rooms_for_level(room_takeoff)
            elif active_level:
                matched_rooms = _non_wet_rooms_for_level(room_takeoff, active_level)

        target = hard_floor_rooms if floor_scope == "hard" else carpet_rooms
        for room in matched_rooms:
            target[(room.level, room.name)] = room

    return _round(sum(room.area_m2 for room in hard_floor_rooms.values())), _round(sum(room.area_m2 for room in carpet_rooms.values()))


def _has_client_supply_sanitaryware(sections: list[ExtractedSection]) -> bool:
    for line in _iter_section_lines(sections, "plumbing", "full_scope"):
        normalized = line.lower()
        if "client supply" not in normalized:
            continue
        if any(
            keyword in normalized
            for keyword in ("sanitary", "sanitaryware", "wc", "toilet", "basin", "bath", "shower", "sink", "vanity")
        ):
            return True
    return False


def _has_client_supply_appliances(sections: list[ExtractedSection]) -> bool:
    for line in _iter_section_lines(sections, "electrical", "plumbing", "full_scope", "joinery"):
        normalized = line.lower()
        if "client supply" not in normalized:
            continue
        if any(
            keyword in normalized
            for keyword in ("appliance", "appliances", "white goods", "oven", "hob", "extractor", "dishwasher", "fridge", "freezer", "washing machine", "dryer")
        ):
            return True
    return False


def _extract_explicit_count(line: str) -> float | None:
    working = re.sub(r"£\s*\d+(?:\.\d+)?", "", line)
    match = re.search(r"(\d+(?:\.\d+)?)", working)
    if match:
        return float(match.group(1))
    return None


def _has_any_keyword(normalized: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in normalized for keyword in keywords)


def _extract_explicit_feature_count(
    sections: list[ExtractedSection],
    section_keys: tuple[str, ...],
    *,
    required_keywords: tuple[str, ...],
    secondary_keywords: tuple[str, ...] = (),
    default_when_mentioned: float = 0.0,
) -> float:
    total = 0.0
    matched = False
    for line in _iter_section_lines(sections, *section_keys):
        normalized = line.lower()
        if not _has_any_keyword(normalized, required_keywords):
            continue
        if secondary_keywords and not _has_any_keyword(normalized, secondary_keywords):
            continue
        matched = True
        explicit_count = _extract_explicit_count(normalized)
        if explicit_count is not None:
            total += explicit_count
        elif default_when_mentioned:
            total += default_when_mentioned
    return _round(total) if matched else 0.0


def _extract_contextual_line_count(
    sections: list[ExtractedSection],
    section_keys: tuple[str, ...],
    *,
    predicate,
    default_when_mentioned: float = 0.0,
) -> float:
    total = 0.0
    matched = False
    for section in sections:
        if section.key not in section_keys:
            continue
        active_context = ""
        for line in section.lines:
            normalized = line.lower().strip()
            if not normalized:
                continue
            if normalized.endswith(":"):
                active_context = normalized.rstrip(":").strip()
                continue
            if not predicate(normalized, active_context):
                continue
            matched = True
            explicit_count = _extract_explicit_count(normalized)
            if explicit_count is not None:
                total += explicit_count
            elif default_when_mentioned:
                total += default_when_mentioned
    return _round(total) if matched else 0.0


def _rebuild_room_second_fix_points(room: ExtractedRoomTakeoff) -> None:
    room_fixture_points = (
        room.estimated_wc_count
        + room.estimated_basin_count
        + room.estimated_bath_count
        + room.estimated_shower_count
        + room.estimated_sink_count
    )
    room.estimated_plumbing_points = _round(room_fixture_points)
    room.estimated_plumbing_second_fix_points = _round(room_fixture_points)
    room.estimated_electrical_second_fix_points = _round(
        room.estimated_socket_count
        + room.estimated_switch_count
        + room.estimated_lighting_point_count
        + room.estimated_extractor_fan_count
        + room.estimated_hardwired_appliance_count
    )


def _apply_explicit_sanitary_overrides(
    room_takeoff: list[ExtractedRoomTakeoff],
    sections: list[ExtractedSection],
) -> list[ExtractedRoomTakeoff]:
    for line in _iter_section_lines(sections, "plumbing", "electrical", "joinery"):
        normalized = line.lower()
        matched_rooms = _match_room_takeoff_for_line(room_takeoff, normalized)
        if not matched_rooms:
            continue

        if "double basin" in normalized or "double vanity" in normalized:
            for room in matched_rooms:
                room.estimated_basin_count = max(room.estimated_basin_count, 2.0)
                room.estimated_vanity_unit_count = max(room.estimated_vanity_unit_count, 1.0)

        if "vanity unit" in normalized or "vanity" in normalized:
            for room in matched_rooms:
                room.estimated_vanity_unit_count = max(room.estimated_vanity_unit_count, 1.0)
                room.estimated_basin_count = max(room.estimated_basin_count, 2.0 if "double" in normalized else 1.0)

        if "double sink" in normalized:
            for room in matched_rooms:
                room.estimated_sink_count = max(room.estimated_sink_count, 2.0)

        if "separate shower" in normalized or ("bath" in normalized and "shower" in normalized):
            for room in matched_rooms:
                if room.room_type in {"bathroom", "ensuite", "shower_room"}:
                    room.estimated_shower_count = max(room.estimated_shower_count, 1.0)
                    room.estimated_bath_count = max(room.estimated_bath_count, 1.0) if "bath" in normalized else room.estimated_bath_count

        if "shower only" in normalized or "walk in shower" in normalized or "walk-in shower" in normalized:
            for room in matched_rooms:
                if room.room_type in {"bathroom", "ensuite", "shower_room"}:
                    room.estimated_bath_count = 0.0
                    room.estimated_shower_count = max(room.estimated_shower_count, 1.0)

        if "bath only" in normalized:
            for room in matched_rooms:
                if room.room_type == "bathroom":
                    room.estimated_bath_count = max(room.estimated_bath_count, 1.0)
                    room.estimated_shower_count = 0.0

        if "wc only" in normalized:
            for room in matched_rooms:
                if room.room_type == "wc":
                    room.estimated_basin_count = 0.0
                    room.estimated_bath_count = 0.0
                    room.estimated_shower_count = 0.0
                    room.estimated_plumbing_points = min(room.estimated_plumbing_points, 1.0)
                    room.estimated_plumbing_second_fix_points = min(room.estimated_plumbing_second_fix_points, 1.0)

        if "concealed cistern" in normalized or "wall hung wc" in normalized or "wall-hung wc" in normalized or "wall hung toilet" in normalized or "wall-hung toilet" in normalized:
            for room in matched_rooms:
                if room.room_type in {"bathroom", "ensuite", "shower_room", "wc"}:
                    room.estimated_wc_count = max(room.estimated_wc_count, 1.0)
                    room.estimated_concealed_cistern_count = max(room.estimated_concealed_cistern_count, 1.0)

        if "appliance connection" in normalized or "appliance connections" in normalized:
            explicit_count = _extract_explicit_count(normalized)
            if matched_rooms:
                if explicit_count and len(matched_rooms) == 1:
                    matched_rooms[0].estimated_appliance_connection_count = max(
                        matched_rooms[0].estimated_appliance_connection_count,
                        explicit_count,
                    )
                else:
                    for room in matched_rooms:
                        baseline = explicit_count if explicit_count and len(matched_rooms) == 1 else room.estimated_appliance_connection_count
                        room.estimated_appliance_connection_count = max(room.estimated_appliance_connection_count, baseline)

        if any(keyword in normalized for keyword in ("downlight", "downlights", "spotlight", "spotlights")):
            explicit_count = _extract_explicit_count(normalized)
            for room in matched_rooms:
                if explicit_count is not None and len(matched_rooms) == 1:
                    room.estimated_downlight_count = max(room.estimated_downlight_count, explicit_count)
                    room.estimated_lighting_point_count = max(room.estimated_lighting_point_count, explicit_count)
                elif explicit_count is not None:
                    room.estimated_downlight_count = max(room.estimated_downlight_count, explicit_count / len(matched_rooms))
                    room.estimated_lighting_point_count = max(room.estimated_lighting_point_count, explicit_count / len(matched_rooms))
                elif room.estimated_downlight_count <= 0:
                    room.estimated_downlight_count = max(room.estimated_downlight_count, room.estimated_lighting_point_count)

        if ("extractor" in normalized or "fan" in normalized) and any(
            keyword in normalized for keyword in ("duct", "ducting", "ducted", "vent to outside")
        ):
            explicit_count = _extract_explicit_count(normalized)
            for room in matched_rooms:
                baseline = explicit_count if explicit_count and len(matched_rooms) == 1 else 1.0
                room.estimated_ducted_extractor_count = max(room.estimated_ducted_extractor_count, baseline)
                room.estimated_extractor_fan_count = max(room.estimated_extractor_fan_count, baseline)

        if any(keyword in normalized for keyword in ("hardwire", "hardwired", "hardwiring")) and any(
            keyword in normalized for keyword in ("appliance", "appliances", "oven", "hob", "extractor", "dishwasher", "fridge", "freezer", "washing", "dryer")
        ):
            explicit_count = _extract_explicit_count(normalized)
            for room in matched_rooms:
                baseline = (
                    explicit_count
                    if explicit_count is not None and len(matched_rooms) == 1
                    else max(room.estimated_hardwired_appliance_count, 1.0)
                )
                room.estimated_hardwired_appliance_count = max(room.estimated_hardwired_appliance_count, baseline)
                room.estimated_appliance_connection_count = max(room.estimated_appliance_connection_count, baseline)

        if "client supply" in normalized and any(
            keyword in normalized for keyword in ("sanitary", "sanitaryware", "wc", "toilet", "basin", "bath", "shower", "sink", "vanity")
        ):
            for room in matched_rooms:
                room.estimated_sanitary_set_count = max(room.estimated_sanitary_set_count, 1.0)

        for room in matched_rooms:
            _rebuild_room_second_fix_points(room)

    return room_takeoff


def _has_project_scope(project_scopes: list[str], *keywords: str) -> bool:
    normalized_scopes = [scope.lower() for scope in project_scopes]
    return any(all(keyword in scope for keyword in keywords) for scope in normalized_scopes)


def _extension_candidate_rooms(rooms: list[ExtractedRoom], project_scopes: list[str]) -> list[ExtractedRoom]:
    candidates = [
        room
        for room in rooms
        if room.level.lower() == "ground floor" and any(token in room.name.lower() for token in {"rear", "extension"})
    ]
    if candidates:
        return candidates
    if _has_project_scope(project_scopes, "rear", "extension"):
        return [
            room
            for room in rooms
            if room.level.lower() == "ground floor" and any(token in room.name.lower() for token in {"kitchen", "dining"})
        ][:1]
    return []


def build_room_takeoff(rooms: list[ExtractedRoom]) -> list[ExtractedRoomTakeoff]:
    room_takeoff: list[ExtractedRoomTakeoff] = []
    for room in rooms:
        room_type = _classify_room_type(room.name)
        fixture_counts = _estimate_room_fixture_counts(room_type)
        estimated_socket_count = _estimate_room_socket_count(room, room_type)
        estimated_switch_count = _estimate_room_switch_count(room, room_type)
        estimated_lighting_point_count = _estimate_room_lighting_points(room, room_type)
        estimated_extractor_fan_count = _estimate_room_extractor_fan_count(room_type)
        estimated_plumbing_points = _estimate_room_plumbing_points(room_type)
        room_takeoff.append(
            ExtractedRoomTakeoff(
                level=room.level,
                name=room.name,
                room_type=room_type,
                area_m2=room.area_m2,
                estimated_wall_finish_area_m2=_estimate_room_wall_finish_area(room),
                estimated_ceiling_finish_area_m2=_round(room.area_m2),
                estimated_skirting_lm=_estimate_room_skirting_lm(room),
                estimated_wall_tiling_area_m2=_estimate_room_wall_tiling_area(room, room_type),
                estimated_floor_finish_area_m2=_round(room.area_m2),
                estimated_plumbing_points=estimated_plumbing_points,
                estimated_wc_count=fixture_counts["estimated_wc_count"],
                estimated_basin_count=fixture_counts["estimated_basin_count"],
                estimated_bath_count=fixture_counts["estimated_bath_count"],
                estimated_shower_count=fixture_counts["estimated_shower_count"],
                estimated_sink_count=fixture_counts["estimated_sink_count"],
                estimated_sanitary_set_count=fixture_counts["estimated_sanitary_set_count"],
                estimated_vanity_unit_count=fixture_counts["estimated_vanity_unit_count"],
                estimated_concealed_cistern_count=fixture_counts["estimated_concealed_cistern_count"],
                estimated_appliance_connection_count=fixture_counts["estimated_appliance_connection_count"],
                estimated_socket_count=estimated_socket_count,
                estimated_switch_count=estimated_switch_count,
                estimated_lighting_point_count=estimated_lighting_point_count,
                estimated_downlight_count=0.0,
                estimated_extractor_fan_count=estimated_extractor_fan_count,
                estimated_ducted_extractor_count=0.0,
                estimated_hardwired_appliance_count=0.0,
                estimated_electrical_second_fix_points=_round(
                    estimated_socket_count + estimated_switch_count + estimated_lighting_point_count + estimated_extractor_fan_count
                ),
                estimated_plumbing_second_fix_points=estimated_plumbing_points,
            )
        )
    return room_takeoff


def _iter_section_lines(sections: list[ExtractedSection], *section_keys: str):
    for section in sections:
        if section.key not in section_keys:
            continue
        for line in section.lines:
            yield line


def _find_count(sections: list[ExtractedSection], section_key: str, *keywords: str) -> int:
    total = 0.0
    for section in sections:
        if section.key != section_key:
            continue
        for item in section.count_items:
            name = item.name.lower()
            if all(keyword in name for keyword in keywords):
                total += item.quantity
    return int(total)


def _find_measure(sections: list[ExtractedSection], section_key: str, *keywords: str) -> float:
    total = 0.0
    for section in sections:
        if section.key != section_key:
            continue
        for item in section.measure_items:
            name = item.name.lower()
            if all(keyword in name for keyword in keywords):
                total += item.value
    return total


def _sum_dimension_volume(sections: list[ExtractedSection], section_key: str, *keywords: str) -> float:
    total = 0.0
    for section in sections:
        if section.key != section_key:
            continue
        for item in section.dimension_items:
            name = item.name.lower()
            if item.volume_m3 and all(keyword in name for keyword in keywords):
                total += item.volume_m3
    return total


def _sum_dimension_area(sections: list[ExtractedSection], section_key: str, *keywords: str) -> float:
    total = 0.0
    for section in sections:
        if section.key != section_key:
            continue
        for item in section.dimension_items:
            name = item.name.lower()
            if item.area_m2 and all(keyword in name for keyword in keywords):
                total += item.area_m2
    return total


def _extract_percent_from_lines(sections: list[ExtractedSection], section_keys: tuple[str, ...], *keywords: str) -> float:
    for line in _iter_section_lines(sections, *section_keys):
        normalized = line.lower()
        if not all(keyword in normalized for keyword in keywords):
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", normalized)
        if match:
            return float(match.group(1))
    return 0.0


def _collect_level_hits(
    sections: list[ExtractedSection],
    section_keys: tuple[str, ...],
    predicate,
) -> tuple[set[str], bool]:
    levels: set[str] = set()
    has_general_hit = False

    for section in sections:
        if section.key not in section_keys:
            continue
        active_level: str | None = None
        for line in section.lines:
            normalized = line.strip().lower().rstrip(":")
            if normalized in _LEVEL_AREA_LOOKUP:
                active_level = normalized
                continue
            if not predicate(normalized):
                continue
            explicit_levels = {level for level in _LEVEL_AREA_LOOKUP if level in normalized}
            if explicit_levels:
                levels.update(explicit_levels)
                continue
            if active_level:
                levels.add(active_level)
            else:
                has_general_hit = True

    return levels, has_general_hit


def _sum_levels(
    levels: set[str],
    *,
    basement_area_m2: float,
    ground_floor_area_m2: float,
    first_floor_area_m2: float,
    second_floor_area_m2: float,
    loft_area_m2: float,
) -> float:
    total = 0.0
    if "basement" in levels or "lower ground floor" in levels:
        total += basement_area_m2
    if "ground floor" in levels:
        total += ground_floor_area_m2
    if "first floor" in levels:
        total += first_floor_area_m2
    if "second floor" in levels:
        total += second_floor_area_m2
    if "loft" in levels:
        total += loft_area_m2
    return total


def _estimate_boarding_coverage(
    sections: list[ExtractedSection],
    *,
    basement_area_m2: float,
    ground_floor_area_m2: float,
    first_floor_area_m2: float,
    second_floor_area_m2: float,
    loft_area_m2: float,
    total_internal_floor_area_m2: float,
    roof_area_m2: float,
) -> float:
    ceiling_levels, ceiling_general = _collect_level_hits(
        sections,
        ("ceilings_and_insulation", "structure"),
        lambda line: ("plasterboard" in line or "board" in line) and "ceiling" in line,
    )
    roof_board_levels, roof_board_general = _collect_level_hits(
        sections,
        ("ceilings_and_insulation", "structure", "carpentry"),
        lambda line: ("plasterboard" in line or "board" in line)
        and any(keyword in line for keyword in ("rafter", "roof", "dormer", "insulated board", "board under")),
    )

    coverage = _sum_levels(
        ceiling_levels,
        basement_area_m2=basement_area_m2,
        ground_floor_area_m2=ground_floor_area_m2,
        first_floor_area_m2=first_floor_area_m2,
        second_floor_area_m2=second_floor_area_m2,
        loft_area_m2=loft_area_m2,
    )
    if ceiling_general and not ceiling_levels:
        coverage += total_internal_floor_area_m2
    if roof_area_m2 and (roof_board_general or roof_board_levels):
        coverage += roof_area_m2
    return _round(coverage)


def _estimate_insulation_coverage(
    sections: list[ExtractedSection],
    *,
    basement_area_m2: float,
    ground_floor_area_m2: float,
    first_floor_area_m2: float,
    second_floor_area_m2: float,
    loft_area_m2: float,
    total_internal_floor_area_m2: float,
    roof_area_m2: float,
) -> float:
    floor_levels, floor_general = _collect_level_hits(
        sections,
        ("structure", "carpentry", "ceilings_and_insulation", "plumbing"),
        lambda line: any(keyword in line for keyword in ("insulation", "acoustic", "pir", "same floor build-up", "same floor buildup"))
        and not any(keyword in line for keyword in ("rafter", "warm roof", "roof insulation", "insulated board under")),
    )
    roof_levels, roof_general = _collect_level_hits(
        sections,
        ("structure", "ceilings_and_insulation"),
        lambda line: any(keyword in line for keyword in ("warm roof", "between rafters", "under rafters", "roof insulation", "insulated board", "rafter"))
        or ("pir" in line and "roof" in line),
    )

    coverage = _sum_levels(
        floor_levels,
        basement_area_m2=basement_area_m2,
        ground_floor_area_m2=ground_floor_area_m2,
        first_floor_area_m2=first_floor_area_m2,
        second_floor_area_m2=second_floor_area_m2,
        loft_area_m2=loft_area_m2,
    )
    if floor_general and not floor_levels:
        coverage += total_internal_floor_area_m2
    if roof_area_m2 and (roof_general or roof_levels):
        coverage += roof_area_m2
    return _round(coverage)


def _estimate_loft_roof_finish_area(
    sections: list[ExtractedSection],
    *,
    loft_area_m2: float,
    dormer_area_m2: float,
) -> float:
    if not loft_area_m2 and not dormer_area_m2:
        return 0.0
    has_loft_roof_scope = False
    for line in _iter_section_lines(sections, "structure", "ceilings_and_insulation", "carpentry"):
        normalized = line.lower()
        if any(keyword in normalized for keyword in ("loft", "dormer", "rafter", "warm roof", "roof", "skylight")):
            has_loft_roof_scope = True
            break
    if not has_loft_roof_scope:
        return 0.0
    return _round(loft_area_m2 + dormer_area_m2)


def _estimate_extension_slab_area(
    rooms: list[ExtractedRoom],
    project_scopes: list[str],
    *,
    roof_area_m2: float,
) -> float:
    extension_rooms = _extension_candidate_rooms(rooms, project_scopes)
    if extension_rooms:
        return _round(sum(room.area_m2 for room in extension_rooms))
    if _has_project_scope(project_scopes, "rear", "extension") and roof_area_m2:
        return _round(roof_area_m2)
    return 0.0


def _estimate_extension_flat_roof_area(
    sections: list[ExtractedSection],
    project_scopes: list[str],
    *,
    roof_area_m2: float,
    extension_slab_area_m2: float,
) -> float:
    has_flat_roof_scope = any(
        any(keyword in line.lower() for keyword in ("flat roof", "warm roof", "grp"))
        for line in _iter_section_lines(sections, "structure")
    )
    if not has_flat_roof_scope:
        return 0.0
    if roof_area_m2:
        return _round(roof_area_m2)
    if _has_project_scope(project_scopes, "rear", "extension") and extension_slab_area_m2:
        return _round(extension_slab_area_m2)
    return 0.0


def _estimate_extension_external_wall_area(
    rooms: list[ExtractedRoom],
    project_scopes: list[str],
    *,
    extension_slab_area_m2: float,
) -> float:
    extension_rooms = _extension_candidate_rooms(rooms, project_scopes)
    total = 0.0
    for room in extension_rooms:
        width_m = max(room.width_m, room.length_m)
        depth_m = min(room.width_m, room.length_m)
        exposed_perimeter = width_m + (2 * depth_m)
        total += exposed_perimeter * _LEVEL_WALL_HEIGHTS_M["ground floor"] * _OPENING_DEDUCTION_FACTOR
    if total > 0:
        return _round(total)
    if extension_slab_area_m2 and _has_project_scope(project_scopes, "rear", "extension"):
        estimated_open_perimeter = 3 * (extension_slab_area_m2 ** 0.5)
        return _round(estimated_open_perimeter * _LEVEL_WALL_HEIGHTS_M["ground floor"] * _OPENING_DEDUCTION_FACTOR)
    return 0.0


def _estimate_roof_removal_area(
    sections: list[ExtractedSection],
    *,
    loft_area_m2: float,
    dormer_area_m2: float,
) -> float:
    explicit_demolition_area = _round(_sum_dimension_area(sections, "demolition", "roof"))
    removal_pct = _extract_percent_from_lines(sections, ("structure", "demolition"), "roof", "removal")
    if removal_pct:
        removal_base = loft_area_m2 + dormer_area_m2
        if removal_base:
            explicit_demolition_area += removal_base * (removal_pct / 100)
    return _round(explicit_demolition_area)


def _extract_trench_foundation_volume(sections: list[ExtractedSection]) -> float:
    total_length = _find_measure(sections, "groundworks", "trench foundations", "total")
    depth = _find_measure(sections, "groundworks", "trench foundations", "depth")
    width = _find_measure(sections, "groundworks", "trench foundations", "width")
    if total_length and depth and width:
        return _round(total_length * depth * width)
    return 0.0


def _extract_door_allowance(sections: list[ExtractedSection], door_set_count: int) -> float:
    for section in sections:
        if section.key != "doors":
            continue
        for line in section.lines:
            match = re.search(r"£\s*(\d+(?:\.\d+)?)\s*per door", line, re.IGNORECASE)
            if match:
                return _round(float(match.group(1)) * door_set_count)
    return 0.0


def _extract_fire_rated_door_set_count(sections: list[ExtractedSection]) -> float:
    return _extract_contextual_line_count(
        sections,
        ("doors",),
        predicate=lambda line, _context: "door" in line
        and any(keyword in line for keyword in ("fire-rated", "fire rated", "fd30", "fd60", "fd90")),
        default_when_mentioned=1.0,
    )


def _extract_pocket_door_set_count(sections: list[ExtractedSection]) -> float:
    return _extract_contextual_line_count(
        sections,
        ("doors",),
        predicate=lambda line, _context: "pocket" in line and "door" in line,
        default_when_mentioned=1.0,
    )


def _extract_fire_rated_pocket_door_set_count(sections: list[ExtractedSection]) -> float:
    return _extract_contextual_line_count(
        sections,
        ("doors",),
        predicate=lambda line, _context: "pocket" in line
        and "door" in line
        and any(keyword in line for keyword in ("fire-rated", "fire rated", "fd30", "fd60", "fd90")),
        default_when_mentioned=1.0,
    )


def _extract_double_pocket_door_set_count(sections: list[ExtractedSection]) -> float:
    return _extract_contextual_line_count(
        sections,
        ("doors",),
        predicate=lambda line, _context: "double" in line and "pocket" in line and "door" in line,
        default_when_mentioned=1.0,
    )


def _extract_staircase_count(sections: list[ExtractedSection]) -> float:
    return _extract_contextual_line_count(
        sections,
        ("joinery", "carpentry"),
        predicate=lambda line, _context: any(keyword in line for keyword in ("staircase", "stairs", "stair")),
        default_when_mentioned=1.0,
    )


def _extract_storage_door_set_count(sections: list[ExtractedSection]) -> float:
    return _extract_contextual_line_count(
        sections,
        ("joinery",),
        predicate=lambda line, context: "storage" in context
        and "door" in line
        and any(keyword in line for keyword in ("set", "sets", "double")),
        default_when_mentioned=1.0,
    )


def _has_explicit_line_scope(
    sections: list[ExtractedSection],
    section_keys: tuple[str, ...] | None,
    *,
    keywords: tuple[str, ...],
) -> bool:
    if section_keys:
        for line in _iter_section_lines(sections, *section_keys):
            normalized = line.lower()
            if all(keyword in normalized for keyword in keywords):
                return True
        return False

    for section in sections:
        for line in section.lines:
            normalized = line.lower()
            if all(keyword in normalized for keyword in keywords):
                return True
    return False


def _derive_architrave_set_count(sections: list[ExtractedSection], door_set_count: int) -> float:
    if not door_set_count:
        return 0.0
    if _has_explicit_line_scope(
        sections,
        None,
        keywords=("architrave",),
    ):
        return float(door_set_count)
    return 0.0


def _derive_window_board_metrics(
    sections: list[ExtractedSection],
    *,
    upvc_window_count: int,
) -> tuple[float, float]:
    if not _has_explicit_line_scope(
        sections,
        None,
        keywords=("window", "board"),
    ):
        return 0.0, 0.0

    total_lm = 0.0
    total_count = 0.0
    for section in sections:
        if section.key != "windows":
            continue
        for item in section.dimension_items:
            name = item.name.lower()
            if "window" not in name or "velux" in name:
                continue
            if item.length_m:
                total_lm += item.length_m * item.count
            total_count += item.count
        if total_lm <= 0:
            for item in section.count_items:
                name = item.name.lower()
                if "window" not in name or "velux" in name:
                    continue
                dims_match = re.search(r"\((\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\)", item.raw_text, re.IGNORECASE)
                if dims_match:
                    total_lm += float(dims_match.group(1)) * item.quantity
                total_count += item.quantity

    if total_count <= 0 and upvc_window_count:
        total_count = float(upvc_window_count)
    return _round(total_count), _round(total_lm)


def build_takeoff_summary(
    rooms: list[ExtractedRoom],
    sections: list[ExtractedSection],
    *,
    project_scopes: list[str] | None = None,
    room_takeoff: list[ExtractedRoomTakeoff] | None = None,
) -> StructuredTakeoffSummary:
    project_scopes = project_scopes or []
    basement_area_m2 = _sum_room_area(rooms, "Basement") + _sum_room_area(rooms, "Lower Ground Floor")
    ground_floor_area_m2 = _sum_room_area(rooms, "Ground Floor")
    first_floor_area_m2 = _sum_room_area(rooms, "First Floor")
    second_floor_area_m2 = _sum_room_area(rooms, "Second Floor")
    loft_area_m2 = _sum_room_area(rooms, "Loft")
    total_internal_floor_area_m2 = _round(sum(room.area_m2 for room in rooms))
    estimated_wall_finish_area_m2 = _sum_estimated_wall_finish_area(rooms)
    estimated_ceiling_finish_area_m2 = _sum_estimated_ceiling_area(rooms)
    estimated_skirting_lm = _sum_estimated_skirting_lm(rooms)
    room_takeoff = room_takeoff or build_room_takeoff(rooms)
    estimated_hard_floor_scope_area_m2, estimated_carpet_scope_area_m2 = _estimate_flooring_scope_areas(sections, room_takeoff)
    wet_rooms = [room for room in room_takeoff if room.room_type in {"bathroom", "ensuite", "shower_room", "wc"}]
    client_supply_sanitaryware = _has_client_supply_sanitaryware(sections)
    client_supply_appliances = _has_client_supply_appliances(sections)

    pad_foundation_volume_m3 = _round(_sum_dimension_volume(sections, "groundworks", "pad foundations"))
    trench_foundation_volume_m3 = _extract_trench_foundation_volume(sections)
    internal_foundation_volume_m3 = _round(
        _sum_dimension_volume(sections, "groundworks", "internal foundation")
    )

    patio_area_m2 = _round(_sum_dimension_area(sections, "demolition", "patio"))
    roof_area_m2 = _round(
        _sum_dimension_area(sections, "demolition", "roof")
        + _sum_dimension_area(sections, "structure", "roof")
    )
    estimated_extension_slab_area_m2 = _estimate_extension_slab_area(
        rooms,
        project_scopes,
        roof_area_m2=roof_area_m2,
    )
    estimated_extension_flat_roof_area_m2 = _estimate_extension_flat_roof_area(
        sections,
        project_scopes,
        roof_area_m2=roof_area_m2,
        extension_slab_area_m2=estimated_extension_slab_area_m2,
    )
    estimated_extension_external_wall_area_m2 = _estimate_extension_external_wall_area(
        rooms,
        project_scopes,
        extension_slab_area_m2=estimated_extension_slab_area_m2,
    )
    estimated_boarding_coverage_m2 = _estimate_boarding_coverage(
        sections,
        basement_area_m2=basement_area_m2,
        ground_floor_area_m2=ground_floor_area_m2,
        first_floor_area_m2=first_floor_area_m2,
        second_floor_area_m2=second_floor_area_m2,
        loft_area_m2=loft_area_m2,
        total_internal_floor_area_m2=total_internal_floor_area_m2,
        roof_area_m2=roof_area_m2,
    )
    estimated_insulation_coverage_m2 = _estimate_insulation_coverage(
        sections,
        basement_area_m2=basement_area_m2,
        ground_floor_area_m2=ground_floor_area_m2,
        first_floor_area_m2=first_floor_area_m2,
        second_floor_area_m2=second_floor_area_m2,
        loft_area_m2=loft_area_m2,
        total_internal_floor_area_m2=total_internal_floor_area_m2,
        roof_area_m2=roof_area_m2,
    )
    dormer_area_m2 = _round(_sum_dimension_area(sections, "structure", "dormer"))
    estimated_loft_roof_finish_area_m2 = _estimate_loft_roof_finish_area(
        sections,
        loft_area_m2=loft_area_m2,
        dormer_area_m2=dormer_area_m2,
    )
    estimated_roof_removal_area_m2 = _estimate_roof_removal_area(
        sections,
        loft_area_m2=loft_area_m2,
        dormer_area_m2=dormer_area_m2,
    )
    skylight_area_m2 = _round(_sum_dimension_area(sections, "structure", "skylight"))

    skylight_count = _find_count(sections, "structure", "skylight")
    velux_count = _find_count(sections, "windows", "velux")
    upvc_window_count = _find_count(sections, "windows", "upvc", "window")
    steel_post_count = _find_count(sections, "steelworks", "total posts")
    steel_junction_count = _find_count(sections, "steelworks", "junctions")
    radiator_removal_count = _find_count(sections, "demolition", "radiators")
    radiator_install_count = _find_count(sections, "plumbing", "radiators")
    socket_count = _find_count(sections, "electrical", "sockets")
    switch_count = _find_count(sections, "electrical", "switches")
    extractor_fan_count = _find_count(sections, "electrical", "extractor fans")
    consumer_unit_count = _extract_explicit_feature_count(
        sections,
        ("electrical", "full_scope"),
        required_keywords=("consumer unit", "distribution board"),
        default_when_mentioned=1.0,
    )
    downlight_count = _extract_explicit_feature_count(
        sections,
        ("electrical", "full_scope"),
        required_keywords=("downlight", "downlights", "spotlight", "spotlights"),
    )
    if downlight_count <= 0:
        downlight_count = _round(sum(room.estimated_downlight_count for room in room_takeoff))
    ducted_extractor_count = _extract_explicit_feature_count(
        sections,
        ("electrical", "full_scope"),
        required_keywords=("extractor", "fan"),
        secondary_keywords=("duct", "ducting", "ducted", "vent to outside"),
    )
    if ducted_extractor_count <= 0:
        ducted_extractor_count = _round(sum(room.estimated_ducted_extractor_count for room in room_takeoff))
    hardwired_appliance_count = _extract_explicit_feature_count(
        sections,
        ("electrical", "full_scope", "joinery"),
        required_keywords=("hardwire", "hardwired", "hardwiring"),
        secondary_keywords=("appliance", "appliances", "oven", "hob", "extractor", "dishwasher", "fridge", "freezer", "washing", "dryer"),
    )
    if hardwired_appliance_count <= 0:
        hardwired_appliance_count = _round(sum(room.estimated_hardwired_appliance_count for room in room_takeoff))
    door_set_count = _find_count(sections, "doors", "door")
    fire_rated_door_set_count = _extract_fire_rated_door_set_count(sections)
    pocket_door_set_count = _extract_pocket_door_set_count(sections)
    fire_rated_pocket_door_set_count = _extract_fire_rated_pocket_door_set_count(sections)
    double_pocket_door_set_count = _extract_double_pocket_door_set_count(sections)
    staircase_count = _extract_staircase_count(sections)
    storage_door_set_count = _extract_storage_door_set_count(sections)
    architrave_set_count = _derive_architrave_set_count(sections, door_set_count)
    estimated_architrave_lm = _round(architrave_set_count * _ARCHITRAVE_LM_PER_DOOR_SET)
    tree_count = _find_count(sections, "demolition", "tree")
    fence_count = _find_count(sections, "demolition", "fence")
    soil_volume_m3 = _round(_find_measure(sections, "demolition", "soil"))
    door_supply_allowance_total = _extract_door_allowance(sections, door_set_count)
    window_board_count, estimated_window_board_lm = _derive_window_board_metrics(
        sections,
        upvc_window_count=upvc_window_count,
    )

    return StructuredTakeoffSummary(
        room_count=len(rooms),
        basement_area_m2=basement_area_m2,
        ground_floor_area_m2=ground_floor_area_m2,
        first_floor_area_m2=first_floor_area_m2,
        second_floor_area_m2=second_floor_area_m2,
        loft_area_m2=loft_area_m2,
        total_internal_floor_area_m2=total_internal_floor_area_m2,
        estimated_wall_finish_area_m2=estimated_wall_finish_area_m2,
        estimated_ceiling_finish_area_m2=estimated_ceiling_finish_area_m2,
        estimated_skirting_lm=estimated_skirting_lm,
        estimated_hard_floor_scope_area_m2=estimated_hard_floor_scope_area_m2,
        estimated_carpet_scope_area_m2=estimated_carpet_scope_area_m2,
        estimated_boarding_coverage_m2=estimated_boarding_coverage_m2,
        estimated_insulation_coverage_m2=estimated_insulation_coverage_m2,
        estimated_extension_slab_area_m2=estimated_extension_slab_area_m2,
        estimated_extension_flat_roof_area_m2=estimated_extension_flat_roof_area_m2,
        estimated_extension_external_wall_area_m2=estimated_extension_external_wall_area_m2,
        estimated_loft_roof_finish_area_m2=estimated_loft_roof_finish_area_m2,
        estimated_roof_removal_area_m2=estimated_roof_removal_area_m2,
        estimated_dormer_build_area_m2=dormer_area_m2,
        wet_room_count=len(wet_rooms),
        estimated_wet_room_wall_tiling_area_m2=_round(sum(room.estimated_wall_tiling_area_m2 for room in wet_rooms)),
        estimated_wet_room_floor_tiling_area_m2=_round(sum(room.estimated_floor_finish_area_m2 for room in wet_rooms)),
        total_wc_count=_round(sum(room.estimated_wc_count for room in room_takeoff)),
        total_basin_count=_round(sum(room.estimated_basin_count for room in room_takeoff)),
        total_bath_count=_round(sum(room.estimated_bath_count for room in room_takeoff)),
        total_shower_count=_round(sum(room.estimated_shower_count for room in room_takeoff)),
        total_sink_count=_round(sum(room.estimated_sink_count for room in room_takeoff)),
        total_sanitary_set_count=_round(sum(room.estimated_sanitary_set_count for room in room_takeoff)),
        total_vanity_unit_count=_round(sum(room.estimated_vanity_unit_count for room in room_takeoff)),
        total_concealed_cistern_count=_round(sum(room.estimated_concealed_cistern_count for room in room_takeoff)),
        total_appliance_connection_count=_round(sum(room.estimated_appliance_connection_count for room in room_takeoff)),
        consumer_unit_count=consumer_unit_count,
        downlight_count=downlight_count,
        ducted_extractor_count=ducted_extractor_count,
        hardwired_appliance_count=hardwired_appliance_count,
        client_supply_sanitaryware=client_supply_sanitaryware,
        client_supply_appliances=client_supply_appliances,
        estimated_electrical_second_fix_points=_round(sum(room.estimated_electrical_second_fix_points for room in room_takeoff)),
        estimated_plumbing_second_fix_points=_round(sum(room.estimated_plumbing_second_fix_points for room in room_takeoff)),
        pad_foundation_volume_m3=pad_foundation_volume_m3,
        trench_foundation_volume_m3=trench_foundation_volume_m3,
        internal_foundation_volume_m3=internal_foundation_volume_m3,
        total_foundation_concrete_m3=_round(
            pad_foundation_volume_m3 + trench_foundation_volume_m3 + internal_foundation_volume_m3
        ),
        patio_area_m2=patio_area_m2,
        roof_area_m2=roof_area_m2,
        dormer_area_m2=dormer_area_m2,
        skylight_count=skylight_count,
        skylight_area_m2=skylight_area_m2,
        velux_count=velux_count,
        upvc_window_count=upvc_window_count,
        steel_post_count=steel_post_count,
        steel_junction_count=steel_junction_count,
        radiator_removal_count=radiator_removal_count,
        radiator_install_count=radiator_install_count,
        socket_count=socket_count,
        switch_count=switch_count,
        extractor_fan_count=extractor_fan_count,
        door_set_count=door_set_count,
        fire_rated_door_set_count=fire_rated_door_set_count,
        pocket_door_set_count=pocket_door_set_count,
        fire_rated_pocket_door_set_count=fire_rated_pocket_door_set_count,
        double_pocket_door_set_count=double_pocket_door_set_count,
        staircase_count=staircase_count,
        storage_door_set_count=storage_door_set_count,
        architrave_set_count=architrave_set_count,
        estimated_architrave_lm=estimated_architrave_lm,
        window_board_count=window_board_count,
        estimated_window_board_lm=estimated_window_board_lm,
        door_supply_allowance_total=door_supply_allowance_total,
        tree_count=tree_count,
        fence_count=fence_count,
        soil_volume_m3=soil_volume_m3,
    )


def build_room_takeoff_with_overrides(
    rooms: list[ExtractedRoom],
    sections: list[ExtractedSection],
) -> list[ExtractedRoomTakeoff]:
    return _apply_explicit_sanitary_overrides(build_room_takeoff(rooms), sections)
