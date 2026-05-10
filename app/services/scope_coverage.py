from __future__ import annotations

from dataclasses import dataclass

from app.schemas import PreviewAssumption, PreviewMatchedRow, ScopeExtractionResponse
from app.services.normalizer import tokenize


@dataclass(frozen=True)
class CoverageBucket:
    key: str
    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class SectionDetailBucket:
    key: str
    label: str
    trigger_keywords: tuple[str, ...]
    expected_keywords: tuple[str, ...]


@dataclass(frozen=True)
class ScopeProfile:
    key: str
    label: str
    trigger_keywords: tuple[str, ...]
    room_keywords: tuple[str, ...]
    expected_buckets: tuple[CoverageBucket, ...]


COMMON_BUCKETS = {
    "demolition": CoverageBucket("demolition", "strip out / demolition", ("strip", "stripout", "strip-out", "demolition", "demolish", "remove", "removal", "hack", "hackoff")),
    "plastering": CoverageBucket("plastering", "plastering / wall preparation", ("plaster", "skim", "render", "makegood", "make-good", "patch", "repair")),
    "decorations": CoverageBucket("decorations", "decorations", ("paint", "painting", "decorate", "decorating")),
    "flooring": CoverageBucket("flooring", "flooring", ("floor", "flooring", "laminate", "vinyl", "carpet", "underlay", "timber", "wood", "engineered")),
    "tiling": CoverageBucket("tiling", "tiling / waterproofing", ("tile", "tiling", "grout", "waterproof", "tank", "tanking")),
    "electrics": CoverageBucket("electrics", "electrics", ("electric", "electrical", "socket", "light", "lighting", "downlight", "switch", "consumer", "alarm")),
    "plumbing": CoverageBucket("plumbing", "plumbing", ("plumb", "plumbing", "pipe", "sink", "wc", "toilet", "basin", "radiator", "bath", "shower")),
    "joinery": CoverageBucket("joinery", "joinery / fit-out", ("joinery", "cabinet", "cabinetry", "worktop", "door", "frame", "skirting", "architrave", "kitchen")),
    "structure": CoverageBucket("structure", "structure / steel / framing", ("steel", "beam", "joist", "structure", "structural", "framing", "frame")),
    "insulation": CoverageBucket("insulation", "insulation", ("insulation", "insulate")),
    "partitions": CoverageBucket("partitions", "partitions / plasterboard", ("partition", "stud", "plasterboard", "board", "drywall")),
    "groundworks": CoverageBucket("groundworks", "groundworks / excavation", ("excavate", "excavation", "dig", "trench", "foundation", "concrete", "slab", "groundwork")),
    "brickwork": CoverageBucket("brickwork", "brickwork / blockwork", ("brick", "brickwork", "block", "blockwork", "masonry")),
    "roofing": CoverageBucket("roofing", "roofing", ("roof", "roofing", "joist", "rafter", "felt", "membrane", "tile", "slate", "flatroof", "flat-roof")),
}


SCOPE_PROFILES: tuple[ScopeProfile, ...] = (
    ScopeProfile(
        key="kitchen_refurbishment",
        label="kitchen refurbishment",
        trigger_keywords=("kitchen refurbishment", "kitchen renovation", "new kitchen", "kitchen refit"),
        room_keywords=("kitchen",),
        expected_buckets=(
            COMMON_BUCKETS["demolition"],
            COMMON_BUCKETS["plastering"],
            COMMON_BUCKETS["decorations"],
            COMMON_BUCKETS["flooring"],
            COMMON_BUCKETS["electrics"],
            COMMON_BUCKETS["plumbing"],
            COMMON_BUCKETS["joinery"],
        ),
    ),
    ScopeProfile(
        key="bathroom_renovation",
        label="bathroom renovation",
        trigger_keywords=("bathroom renovation", "bathroom refurb", "new bathroom", "bathroom refit"),
        room_keywords=("bathroom",),
        expected_buckets=(
            COMMON_BUCKETS["demolition"],
            COMMON_BUCKETS["plumbing"],
            COMMON_BUCKETS["electrics"],
            COMMON_BUCKETS["tiling"],
            COMMON_BUCKETS["decorations"],
        ),
    ),
    ScopeProfile(
        key="loft_conversion",
        label="loft conversion",
        trigger_keywords=("loft conversion", "loft refurb", "attic conversion"),
        room_keywords=("loft", "attic"),
        expected_buckets=(
            COMMON_BUCKETS["structure"],
            COMMON_BUCKETS["insulation"],
            COMMON_BUCKETS["partitions"],
            COMMON_BUCKETS["electrics"],
            COMMON_BUCKETS["plumbing"],
            COMMON_BUCKETS["plastering"],
            COMMON_BUCKETS["decorations"],
        ),
    ),
    ScopeProfile(
        key="full_property_refresh",
        label="full property refresh",
        trigger_keywords=("full flat refresh", "full house refresh", "full property refresh", "full redecoration"),
        room_keywords=("flat", "house", "property"),
        expected_buckets=(
            COMMON_BUCKETS["decorations"],
            COMMON_BUCKETS["flooring"],
            COMMON_BUCKETS["joinery"],
            COMMON_BUCKETS["electrics"],
            COMMON_BUCKETS["plumbing"],
            COMMON_BUCKETS["plastering"],
        ),
    ),
    ScopeProfile(
        key="full_house_refurbishment",
        label="full house refurbishment",
        trigger_keywords=("full refurb", "full house refurb", "whole house renovation", "whole house refurb", "full renovation"),
        room_keywords=("house", "property"),
        expected_buckets=(
            COMMON_BUCKETS["demolition"],
            COMMON_BUCKETS["plastering"],
            COMMON_BUCKETS["decorations"],
            COMMON_BUCKETS["flooring"],
            COMMON_BUCKETS["electrics"],
            COMMON_BUCKETS["plumbing"],
            COMMON_BUCKETS["joinery"],
        ),
    ),
    ScopeProfile(
        key="rear_extension",
        label="rear extension",
        trigger_keywords=("rear extension", "side return extension", "kitchen extension", "ground floor extension"),
        room_keywords=(),
        expected_buckets=(
            COMMON_BUCKETS["groundworks"],
            COMMON_BUCKETS["structure"],
            COMMON_BUCKETS["brickwork"],
            COMMON_BUCKETS["roofing"],
            COMMON_BUCKETS["electrics"],
            COMMON_BUCKETS["plumbing"],
            COMMON_BUCKETS["plastering"],
            COMMON_BUCKETS["decorations"],
        ),
    ),
)


def _matched_scope_tokens(rows: list[PreviewMatchedRow]) -> set[str]:
    values: list[str] = []
    for row in rows:
        values.extend([row.WorkName, row.Unit or "", row.ReviewReason or ""])
    return tokenize(" ".join(values))


SECTION_EXPECTED_BUCKETS = {
    "demolition": (COMMON_BUCKETS["demolition"],),
    "groundworks": (COMMON_BUCKETS["groundworks"],),
    "steelworks": (COMMON_BUCKETS["structure"],),
    "structure": (COMMON_BUCKETS["structure"], COMMON_BUCKETS["brickwork"], COMMON_BUCKETS["roofing"]),
    "carpentry": (COMMON_BUCKETS["joinery"], COMMON_BUCKETS["partitions"]),
    "ceilings_and_insulation": (COMMON_BUCKETS["insulation"], COMMON_BUCKETS["plastering"]),
    "doors": (COMMON_BUCKETS["joinery"],),
    "joinery": (COMMON_BUCKETS["joinery"],),
    "plastering": (COMMON_BUCKETS["plastering"],),
    "tiling": (COMMON_BUCKETS["tiling"],),
    "plumbing": (COMMON_BUCKETS["plumbing"],),
    "heating": (COMMON_BUCKETS["plumbing"],),
    "electrical": (COMMON_BUCKETS["electrics"],),
    "decorating": (COMMON_BUCKETS["decorations"],),
    "flooring": (COMMON_BUCKETS["flooring"],),
    "windows": (COMMON_BUCKETS["joinery"],),
    "woodwork_painting": (COMMON_BUCKETS["decorations"], COMMON_BUCKETS["joinery"]),
}


SECTION_DETAIL_BUCKETS: dict[str, tuple[SectionDetailBucket, ...]] = {
    "doors": (
        SectionDetailBucket(
            "fire_rated_doors",
            "fire-rated door sets",
            ("fire-rated", "fire rated", "fd30", "fd60", "fd90"),
            ("fire", "door"),
        ),
        SectionDetailBucket(
            "pocket_doors",
            "pocket door sets",
            ("pocket door", "pocket doors"),
            ("pocket", "door"),
        ),
        SectionDetailBucket(
            "door_allowance",
            "door supply allowance",
            ("per door", "allowance"),
            ("door", "allowance", "supply"),
        ),
    ),
    "joinery": (
        SectionDetailBucket(
            "staircase",
            "staircase joinery",
            ("staircase", "stairs"),
            ("stair", "staircase"),
        ),
        SectionDetailBucket(
            "storage_joinery",
            "storage / fitted joinery door sets",
            ("storage", "double door set", "double door sets"),
            ("storage", "cupboard", "wardrobe", "door", "set"),
        ),
    ),
    "decorating": (
        SectionDetailBucket(
            "wall_decoration",
            "wall decoration",
            ("wall", "walls"),
            ("paint", "decorate", "skim", "mist", "coat"),
        ),
        SectionDetailBucket(
            "ceiling_decoration",
            "ceiling decoration",
            ("ceiling", "ceilings"),
            ("paint", "decorate", "mist", "coat"),
        ),
        SectionDetailBucket(
            "mist_coats",
            "mist / finish coats",
            ("mist", "coat", "coats"),
            ("mist", "coat", "decorate", "paint"),
        ),
    ),
    "flooring": (
        SectionDetailBucket(
            "hard_floor_finishes",
            "hard-floor finishes",
            ("engineered", "laminate", "timber", "vinyl", "lvt"),
            ("engineered", "laminate", "timber", "vinyl", "wood", "floor"),
        ),
        SectionDetailBucket(
            "carpet_underlay",
            "carpet / underlay",
            ("carpet", "underlay"),
            ("carpet", "underlay"),
        ),
    ),
    "electrical": (
        SectionDetailBucket(
            "consumer_unit",
            "consumer unit / distribution board",
            ("consumer unit", "distribution board"),
            ("consumer", "distribution", "board", "unit"),
        ),
        SectionDetailBucket(
            "rewire_circuits",
            "rewire / circuit work",
            ("rewire", "circuit", "first fix", "second fix"),
            ("rewire", "circuit", "first", "second", "fix"),
        ),
        SectionDetailBucket(
            "lighting_points",
            "lighting / downlights",
            ("downlight", "spotlight", "light", "lighting"),
            ("downlight", "spotlight", "light", "lighting"),
        ),
        SectionDetailBucket(
            "extract_ventilation",
            "extract ventilation",
            ("extractor", "fan", "duct"),
            ("extractor", "fan", "duct"),
        ),
        SectionDetailBucket(
            "appliance_connections",
            "appliance connections / hardwiring",
            ("appliance", "hardwire", "connection"),
            ("appliance", "hardwire", "connect"),
        ),
    ),
    "plumbing": (
        SectionDetailBucket(
            "sanitary_sets",
            "sanitary fixtures",
            ("wc", "toilet", "basin", "bath", "shower", "sanitary"),
            ("wc", "toilet", "basin", "bath", "shower", "sanitary", "sink"),
        ),
        SectionDetailBucket(
            "radiators",
            "radiators / emitters",
            ("radiator", "towel rail"),
            ("radiator", "towel", "rail"),
        ),
        SectionDetailBucket(
            "ufh",
            "underfloor heating",
            ("ufh", "underfloor heating"),
            ("ufh", "underfloor", "heating"),
        ),
        SectionDetailBucket(
            "hot_water",
            "boiler / cylinder plant",
            ("boiler", "cylinder", "megaflo"),
            ("boiler", "cylinder", "megaflo"),
        ),
    ),
    "heating": (
        SectionDetailBucket(
            "boiler_cylinder",
            "boiler / cylinder plant",
            ("boiler", "cylinder", "megaflo"),
            ("boiler", "cylinder", "megaflo"),
        ),
        SectionDetailBucket(
            "radiators",
            "radiators / emitters",
            ("radiator", "towel rail"),
            ("radiator", "towel", "rail"),
        ),
        SectionDetailBucket(
            "gas_line",
            "gas line / pipework",
            ("gas line", "gas"),
            ("gas", "pipe", "line"),
        ),
    ),
    "structure": (
        SectionDetailBucket(
            "extension_slab",
            "extension slab / floor build-up",
            ("slab", "screed", "concrete"),
            ("slab", "screed", "concrete", "floor"),
        ),
        SectionDetailBucket(
            "extension_walls",
            "extension walls / cavity shell",
            ("brick", "block", "cavity"),
            ("brick", "block", "cavity", "wall"),
        ),
        SectionDetailBucket(
            "flat_roof",
            "flat roof / warm roof",
            ("flat roof", "warm roof", "grp"),
            ("flat", "roof", "warm", "grp"),
        ),
        SectionDetailBucket(
            "dormer_build",
            "dormer build",
            ("dormer",),
            ("dormer",),
        ),
        SectionDetailBucket(
            "roof_removal_lining",
            "roof removal / loft roof lining",
            ("roof removal", "rafter", "roof", "skylight"),
            ("roof", "loft", "rafter", "skylight", "velux"),
        ),
        SectionDetailBucket(
            "rooflights",
            "rooflights / skylights",
            ("velux", "skylight", "rooflight"),
            ("velux", "skylight", "rooflight"),
        ),
        SectionDetailBucket(
            "dormer_cheeks",
            "dormer cheeks / cladding",
            ("dormer cheek", "dormer cheeks", "cheek", "cladding"),
            ("dormer", "cheek", "cladding"),
        ),
        SectionDetailBucket(
            "roof_edges",
            "roof edges / parapets / upstands",
            ("parapet", "upstand", "edge trim", "drip trim", "fascia", "soffit", "verge", "coping"),
            ("parapet", "upstand", "trim", "fascia", "soffit", "verge", "coping"),
        ),
    ),
    "ceilings_and_insulation": (
        SectionDetailBucket(
            "roof_build_up_insulation",
            "roof insulation / rafter build-up",
            ("between rafters", "under rafters", "rafter", "roof insulation", "insulated board"),
            ("rafter", "insulation", "pir", "board"),
        ),
    ),
    "windows": (
        SectionDetailBucket(
            "rooflights",
            "rooflights / skylights",
            ("velux", "skylight", "rooflight"),
            ("velux", "skylight", "rooflight"),
        ),
        SectionDetailBucket(
            "windows",
            "window sets",
            ("window", "upvc"),
            ("window", "upvc"),
        ),
    ),
    "woodwork_painting": (
        SectionDetailBucket(
            "painted_doors",
            "painted door sets",
            ("door",),
            ("door",),
        ),
        SectionDetailBucket(
            "skirting",
            "skirting",
            ("skirting",),
            ("skirting",),
        ),
        SectionDetailBucket(
            "architraves",
            "architraves",
            ("architrave",),
            ("architrave",),
        ),
        SectionDetailBucket(
            "staircase",
            "staircase woodwork",
            ("staircase", "stairs"),
            ("stair", "staircase"),
        ),
        SectionDetailBucket(
            "window_boards",
            "window boards",
            ("window boards", "window board"),
            ("window", "board"),
        ),
    ),
}


def _detect_scope_profiles(prompt: str) -> list[ScopeProfile]:
    lowered = prompt.lower()
    detected: list[ScopeProfile] = []
    for profile in SCOPE_PROFILES:
        if any(keyword in lowered for keyword in profile.trigger_keywords):
            detected.append(profile)
            continue
        if profile.key in {"kitchen_refurbishment", "bathroom_renovation", "loft_conversion"} and any(
            keyword in lowered for keyword in profile.room_keywords
        ):
            detected.append(profile)
    return detected


def _detect_scope_profiles_from_extraction(extracted_scope: ScopeExtractionResponse) -> list[ScopeProfile]:
    lowered_values = {
        value.lower()
        for value in (
            list(extracted_scope.property_context.project_scopes)
            + [extracted_scope.property_context.property_type, extracted_scope.property_context.location]
        )
        if value
    }
    detected: list[ScopeProfile] = []
    for profile in SCOPE_PROFILES:
        if any(keyword in value for value in lowered_values for keyword in profile.trigger_keywords):
            detected.append(profile)
            continue
        if profile.key in {"kitchen_refurbishment", "bathroom_renovation", "loft_conversion"} and any(
            keyword in value for value in lowered_values for keyword in profile.room_keywords
        ):
            detected.append(profile)
    return detected


def _section_completeness_assumptions(
    extracted_scope: ScopeExtractionResponse,
    matched_tokens: set[str],
) -> list[PreviewAssumption]:
    assumptions: list[PreviewAssumption] = []
    seen_sections: set[str] = set()
    for section in extracted_scope.sections:
        if not section.lines or section.key in seen_sections:
            continue
        buckets = SECTION_EXPECTED_BUCKETS.get(section.key, ())
        if not buckets:
            continue
        seen_sections.add(section.key)
        missing_labels = [
            bucket.label
            for bucket in buckets
            if not any(keyword in matched_tokens for keyword in bucket.keywords)
        ]
        if missing_labels:
            assumptions.append(
                PreviewAssumption(
                    text=(
                        f"The extracted {section.title.lower()} section contains scope lines, but no matched rows were found yet for "
                        f"{', '.join(missing_labels)}. Confirm whether those work items are still missing from the draft."
                    ),
                    kind="section_coverage_gap",
                    severity="warning",
                )
            )
    return assumptions


def _section_detail_coverage_assumptions(
    extracted_scope: ScopeExtractionResponse,
    matched_tokens: set[str],
) -> list[PreviewAssumption]:
    assumptions: list[PreviewAssumption] = []
    for section in extracted_scope.sections:
        detail_buckets = SECTION_DETAIL_BUCKETS.get(section.key, ())
        if not detail_buckets or not section.lines:
            continue

        section_text = " ".join(section.lines).lower()
        missing_labels: list[str] = []
        for bucket in detail_buckets:
            if not any(keyword in section_text for keyword in bucket.trigger_keywords):
                continue
            if any(keyword in matched_tokens for keyword in bucket.expected_keywords):
                continue
            missing_labels.append(bucket.label)

        if missing_labels:
            assumptions.append(
                PreviewAssumption(
                    text=(
                        f"The extracted {section.title.lower()} section includes explicit scope for "
                        f"{', '.join(missing_labels)}, but the draft has no obvious matched rows for those detail items yet."
                    ),
                    kind="section_detail_coverage_gap",
                    severity="warning",
                )
            )

    return assumptions


def _room_finish_coverage_assumptions(
    extracted_scope: ScopeExtractionResponse,
    matched_rows: list[PreviewMatchedRow],
) -> list[PreviewAssumption]:
    assumptions: list[PreviewAssumption] = []
    finish_section_keys = {"decorating", "flooring", "woodwork_painting"}
    rooms = sorted(extracted_scope.room_takeoff, key=lambda room: len(room.name), reverse=True)

    for section in extracted_scope.sections:
        if section.key not in finish_section_keys or not section.lines:
            continue

        explicitly_named_rooms: list[str] = []
        for line in section.lines:
            normalized = line.lower()
            if not normalized or normalized.endswith(":"):
                continue
            if any(keyword in normalized for keyword in ("by others", "excluded", "exclude", "by owner", "not in scope")):
                continue
            for room in rooms:
                room_name = room.name.lower()
                if room_name in normalized and room.name not in explicitly_named_rooms:
                    explicitly_named_rooms.append(room.name)

        if not explicitly_named_rooms:
            continue

        section_rows = [row for row in matched_rows if row.MatchedSectionKey == section.key]
        if not section_rows:
            continue

        covered_rooms: set[str] = set()
        for row in section_rows:
            row_text = " ".join(filter(None, [row.WorkName, row.AREA or "", row.ReviewReason or ""])).lower()
            for room_name in explicitly_named_rooms:
                if room_name.lower() in row_text:
                    covered_rooms.add(room_name)

        missing_rooms = [room_name for room_name in explicitly_named_rooms if room_name not in covered_rooms]
        if missing_rooms:
            assumptions.append(
                PreviewAssumption(
                    text=(
                        f"The extracted {section.title.lower()} section explicitly mentions "
                        f"{', '.join(missing_rooms)}, but the current draft has no obvious {section.title.lower()} rows tied to those rooms yet."
                    ),
                    kind="room_finish_coverage_gap",
                    severity="warning",
                )
            )

    return assumptions


def _level_joinery_coverage_assumptions(
    extracted_scope: ScopeExtractionResponse,
    matched_rows: list[PreviewMatchedRow],
) -> list[PreviewAssumption]:
    assumptions: list[PreviewAssumption] = []
    supported_sections = {"doors", "joinery", "woodwork_painting"}
    supported_levels = {"ground floor", "first floor", "loft"}

    for section in extracted_scope.sections:
        if section.key not in supported_sections or not section.lines:
            continue

        section_rows = [row for row in matched_rows if row.MatchedSectionKey == section.key]
        if not section_rows:
            continue

        explicit_levels: list[str] = []
        active_level = ""
        for line in section.lines:
            normalized = line.lower().strip()
            if not normalized:
                continue

            heading = normalized.rstrip(":")
            if heading in supported_levels:
                active_level = heading
                if heading not in explicit_levels:
                    explicit_levels.append(heading)
                continue

            if normalized.endswith(":"):
                continue

            if any(keyword in normalized for keyword in ("by others", "excluded", "exclude", "by owner", "not in scope")):
                continue

            line_levels = [level for level in supported_levels if level in normalized]
            for level in line_levels:
                if level not in explicit_levels:
                    explicit_levels.append(level)

            if active_level and active_level not in explicit_levels:
                explicit_levels.append(active_level)

        if not explicit_levels:
            continue

        covered_levels: set[str] = set()
        for row in section_rows:
            row_text = " ".join(filter(None, [row.WorkName, row.AREA or "", row.ReviewReason or ""])).lower()
            for level in explicit_levels:
                if level in row_text:
                    covered_levels.add(level)

        missing_levels = [level.title() for level in explicit_levels if level not in covered_levels]
        if missing_levels:
            assumptions.append(
                PreviewAssumption(
                    text=(
                        f"The extracted {section.title.lower()} section explicitly breaks scope down by "
                        f"{', '.join(missing_levels)}, but the current draft has no obvious {section.title.lower()} rows tied to those levels yet."
                    ),
                    kind="level_joinery_coverage_gap",
                    severity="warning",
                )
            )

    return assumptions


def build_scope_coverage_assumptions(
    prompt: str,
    matched_rows: list[PreviewMatchedRow],
    extracted_scope: ScopeExtractionResponse | None = None,
) -> list[PreviewAssumption]:
    matched_tokens = _matched_scope_tokens(matched_rows)
    assumptions: list[PreviewAssumption] = []
    if extracted_scope:
        assumptions.extend(_section_completeness_assumptions(extracted_scope, matched_tokens))
        assumptions.extend(_section_detail_coverage_assumptions(extracted_scope, matched_tokens))
        assumptions.extend(_room_finish_coverage_assumptions(extracted_scope, matched_rows))
        assumptions.extend(_level_joinery_coverage_assumptions(extracted_scope, matched_rows))

    profiles = _detect_scope_profiles_from_extraction(extracted_scope) if extracted_scope else []
    if not profiles:
        profiles = _detect_scope_profiles(prompt)
    if not profiles:
        return assumptions

    for profile in profiles:
        missing_labels: list[str] = []
        for bucket in profile.expected_buckets:
            if not any(keyword in matched_tokens for keyword in bucket.keywords):
                missing_labels.append(bucket.label)

        if missing_labels:
            assumptions.append(
                PreviewAssumption(
                    text=(
                        f"This looks like a {profile.label} scope. No matched rows were found yet for "
                        f"{', '.join(missing_labels)}. Confirm whether those items are intentionally excluded or still missing from the draft."
                    ),
                    kind="coverage_gap",
                    severity="warning",
                )
            )

    return assumptions
