from __future__ import annotations

import pytest

from app.schemas import (
    CandidateRow,
    ExtractedPropertyContext,
    ExtractedRoomTakeoff,
    ExtractedSection,
    ScopeExtractionResponse,
    StructuredTakeoffSummary,
)
from app.services.scope_matching import (
    _has_extracted_scope_line,
    _match_room_takeoffs_for_segment,
    _room_area_label,
    build_scope_matching_context,
    is_extracted_context_segment,
    match_row_to_extracted_section,
    suggest_quantity_from_takeoff,
)


def _row(unit: str = "pcs", work_name: str = "Test work item", **overrides) -> CandidateRow:
    return CandidateRow(
        INSIDEQUOTESGUID="row-1",
        WorkName=work_name,
        Unit=unit,
        **overrides,
    )


def _scope(
    *,
    rooms=None,
    room_takeoff=None,
    sections=None,
    property_context: ExtractedPropertyContext | None = None,
    **summary_overrides,
) -> ScopeExtractionResponse:
    return ScopeExtractionResponse(
        property_context=property_context or ExtractedPropertyContext(),
        rooms=rooms or [],
        room_takeoff=room_takeoff or [],
        sections=sections or [],
        takeoff_summary=StructuredTakeoffSummary(**summary_overrides),
        assumptions=[],
    )


def _room_takeoff(name: str = "Bathroom", room_type: str = "bathroom", **overrides) -> ExtractedRoomTakeoff:
    defaults = dict(
        level="ground floor",
        name=name,
        room_type=room_type,
        area_m2=5.0,
        estimated_wall_finish_area_m2=10.0,
        estimated_ceiling_finish_area_m2=5.0,
        estimated_skirting_lm=6.0,
        estimated_wall_tiling_area_m2=8.0,
        estimated_floor_finish_area_m2=5.0,
    )
    defaults.update(overrides)
    return ExtractedRoomTakeoff(**defaults)


class TestBuildScopeMatchingContext:
    def test_includes_location_and_property_type_and_scopes(self):
        scope = _scope(
            property_context=ExtractedPropertyContext(
                property_type="Victorian house",
                location="North London",
                project_scopes=["full refurbishment"],
            ),
        )
        result = build_scope_matching_context(scope)

        assert "Victorian house" in result
        assert "North London" in result
        assert "full refurbishment" in result

    def test_includes_section_titles_and_items(self):
        scope = _scope(
            sections=[
                ExtractedSection(key="electrical", title="Electrical", lines=["12 downlights"]),
            ],
        )
        result = build_scope_matching_context(scope)

        assert "Electrical" in result


class TestIsExtractedContextSegment:
    def test_blank_segment_is_context(self):
        assert is_extracted_context_segment("   ", _scope()) is True

    def test_known_label_is_context(self):
        assert is_extracted_context_segment("Ground floor", _scope()) is True

    def test_section_title_is_context(self):
        scope = _scope(sections=[ExtractedSection(key="electrical", title="Electrical")])
        assert is_extracted_context_segment("electrical", scope) is True

    def test_project_scope_label_is_context(self):
        scope = _scope(
            property_context=ExtractedPropertyContext(project_scopes=["full refurbishment"])
        )
        assert is_extracted_context_segment("full refurbishment", scope) is True

    def test_room_dimension_suffix_is_context(self):
        from app.schemas import ExtractedRoom

        scope = _scope(rooms=[ExtractedRoom(level="ground floor", name="Kitchen", width_m=3, length_m=4, area_m2=12)])
        assert is_extracted_context_segment("kitchen: 3.5 x 5.8", scope) is True

    def test_room_name_with_dimensions_elsewhere_is_context(self):
        from app.schemas import ExtractedRoom

        scope = _scope(rooms=[ExtractedRoom(level="ground floor", name="Kitchen", width_m=3, length_m=4, area_m2=12)])
        assert is_extracted_context_segment("kitchen area 3.5 x 5.8", scope) is True

    def test_unrelated_segment_is_not_context(self):
        assert is_extracted_context_segment("Paint walls 20m2", _scope()) is False


class TestMatchRoomTakeoffsForSegment:
    def test_blank_segment_returns_empty(self):
        assert _match_room_takeoffs_for_segment("", _scope()) == []

    def test_room_without_shared_tokens_is_skipped(self):
        scope = _scope(room_takeoff=[_room_takeoff(name="")])
        assert _match_room_takeoffs_for_segment("bathroom tiling", scope) == []

    def test_matches_room_by_name_token(self):
        room = _room_takeoff(name="Bathroom")
        scope = _scope(room_takeoff=[room])
        result = _match_room_takeoffs_for_segment("Bathroom wall tiling", scope)
        assert result == [room]

    def test_no_shared_tokens_is_skipped(self):
        room = _room_takeoff(name="Bathroom")
        scope = _scope(room_takeoff=[room])
        assert _match_room_takeoffs_for_segment("kitchen sink", scope) == []


class TestRoomAreaLabel:
    def test_empty_rooms_returns_empty_string(self):
        assert _room_area_label([]) == ""

    def test_single_room_name(self):
        room = _room_takeoff(name="Bathroom")
        assert _room_area_label([room]) == "Bathroom"

    def test_multiple_rooms_same_level_returns_level(self):
        rooms = [
            _room_takeoff(name="Bathroom", level="ground floor"),
            _room_takeoff(name="Ensuite", level="ground floor"),
        ]
        assert _room_area_label(rooms) == "Ground Floor"

    def test_multiple_rooms_different_levels_returns_multiple(self):
        rooms = [
            _room_takeoff(name="Bathroom", level="ground floor"),
            _room_takeoff(name="Ensuite", level="first floor"),
        ]
        assert _room_area_label(rooms) == "Multiple rooms"


class TestHasExtractedScopeLine:
    def test_returns_false_when_no_section_matches(self):
        scope = _scope(sections=[ExtractedSection(key="electrical", title="Electrical", lines=["12 downlights"])])
        assert (
            _has_extracted_scope_line(scope, section_keys=("joinery",), keywords=("architrave",))
            is False
        )

    def test_returns_true_when_keywords_all_present(self):
        scope = _scope(
            sections=[
                ExtractedSection(key="joinery", title="Joinery", lines=["fit architrave to all doors"]),
            ]
        )
        assert (
            _has_extracted_scope_line(scope, section_keys=("joinery",), keywords=("architrave",))
            is True
        )


class TestMatchRowToExtractedSection:
    def test_returns_none_when_section_has_no_tokens(self):
        row = _row(work_name="Paint walls")
        scope = _scope(sections=[ExtractedSection(key="electrical", title="")])
        assert match_row_to_extracted_section(row, scope, "paint walls") is None

    def test_skips_non_action_sections(self):
        row = _row(work_name="Paint walls")
        scope = _scope(
            sections=[ExtractedSection(key="property", title="Property", lines=["paint walls"])]
        )
        assert match_row_to_extracted_section(row, scope, "paint walls") is None

    def test_matches_best_scoring_section(self):
        row = _row(work_name="Fit downlights", WorkGroupName="Electrical")
        scope = _scope(
            sections=[
                ExtractedSection(key="electrical", title="Electrical", lines=["12 downlights and sockets"]),
                ExtractedSection(key="plastering", title="Plastering", lines=["skim walls"]),
            ]
        )
        match = match_row_to_extracted_section(row, scope, "fit downlights")
        assert match is not None
        assert match.key == "electrical"


class TestSanitaryAndApplianceSupplyReview:
    def test_sanitaryware_defaults_to_one_when_no_takeoff(self):
        row = _row(unit="pcs", work_name="Supply sanitaryware WC")
        scope = _scope(client_supply_sanitaryware=True)
        result = suggest_quantity_from_takeoff(row, scope, "supply wc")

        assert result.quantity == 1.0
        assert result.needs_review is True

    def test_appliances_supply_review(self):
        row = _row(unit="pcs", work_name="Supply oven appliance")
        scope = _scope(client_supply_appliances=True)
        result = suggest_quantity_from_takeoff(row, scope, "supply oven appliance")

        assert result.quantity == 1.0
        assert result.needs_review is True


class TestM2WholePropertyBranches:
    def test_engineered_floor_scope(self):
        row = _row(unit="m2", work_name="Fit engineered floor")
        scope = _scope(estimated_hard_floor_scope_area_m2=42.0)
        result = suggest_quantity_from_takeoff(row, scope, "engineered floor")
        assert result.quantity == 42.0

    def test_carpet_scope(self):
        row = _row(unit="m2", work_name="Fit carpet and underlay")
        scope = _scope(estimated_carpet_scope_area_m2=18.0)
        result = suggest_quantity_from_takeoff(row, scope, "carpet underlay")
        assert result.quantity == 18.0

    def test_extension_slab(self):
        row = _row(unit="m2", work_name="Pour concrete slab")
        scope = _scope(estimated_extension_slab_area_m2=25.0)
        result = suggest_quantity_from_takeoff(row, scope, "concrete slab")
        assert result.quantity == 25.0

    def test_extension_flat_roof(self):
        row = _row(unit="m2", work_name="Flat roof warm build-up")
        scope = _scope(estimated_extension_flat_roof_area_m2=15.0)
        result = suggest_quantity_from_takeoff(row, scope, "flat roof")
        assert result.quantity == 15.0

    def test_extension_external_wall(self):
        row = _row(unit="m2", work_name="Build external cavity wall")
        scope = _scope(estimated_extension_external_wall_area_m2=30.0)
        result = suggest_quantity_from_takeoff(row, scope, "extension external cavity wall")
        assert result.quantity == 30.0

    def test_roof_removal(self):
        row = _row(unit="m2", work_name="Strip existing roof")
        scope = _scope(estimated_roof_removal_area_m2=20.0)
        result = suggest_quantity_from_takeoff(row, scope, "strip roof")
        assert result.quantity == 20.0

    def test_dormer_build(self):
        row = _row(unit="m2", work_name="Build dormer")
        scope = _scope(estimated_dormer_build_area_m2=12.0)
        result = suggest_quantity_from_takeoff(row, scope, "dormer")
        assert result.quantity == 12.0

    def test_loft_roof_finish(self):
        row = _row(unit="m2", work_name="Board and insulate loft roof")
        scope = _scope(estimated_loft_roof_finish_area_m2=22.0)
        result = suggest_quantity_from_takeoff(row, scope, "loft roof board insulation")
        assert result.quantity == 22.0

    def test_ceiling_paint(self):
        row = _row(unit="m2", work_name="Paint ceiling")
        scope = _scope(estimated_ceiling_finish_area_m2=33.0)
        result = suggest_quantity_from_takeoff(row, scope, "paint ceiling")
        assert result.quantity == 33.0

    def test_wall_paint(self):
        row = _row(unit="m2", work_name="Paint walls")
        scope = _scope(estimated_wall_finish_area_m2=44.0)
        result = suggest_quantity_from_takeoff(row, scope, "paint walls")
        assert result.quantity == 44.0

    def test_ceiling_board(self):
        row = _row(unit="m2", work_name="Plasterboard ceiling")
        scope = _scope(estimated_ceiling_finish_area_m2=17.0)
        result = suggest_quantity_from_takeoff(row, scope, "plasterboard ceiling")
        assert result.quantity == 17.0

    def test_boarding_coverage(self):
        row = _row(unit="m2", work_name="Board roof slopes")
        scope = _scope(estimated_boarding_coverage_m2=19.0)
        result = suggest_quantity_from_takeoff(row, scope, "board roof slopes")
        assert result.quantity == 19.0

    def test_roof_insulation(self):
        row = _row(unit="m2", work_name="Insulate roof rafters")
        scope = _scope(roof_area_m2=28.0)
        result = suggest_quantity_from_takeoff(row, scope, "insulation roof rafters")
        assert result.quantity == 28.0

    def test_insulation_coverage(self):
        row = _row(unit="m2", work_name="Fit acoustic insulation")
        scope = _scope(estimated_insulation_coverage_m2=9.0)
        result = suggest_quantity_from_takeoff(row, scope, "acoustic insulation")
        assert result.quantity == 9.0

    def test_roof_area(self):
        row = _row(unit="m2", work_name="Re-roof")
        scope = _scope(roof_area_m2=50.0)
        result = suggest_quantity_from_takeoff(row, scope, "roof")
        assert result.quantity == 50.0
        assert result.needs_review is False

    def test_patio_area(self):
        row = _row(unit="m2", work_name="Lay patio")
        scope = _scope(patio_area_m2=14.0)
        result = suggest_quantity_from_takeoff(row, scope, "patio")
        assert result.quantity == 14.0

    def test_total_internal_floor_area(self):
        row = _row(unit="m2", work_name="Fit laminate floor")
        scope = _scope(total_internal_floor_area_m2=80.0)
        result = suggest_quantity_from_takeoff(row, scope, "laminate floor")
        assert result.quantity == 80.0


class TestM2MatchedRoomBranches:
    def test_wall_tiling(self):
        room = _room_takeoff(estimated_wall_tiling_area_m2=6.0)
        row = _row(unit="m2", work_name="Tile walls")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom tile wall")
        assert result.quantity == 6.0

    def test_floor_tiling(self):
        room = _room_takeoff(estimated_floor_finish_area_m2=7.0)
        row = _row(unit="m2", work_name="Tile floor")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom tile floor")
        assert result.quantity == 7.0

    def test_ceiling_paint_matched_room(self):
        room = _room_takeoff(estimated_ceiling_finish_area_m2=4.0)
        row = _row(unit="m2", work_name="Paint ceiling")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom paint ceiling")
        assert result.quantity == 4.0

    def test_wall_paint_matched_room(self):
        room = _room_takeoff(estimated_wall_finish_area_m2=11.0)
        row = _row(unit="m2", work_name="Paint walls")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom paint wall")
        assert result.quantity == 11.0

    def test_wall_paint_matched_room_assumption_summarizes_more_than_four_rooms(self):
        rooms = [_room_takeoff(name="Bedroom", estimated_wall_finish_area_m2=5.0) for _ in range(5)]
        row = _row(unit="m2", work_name="Paint walls")
        scope = _scope(room_takeoff=rooms)
        result = suggest_quantity_from_takeoff(row, scope, "bedroom paint wall")
        assert result.quantity == 25.0
        assert "+1 more" in result.assumption.text

    def test_ceiling_board_matched_room(self):
        room = _room_takeoff(estimated_ceiling_finish_area_m2=3.0)
        row = _row(unit="m2", work_name="Board ceiling")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom board ceiling")
        assert result.quantity == 3.0

    def test_floor_finish_matched_room(self):
        room = _room_takeoff(area_m2=9.0)
        row = _row(unit="m2", work_name="Fit laminate floor")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom laminate floor")
        assert result.quantity == 9.0


class TestPcsWholePropertyNonRoomBranches:
    def test_electrical_second_fix(self):
        row = _row(unit="pcs", work_name="Electrical second fix")
        scope = _scope(estimated_electrical_second_fix_points=15.0)
        result = suggest_quantity_from_takeoff(row, scope, "electric second fix")
        assert result.quantity == 15.0

    def test_plumbing_second_fix(self):
        row = _row(unit="pcs", work_name="Plumbing second fix")
        scope = _scope(estimated_plumbing_second_fix_points=8.0)
        result = suggest_quantity_from_takeoff(row, scope, "plumbing second fix")
        assert result.quantity == 8.0


class TestPcsMatchedRoomBranches:
    def test_downlight_from_downlight_count(self):
        room = _room_takeoff(estimated_downlight_count=6.0)
        row = _row(unit="pcs", work_name="Fit downlights")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom downlight")
        assert result.quantity == 6.0

    def test_downlight_falls_back_to_lighting_point_count(self):
        room = _room_takeoff(estimated_downlight_count=0.0, estimated_lighting_point_count=3.0)
        row = _row(unit="pcs", work_name="Fit spotlights")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom spotlight")
        assert result.quantity == 3.0

    def test_socket_count(self):
        room = _room_takeoff(estimated_socket_count=4.0)
        row = _row(unit="pcs", work_name="Fit sockets")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom socket")
        assert result.quantity == 4.0

    def test_switch_count(self):
        room = _room_takeoff(estimated_switch_count=2.0)
        row = _row(unit="pcs", work_name="Fit switches")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom switch")
        assert result.quantity == 2.0

    def test_lighting_point_count(self):
        room = _room_takeoff(estimated_lighting_point_count=5.0)
        row = _row(unit="pcs", work_name="Fit light fittings")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom light")
        assert result.quantity == 5.0

    def test_ducted_extractor_count(self):
        room = _room_takeoff(estimated_ducted_extractor_count=2.0)
        row = _row(unit="pcs", work_name="Fit ducted extractor fan")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom extractor fan duct")
        assert result.quantity == 2.0

    def test_ducted_extractor_falls_back_to_extractor_fan_count(self):
        room = _room_takeoff(estimated_ducted_extractor_count=0.0, estimated_extractor_fan_count=1.0)
        row = _row(unit="pcs", work_name="Fit ducted fan")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom fan duct")
        assert result.quantity == 1.0

    def test_extractor_fan_count(self):
        room = _room_takeoff(estimated_extractor_fan_count=1.0)
        row = _row(unit="pcs", work_name="Fit extractor")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom extractor")
        assert result.quantity == 1.0

    def test_hardwired_appliance_count(self):
        room = _room_takeoff(
            name="Kitchen", room_type="kitchen", estimated_hardwired_appliance_count=2.0
        )
        row = _row(unit="pcs", work_name="Hardwire appliance")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "kitchen appliance hardwire")
        assert result.quantity == 2.0

    def test_hardwired_appliance_falls_back_to_connection_count(self):
        room = _room_takeoff(
            name="Kitchen",
            room_type="kitchen",
            estimated_hardwired_appliance_count=0.0,
            estimated_appliance_connection_count=3.0,
        )
        row = _row(unit="pcs", work_name="Wire appliance")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "kitchen appliance wire")
        assert result.quantity == 3.0

    def test_electrical_second_fix_matched_rooms(self):
        room = _room_takeoff(estimated_electrical_second_fix_points=9.0)
        row = _row(unit="pcs", work_name="Electric fit off")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom electric fit off")
        assert result.quantity == 9.0

    def test_concealed_cistern_matched_rooms(self):
        room = _room_takeoff(estimated_concealed_cistern_count=1.0)
        row = _row(unit="pcs", work_name="Concealed cistern")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom concealed cistern")
        assert result.quantity == 1.0

    def test_wc_count_matched_rooms(self):
        room = _room_takeoff(estimated_wc_count=1.0)
        row = _row(unit="pcs", work_name="Fit WC")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom wc")
        assert result.quantity == 1.0

    def test_basin_matched_rooms(self):
        room = _room_takeoff(estimated_basin_count=1.0)
        row = _row(unit="pcs", work_name="Fit basin")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom basin")
        assert result.quantity == 1.0

    def test_vanity_matched_rooms(self):
        room = _room_takeoff(estimated_vanity_unit_count=1.0)
        row = _row(unit="pcs", work_name="Fit vanity unit")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom vanity")
        assert result.quantity == 1.0

    def test_sink_matched_rooms(self):
        room = _room_takeoff(name="Kitchen", room_type="kitchen", estimated_sink_count=1.0)
        row = _row(unit="pcs", work_name="Fit sink")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "kitchen sink")
        assert result.quantity == 1.0

    def test_appliance_connection_matched_rooms(self):
        room = _room_takeoff(
            name="Kitchen", room_type="kitchen", estimated_appliance_connection_count=2.0
        )
        row = _row(unit="pcs", work_name="Connect appliance")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "kitchen appliance connection")
        assert result.quantity == 2.0

    def test_bath_matched_rooms(self):
        room = _room_takeoff(estimated_bath_count=1.0)
        row = _row(unit="pcs", work_name="Fit bath")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom bath")
        assert result.quantity == 1.0

    def test_shower_matched_rooms(self):
        room = _room_takeoff(estimated_shower_count=1.0)
        row = _row(unit="pcs", work_name="Fit shower")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom shower")
        assert result.quantity == 1.0

    def test_sanitary_set_matched_wet_rooms(self):
        room = _room_takeoff(estimated_sanitary_set_count=1.0)
        row = _row(unit="pcs", work_name="Fit sanitary suite")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom sanitary suite")
        assert result.quantity == 1.0

    def test_plumbing_matched_rooms(self):
        room = _room_takeoff(estimated_plumbing_second_fix_points=4.0)
        row = _row(unit="pcs", work_name="Plumbing first fix")
        scope = _scope(room_takeoff=[room])
        result = suggest_quantity_from_takeoff(row, scope, "bathroom plumb first fix")
        assert result.quantity == 4.0


class TestPcsWholePropertyFixtureBranches:
    def test_staircase_count(self):
        row = _row(unit="pcs", work_name="Fit staircase")
        scope = _scope(staircase_count=1.0)
        result = suggest_quantity_from_takeoff(row, scope, "staircase")
        assert result.quantity == 1.0

    def test_window_board_count(self):
        row = _row(unit="pcs", work_name="Fit window board")
        scope = _scope(window_board_count=5.0)
        result = suggest_quantity_from_takeoff(row, scope, "window board")
        assert result.quantity == 5.0

    def test_architrave_set_count(self):
        row = _row(unit="pcs", work_name="Fit architrave")
        scope = _scope(
            architrave_set_count=6.0,
            sections=[ExtractedSection(key="joinery", title="Joinery", lines=["fit architrave to all doors"])],
        )
        result = suggest_quantity_from_takeoff(row, scope, "architrave")
        assert result.quantity == 6.0

    def test_pocket_door_plain_set_count(self):
        row = _row(unit="pcs", work_name="Fit pocket door")
        scope = _scope(pocket_door_set_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "pocket door")
        assert result.quantity == 2.0

    def test_pocket_door_double_set_count(self):
        row = _row(unit="pcs", work_name="Fit double pocket door")
        scope = _scope(double_pocket_door_set_count=1.0, pocket_door_set_count=3.0)
        result = suggest_quantity_from_takeoff(row, scope, "double pocket door")
        assert result.quantity == 1.0

    def test_pocket_door_fire_rated_set_count(self):
        row = _row(unit="pcs", work_name="Fit fire rated pocket door")
        scope = _scope(fire_rated_pocket_door_set_count=1.0, double_pocket_door_set_count=1.0)
        result = suggest_quantity_from_takeoff(row, scope, "fire pocket door")
        assert result.quantity == 1.0

    def test_fire_rated_door_set_count(self):
        row = _row(unit="pcs", work_name="Fit fire door")
        scope = _scope(fire_rated_door_set_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "fire door")
        assert result.quantity == 2.0

    def test_storage_door_set_count(self):
        row = _row(unit="pcs", work_name="Fit wardrobe doors")
        scope = _scope(storage_door_set_count=4.0)
        result = suggest_quantity_from_takeoff(row, scope, "wardrobe storage door")
        assert result.quantity == 4.0

    def test_consumer_unit_count(self):
        row = _row(unit="pcs", work_name="Replace consumer unit")
        scope = _scope(consumer_unit_count=1.0)
        result = suggest_quantity_from_takeoff(row, scope, "consumer unit")
        assert result.quantity == 1.0

    def test_downlight_count(self):
        row = _row(unit="pcs", work_name="Fit downlights")
        scope = _scope(downlight_count=10.0)
        result = suggest_quantity_from_takeoff(row, scope, "downlight")
        assert result.quantity == 10.0

    def test_ducted_extractor_count(self):
        row = _row(unit="pcs", work_name="Fit ducted extractor")
        scope = _scope(ducted_extractor_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "extractor duct")
        assert result.quantity == 2.0

    def test_ducted_extractor_falls_back_to_extractor_fan_count(self):
        row = _row(unit="pcs", work_name="Fit fan duct")
        scope = _scope(ducted_extractor_count=0.0, extractor_fan_count=3)
        result = suggest_quantity_from_takeoff(row, scope, "fan duct")
        assert result.quantity == 3.0

    def test_hardwired_appliance_count(self):
        row = _row(unit="pcs", work_name="Hardwire appliance")
        scope = _scope(hardwired_appliance_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "appliance hardwire")
        assert result.quantity == 2.0

    def test_hardwired_appliance_falls_back_to_connection_count(self):
        row = _row(unit="pcs", work_name="Wire appliance")
        scope = _scope(hardwired_appliance_count=0.0, total_appliance_connection_count=4.0)
        result = suggest_quantity_from_takeoff(row, scope, "appliance wire")
        assert result.quantity == 4.0

    def test_concealed_cistern_count(self):
        row = _row(unit="pcs", work_name="Concealed cistern")
        scope = _scope(total_concealed_cistern_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "concealed cistern")
        assert result.quantity == 2.0

    def test_total_wc_count(self):
        row = _row(unit="pcs", work_name="Fit WC")
        scope = _scope(total_wc_count=3.0)
        result = suggest_quantity_from_takeoff(row, scope, "wc")
        assert result.quantity == 3.0

    def test_total_basin_count(self):
        row = _row(unit="pcs", work_name="Fit basin")
        scope = _scope(total_basin_count=3.0)
        result = suggest_quantity_from_takeoff(row, scope, "basin")
        assert result.quantity == 3.0

    def test_total_vanity_unit_count(self):
        row = _row(unit="pcs", work_name="Fit vanity")
        scope = _scope(total_vanity_unit_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "vanity")
        assert result.quantity == 2.0

    def test_total_sink_count(self):
        row = _row(unit="pcs", work_name="Fit sink")
        scope = _scope(total_sink_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "sink")
        assert result.quantity == 2.0

    def test_total_appliance_connection_count(self):
        row = _row(unit="pcs", work_name="Connect appliances")
        scope = _scope(total_appliance_connection_count=3.0)
        result = suggest_quantity_from_takeoff(row, scope, "appliance connection")
        assert result.quantity == 3.0

    def test_total_bath_count(self):
        row = _row(unit="pcs", work_name="Fit bath")
        scope = _scope(total_bath_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "bath")
        assert result.quantity == 2.0

    def test_total_shower_count(self):
        row = _row(unit="pcs", work_name="Fit shower")
        scope = _scope(total_shower_count=2.0)
        result = suggest_quantity_from_takeoff(row, scope, "shower")
        assert result.quantity == 2.0

    def test_total_sanitary_set_count(self):
        row = _row(unit="pcs", work_name="Fit sanitary suite")
        scope = _scope(total_sanitary_set_count=3.0)
        result = suggest_quantity_from_takeoff(row, scope, "bathroom suite")
        assert result.quantity == 3.0


class TestExactCountCases:
    @pytest.mark.parametrize(
        "segment,summary_field,summary_value",
        [
            ("fit sockets", "socket_count", 8),
            ("fit switches", "switch_count", 4),
            ("fit extractor fan", "extractor_fan_count", 2),
            ("fit velux window", "velux_count", 1),
            ("fit upvc window", "upvc_window_count", 6),
            ("remove radiator", "radiator_removal_count", 3),
            ("install radiator", "radiator_install_count", 5),
            ("fit door set", "door_set_count", 4),
            ("fit steel junction", "steel_junction_count", 2),
            ("fit steel post", "steel_post_count", 3),
            ("remove tree", "tree_count", 1),
            ("fit fence", "fence_count", 1),
        ],
    )
    def test_exact_count_branch_returns_summary_value(self, segment, summary_field, summary_value):
        row = _row(unit="pcs", work_name=segment.title())
        scope = _scope(**{summary_field: summary_value})
        result = suggest_quantity_from_takeoff(row, scope, segment)
        assert result.quantity == float(summary_value)


class TestM3Branches:
    def test_soil_volume(self):
        row = _row(unit="m3", work_name="Remove soil")
        scope = _scope(soil_volume_m3=12.0)
        result = suggest_quantity_from_takeoff(row, scope, "soil")
        assert result.quantity == 12.0

    def test_trench_foundation_volume(self):
        row = _row(unit="m3", work_name="Pour trench foundation")
        scope = _scope(trench_foundation_volume_m3=8.0)
        result = suggest_quantity_from_takeoff(row, scope, "trench foundation")
        assert result.quantity == 8.0

    def test_total_foundation_concrete(self):
        row = _row(unit="m3", work_name="Pour foundation concrete")
        scope = _scope(total_foundation_concrete_m3=20.0)
        result = suggest_quantity_from_takeoff(row, scope, "foundation concrete excavate")
        assert result.quantity == 20.0
        assert result.needs_review is True


class TestMUnitBranches:
    def test_skirting_matched_rooms(self):
        room = _room_takeoff(estimated_skirting_lm=6.0)
        row = _row(unit="m", work_name="Fit skirting")
        scope = _scope(room_takeoff=[room], estimated_skirting_lm=6.0)
        result = suggest_quantity_from_takeoff(row, scope, "bathroom skirting")
        assert result.quantity == 6.0
        assert result.area == "Bathroom"

    def test_skirting_whole_property(self):
        row = _row(unit="m", work_name="Fit skirting")
        scope = _scope(estimated_skirting_lm=45.0)
        result = suggest_quantity_from_takeoff(row, scope, "skirting")
        assert result.quantity == 45.0

    def test_architrave_length(self):
        row = _row(unit="m", work_name="Fit architrave")
        scope = _scope(
            estimated_architrave_lm=30.0,
            sections=[ExtractedSection(key="joinery", title="Joinery", lines=["fit architrave to all doors"])],
        )
        result = suggest_quantity_from_takeoff(row, scope, "architrave")
        assert result.quantity == 30.0

    def test_window_board_length(self):
        row = _row(unit="m", work_name="Fit window board")
        scope = _scope(estimated_window_board_lm=12.0)
        result = suggest_quantity_from_takeoff(row, scope, "window board")
        assert result.quantity == 12.0


class TestNoMatchFallback:
    def test_returns_empty_suggestion_when_nothing_matches(self):
        row = _row(unit="pcs", work_name="Unrelated bespoke work")
        result = suggest_quantity_from_takeoff(row, _scope(), "something with no takeoff hooks")
        assert result.quantity is None
