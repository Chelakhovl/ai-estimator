from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services.scope_extractor import extract_scope


STRUCTURED_PROMPT = """
Property
Type: Semi-detached house
Location: Outskirts of London (simple scenario – driveway available)

FULL SCOPE
Full house refurbishment
Rear extension
Loft conversion

FLOOR AREAS & ROOMS
Ground Floor
Formal Living: 4.5 × 3.5
Utility: 3.5 × 1.5
Kitchen Dining: 3.5 × 5.8
Hallway: 1.7 × 6.5
WC: 1.2 × 1.5
Study: 1.5 × 1.8
Rear Living: 3.5 × 3.2
Dining Room: 3.7 × 7
First Floor
Bedroom 1: 3.5 × 4.2
Bedroom 2: 4.5 × 3.5
Bedroom 3: 3.5 × 3.5
Bedroom 4: 3.5 × 3
Store: 2 × 1.2
Ensuite: 2 × 1.7
Bathroom: 2.5 × 2.5
Landing: 3 × 3
Loft
Bedroom: 5 × 3
Bathroom: 2.3 × 4.5
Landing: 2.5 × 1.3

DEMOLITION
Full strip-out (floors, ceilings, doors)
Remove kitchen + utility appliances
Remove 8 radiators
Existing extension roof (3.5 × 3.2)
Tree + roots
Patio (5 × 3)
Brick wall (4m × 0.5m)
2m³ soil
2 fences

GROUNDWORKS
Pad foundations:
2 × (1 × 1 × 1 m)
Trench foundations:
Total: 11.5 m
Depth: 1 m
Width: 0.6 m
Internal foundation:
3.5 m × 1 m × 0.5 m

STEELWORKS
Ground beam (encased) - 5 m
Total posts: 10
Junctions: 30

DOORS
Ground floor:
2 × double pocket doors (fire-rated)
1 × single pocket door
6 × fire-rated doors
First floor:
6 × fire-rated doors
Loft:
2 × pocket doors (fire-rated)
Allowance:
£200 per door (supply)

PLUMBING
Radiators:
5 (first floor)
2 (loft)

ELECTRICAL
Switches (32 total)
Sockets (48 total)
Extractor fans (6)

WINDOWS
4 × Velux (MK04)
1 × uPVC window (1×1)
1 × uPVC window (2×1)
""".strip()


def test_extract_scope_parses_large_house_prompt() -> None:
    response = extract_scope(STRUCTURED_PROMPT)

    assert response.error_text == ""
    assert response.property_context.property_type == "Semi-detached house"
    assert response.property_context.location.startswith("Outskirts of London")
    assert response.property_context.project_scopes == [
        "Full house refurbishment",
        "Rear extension",
        "Loft conversion",
    ]

    assert len(response.rooms) == 19
    assert response.takeoff_summary.ground_floor_area_m2 == 93.95
    assert response.takeoff_summary.first_floor_area_m2 == 74.25
    assert response.takeoff_summary.loft_area_m2 == 28.6
    assert response.takeoff_summary.total_internal_floor_area_m2 == 196.8
    assert response.takeoff_summary.estimated_wall_finish_area_m2 == 484.81
    assert response.takeoff_summary.estimated_ceiling_finish_area_m2 == 196.8
    assert response.takeoff_summary.estimated_skirting_lm == 196.14

    assert response.takeoff_summary.pad_foundation_volume_m3 == 2.0
    assert response.takeoff_summary.trench_foundation_volume_m3 == 6.9
    assert response.takeoff_summary.internal_foundation_volume_m3 == 1.75
    assert response.takeoff_summary.total_foundation_concrete_m3 == 10.65

    assert response.takeoff_summary.patio_area_m2 == 15.0
    assert response.takeoff_summary.roof_area_m2 == 11.2
    assert response.takeoff_summary.estimated_extension_slab_area_m2 == 11.2
    assert response.takeoff_summary.estimated_extension_flat_roof_area_m2 == 0.0
    assert response.takeoff_summary.estimated_extension_external_wall_area_m2 == 20.2
    assert response.takeoff_summary.steel_post_count == 10
    assert response.takeoff_summary.steel_junction_count == 30
    assert response.takeoff_summary.radiator_removal_count == 8
    assert response.takeoff_summary.radiator_install_count == 7
    assert response.takeoff_summary.socket_count == 48
    assert response.takeoff_summary.switch_count == 32
    assert response.takeoff_summary.extractor_fan_count == 6
    assert response.takeoff_summary.velux_count == 4
    assert response.takeoff_summary.upvc_window_count == 2
    assert response.takeoff_summary.door_set_count == 17
    assert response.takeoff_summary.fire_rated_door_set_count == 16.0
    assert response.takeoff_summary.pocket_door_set_count == 5.0
    assert response.takeoff_summary.fire_rated_pocket_door_set_count == 4.0
    assert response.takeoff_summary.double_pocket_door_set_count == 2.0
    assert response.takeoff_summary.door_supply_allowance_total == 3400.0

    warning_text = " ".join(assumption.text for assumption in response.assumptions)
    assert "Door supply allowance was calculated from counted door sets" in warning_text
    derived_finish_warning = next(
        assumption
        for assumption in response.assumptions
        if assumption.kind == "derived_finish_takeoff"
    )
    assert (
        "Derived wall, ceiling, skirting, boarding, insulation, and loft/roof takeoff uses"
        in derived_finish_warning.text
    )
    assert "AI-side room-height and build-up heuristics" in derived_finish_warning.text
    assert (
        "Treat these values as a strong draft baseline" in derived_finish_warning.text
    )


def test_extract_scope_derives_joinery_detail_counts() -> None:
    prompt = """
Property
Type: Semi-detached house

FULL SCOPE
Loft conversion

JOINERY
Staircase (softwood, painted)
Loft storage:
3 × double door sets
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert response.takeoff_summary.staircase_count == 1.0
    assert response.takeoff_summary.storage_door_set_count == 3.0


def test_extract_scope_recovers_inline_headings_and_room_schedule() -> None:
    prompt = (
        "Property Type: Semi-detached house "
        "Location: Outskirts of London "
        "FULL SCOPE Full house refurbishment Rear extension "
        "FLOOR AREAS & ROOMS Ground Floor Kitchen: 3.5 x 5.8 Utility: 3.5 x 1.5 "
        "DEMOLITION Strip out kitchen"
    )

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert response.property_context.property_type == "Semi-detached house"
    assert response.property_context.location == "Outskirts of London"
    assert "Full house refurbishment" in response.property_context.project_scopes
    assert "Rear extension" in response.property_context.project_scopes
    assert len(response.rooms) == 2
    assert response.rooms[0].name == "Kitchen"
    assert response.rooms[1].name == "Utility"
    demolition_section = next(
        section for section in response.sections if section.key == "demolition"
    )
    assert any("Strip out kitchen" in line for line in demolition_section.lines)


def test_extract_scope_derives_woodwork_finish_metrics() -> None:
    prompt = """
Property
Type: Semi-detached house

FULL SCOPE
Full house refurbishment

DOORS
6 × fire-rated doors

WINDOWS
1 × uPVC window (1 × 1)
1 × uPVC window (2 × 1)

JOINERY
Staircase (softwood, painted)

WOODWORK PAINTING
All:
Doors
Skirting
Architraves
Staircase
Window boards
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert response.takeoff_summary.architrave_set_count == 6.0
    assert response.takeoff_summary.estimated_architrave_lm == 32.4
    assert response.takeoff_summary.window_board_count == 2.0
    assert response.takeoff_summary.estimated_window_board_lm == 3.0
    assert response.takeoff_summary.staircase_count == 1.0


def test_extract_scope_derives_flooring_scope_areas() -> None:
    prompt = """
Property
Type: Semi-detached house

FULL SCOPE
Full house refurbishment

FLOOR AREAS & ROOMS
Ground Floor
Kitchen: 4 × 5
Hallway: 2 × 5
WC: 1 × 2
First Floor
Bedroom 1: 4 × 4
Bathroom: 2 × 2

FLOORING
Ground Floor:
Engineered wood (client supply)
Carpet:
By others
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert response.takeoff_summary.estimated_hard_floor_scope_area_m2 == 30.0
    assert response.takeoff_summary.estimated_carpet_scope_area_m2 == 0.0


def test_extract_scope_derives_boarding_and_insulation_coverage() -> None:
    prompt = """
Property
Type: Terraced house

FULL SCOPE
Loft conversion

FLOOR AREAS & ROOMS
Ground Floor
Kitchen: 4 × 5
First Floor
Bedroom 1: 4 × 4
Loft
Bedroom: 5 × 3

STRUCTURE
Ground Floor
100 mm PIR
Flat roof:
Warm roof (140 mm PIR)

CEILINGS & INSULATION
All new plasterboard ceilings
First Floor:
100 mm acoustic insulation
Loft:
50 mm insulated board under
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert response.takeoff_summary.total_internal_floor_area_m2 == 51.0
    assert response.takeoff_summary.roof_area_m2 == 0.0
    assert response.takeoff_summary.estimated_boarding_coverage_m2 == 51.0
    assert response.takeoff_summary.estimated_insulation_coverage_m2 == 36.0
    assert response.takeoff_summary.estimated_electrical_second_fix_points == 33.0
    assert len(response.room_takeoff) == 3
    kitchen = next(room for room in response.room_takeoff if room.name == "Kitchen")
    assert kitchen.area_m2 == 20.0
    assert kitchen.estimated_wall_finish_area_m2 == 36.72
    assert kitchen.estimated_ceiling_finish_area_m2 == 20.0
    assert kitchen.estimated_skirting_lm == 14.76
    assert kitchen.estimated_socket_count == 7.0
    assert kitchen.estimated_switch_count == 2.0
    assert kitchen.estimated_lighting_point_count == 5.0
    assert kitchen.estimated_extractor_fan_count == 1.0
    assert kitchen.estimated_electrical_second_fix_points == 15.0


def test_extract_scope_derives_loft_roof_and_dormer_metrics() -> None:
    prompt = """
Property
Type: Semi-detached house

FULL SCOPE
Loft conversion

FLOOR AREAS & ROOMS
Loft
Bedroom: 5 × 3
Bathroom: 2 × 2

STRUCTURE
Partial roof removal (50%)
Rear dormer: 3.5 × 5.5
Side dormer: 3 × 2.5

CEILINGS & INSULATION
Loft insulation:
120 mm between rafters
50 mm insulated board under
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert len(response.rooms) == 2
    assert {room.name for room in response.rooms} == {"Bedroom", "Bathroom"}
    assert response.takeoff_summary.loft_area_m2 == 19.0
    assert response.takeoff_summary.dormer_area_m2 == 26.75
    assert response.takeoff_summary.estimated_dormer_build_area_m2 == 26.75
    assert response.takeoff_summary.estimated_loft_roof_finish_area_m2 == 45.75
    assert response.takeoff_summary.estimated_roof_removal_area_m2 == 22.88


def test_extract_scope_derives_rear_extension_structure_metrics() -> None:
    prompt = """
Property
Type: Semi-detached house

FULL SCOPE
Rear extension

FLOOR AREAS & ROOMS
Ground Floor
Rear Living: 3.5 × 3.2
Kitchen: 4 × 3

STRUCTURE
Brick & block cavity walls
Flat roof:
Warm roof (140 mm PIR)
GRP finish
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert response.takeoff_summary.estimated_extension_slab_area_m2 == 11.2
    assert response.takeoff_summary.estimated_extension_flat_roof_area_m2 == 11.2
    assert response.takeoff_summary.estimated_extension_external_wall_area_m2 == 20.2


def test_extract_scope_derives_wet_room_metrics() -> None:
    prompt = """
Property
Type: Terraced house

FULL SCOPE
Bathroom renovation

FLOOR AREAS & ROOMS
Ground Floor
Bathroom: 2 × 2
WC: 1.2 × 1.5
First Floor
Ensuite: 2 × 1.5
Kitchen: 4 × 3
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert response.takeoff_summary.wet_room_count == 3
    assert response.takeoff_summary.estimated_wet_room_floor_tiling_area_m2 == 8.8
    assert response.takeoff_summary.estimated_wet_room_wall_tiling_area_m2 == 22.89

    bathroom = next(room for room in response.room_takeoff if room.name == "Bathroom")
    wc = next(room for room in response.room_takeoff if room.name == "WC")
    ensuite = next(room for room in response.room_takeoff if room.name == "Ensuite")

    assert bathroom.room_type == "bathroom"
    assert bathroom.estimated_wall_tiling_area_m2 == 8.98
    assert bathroom.estimated_plumbing_points == 3.0
    assert bathroom.estimated_wc_count == 1.0
    assert bathroom.estimated_basin_count == 1.0
    assert bathroom.estimated_bath_count == 1.0
    assert bathroom.estimated_shower_count == 0.0
    assert wc.room_type == "wc"
    assert wc.estimated_wall_tiling_area_m2 == 6.06
    assert wc.estimated_plumbing_points == 2.0
    assert wc.estimated_wc_count == 1.0
    assert wc.estimated_basin_count == 1.0
    assert ensuite.room_type == "ensuite"
    assert ensuite.estimated_wall_tiling_area_m2 == 7.85
    assert ensuite.estimated_plumbing_points == 3.0
    assert ensuite.estimated_wc_count == 1.0
    assert ensuite.estimated_basin_count == 1.0
    assert ensuite.estimated_shower_count == 1.0

    kitchen = next(room for room in response.room_takeoff if room.name == "Kitchen")
    assert kitchen.room_type == "kitchen"
    assert kitchen.estimated_sink_count == 1.0

    assert response.takeoff_summary.total_wc_count == 3.0
    assert response.takeoff_summary.total_basin_count == 3.0
    assert response.takeoff_summary.total_bath_count == 1.0
    assert response.takeoff_summary.total_shower_count == 1.0
    assert response.takeoff_summary.total_sink_count == 1.0
    assert response.takeoff_summary.total_sanitary_set_count == 3.0


def test_extract_scope_applies_explicit_sanitary_fixture_overrides() -> None:
    prompt = """
Property
Type: Terraced house

FULL SCOPE
Bathroom renovation

FLOOR AREAS & ROOMS
Ground Floor
Bathroom: 2 × 2
WC: 1.2 × 1.5
Kitchen: 4 × 3

PLUMBING
Bathroom: double basin and separate shower
WC: WC only
WC: wall-hung WC with concealed cistern
Kitchen: double sink
Kitchen: 5 appliance connections
Full sanitary install (client supply)
Client supply appliances
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    bathroom = next(room for room in response.room_takeoff if room.name == "Bathroom")
    wc = next(room for room in response.room_takeoff if room.name == "WC")
    kitchen = next(room for room in response.room_takeoff if room.name == "Kitchen")

    assert bathroom.estimated_basin_count == 2.0
    assert bathroom.estimated_bath_count == 1.0
    assert bathroom.estimated_shower_count == 1.0
    assert wc.estimated_basin_count == 0.0
    assert wc.estimated_plumbing_points == 1.0
    assert wc.estimated_concealed_cistern_count == 1.0
    assert kitchen.estimated_sink_count == 2.0
    assert kitchen.estimated_appliance_connection_count == 5.0
    assert response.takeoff_summary.total_basin_count == 2.0
    assert response.takeoff_summary.total_shower_count == 1.0
    assert response.takeoff_summary.total_sink_count == 2.0
    assert response.takeoff_summary.total_vanity_unit_count == 1.0
    assert response.takeoff_summary.total_concealed_cistern_count == 1.0
    assert response.takeoff_summary.total_appliance_connection_count == 5.0
    assert response.takeoff_summary.client_supply_sanitaryware is True
    assert response.takeoff_summary.client_supply_appliances is True
    assert any(
        "client supplied" in assumption.text.lower()
        for assumption in response.assumptions
        if assumption.text
    )


def test_extract_scope_applies_explicit_electrical_detail_overrides() -> None:
    prompt = """
Property
Type: Terraced house

FULL SCOPE
Kitchen and bathroom refurbishment

FLOOR AREAS & ROOMS
Ground Floor
Kitchen: 4 × 3
Utility: 2 × 2
Bathroom: 2 × 2

ELECTRICAL
Consumer unit (£700 labour + £500 materials)
Kitchen: 8 downlights
Bathroom: extractor fan with ducting
Kitchen: 3 appliance hardwire connections
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    kitchen = next(room for room in response.room_takeoff if room.name == "Kitchen")
    bathroom = next(room for room in response.room_takeoff if room.name == "Bathroom")

    assert kitchen.estimated_downlight_count == 8.0
    assert kitchen.estimated_lighting_point_count == 8.0
    assert kitchen.estimated_hardwired_appliance_count == 3.0
    assert kitchen.estimated_electrical_second_fix_points == 20.0
    assert bathroom.estimated_ducted_extractor_count == 1.0
    assert response.takeoff_summary.consumer_unit_count == 1.0
    assert response.takeoff_summary.downlight_count == 8.0
    assert response.takeoff_summary.ducted_extractor_count == 1.0
    assert response.takeoff_summary.hardwired_appliance_count == 3.0


def test_extract_scope_endpoint_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security

    reload(app_config)
    reload(app_security)

    client = TestClient(app)
    response = client.post(
        "/v1/estimate/extract-scope", json={"prompt": STRUCTURED_PROMPT}
    )

    assert response.status_code == 401


def test_extract_scope_endpoint_returns_structured_payload(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security

    reload(app_config)
    reload(app_security)

    client = TestClient(app)
    response = client.post(
        "/v1/estimate/extract-scope",
        json={"prompt": STRUCTURED_PROMPT},
        headers={"x-api-key": "test-secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["property_context"]["property_type"] == "Semi-detached house"
    assert body["takeoff_summary"]["total_internal_floor_area_m2"] == 196.8
    assert len(body["sections"]) >= 6


def test_extract_scope_supports_basement_second_floor_and_shower_room() -> None:
    prompt = """
Property
Type: Flat

FLOOR AREAS & ROOMS
Basement
Utility: 2 x 3
Second Floor
Shower Room: 2 x 2
Bedroom 1: 3 x 4
""".strip()

    response = extract_scope(prompt)

    assert response.error_text == ""
    assert [room.level for room in response.rooms] == [
        "Basement",
        "Second Floor",
        "Second Floor",
    ]
    assert response.takeoff_summary.basement_area_m2 == 6.0
    assert response.takeoff_summary.second_floor_area_m2 == 16.0
    assert response.takeoff_summary.total_internal_floor_area_m2 == 22.0

    shower_room = next(
        room for room in response.room_takeoff if room.name == "Shower Room"
    )
    assert shower_room.room_type == "shower_room"
    assert shower_room.estimated_shower_count == 1.0
    assert shower_room.estimated_bath_count == 0.0
    assert shower_room.estimated_wc_count == 1.0
