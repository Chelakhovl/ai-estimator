from importlib import reload
import json
from pathlib import Path

import app.config as app_config
import app.services.llm_client as app_llm_client
import app.services.matcher as app_matcher
from app.schemas import (
    CandidateRow,
    PreviewAcceptedExample,
    PreviewAcceptedExampleRow,
    PreviewRequest,
)
from app.services.candidate_shortlist import shortlist_candidates
from app.services.matcher import generate_preview
from app.services.normalizer import parse_quantity_and_unit, split_prompt_segments
from app.services.prompt_builder import (
    build_preview_system_prompt,
    build_preview_user_payload,
)
from app.services.scope_extractor import extract_scope
from app.services.scope_matching import match_row_to_extracted_section
from app.services.scope_takeoff import (
    TAKEOFF_HEURISTICS_SOURCE,
    TAKEOFF_HEURISTICS_VERSION,
)
import pytest


def load_eval_cases() -> list[dict]:
    fixtures_dir = Path(__file__).parent / "fixtures"
    cases: list[dict] = []
    for fixture_path in sorted(fixtures_dir.glob("*.json")):
        cases.extend(json.loads(fixture_path.read_text()))
    return cases


def test_preview_contract_returns_expected_shape(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-1",
        prompt="Kitchen renovation, paint walls 120 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-1",
                "WORKGUID": "work-1",
                "WorkName": "Paint walls",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert response.service_mode == "mock"
    assert response.review_summary.matched_count == 1
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].WorkName == "Paint walls"
    assert response.matched_rows[0].QUANTITY == 120
    assert response.telemetry.candidate_row_count == 1
    assert response.telemetry.shortlist_size == 1
    assert response.telemetry.fallback_reason == "llm_not_configured"
    assert response.review_queue == []
    assert isinstance(response.coverage_prompts, list)
    assert all(prompt.title for prompt in response.coverage_prompts)


def test_preview_without_llm_config_fails_fast_outside_local_or_test(
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-prod-no-llm",
        prompt="Paint walls 20 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-1",
                "WORKGUID": "work-1",
                "WorkName": "Paint walls",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="LLM preview is not configured"):
        app_matcher.generate_preview(request)


def test_preview_system_prompt_discloses_takeoff_heuristics_version() -> None:
    prompt = build_preview_system_prompt()

    assert TAKEOFF_HEURISTICS_VERSION in prompt
    assert TAKEOFF_HEURISTICS_SOURCE in prompt


def test_preview_builds_review_queue_and_coverage_prompts_from_unmatched_scope(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-queue",
        prompt="Install gas cooker",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-gas",
                "WORKGUID": "work-gas",
                "WorkItemCode": "install-gas-cooker",
                "GuardrailCode": "do_not_include",
                "WorkName": "Install gas cooker",
                "WorkGroupName": "Plumbing",
                "Unit": "item",
                "WorkLabourCost": 10,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.review_queue
    assert response.coverage_prompts
    assert any(item.title == "Install gas cooker" for item in response.review_queue)
    assert any(
        prompt.title == "Install gas cooker" for prompt in response.coverage_prompts
    )


def test_preview_uses_group_context_to_improve_mock_matching(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-2",
        prompt="Bathroom demolition, strip out 15 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-1",
                "WORKGUID": "work-1",
                "WorkItemCode": "demo-strip-out",
                "WorkName": "Strip out",
                "WorkGroupName": "Demolition",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 1,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            },
            {
                "INSIDEQUOTESGUID": "inside-2",
                "WORKGUID": "work-2",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            },
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].INSIDEQUOTESGUID == "inside-1"


def test_preview_uses_takeoff_quantity_for_structured_switch_counts(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-3",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "ELECTRICAL\n"
            "Switches (32 total)\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-switches",
                "WORKGUID": "work-switches",
                "WorkItemCode": "install-switches",
                "WorkName": "Install switches",
                "WorkGroupName": "Electrics",
                "Unit": "pcs",
                "WorkLabourCost": 15,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].INSIDEQUOTESGUID == "inside-switches"
    assert response.matched_rows[0].QUANTITY == 32
    assert any(
        "derived from the structured scope takeoff" in (assumption.text or "")
        for assumption in response.assumptions
    )


def test_preview_skips_measured_rows_without_explicit_or_recoverable_quantity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-measured-safety",
        prompt="Paint walls throughout",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-paint",
                "WORKGUID": "work-paint",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorating",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
                "QUANTITY": 0,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert response.matched_rows == []
    assert response.unmatched_items
    assert "could not be safely inferred" in response.unmatched_items[0].reason
    assert any(
        "no defensible takeoff" in (assumption.text or "")
        for assumption in response.assumptions
    )


def test_preview_uses_extracted_section_completeness_warnings(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-4",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "PLUMBING\n"
            "Install basin 1 ea\n"
            "ELECTRICAL\n"
            "Switches (32 total)\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-basin",
                "WORKGUID": "work-basin",
                "WorkItemCode": "install-basin",
                "WorkName": "Install basin",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 85,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    )

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert "extracted electrical section" in warning_text.lower()


def test_preview_flags_missing_electrical_detail_blocks(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-electrical-detail-gaps",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "ELECTRICAL\n"
            "Consumer unit (£700 labour + £500 materials)\n"
            "Kitchen: 12 downlights\n"
            "Bathroom: extractor fan with ducting\n"
            "Kitchen: 3 appliance hardwire connections\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-sockets-only",
                "WORKGUID": "work-sockets-only",
                "WorkItemCode": "install-sockets",
                "WorkName": "Install sockets",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 20,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    )

    assert response.error_text == ""
    assert "consumer unit / distribution board" in warning_text.lower()
    assert "lighting / downlights" in warning_text.lower()
    assert "extract ventilation" in warning_text.lower()
    assert "appliance connections / hardwiring" in warning_text.lower()


def test_preview_flags_missing_structure_detail_blocks(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-structure-detail-gaps",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Rear extension\n"
            "Loft conversion\n"
            "STRUCTURE\n"
            "Brick & block cavity walls\n"
            "Flat roof:\n"
            "Warm roof (140 mm PIR)\n"
            "GRP finish\n"
            "Rear dormer: 3.5 × 5.5\n"
            "4 × Velux (MK04)\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-extension-walls-only",
                "WORKGUID": "work-extension-walls-only",
                "WorkItemCode": "build-extension-walls",
                "WorkName": "Build rear extension cavity walls",
                "WorkGroupName": "Brickwork",
                "Unit": "m2",
                "WorkLabourCost": 55,
                "WorkMatCost": 35,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    )

    assert response.error_text == ""
    assert "flat roof / warm roof" in warning_text.lower()
    assert "dormer build" in warning_text.lower()
    assert "rooflights / skylights" in warning_text.lower()


def test_preview_flags_missing_deeper_roof_build_detail_blocks(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-roof-detail-gaps",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Rear extension\n"
            "Loft conversion\n"
            "STRUCTURE\n"
            "Flat roof:\n"
            "Warm roof (140 mm PIR)\n"
            "GRP finish\n"
            "Parapet upstands\n"
            "Rear dormer: 3.5 × 5.5\n"
            "Dormer cheeks in cladding\n"
            "WINDOWS\n"
            "4 × Velux (MK04)\n"
            "CEILINGS & INSULATION\n"
            "120 mm between rafters\n"
            "50 mm insulated board under\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-flat-roof-only",
                "WORKGUID": "work-flat-roof-only",
                "WorkItemCode": "install-flat-roof",
                "WorkName": "Install flat roof",
                "WorkGroupName": "Roofing",
                "Unit": "m2",
                "WorkLabourCost": 55,
                "WorkMatCost": 35,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    ).lower()

    assert response.error_text == ""
    assert "dormer cheeks / cladding" in warning_text
    assert "roof edges / parapets / upstands" in warning_text
    assert "roof insulation / rafter build-up" in warning_text


def test_preview_uses_derived_wall_finish_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-5",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 5\n"
            "DECORATING\n"
            "Paint walls throughout\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-paint-walls",
                "WORKGUID": "work-paint-walls",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 36.72
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "derived wall finish area" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_room_specific_wall_takeoff_when_room_named(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-wall-1",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 5\n"
            "Bathroom: 2 × 2\n"
            "DECORATING\n"
            "Paint kitchen walls\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-paint-kitchen-walls",
                "WORKGUID": "work-paint-kitchen-walls",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 36.72
    assert any(
        "matched rooms: kitchen" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_room_specific_floor_takeoff_when_room_named(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-floor-1",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen Dining: 3.5 × 5.8\n"
            "Study: 1.5 × 1.8\n"
            "FLOORING\n"
            "Install engineered timber floor to kitchen dining\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-floor-kitchen",
                "WORKGUID": "work-floor-kitchen",
                "WorkItemCode": "install-floor",
                "WorkName": "Install engineered timber floor",
                "WorkGroupName": "Flooring",
                "Unit": "m2",
                "WorkLabourCost": 12,
                "WorkMatCost": 9,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 20.3
    assert any(
        "matched rooms: kitchen dining" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_flags_room_specific_finish_gaps_when_finish_section_is_only_partially_matched(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-finish-gap-1",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen Dining: 3.5 × 5.8\n"
            "Study: 1.5 × 1.8\n"
            "FLOORING\n"
            "Kitchen Dining: Engineered wood\n"
            "Study: Engineered wood\n"
            "Install engineered timber floor to kitchen dining\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-floor-kitchen-gap",
                "WORKGUID": "work-floor-kitchen-gap",
                "WorkItemCode": "install-floor",
                "WorkName": "Install engineered timber floor",
                "WorkGroupName": "Flooring",
                "Unit": "m2",
                "WorkLabourCost": 12,
                "WorkMatCost": 9,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    gap_warnings = [
        assumption
        for assumption in response.assumptions
        if assumption.kind == "room_finish_coverage_gap"
    ]
    assert gap_warnings
    assert "study" in gap_warnings[0].text.lower()


def test_preview_skips_room_finish_gap_warning_when_no_finish_rows_matched_yet(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-finish-gap-2",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen Dining: 3.5 × 5.8\n"
            "Study: 1.5 × 1.8\n"
            "FLOORING\n"
            "Kitchen Dining: Engineered wood\n"
            "Study: Engineered wood\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-generic-prep",
                "WORKGUID": "work-generic-prep",
                "WorkItemCode": "prep-walls",
                "WorkName": "Prepare walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 7,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert not any(
        assumption.kind == "room_finish_coverage_gap"
        for assumption in response.assumptions
    )


def test_preview_uses_room_specific_socket_takeoff_when_room_named(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-sockets-1",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen Dining: 3.5 × 5.8\n"
            "Study: 1.5 × 1.8\n"
            "ELECTRICAL\n"
            "Install kitchen dining sockets\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-room-sockets",
                "WORKGUID": "work-room-sockets",
                "WorkItemCode": "install-sockets",
                "WorkName": "Install sockets",
                "WorkGroupName": "Electrics",
                "Unit": "pcs",
                "WorkLabourCost": 20,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 7
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "socket allowances" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_wet_room_wall_tiling_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-tile-1",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "WC: 1.2 × 1.5\n"
            "TILING\n"
            "Tile bathroom walls\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-tile-bathroom-walls",
                "WORKGUID": "work-tile-bathroom-walls",
                "WorkItemCode": "tile-walls",
                "WorkName": "Tile walls",
                "WorkGroupName": "Tiling",
                "Unit": "m2",
                "WorkLabourCost": 22,
                "WorkMatCost": 18,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 8.98
    assert any(
        "wet-room wall tiling" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_wet_room_plumbing_points(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-plumb-1",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "First Floor\n"
            "Ensuite: 2 × 1.5\n"
            "PLUMBING\n"
            "Bathroom ensuite plumbing first fix\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-bathroom-plumb-first-fix",
                "WORKGUID": "work-bathroom-plumb-first-fix",
                "WorkItemCode": "bathroom-plumbing-first-fix",
                "WorkName": "Bathroom plumbing first fix",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 55,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 6
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "room-specific plumbing points" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_room_specific_plumbing_second_fix_points(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-plumb-second-fix",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "Utility: 2 × 1.5\n"
            "PLUMBING\n"
            "Bathroom utility plumbing second fix\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-room-plumb-second-fix",
                "WORKGUID": "work-room-plumb-second-fix",
                "WorkItemCode": "plumbing-second-fix",
                "WorkName": "Bathroom plumbing second fix",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 55,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 4
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "room-specific plumbing points" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_room_specific_sanitary_set_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-sanitary-sets",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "First Floor\n"
            "Ensuite: 2 × 1.5\n"
            "PLUMBING\n"
            "Bathroom ensuite sanitary install\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-sanitary-install",
                "WORKGUID": "work-sanitary-install",
                "WorkItemCode": "sanitary-install",
                "WorkName": "Bathroom sanitary install",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 80,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 2
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "sanitary-set counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_whole_property_wc_fixture_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-whole-property-wc",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "WC: 1.2 × 1.5\n"
            "First Floor\n"
            "Ensuite: 2 × 1.5\n"
            "PLUMBING\n"
            "Install toilet pans throughout the property\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-install-wc",
                "WORKGUID": "work-install-wc",
                "WorkItemCode": "install-wc",
                "WorkName": "Install WC pan",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 40,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 3
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "whole-property wc fixture counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_explicit_double_basin_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-double-basin",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "PLUMBING\n"
            "Bathroom: double basin\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-install-basin",
                "WORKGUID": "work-install-basin",
                "WorkItemCode": "install-basin",
                "WorkName": "Install basin",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 40,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 2
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "room-specific basin fixture counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_holds_sanitary_supply_rows_for_client_supply_review(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-client-supply-sanitary",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "PLUMBING\n"
            "Full sanitary install (client supply)\n"
            "Supply bathroom sanitaryware\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-supply-sanitary",
                "WORKGUID": "work-supply-sanitary",
                "WorkItemCode": "supply-bathroom-sanitaryware",
                "WorkName": "Supply bathroom sanitaryware",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 0,
                "WorkMatCost": 500,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 1
    assert response.matched_rows[0].NeedsReview is True
    assert "client supplied" in (response.matched_rows[0].ReviewReason or "").lower()
    assert any(
        "client-supplied sanitaryware" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_vanity_unit_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-vanity-unit",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "PLUMBING\n"
            "Bathroom: double vanity unit\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-vanity",
                "WORKGUID": "work-vanity",
                "WorkItemCode": "install-vanity",
                "WorkName": "Install vanity unit",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 65,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 1
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "vanity unit counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_concealed_cistern_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-concealed-cistern",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom renovation\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "WC: 1.2 × 1.5\n"
            "PLUMBING\n"
            "WC: wall-hung WC with concealed cistern\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-cistern",
                "WORKGUID": "work-cistern",
                "WorkItemCode": "install-concealed-cistern",
                "WorkName": "Install concealed cistern",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 55,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 1
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "concealed cistern counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_appliance_connection_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-appliance-connections",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Kitchen refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 3\n"
            "ELECTRICAL\n"
            "Kitchen: 5 appliance connections\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-appliance-connections",
                "WORKGUID": "work-appliance-connections",
                "WorkItemCode": "connect-appliances",
                "WorkName": "Connect appliances",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 35,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 5
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "appliance connection counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_holds_appliance_supply_rows_for_client_supply_review(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-client-supply-appliances",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Kitchen refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 3\n"
            "ELECTRICAL\n"
            "Kitchen: 4 appliance connections\n"
            "Client supply appliances\n"
            "Supply kitchen appliances\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-supply-appliances",
                "WORKGUID": "work-supply-appliances",
                "WorkItemCode": "supply-kitchen-appliances",
                "WorkName": "Supply kitchen appliances",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 0,
                "WorkMatCost": 900,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 1
    assert response.matched_rows[0].NeedsReview is True
    assert "client supplied" in (response.matched_rows[0].ReviewReason or "").lower()
    assert any(
        "client-supplied appliances" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_consumer_unit_count(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-consumer-unit",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "ELECTRICAL\n"
            "Consumer unit (£700 labour + £500 materials)\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-consumer-unit",
                "WORKGUID": "work-consumer-unit",
                "WorkItemCode": "install-consumer-unit",
                "WorkName": "Install consumer unit",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 700,
                "WorkMatCost": 500,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 1
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "consumer unit counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )
    assert "specialist electrical review" in " ".join(
        assumption.text.lower()
        for assumption in response.assumptions
        if assumption.text
    )


def test_preview_uses_fire_rated_pocket_door_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-fire-pocket-doors",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "DOORS\n"
            "2 × double pocket doors (fire-rated)\n"
            "1 × single pocket door\n"
            "6 × fire-rated doors\n"
            "2 × pocket doors (fire-rated)\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-fire-pocket-door",
                "WORKGUID": "work-fire-pocket-door",
                "WorkItemCode": "install-fire-pocket-door-set",
                "WorkName": "Install fire-rated pocket door set",
                "WorkGroupName": "Doors",
                "Unit": "pcs",
                "WorkLabourCost": 150,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 4
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "fire-rated pocket door set counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_joinery_detail_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    staircase_request = PreviewRequest(
        quote_guid="quote-staircase",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Loft conversion\n"
            "JOINERY\n"
            "Staircase (softwood, painted)\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-staircase",
                "WORKGUID": "work-staircase",
                "WorkItemCode": "install-staircase",
                "WorkName": "Install staircase",
                "WorkGroupName": "Joinery",
                "Unit": "pcs",
                "WorkLabourCost": 650,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    staircase_response = generate_preview(staircase_request)

    assert staircase_response.error_text == ""
    assert len(staircase_response.matched_rows) == 1
    assert staircase_response.matched_rows[0].QUANTITY == 1
    assert staircase_response.matched_rows[0].NeedsReview is True
    assert any(
        "staircase counts" in (assumption.text or "").lower()
        for assumption in staircase_response.assumptions
    )

    storage_request = PreviewRequest(
        quote_guid="quote-storage-doors",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Loft conversion\n"
            "JOINERY\n"
            "Loft storage:\n"
            "3 × double door sets\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-storage-doors",
                "WORKGUID": "work-storage-doors",
                "WorkItemCode": "install-storage-door-set",
                "WorkName": "Install storage door set",
                "WorkGroupName": "Joinery",
                "Unit": "pcs",
                "WorkLabourCost": 120,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    storage_response = generate_preview(storage_request)

    assert storage_response.error_text == ""
    assert len(storage_response.matched_rows) == 1
    assert storage_response.matched_rows[0].QUANTITY == 3
    assert storage_response.matched_rows[0].NeedsReview is True
    assert any(
        "storage door-set counts" in (assumption.text or "").lower()
        for assumption in storage_response.assumptions
    )


def test_preview_flags_missing_door_and_joinery_detail_blocks(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-door-joinery-detail-gaps",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "DOORS\n"
            "2 × double pocket doors (fire-rated)\n"
            "Allowance:\n"
            "£200 per door (supply)\n"
            "JOINERY\n"
            "Staircase (softwood, painted)\n"
            "Loft storage:\n"
            "3 × double door sets\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-generic-strip-out",
                "WORKGUID": "work-generic-strip-out",
                "WorkItemCode": "strip-out",
                "WorkName": "Strip out",
                "WorkGroupName": "Demolition",
                "Unit": "m2",
                "WorkLabourCost": 20,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    ).lower()

    assert response.error_text == ""
    assert "fire-rated door sets" in warning_text
    assert "pocket door sets" in warning_text
    assert "door supply allowance" in warning_text
    assert "staircase joinery" in warning_text
    assert "storage / fitted joinery door sets" in warning_text


def test_preview_uses_woodwork_finish_baselines(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    architrave_request = PreviewRequest(
        quote_guid="quote-architraves",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "DOORS\n"
            "6 × fire-rated doors\n"
            "WOODWORK PAINTING\n"
            "All:\n"
            "Architraves\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-architraves",
                "WORKGUID": "work-architraves",
                "WorkItemCode": "paint-architraves",
                "WorkName": "Paint architraves",
                "WorkGroupName": "Woodwork Painting",
                "Unit": "m",
                "WorkLabourCost": 8,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    architrave_response = generate_preview(architrave_request)

    assert architrave_response.error_text == ""
    assert len(architrave_response.matched_rows) == 1
    assert architrave_response.matched_rows[0].QUANTITY == 32.4
    assert architrave_response.matched_rows[0].NeedsReview is True
    assert any(
        "architrave length" in (assumption.text or "").lower()
        for assumption in architrave_response.assumptions
    )

    window_board_request = PreviewRequest(
        quote_guid="quote-window-boards",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "WINDOWS\n"
            "1 × uPVC window (1 × 1)\n"
            "1 × uPVC window (2 × 1)\n"
            "WOODWORK PAINTING\n"
            "All:\n"
            "Window boards\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-window-boards",
                "WORKGUID": "work-window-boards",
                "WorkItemCode": "paint-window-boards",
                "WorkName": "Paint window boards",
                "WorkGroupName": "Woodwork Painting",
                "Unit": "m",
                "WorkLabourCost": 6,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    window_board_response = generate_preview(window_board_request)

    assert window_board_response.error_text == ""
    assert len(window_board_response.matched_rows) == 1
    assert window_board_response.matched_rows[0].QUANTITY == 3.0
    assert window_board_response.matched_rows[0].NeedsReview is True
    assert any(
        "window board length" in (assumption.text or "").lower()
        for assumption in window_board_response.assumptions
    )

    skirting_request = PreviewRequest(
        quote_guid="quote-room-skirting",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 5\n"
            "Study: 1.5 × 1.8\n"
            "WOODWORK PAINTING\n"
            "Skirting to kitchen\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-skirting-kitchen",
                "WORKGUID": "work-skirting-kitchen",
                "WorkItemCode": "paint-skirting",
                "WorkName": "Paint skirting",
                "WorkGroupName": "Woodwork Painting",
                "Unit": "m",
                "WorkLabourCost": 6,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    skirting_response = generate_preview(skirting_request)

    assert skirting_response.error_text == ""
    assert len(skirting_response.matched_rows) == 1
    assert skirting_response.matched_rows[0].AREA == "Kitchen"
    assert skirting_response.matched_rows[0].NeedsReview is True
    assert any(
        "room-specific skirting length" in (assumption.text or "").lower()
        for assumption in skirting_response.assumptions
    )


def test_preview_flags_missing_woodwork_painting_detail_blocks(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-woodwork-painting-detail-gaps",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "DOORS\n"
            "6 × fire-rated doors\n"
            "WINDOWS\n"
            "1 × uPVC window (1 × 1)\n"
            "JOINERY\n"
            "Staircase (softwood, painted)\n"
            "WOODWORK PAINTING\n"
            "All:\n"
            "Doors\n"
            "Skirting\n"
            "Architraves\n"
            "Staircase\n"
            "Window boards\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-generic-paint",
                "WORKGUID": "work-generic-paint",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 7,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    ).lower()

    assert response.error_text == ""
    assert "painted door sets" in warning_text
    assert "skirting" in warning_text
    assert "architraves" in warning_text
    assert "staircase woodwork" in warning_text
    assert "window boards" in warning_text


def test_preview_flags_level_specific_door_scope_when_matched_rows_stay_generic(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-level-door-gap",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "DOORS\n"
            "Ground Floor:\n"
            "6 × fire-rated doors\n"
            "Loft:\n"
            "2 × pocket doors (fire-rated)\n"
            "Install fire-rated doors\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-fire-doors-generic",
                "WORKGUID": "work-fire-doors-generic",
                "WorkItemCode": "install-fire-door-set",
                "WorkName": "Install fire-rated doors",
                "WorkGroupName": "Doors",
                "Unit": "pcs",
                "WorkLabourCost": 140,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.kind == "level_joinery_coverage_gap"
    ).lower()

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert "ground floor" in warning_text
    assert "loft" in warning_text


def test_preview_uses_structured_flooring_scope_area(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-structured-flooring-scope",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 5\n"
            "Hallway: 2 × 5\n"
            "WC: 1 × 2\n"
            "First Floor\n"
            "Bedroom 1: 4 × 4\n"
            "Bathroom: 2 × 2\n"
            "FLOORING\n"
            "Ground Floor:\n"
            "Engineered wood (client supply)\n"
            "Carpet:\n"
            "By others\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-engineered-floor",
                "WORKGUID": "work-engineered-floor",
                "WorkItemCode": "install-engineered-floor",
                "WorkName": "Install engineered timber floor",
                "WorkGroupName": "Flooring",
                "Unit": "m2",
                "WorkLabourCost": 12,
                "WorkMatCost": 9,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 30.0
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "hard-floor scope area" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_flags_missing_decorating_and_flooring_detail_blocks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-finish-detail-gaps",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "DECORATING\n"
            "Walls + ceilings\n"
            "1 mist + 2 coats\n"
            "FLOORING\n"
            "Ground Floor:\n"
            "Engineered wood\n"
            "First Floor:\n"
            "Carpet + underlay\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-generic-prep",
                "WORKGUID": "work-generic-prep",
                "WorkItemCode": "prep-walls",
                "WorkName": "Prepare walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 7,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    ).lower()

    assert response.error_text == ""
    assert "wall decoration" in warning_text
    assert "ceiling decoration" in warning_text
    assert "mist / finish coats" in warning_text
    assert "hard-floor finishes" in warning_text
    assert "carpet / underlay" in warning_text


def test_preview_holds_rewire_and_circuit_scope_on_specialist_review(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-rewire-circuits",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "ELECTRICAL\n"
            "Full rewire\n"
            "New circuit connections\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-rewire",
                "WORKGUID": "work-rewire",
                "WorkItemCode": "full-rewire",
                "WorkName": "Full rewire with circuit alterations",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 500,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].NeedsReview is True
    assert (
        "rewire scope should stay on specialist review"
        in (response.matched_rows[0].ReviewReason or "").lower()
    )
    assert (
        "circuit-based electrical scope should stay on specialist review"
        in (response.matched_rows[0].ReviewReason or "").lower()
    )


def test_preview_holds_boiler_ufh_and_gas_scope_on_specialist_review(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-specialist-heating",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "HEATING\n"
            "System boiler\n"
            "Megaflo 300L\n"
            "Ground floor water UFH\n"
            "Gas line\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-specialist-heating",
                "WORKGUID": "work-specialist-heating",
                "WorkItemCode": "install-boiler-cylinder-ufh-gas",
                "WorkName": "Install boiler cylinder and UFH gas line package",
                "WorkGroupName": "Heating",
                "Unit": "pcs",
                "WorkLabourCost": 1200,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)
    review_reason = (response.matched_rows[0].ReviewReason or "").lower()
    assumption_text = " ".join(
        assumption.text.lower()
        for assumption in response.assumptions
        if assumption.text
    )

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].NeedsReview is True
    assert "boiler / cylinder work should stay on specialist review" in review_reason
    assert "underfloor heating scope should stay on specialist review" in review_reason
    assert "gas line scope should stay on specialist review" in review_reason
    assert "specialist plumbing/heating review retained" in assumption_text


def test_preview_uses_room_specific_downlight_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-room-downlights",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Kitchen refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 3\n"
            "ELECTRICAL\n"
            "Kitchen: 8 downlights\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-downlights",
                "WORKGUID": "work-downlights",
                "WorkItemCode": "install-downlights",
                "WorkName": "Install downlights",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 35,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 8
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "downlight counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_room_specific_ducted_extractor_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-ducted-extractor",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Bathroom refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Bathroom: 2 × 2\n"
            "ELECTRICAL\n"
            "Bathroom: extractor fan with ducting\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-ducted-extractor",
                "WORKGUID": "work-ducted-extractor",
                "WorkItemCode": "install-ducted-extractor",
                "WorkName": "Install extractor fan with ducting",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 55,
                "WorkMatCost": 15,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 1
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "ducted extractor counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_room_specific_hardwired_appliance_counts(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-hardwired-appliances",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Kitchen refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 3\n"
            "ELECTRICAL\n"
            "Kitchen: 3 appliance hardwire connections\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-hardwire-appliances",
                "WORKGUID": "work-hardwire-appliances",
                "WorkItemCode": "hardwire-appliances",
                "WorkName": "Hardwire kitchen appliances",
                "WorkGroupName": "Electrical",
                "Unit": "pcs",
                "WorkLabourCost": 40,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 3
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "hardwired appliance counts" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_assigns_section_metadata_and_boarding_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-6",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Loft conversion\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 5\n"
            "CEILINGS & INSULATION\n"
            "All new plasterboard ceilings\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-board-ceilings",
                "WORKGUID": "work-board-ceilings",
                "WorkItemCode": "board-ceilings",
                "WorkName": "Fix plasterboard ceilings",
                "WorkGroupName": "Ceilings & Insulation",
                "Unit": "m2",
                "WorkLabourCost": 12,
                "WorkMatCost": 5,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 20
    assert response.matched_rows[0].MatchedSectionKey == "ceilings_and_insulation"
    assert response.matched_rows[0].MatchedSectionTitle == "Ceilings & Insulation"


def test_preview_uses_derived_insulation_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-7",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Rear extension\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 5\n"
            "First Floor\n"
            "Bedroom: 4 × 4\n"
            "STRUCTURE\n"
            "Ground Floor\n"
            "100 mm PIR\n"
            "CEILINGS & INSULATION\n"
            "First Floor:\n"
            "100 mm acoustic insulation\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-insulation",
                "WORKGUID": "work-insulation",
                "WorkItemCode": "install-insulation",
                "WorkName": "Install acoustic insulation",
                "WorkGroupName": "Ceilings & Insulation",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 8,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 36
    assert response.matched_rows[0].NeedsReview is True
    assert "derived insulation coverage" in " ".join(
        assumption.text.lower()
        for assumption in response.assumptions
        if assumption.text
    )


def test_preview_uses_extension_flat_roof_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-extension-flat-roof",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Rear extension\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Rear Living: 3.5 × 3.2\n"
            "STRUCTURE\n"
            "Flat roof:\n"
            "Warm roof (140 mm PIR)\n"
            "GRP finish\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-flat-roof",
                "WORKGUID": "work-flat-roof",
                "WorkItemCode": "install-flat-roof",
                "WorkName": "Install flat roof",
                "WorkGroupName": "Roofing",
                "Unit": "m2",
                "WorkLabourCost": 75,
                "WorkMatCost": 60,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 11.2
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "rear extension flat-roof area" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_extension_wall_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-extension-walls",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Rear extension\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Rear Living: 3.5 × 3.2\n"
            "STRUCTURE\n"
            "Brick & block cavity walls\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-extension-walls",
                "WORKGUID": "work-extension-walls",
                "WorkItemCode": "build-extension-walls",
                "WorkName": "Build rear extension cavity walls",
                "WorkGroupName": "Brickwork",
                "Unit": "m2",
                "WorkLabourCost": 55,
                "WorkMatCost": 35,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 20.2
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "rear extension external wall area" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_whole_property_electrical_second_fix_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-electrical-second-fix-whole-house",
        prompt=(
            "Property\n"
            "Type: Terraced house\n"
            "FULL SCOPE\n"
            "Full house refurbishment\n"
            "FLOOR AREAS & ROOMS\n"
            "Ground Floor\n"
            "Kitchen: 4 × 5\n"
            "Living Room: 4 × 4\n"
            "First Floor\n"
            "Bedroom 1: 4 × 4\n"
            "Bathroom: 2 × 2\n"
            "ELECTRICAL\n"
            "Electrical second fix\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-electrical-second-fix",
                "WORKGUID": "work-electrical-second-fix",
                "WorkItemCode": "electrical-second-fix",
                "WorkName": "Electrical second fix",
                "WorkGroupName": "Electrics",
                "Unit": "pcs",
                "WorkLabourCost": 30,
                "WorkMatCost": 0,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 40
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "whole-property electrical second-fix allowances"
        in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_preview_uses_derived_loft_roof_finish_takeoff(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid="quote-loft-roof-finish",
        prompt=(
            "Property\n"
            "Type: Semi-detached house\n"
            "FULL SCOPE\n"
            "Loft conversion\n"
            "FLOOR AREAS & ROOMS\n"
            "Loft\n"
            "Bedroom: 5 × 3\n"
            "Bathroom: 2 × 2\n"
            "STRUCTURE\n"
            "Rear dormer: 3.5 × 5.5\n"
            "Side dormer: 3 × 2.5\n"
            "CEILINGS & INSULATION\n"
            "120 mm between rafters\n"
            "50 mm insulated board under\n"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-loft-roof-board",
                "WORKGUID": "work-loft-roof-board",
                "WorkItemCode": "board-loft-roof",
                "WorkName": "Board loft roof slopes",
                "WorkGroupName": "Ceilings & Insulation",
                "Unit": "m2",
                "WorkLabourCost": 12,
                "WorkMatCost": 5,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].QUANTITY == 45.75
    assert response.matched_rows[0].NeedsReview is True
    assert any(
        "loft / dormer roof finish area" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_segment_aware_shortlist_keeps_large_mixed_scope_coverage() -> None:
    candidate_rows = []
    for index in range(30):
        candidate_rows.append(
            {
                "INSIDEQUOTESGUID": f"noise-{index}",
                "WORKGUID": f"noise-work-{index}",
                "WorkItemCode": f"noise-task-{index}",
                "WorkName": f"Install replacement feature {index}",
                "WorkGroupName": "General works",
                "Unit": "pcs",
                "WorkLabourCost": 10,
                "WorkMatCost": 1,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        )

    relevant_rows = [
        {
            "INSIDEQUOTESGUID": "rel-stripout",
            "WORKGUID": "work-stripout",
            "WorkItemCode": "demo-strip-out",
            "WorkName": "Strip out kitchen",
            "WorkGroupName": "Demolition",
            "Unit": "m2",
            "WorkLabourCost": 10,
            "WorkMatCost": 1,
            "WorkOtherCost": 0,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        },
        {
            "INSIDEQUOTESGUID": "rel-paint",
            "WORKGUID": "work-paint",
            "WorkItemCode": "paint-walls",
            "WorkName": "Paint walls",
            "WorkGroupName": "Decorations",
            "Unit": "m2",
            "WorkLabourCost": 10,
            "WorkMatCost": 4,
            "WorkOtherCost": 1,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        },
        {
            "INSIDEQUOTESGUID": "rel-floor",
            "WORKGUID": "work-floor",
            "WorkItemCode": "install-laminate",
            "WorkName": "Install laminate floor",
            "WorkGroupName": "Flooring",
            "Unit": "m2",
            "WorkLabourCost": 12,
            "WorkMatCost": 9,
            "WorkOtherCost": 0,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        },
        {
            "INSIDEQUOTESGUID": "rel-sockets",
            "WORKGUID": "work-sockets",
            "WorkItemCode": "replace-sockets",
            "WorkName": "Replace sockets",
            "WorkGroupName": "Electrics",
            "Unit": "pcs",
            "WorkLabourCost": 20,
            "WorkMatCost": 0,
            "WorkOtherCost": 0,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        },
        {
            "INSIDEQUOTESGUID": "rel-sink",
            "WORKGUID": "work-sink",
            "WorkItemCode": "reconnect-sink",
            "WorkName": "Reconnect sink",
            "WorkGroupName": "Plumbing",
            "Unit": "pcs",
            "WorkLabourCost": 75,
            "WorkMatCost": 0,
            "WorkOtherCost": 0,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        },
        {
            "INSIDEQUOTESGUID": "rel-tiles",
            "WORKGUID": "work-tiles",
            "WorkItemCode": "tile-splashback",
            "WorkName": "Tile splashback",
            "WorkGroupName": "Tiling",
            "Unit": "m2",
            "WorkLabourCost": 20,
            "WorkMatCost": 18,
            "WorkOtherCost": 0,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        },
    ]

    prompt = (
        "Large kitchen refurbishment, strip out kitchen 20 m2, paint walls 120 m2, install laminate floor 60 m2, "
        "replace sockets 8 pc, reconnect sink 1 ea, tile splashback 8 m2"
    )

    shortlist_input = [
        CandidateRow.model_validate(row) for row in (relevant_rows + candidate_rows)
    ]
    shortlisted = shortlist_candidates(
        prompt, shortlist_input, limit=12, per_segment_limit=3
    )
    shortlisted_ids = {row.INSIDEQUOTESGUID for row in shortlisted}

    assert {
        "rel-stripout",
        "rel-paint",
        "rel-floor",
        "rel-sockets",
        "rel-sink",
        "rel-tiles",
    }.issubset(shortlisted_ids)


def test_split_prompt_segments_keeps_shared_quantity_scope_together() -> None:
    prompt = "Full house refurb, strip out walls and floors 140 m2, paint walls 220 m2"

    assert split_prompt_segments(prompt) == [
        "Full house refurb",
        "strip out walls and floors 140 m2",
        "paint walls 220 m2",
    ]


def test_preview_user_payload_includes_scope_summary_and_norm_basis() -> None:
    prompt = (
        "PROPERTY:\n"
        "Type: Flat\n"
        "\n"
        "FLOOR AREAS & ROOMS:\n"
        "Ground Floor\n"
        "Kitchen: 4.5 x 3.2\n"
        "\n"
        "ELECTRICAL:\n"
        "Kitchen: 8 downlights\n"
    )
    extracted_scope = extract_scope(prompt)
    payload = build_preview_user_payload(
        prompt,
        [
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": "inside-1",
                    "WORKGUID": "work-1",
                    "WorkItemCode": "ELEC-DL-01",
                    "WorkName": "Install LED downlight",
                    "WorkGroupName": "Electrical",
                    "Unit": "pcs",
                    "WorkQTYforNorm": 10,
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        ],
        extracted_scope,
    )

    assert payload["candidate_rows"][0]["nb"] == 10
    assert payload["candidate_row_key_legend"]["id"] == "inside_quote_guid"
    assert payload["scope_summary"]["rooms"][0]["name"] == "Kitchen"
    assert payload["scope_summary"]["takeoff_summary"]["room_count"] == 1
    assert payload["scope_summary"]["room_takeoff"][0]["estimated_downlight_count"] == 8
    assert "Total internal floor area: 14.4 m2" in payload["scope_summary_text"]
    assert "Downlights: 8" in payload["scope_summary_text"]
    assert "Ground Floor: Kitchen 14.4 m2" in payload["scope_summary_text"]


def test_preview_user_payload_includes_active_scope_sections_and_group_alias_scope_hints() -> (
    None
):
    prompt = (
        "PROPERTY:\n"
        "Type: Flat\n"
        "\n"
        "DECORATING:\n"
        "Paint walls throughout\n"
        "\n"
        "FLOORING:\n"
        "Lay engineered timber flooring\n"
    )
    extracted_scope = extract_scope(prompt)
    payload = build_preview_user_payload(
        prompt,
        [
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": "inside-decorating",
                    "WORKGUID": "work-decorating",
                    "WorkItemCode": "DEC-01",
                    "WorkName": "General finish package",
                    "WorkGroupName": "Decorations",
                    "Unit": "m2",
                    "WorkQTYforNorm": 1,
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            ),
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": "inside-flooring",
                    "WORKGUID": "work-flooring",
                    "WorkItemCode": "FLOOR-01",
                    "WorkName": "Engineered floor fitting",
                    "WorkGroupName": "Flooring",
                    "Unit": "m2",
                    "WorkQTYforNorm": 1,
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            ),
        ],
        extracted_scope,
    )

    assert payload["candidate_row_key_legend"]["wc"] == "work_code"
    assert payload["candidate_rows"][0]["wc"] == "DEC-01"
    assert payload["candidate_rows"][0]["ss"] == "decorating"
    assert payload["candidate_rows"][1]["ss"] == "flooring"
    assert {section["title"] for section in payload["active_scope_sections"]} == {
        "Decorating",
        "Flooring",
    }


def test_preview_user_payload_can_include_accepted_examples() -> None:
    payload = build_preview_user_payload(
        "Kitchen refresh",
        [
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": "inside-1",
                    "WORKGUID": "work-1",
                    "WorkItemCode": "PAINT-01",
                    "WorkName": "Paint walls",
                    "WorkGroupName": "Decorating",
                    "Unit": "m2",
                    "WorkQTYforNorm": 1,
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        ],
        accepted_examples=[
            PreviewAcceptedExample(
                prompt="Previous kitchen example",
                prompt_template_title="Kitchen baseline",
                accepted_rows=[
                    PreviewAcceptedExampleRow(
                        work_name="Paint walls",
                        unit="m2",
                        area="Kitchen",
                        quantity=42,
                        section_title="Decorating",
                    )
                ],
            )
        ],
    )

    assert (
        payload["accepted_examples"][0]["prompt_template_title"] == "Kitchen baseline"
    )
    assert payload["accepted_examples"][0]["accepted_rows"][0]["quantity"] == 42


@pytest.mark.parametrize(
    ("segment", "expected_quantity", "expected_unit"),
    [
        ("Install skirting 12 lm", 12, "m"),
        ("Fit architrave 8 linear metre", 8, "m"),
        ("Install window board 4 linear meter", 4, "m"),
    ],
)
def test_parse_quantity_and_unit_supports_linear_metre_aliases(
    segment: str,
    expected_quantity: float,
    expected_unit: str,
) -> None:
    quantity, unit = parse_quantity_and_unit(segment)

    assert quantity == expected_quantity
    assert unit == expected_unit


def test_preview_user_payload_includes_guardrails_and_scope_hints() -> None:
    prompt = (
        "PROPERTY:\n"
        "Type: Flat\n\n"
        "ELECTRICAL:\n"
        "Kitchen: 8 downlights\n\n"
        "JOINERY:\n"
        "Bespoke media wall\n"
    )
    extracted_scope = extract_scope(prompt)
    payload = build_preview_user_payload(
        prompt,
        [
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": "joinery-row",
                    "WORKGUID": "work-joinery",
                    "WorkItemCode": "JOIN-01",
                    "WorkName": "Install bespoke media wall",
                    "WorkGroupName": "Joinery",
                    "Unit": "pcs",
                    "WorkQTYforNorm": 1,
                    "GuardrailCode": "review_required_specialist_scope",
                    "GuardrailStatus": "Pricing review",
                    "GuardrailDetail": "Bespoke joinery needs manual pricing review.",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            ),
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": "electrical-row",
                    "WORKGUID": "work-electrical",
                    "WorkItemCode": "ELEC-DL-01",
                    "WorkName": "Install LED downlight",
                    "WorkGroupName": "Electrical",
                    "Unit": "pcs",
                    "WorkQTYforNorm": 1,
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            ),
        ],
        extracted_scope,
    )

    joinery_row = next(
        row for row in payload["candidate_rows"] if row["id"] == "joinery-row"
    )
    electrical_row = next(
        row for row in payload["candidate_rows"] if row["id"] == "electrical-row"
    )

    assert joinery_row["gc"] == "review_required_specialist_scope"
    assert joinery_row["gd"] == "Bespoke joinery needs manual pricing review."
    assert joinery_row["ss"] == "joinery"
    assert joinery_row["sst"] == "Joinery"
    assert electrical_row["ss"] == "electrical"


def test_preview_system_prompt_includes_norm_basis_and_bathroom_examples() -> None:
    prompt_text = build_preview_system_prompt()

    assert (
        'work_qty_norm_basis example: if a "Paint walls" row has work_qty_norm_basis=14.4'
        in prompt_text
    )
    assert "Install WC" in prompt_text
    assert "Full-height wall tiling to bathroom" in prompt_text
    assert "Solar panels are not mentioned in the scope." in prompt_text
    assert '"NeedsReview": true' in prompt_text
    assert (
        "Gas Safe registered engineer required. Confirm pricing separately."
        in prompt_text
    )


def test_shortlist_expands_demolition_synonyms_for_removal_scope() -> None:
    shortlist_input = [
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "remove-partition",
                "WORKGUID": "work-remove-partition",
                "WorkItemCode": "remove-partition-wall",
                "WorkName": "Remove partition wall",
                "WorkGroupName": "Demolition",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 1,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "paint-wall",
                "WORKGUID": "work-paint-wall",
                "WorkItemCode": "paint-wall",
                "WorkName": "Paint wall",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
    ]

    shortlisted = shortlist_candidates(
        "Demolish internal partition wall 12 m2",
        shortlist_input,
        limit=3,
        per_segment_limit=2,
    )

    assert shortlisted[0].INSIDEQUOTESGUID == "remove-partition"


def test_shortlist_demotes_kitchen_unit_removal_for_internal_wall_scope() -> None:
    shortlist_input = [
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "remove-partition",
                "WORKGUID": "work-remove-partition",
                "WorkItemCode": "remove-partition",
                "WorkName": "Remove stud partition; including finishings",
                "WorkGroupName": "Demolition / Stripping",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 1,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "remove-kitchen-wall-units",
                "WORKGUID": "work-remove-kitchen-wall-units",
                "WorkItemCode": "remove-kitchen-wall-units",
                "WorkName": "Remove kitchen wall and  floor units",
                "WorkGroupName": "Demolition / Stripping",
                "Unit": "each",
                "WorkLabourCost": 10,
                "WorkMatCost": 1,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
    ]

    shortlisted = shortlist_candidates(
        "Remove internal walls 15 lm", shortlist_input, limit=2, per_segment_limit=2
    )

    assert shortlisted[0].INSIDEQUOTESGUID == "remove-partition"


def test_shortlist_expands_install_and_heating_synonyms() -> None:
    shortlist_input = [
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "install-radiator",
                "WORKGUID": "work-install-radiator",
                "WorkItemCode": "install-radiator",
                "WorkName": "Install radiator",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "WorkLabourCost": 95,
                "WorkMatCost": 35,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "paint-radiator",
                "WORKGUID": "work-paint-radiator",
                "WorkItemCode": "paint-radiator",
                "WorkName": "Paint radiator",
                "WorkGroupName": "Decorations",
                "Unit": "pcs",
                "WorkLabourCost": 20,
                "WorkMatCost": 5,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
    ]

    shortlisted = shortlist_candidates(
        "Heating plant replacement, fit new radiator 2 pcs",
        shortlist_input,
        limit=3,
        per_segment_limit=2,
    )

    assert shortlisted[0].INSIDEQUOTESGUID == "install-radiator"


def test_shortlist_matches_heating_plant_to_combi_boiler_supply_and_fit() -> None:
    shortlist_input = [
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "boiler-fit",
                "WORKGUID": "work-boiler-fit",
                "WorkItemCode": "boiler-fit",
                "WorkName": "Combi boiler supply and fit",
                "WorkGroupName": "Heating",
                "Unit": "pcs",
                "WorkLabourCost": 500,
                "WorkMatCost": 1200,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
        CandidateRow.model_validate(
            {
                "INSIDEQUOTESGUID": "kitchen-generic",
                "WORKGUID": "work-kitchen-generic",
                "WorkItemCode": "kitchen-generic",
                "WorkName": "Kitchen refurbishment item",
                "WorkGroupName": "General",
                "Unit": "item",
                "WorkLabourCost": 100,
                "WorkMatCost": 50,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ),
    ]

    shortlisted = shortlist_candidates(
        "Heating plant replacement with new boiler installation",
        shortlist_input,
        limit=2,
        per_segment_limit=2,
    )

    assert shortlisted[0].INSIDEQUOTESGUID == "boiler-fit"


def test_shortlist_guarantees_minimum_rows_for_active_sections() -> None:
    shortlist_input: list[CandidateRow] = []
    for index in range(12):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"electrical-{index}",
                    "WORKGUID": f"work-electrical-{index}",
                    "WorkItemCode": f"ELEC-{index}",
                    "WorkName": f"Electrical scope row {index}",
                    "WorkGroupName": "Electrical",
                    "Unit": "pcs",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )
    for index in range(14):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"decorating-{index}",
                    "WORKGUID": f"work-decorating-{index}",
                    "WorkItemCode": f"DEC-{index}",
                    "WorkName": f"Decorating scope row {index}",
                    "WorkGroupName": "Decorating",
                    "Unit": "m2",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )

    shortlisted = shortlist_candidates(
        "DECORATING:\nPaint walls throughout\n\nELECTRICAL:\nInstall sockets and downlights",
        shortlist_input,
        limit=20,
        per_segment_limit=4,
    )

    electrical_count = sum(
        1 for row in shortlisted if row.WorkGroupName == "Electrical"
    )
    decorating_count = sum(
        1 for row in shortlisted if row.WorkGroupName == "Decorating"
    )
    assert electrical_count >= 8
    assert decorating_count >= 8


def test_shortlist_guarantees_minimum_rows_for_alias_group_sections() -> None:
    shortlist_input: list[CandidateRow] = []
    for index in range(12):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"electrics-{index}",
                    "WORKGUID": f"work-electrics-{index}",
                    "WorkItemCode": f"ELEC-{index}",
                    "WorkName": f"Socket package row {index}",
                    "WorkGroupName": "Electrics",
                    "Unit": "pcs",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )
    for index in range(12):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"decorations-{index}",
                    "WORKGUID": f"work-decorations-{index}",
                    "WorkItemCode": f"DEC-{index}",
                    "WorkName": f"Paint package row {index}",
                    "WorkGroupName": "Decorations",
                    "Unit": "m2",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )

    shortlisted = shortlist_candidates(
        "DECORATING:\nPaint walls throughout\n\nELECTRICAL:\nInstall sockets and downlights",
        shortlist_input,
        limit=20,
        per_segment_limit=4,
    )

    electrical_count = sum(1 for row in shortlisted if row.WorkGroupName == "Electrics")
    decorating_count = sum(
        1 for row in shortlisted if row.WorkGroupName == "Decorations"
    )
    assert electrical_count >= 8
    assert decorating_count >= 8


def test_shortlist_allows_more_rows_from_active_section_groups() -> None:
    shortlist_input: list[CandidateRow] = []
    for index in range(30):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"electrical-{index}",
                    "WORKGUID": f"work-electrical-{index}",
                    "WorkItemCode": f"ELEC-{index}",
                    "WorkName": f"Electrical install row {index}",
                    "WorkGroupName": "Electrical",
                    "Unit": "pcs",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )
    for index in range(5):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"decorating-{index}",
                    "WORKGUID": f"work-decorating-{index}",
                    "WorkItemCode": f"DEC-{index}",
                    "WorkName": f"Decorating fallback row {index}",
                    "WorkGroupName": "Decorating",
                    "Unit": "m2",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )

    shortlisted = shortlist_candidates(
        "ELECTRICAL:\nFull rewire with sockets, downlights, switches, fans and hardwired appliances",
        shortlist_input,
        limit=30,
        per_segment_limit=4,
    )

    electrical_count = sum(
        1 for row in shortlisted if row.WorkGroupName == "Electrical"
    )
    assert electrical_count > 20


def test_shortlist_guarantees_minimum_rows_for_extended_active_sections() -> None:
    shortlist_input: list[CandidateRow] = []
    for index in range(12):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"heating-{index}",
                    "WORKGUID": f"work-heating-{index}",
                    "WorkItemCode": f"HEAT-{index}",
                    "WorkName": f"Heating scope row {index}",
                    "WorkGroupName": "Heating",
                    "Unit": "pcs",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )
    for index in range(12):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"windows-{index}",
                    "WORKGUID": f"work-windows-{index}",
                    "WorkItemCode": f"WIN-{index}",
                    "WorkName": f"Windows scope row {index}",
                    "WorkGroupName": "Windows",
                    "Unit": "m2",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )

    shortlisted = shortlist_candidates(
        "HEATING:\nReplace combi boiler and radiators\n\nWINDOWS:\nReplace windows and rooflights",
        shortlist_input,
        limit=20,
        per_segment_limit=4,
    )

    heating_count = sum(1 for row in shortlisted if row.WorkGroupName == "Heating")
    windows_count = sum(1 for row in shortlisted if row.WorkGroupName == "Windows")
    assert heating_count >= 8
    assert windows_count >= 8


def test_dynamic_shortlist_limit_scales_for_multi_section_prompts() -> None:
    prompt = (
        "DEMOLITION:\nStrip out ground floor\n\n"
        "STRUCTURE:\nForm new openings\n\n"
        "PLASTERING:\nSkim walls\n\n"
        "ELECTRICAL:\nRewire kitchen\n\n"
        "PLUMBING:\nReplace radiators\n\n"
        "DECORATING:\nPaint throughout\n\n"
        "FLOORING:\nLay engineered floor"
    )

    assert app_matcher._resolve_shortlist_limit(prompt, None) == 220


def test_dynamic_shortlist_limit_scales_with_large_candidate_sets() -> None:
    prompt = (
        "DEMOLITION:\nStrip out ground floor\n\n"
        "STRUCTURE:\nForm new openings\n\n"
        "PLASTERING:\nSkim walls\n\n"
        "ELECTRICAL:\nRewire kitchen\n\n"
        "PLUMBING:\nReplace radiators\n\n"
        "DECORATING:\nPaint throughout\n\n"
        "FLOORING:\nLay engineered floor"
    )

    assert (
        app_matcher._resolve_shortlist_limit(prompt, None, candidate_count=1000) == 305
    )


def test_budget_fit_trims_large_normalized_preview_payload() -> None:
    normalized_prompt = (
        "PROPERTY:\n"
        "Type: First floor flat\n\n"
        "FLOOR AREAS & ROOMS:\n"
        "First Floor\n"
        "Kitchen: 4 x 3\n"
        "Rear room: 5 x 4\n\n"
        "DEMOLITION:\n"
        "Strip out throughout\n\n"
        "ELECTRICAL:\n"
        "Rewire kitchen and rear room\n\n"
        "PLUMBING:\n"
        "Replace bathroom sanitaryware\n\n"
        "FLOORING:\n"
        "Lay engineered flooring\n"
    )
    extracted_scope = extract_scope(normalized_prompt)
    shortlist_input: list[CandidateRow] = []
    for index in range(320):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"candidate-{index}",
                    "WORKGUID": f"work-{index}",
                    "WorkItemCode": f"ROW-{index}",
                    "WorkName": (
                        f"Detailed refurbishment row {index} for kitchen, flooring, "
                        "electrical, plumbing, demolition and finish coordination"
                    ),
                    "WorkGroupName": "Electrical" if index % 2 == 0 else "Flooring",
                    "Unit": "m2" if index % 3 else "pcs",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )

    trimmed, budget_assumption = app_matcher._fit_shortlist_to_budget(
        prompt=normalized_prompt,
        shortlist=shortlist_input,
        extracted_scope=extracted_scope,
        accepted_examples=[],
    )

    assert len(trimmed) < len(shortlist_input)
    assert budget_assumption is not None
    assert (
        app_matcher._estimate_preview_message_chars(
            prompt=normalized_prompt,
            candidate_rows=trimmed,
            extracted_scope=extracted_scope,
            accepted_examples=[],
        )
        <= app_matcher._PREVIEW_ROUGH_CHAR_BUDGET
    )


def test_fit_accepted_examples_to_budget_can_drop_examples_to_preserve_shortlist() -> (
    None
):
    normalized_prompt = (
        "PROPERTY:\n"
        "Type: First floor flat\n\n"
        "FLOOR AREAS & ROOMS:\n"
        "First Floor\n"
        "Kitchen: 4 x 3\n"
        "Bathroom: 2 x 2\n"
        "Rear room: 4 x 5\n\n"
        "PLUMBING:\n"
        "Install boiler\n"
        "Install electric underfloor heating\n"
    )
    extracted_scope = extract_scope(normalized_prompt)
    shortlist_input: list[CandidateRow] = []
    for index in range(180):
        shortlist_input.append(
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": f"candidate-{index}",
                    "WORKGUID": f"work-{index}",
                    "WorkItemCode": f"ROW-{index}",
                    "WorkName": f"Detailed refurbishment row {index} for kitchen, plumbing, electrical, flooring and finishes",
                    "WorkGroupName": "Plumbing" if index % 2 == 0 else "Electrical",
                    "Unit": "m2" if index % 3 else "pcs",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        )

    accepted_examples = [
        PreviewAcceptedExample.model_validate(
            {
                "prompt": "Structured prompt example "
                + (
                    "boiler and electrical scope with sequencing, client-supply notes and room breakdown "
                    * 180
                ),
                "prompt_template_title": "Heating baseline",
                "accepted_rows": [
                    PreviewAcceptedExampleRow.model_validate(
                        {
                            "work_name": "Install boiler with client-supplied appliance handover and coordination",
                            "unit": "pcs",
                            "area": "Plant room",
                            "quantity": 1,
                            "section_title": "Heating",
                        }
                    )
                ]
                * 40,
            }
        ),
        PreviewAcceptedExample.model_validate(
            {
                "prompt": "Structured prompt example "
                + (
                    "bathroom and flooring scope with UFH zones, tile allowances and electrical points "
                    * 180
                ),
                "prompt_template_title": "Bathroom baseline",
                "accepted_rows": [
                    PreviewAcceptedExampleRow.model_validate(
                        {
                            "work_name": "Install electric underfloor heating with thermostat and commissioning",
                            "unit": "pcs",
                            "area": "Bathroom",
                            "quantity": 1,
                            "section_title": "Electrical",
                        }
                    )
                ]
                * 40,
            }
        ),
    ]

    trimmed_examples, assumption = app_matcher._fit_accepted_examples_to_budget(
        prompt=normalized_prompt,
        shortlist=shortlist_input,
        extracted_scope=extracted_scope,
        accepted_examples=accepted_examples,
    )

    assert len(trimmed_examples) < len(accepted_examples)
    assert assumption is not None


def test_match_row_to_extracted_section_prefers_candidate_section_aliases() -> None:
    extracted_scope = extract_scope(
        "PROPERTY:\n"
        "Type: First floor flat\n\n"
        "FLOOR AREAS & ROOMS:\n"
        "First Floor\n"
        "Kitchen: 4 x 3\n\n"
        "FLOORING:\n"
        "Engineered floor build-up\n\n"
        "ELECTRICAL:\n"
        "General: 6 switches\n"
    )
    row = CandidateRow.model_validate(
        {
            "INSIDEQUOTESGUID": "switch-row",
            "WORKGUID": "switch-work",
            "WorkItemCode": "ELEC-SW-01",
            "WorkName": "Install light switch",
            "WorkGroupName": "Electrics",
            "Unit": "pcs",
            "WorkLabourCost": 10,
            "WorkMatCost": 4,
            "WorkOtherCost": 1,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        }
    )

    section_match = match_row_to_extracted_section(
        row, extracted_scope, "General: 6 switches"
    )

    assert section_match is not None
    assert section_match.key == "electrical"


def test_build_extracted_scope_logs_parser_failures(monkeypatch, caplog) -> None:
    def raise_parser_error(_prompt: str):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(app_matcher, "extract_scope", raise_parser_error)

    with caplog.at_level("ERROR", logger=app_matcher.__name__):
        extracted_scope = app_matcher._build_extracted_scope("DEMOLITION:\nStrip out")

    assert extracted_scope is None
    assert (
        "Structured scope extraction failed during preview generation." in caplog.text
    )
    assert "parser exploded" in caplog.text


def test_direct_scope_support_allows_generic_rows_without_significant_tokens() -> None:
    row = CandidateRow.model_validate(
        {
            "INSIDEQUOTESGUID": "generic-allowance",
            "WORKGUID": "work-generic-allowance",
            "WorkItemCode": "ALLOW-01",
            "WorkName": "General preliminaries allowance",
            "WorkGroupName": "",
            "Unit": "item",
            "WorkLabourCost": 10,
            "WorkMatCost": 4,
            "WorkOtherCost": 1,
            "WorkQTYforNorm": 1,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        }
    )

    assert app_matcher._has_direct_scope_support(
        row,
        scope_text="SITE COSTS:\nGeneral preliminaries allowance",
        extracted_scope=None,
        takeoff_suggestion=None,
    )


def test_llm_preview_recovers_missing_quantity_from_takeoff_before_validation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fake_llm_output = {
        "matched_rows": [
            {
                "INSIDEQUOTESGUID": "paint-walls",
                "AREA": "Kitchen",
                "QUANTITY": None,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.88,
                "NeedsReview": False,
                "ReviewReason": None,
            }
        ],
        "unmatched_items": [],
        "assumptions": [],
    }

    class FakeClient:
        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            from app.schemas import LLMPreviewOutput

            return LLMPreviewOutput.model_validate(fake_llm_output)

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-takeoff-rescue",
        prompt=(
            "PROPERTY:\n"
            "Type: Flat\n\n"
            "FLOOR AREAS & ROOMS:\n"
            "Ground Floor\n"
            "Kitchen: 4.5 x 3.2\n\n"
            "DECORATING:\n"
            "Paint kitchen walls"
        ),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "paint-walls",
                "WORKGUID": "work-1",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorating",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.service_mode == "llm"
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].INSIDEQUOTESGUID == "paint-walls"
    assert response.matched_rows[0].QUANTITY > 0
    assert response.matched_rows[0].AREA == "Kitchen"


@pytest.mark.parametrize(
    "case",
    load_eval_cases(),
    ids=lambda case: case["name"],
)
def test_preview_eval_cases(monkeypatch, case) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    request = PreviewRequest(
        quote_guid=f"eval-{case['name']}",
        prompt=case["prompt"],
        candidate_rows=case["candidate_rows"],
    )

    response = generate_preview(request)

    matched_ids = [row.INSIDEQUOTESGUID for row in response.matched_rows]
    warning_text = " ".join(
        assumption.text
        for assumption in response.assumptions
        if assumption.severity == "warning"
    )

    assert set(matched_ids) == set(case["expected_matched_ids"])
    if "expected_unmatched_count" in case:
        assert len(response.unmatched_items) == case["expected_unmatched_count"]
    if "expected_review_count" in case:
        assert response.review_summary.review_count == case["expected_review_count"]
    for expected_fragment in case["expected_warning_contains"]:
        assert expected_fragment in warning_text


def test_llm_preview_skips_invalid_guid_row_instead_of_crashing(monkeypatch) -> None:
    """If the LLM returns an INSIDEQUOTESGUID that does not exist in the shortlist,
    the row should be added to unmatched_items and the rest of the response should
    still be returned normally — no exception, no mock fallback."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fake_llm_output = {
        "matched_rows": [
            {
                "INSIDEQUOTESGUID": "does-not-exist",
                "AREA": "",
                "QUANTITY": 10,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.9,
                "NeedsReview": False,
                "ReviewReason": None,
            },
            {
                "INSIDEQUOTESGUID": "valid-guid-1",
                "AREA": "Kitchen",
                "QUANTITY": 5,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.92,
                "NeedsReview": False,
                "ReviewReason": None,
            },
        ],
        "unmatched_items": [],
        "assumptions": [],
    }

    class FakeClient:
        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            from app.schemas import LLMPreviewOutput

            return LLMPreviewOutput.model_validate(fake_llm_output)

    # Patch AFTER reload so the patch is not wiped by reload.
    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-invalid-guid",
        prompt="Paint kitchen walls 50 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "valid-guid-1",
                "WORKGUID": "work-1",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.error_text == ""
    assert response.service_mode == "llm"
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].INSIDEQUOTESGUID == "valid-guid-1"
    assert len(response.unmatched_items) == 1
    assert "does-not-exist" in response.unmatched_items[0].source_text
    assert response.telemetry.llm_attempt_count == 1


def test_llm_preview_rejects_rows_without_direct_scope_support(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fake_llm_output = {
        "matched_rows": [
            {
                "INSIDEQUOTESGUID": "handrail-row",
                "AREA": "Kitchen",
                "QUANTITY": 1,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.91,
                "NeedsReview": False,
                "ReviewReason": None,
            }
        ],
        "unmatched_items": [],
        "assumptions": [],
    }

    class FakeClient:
        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            from app.schemas import LLMPreviewOutput

            return LLMPreviewOutput.model_validate(fake_llm_output)

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-direct-scope-support",
        prompt="FLOORING:\nInstall laminate floor to kitchen 20 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "handrail-row",
                "WORKGUID": "work-handrail",
                "WorkItemCode": "JOIN-HANDRAIL-01",
                "WorkName": "Install handrail",
                "WorkGroupName": "Joinery",
                "Unit": "pcs",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.service_mode == "llm"
    assert response.matched_rows == []
    assert len(response.unmatched_items) == 1
    assert response.unmatched_items[0].source_text == "Install handrail"
    assert "not directly supported" in response.unmatched_items[0].reason.lower()
    assert any(
        assumption.kind == "scope_support_missing"
        for assumption in response.assumptions
    )


def test_llm_preview_forces_do_not_include_rows_to_unmatched(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fake_llm_output = {
        "matched_rows": [
            {
                "INSIDEQUOTESGUID": "gas-cooker-row",
                "AREA": "",
                "QUANTITY": 1,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.95,
                "NeedsReview": False,
                "ReviewReason": None,
            }
        ],
        "unmatched_items": [],
        "assumptions": [],
    }

    class FakeClient:
        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            from app.schemas import LLMPreviewOutput

            return LLMPreviewOutput.model_validate(fake_llm_output)

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-do-not-include",
        prompt="KITCHEN:\nInstall gas cooker",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "gas-cooker-row",
                "WORKGUID": "work-gas-cooker",
                "WorkItemCode": "GAS-01",
                "WorkName": "Install gas cooker",
                "WorkGroupName": "Plumbing",
                "Unit": "pcs",
                "GuardrailCode": "do_not_include",
                "GuardrailDetail": "Gas appliance connection must be excluded from the AI draft.",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.service_mode == "llm"
    assert response.matched_rows == []
    assert len(response.unmatched_items) == 1
    assert response.unmatched_items[0].source_text == "Install gas cooker"
    assert "guardrail excluded" in response.assumptions[0].text.lower()


def test_llm_preview_forces_review_for_human_guardrail_status_labels(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fake_llm_output = {
        "matched_rows": [
            {
                "INSIDEQUOTESGUID": "bespoke-joinery-row",
                "AREA": "Living room",
                "QUANTITY": 1,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.93,
                "NeedsReview": False,
                "ReviewReason": None,
            }
        ],
        "unmatched_items": [],
        "assumptions": [],
    }

    class FakeClient:
        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            from app.schemas import LLMPreviewOutput

            return LLMPreviewOutput.model_validate(fake_llm_output)

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-guardrail-label",
        prompt="JOINERY:\nInstall bespoke media wall",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "bespoke-joinery-row",
                "WORKGUID": "work-bespoke-joinery",
                "WorkItemCode": "JOIN-01",
                "WorkName": "Install bespoke media wall",
                "WorkGroupName": "Joinery",
                "Unit": "pcs",
                "GuardrailStatus": "Pricing review",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.service_mode == "llm"
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].NeedsReview is True
    assert response.matched_rows[0].Confidence == 0.74
    assert (
        "pricing review due to guardrail status"
        in (response.matched_rows[0].ReviewReason or "").lower()
    )
    assert any(
        assumption.kind == "guardrail_review" for assumption in response.assumptions
    )


def test_llm_preview_fallback_captures_retry_telemetry(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    class FakeClient:
        def __init__(self):
            self.last_preview_metadata = {
                "llm_attempt_count": 2,
                "llm_latency_ms": 0,
                "parse_retry_used": True,
                "structured_output_mode": "responses.parse",
            }

        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            raise ValueError("Structured output parse failed twice.")

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-fallback",
        prompt="Paint kitchen walls 50 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "valid-guid-1",
                "WORKGUID": "work-1",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.service_mode == "mock-fallback"
    assert response.telemetry.candidate_row_count == 1
    assert response.telemetry.shortlist_size == 1
    assert response.telemetry.llm_attempt_count == 2
    assert response.telemetry.parse_retry_used is True
    assert response.telemetry.structured_output_mode == "responses.parse"
    assert "ValueError" in response.telemetry.fallback_reason


def test_llm_preview_does_not_silent_fallback_outside_local(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    class FakeClient:
        def __init__(self):
            self.last_preview_metadata = {
                "llm_attempt_count": 2,
                "llm_latency_ms": 0,
                "parse_retry_used": True,
                "structured_output_mode": "responses.parse",
            }

        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            raise ValueError("Structured output parse failed twice.")

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-prod-no-fallback",
        prompt="Paint kitchen walls 50 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "valid-guid-1",
                "WORKGUID": "work-1",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="mock fallback is disabled"):
        app_matcher.generate_preview(request)


def test_llm_preview_uses_marked_fallback_for_normalized_intake_in_local(
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    class FakeClient:
        def __init__(self):
            self.last_preview_metadata = {
                "llm_attempt_count": 1,
                "llm_latency_ms": 0,
                "parse_retry_used": False,
                "structured_output_mode": "responses.parse",
            }

        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            raise ValueError("Structured output parse failed.")

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    normalized_prompt = (
        "PROPERTY:\n"
        "Type: First floor flat\n\n"
        "FLOOR AREAS & ROOMS:\n"
        "First Floor\n"
        "Kitchen: 4 x 3\n\n"
        "DECORATING:\n"
        "Paint walls throughout\n"
    )
    request = PreviewRequest(
        quote_guid="quote-normalized-no-fallback",
        prompt="client raw prompt",
        normalized_prompt_markdown=normalized_prompt,
        normalized_scope=extract_scope(normalized_prompt),
        rules_context={
            "rule_version": "2026-04-27",
            "canonical_source": "backend_registry",
            "runtime_source": "code_defaults",
            "published_version": "",
            "published_at": "",
        },
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "valid-guid-1",
                "WORKGUID": "work-1",
                "WorkItemCode": "paint-walls",
                "WorkName": "Paint walls",
                "WorkGroupName": "Decorations",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.service_mode == "mock-fallback"
    assert "ValueError" in response.telemetry.fallback_reason
    assert response.telemetry.takeoff_heuristics_version == TAKEOFF_HEURISTICS_VERSION
    assert response.telemetry.takeoff_heuristics_source == TAKEOFF_HEURISTICS_SOURCE
    assert response.telemetry.rule_version == "2026-04-27"
    assert response.telemetry.runtime_source == "code_defaults"
    assert any(
        assumption.kind == "normalized_service_fallback"
        and "heuristic fallback mode" in (assumption.text or "").lower()
        for assumption in response.assumptions
    )


def test_llm_preview_retries_with_smaller_shortlist_after_token_budget_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fake_llm_output = {
        "matched_rows": [
            {
                "INSIDEQUOTESGUID": "candidate-0",
                "AREA": "Kitchen",
                "QUANTITY": 12,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.88,
                "NeedsReview": False,
                "ReviewReason": None,
            }
        ],
        "unmatched_items": [],
        "assumptions": [],
    }
    preview_calls: list[int] = []

    class FakeClient:
        def __init__(self):
            self.last_preview_metadata = {
                "llm_attempt_count": 1,
                "llm_latency_ms": 0,
                "parse_retry_used": False,
                "structured_output_mode": "responses.parse",
            }

        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            from app.schemas import LLMPreviewOutput

            preview_calls.append(len(candidate_rows))
            if len(preview_calls) == 1:
                raise ValueError(
                    "Request too large for gpt-4o in organization test on tokens per min (TPM): "
                    "Limit 30000, Requested 37279."
                )
            return LLMPreviewOutput.model_validate(fake_llm_output)

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())
    monkeypatch.setattr(
        app_matcher, "_resolve_shortlist_limit", lambda *args, **kwargs: 220
    )
    monkeypatch.setattr(app_matcher, "_PREVIEW_ROUGH_TOKEN_BUDGET", 999999)
    monkeypatch.setattr(app_matcher, "_PREVIEW_ROUGH_CHAR_BUDGET", 999999 * 4)

    normalized_prompt = (
        "PROPERTY:\n"
        "Type: First floor flat\n\n"
        "FLOOR AREAS & ROOMS:\n"
        "First Floor\n"
        "Kitchen: 4 x 3\n\n"
        "DEMOLITION:\n"
        "Strip out throughout\n\n"
        "ELECTRICAL:\n"
        "Install switches and sockets\n"
    )
    request = PreviewRequest(
        quote_guid="quote-normalized-token-retry",
        prompt="client raw prompt",
        normalized_prompt_markdown=normalized_prompt,
        normalized_scope=extract_scope(normalized_prompt),
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": f"candidate-{index}",
                "WORKGUID": f"work-{index}",
                "WorkItemCode": f"ROW-{index}",
                "WorkName": (
                    f"Detailed refurbishment row {index} for kitchen, flooring, "
                    "electrical, plumbing, demolition and finish coordination"
                ),
                "WorkGroupName": "Electrical" if index % 2 == 0 else "Flooring",
                "Unit": "m2" if index % 3 else "pcs",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
            for index in range(260)
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.service_mode == "llm"
    assert preview_calls[0] == 220
    assert preview_calls[1] < preview_calls[0]
    assert any(
        assumption.kind == "token_budget_trim" for assumption in response.assumptions
    )


def test_direct_scope_support_keeps_real_construction_terms() -> None:
    row = CandidateRow.model_validate(
        {
            "INSIDEQUOTESGUID": "paint-row",
            "WORKGUID": "work-paint",
            "WorkItemCode": "DEC-01",
            "WorkName": "Paint walls",
            "WorkGroupName": "Decorating",
            "Unit": "m2",
            "WorkLabourCost": 10,
            "WorkMatCost": 4,
            "WorkOtherCost": 1,
            "WorkQTYforNorm": 14.4,
            "PROFIT": 20,
            "LabourMarkup": 15,
            "MaterialMarkup": 10,
        }
    )

    assert app_matcher._has_direct_scope_support(
        row,
        scope_text="DECORATING:\nPaint walls and ceilings throughout",
        extracted_scope=None,
        takeoff_suggestion=None,
    )


def test_llm_preview_reports_missing_quantity_with_human_readable_unmatched_item(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    reload(app_config)
    reload(app_llm_client)
    reload(app_matcher)

    fake_llm_output = {
        "matched_rows": [
            {
                "INSIDEQUOTESGUID": "tile-wall-row",
                "AREA": "Bathroom",
                "QUANTITY": None,
                "PROFIT": None,
                "LabourMarkup": None,
                "MaterialMarkup": None,
                "Confidence": 0.82,
                "NeedsReview": True,
                "ReviewReason": "Need wall tiling measure.",
            }
        ],
        "unmatched_items": [],
        "assumptions": [],
    }

    class FakeClient:
        def is_enabled(self) -> bool:
            return True

        def preview_match(
            self,
            prompt,
            candidate_rows,
            extracted_scope=None,
            accepted_examples=None,
            document_context=None,
        ):
            from app.schemas import LLMPreviewOutput

            return LLMPreviewOutput.model_validate(fake_llm_output)

    monkeypatch.setattr(app_matcher, "LLMClient", lambda: FakeClient())

    request = PreviewRequest(
        quote_guid="quote-missing-qty",
        prompt="TILING:\nWall tiling to bathroom",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "tile-wall-row",
                "WORKGUID": "work-wall-tile",
                "WorkItemCode": "TILE-01",
                "WorkName": "Wall tiling",
                "WorkGroupName": "Tiling",
                "Unit": "m2",
                "WorkLabourCost": 12,
                "WorkMatCost": 6,
                "WorkOtherCost": 0,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = app_matcher.generate_preview(request)

    assert response.matched_rows == []
    assert response.unmatched_items[0].source_text == "Wall tiling"
    assert "quantity" in response.unmatched_items[0].reason.lower()
    assert any(
        assumption.kind == "quantity_missing" for assumption in response.assumptions
    )


def test_llm_client_falls_back_to_chat_parse_when_responses_api_is_missing() -> None:
    parsed_output = app_llm_client.LLMPreviewOutput.model_validate(
        {
            "matched_rows": [
                {
                    "INSIDEQUOTESGUID": "inside-1",
                    "AREA": "Kitchen",
                    "QUANTITY": 18,
                    "PROFIT": None,
                    "LabourMarkup": None,
                    "MaterialMarkup": None,
                    "Confidence": 0.92,
                    "NeedsReview": False,
                    "ReviewReason": None,
                }
            ],
            "unmatched_items": [],
            "assumptions": [],
        }
    )

    class FakeParsedMessage:
        def __init__(self, parsed):
            self.parsed = parsed

    class FakeChoice:
        def __init__(self, parsed):
            self.message = FakeParsedMessage(parsed)

    class FakeChatCompletions:
        def parse(self, **kwargs):
            return type("FakeResponse", (), {"choices": [FakeChoice(parsed_output)]})()

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeBeta:
        def __init__(self):
            self.chat = FakeChat()

    class FakeOpenAIClient:
        def __init__(self):
            self.beta = FakeBeta()

    client = app_llm_client.LLMClient()
    client.enabled = True
    client.client = FakeOpenAIClient()
    client.model = "gpt-4o-mini"

    response = client.preview_match(
        prompt="Paint kitchen walls 18 m2",
        candidate_rows=[
            CandidateRow.model_validate(
                {
                    "INSIDEQUOTESGUID": "inside-1",
                    "WORKGUID": "work-1",
                    "WorkName": "Paint walls",
                    "Unit": "m2",
                    "WorkLabourCost": 10,
                    "WorkMatCost": 4,
                    "WorkOtherCost": 1,
                    "WorkQTYforNorm": 1,
                    "PROFIT": 20,
                    "LabourMarkup": 15,
                    "MaterialMarkup": 10,
                }
            )
        ],
    )

    assert response.matched_rows[0].INSIDEQUOTESGUID == "inside-1"
    assert (
        client.last_preview_metadata["structured_output_mode"]
        == "chat.completions.parse"
    )
