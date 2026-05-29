from __future__ import annotations

from importlib import reload

from fastapi.testclient import TestClient

import app.config as app_config
import app.security as app_security
from app.main import app
from app.schemas import PromptNormalizationAnswer, PromptNormalizationRequest
from app.services.intake_normalizer import normalize_intake_prompt


def test_normalize_prompt_infers_first_floor_flat_room_schedule() -> None:
    prompt = """
PROJECT DETAILS
Property: First floor flat, mid-terrace house (converted)
Location: Muswell Hill, London
Ceiling height: 2.4m

AREAS
Open plan (living + kitchen): 6.7 x 7.0
Bathroom: 3.0 x 2.8
Rear room: 4.0 x 5.0

WALLS
Skim + decoration only

BATHROOM
Sanitaryware: client supply (toilet, basin, shower, bath)
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.status == "ready"
    assert "FLOOR AREAS & ROOMS:" in response.normalized_prompt_markdown
    assert "First Floor" in response.normalized_prompt_markdown
    assert response.normalized_scope is not None
    assert len(response.normalized_scope.rooms) == 3
    assert all(room.level == "First Floor" for room in response.normalized_scope.rooms)


def test_normalize_prompt_requests_bathroom_fixture_clarification_when_scope_is_ambiguous() -> (
    None
):
    prompt = """
Property
Type: Flat

Full Scope
Bathroom refurbishment

TILING
Full-height wall tiling to bathroom
Floor tiling to bathroom
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.status == "needs_clarification"
    assert any(
        question.id == "bathroom_fixture_scope" for question in response.questions
    )
    assert any(
        blocker.code == "bathroom_fixture_scope" for blocker in response.blockers
    )
    assert response.missing_items


def test_normalize_prompt_splits_short_comma_scope_into_multiple_sections() -> None:
    prompt = "Kitchen renovation, paint walls 120 m2, new flooring"

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.status in {"ready", "needs_clarification"}
    assert "FULL SCOPE:" in response.normalized_prompt_markdown
    assert "DECORATING:" in response.normalized_prompt_markdown
    assert "FLOORING:" in response.normalized_prompt_markdown
    assert "Kitchen renovation" in response.normalized_prompt_markdown
    assert "paint walls 120 m2" in response.normalized_prompt_markdown
    assert "new flooring" in response.normalized_prompt_markdown


def test_normalize_prompt_does_not_ask_supply_split_for_room_schedule_only() -> None:
    prompt = """
AREAS
Kitchen: 4.5 x 3.2
Rear room: 4 x 5
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert all(question.id != "supply_split" for question in response.questions)


def test_normalize_prompt_keeps_inline_room_schedule_clauses_for_flat_prompt() -> None:
    prompt = """
Property: First floor flat
Location: Muswell Hill, London
Kitchen: 3.5 x 4.2, Living room: 4.0 x 5.0, Bathroom: 2.0 x 2.5
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.normalized_scope is not None
    assert len(response.normalized_scope.rooms) == 3
    assert {room.name for room in response.normalized_scope.rooms} == {
        "Kitchen",
        "Living room",
        "Bathroom",
    }
    assert all(room.level == "First Floor" for room in response.normalized_scope.rooms)


def test_normalize_prompt_does_not_ask_supply_split_when_scope_already_says_supply() -> (
    None
):
    prompt = """
KITCHEN
Supply kitchen units
Install kitchen units
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert all(question.id != "supply_split" for question in response.questions)


def test_normalize_prompt_does_not_ask_supply_split_when_scope_says_supply_and_fit() -> (
    None
):
    prompt = """
KITCHEN
Supply and fit kitchen units
Install kitchen worktop
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert all(question.id != "supply_split" for question in response.questions)


def test_normalize_prompt_keeps_full_house_room_schedule_without_false_heading_resets() -> (
    None
):
    prompt = """
Type: Semi-detached house
Location: Outskirts of London

FLOOR AREAS & ROOMS
Ground Floor
Formal Living: 4.5 x 3.5
Utility: 3.5 x 1.5
Kitchen Dining: 3.5 x 5.8
Hallway: 1.7 x 6.5
WC: 1.2 x 1.5
Study: 1.5 x 1.8
Rear Living: 3.5 x 3.2
Dining Room: 3.7 x 7
First Floor
Bedroom 1: 3.5 x 4.2
Bedroom 2: 4.5 x 3.5
Bedroom 3: 3.5 x 3.5
Bedroom 4: 3.5 x 3
Store: 2 x 1.2
Ensuite: 2 x 1.7
Bathroom: 2.5 x 2.5
Landing: 3 x 3
Loft
Bedroom: 5 x 3
Bathroom: 2.3 x 4.5
Landing: 2.5 x 1.3

STRUCTURE
Rear dormer: 3.5 x 5.5
Side dormer: 3 x 2.5
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.normalized_scope is not None
    assert len(response.normalized_scope.rooms) == 19
    room_names = {room.name for room in response.normalized_scope.rooms}
    assert {"WC", "Study", "Rear Living", "Dining Room"} <= room_names
    assert "Rear dormer" not in room_names
    assert "Side dormer" not in room_names


def test_normalize_prompt_keeps_context_labels_inside_electrical_section() -> None:
    prompt = """
Property
Type: First floor flat

FLOOR AREAS
Kitchen: 4 x 3
Bathroom: 2 x 2

ELECTRICS — count points as follows:
Kitchen: 15 spotlights, 10 double sockets, appliance connections
Bathroom: 6 spotlights, 1 extractor fan, UFH thermostat connection
General: 6 switches
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert "ELECTRICAL:" in response.normalized_prompt_markdown
    assert (
        "Kitchen: 15 spotlights, 10 double sockets, appliance connections"
        in response.normalized_prompt_markdown
    )
    assert (
        "Bathroom: 6 spotlights, 1 extractor fan, UFH thermostat connection"
        in response.normalized_prompt_markdown
    )
    assert (
        "FULL SCOPE:\nConnect all appliances" not in response.normalized_prompt_markdown
    )


def test_normalize_prompt_does_not_repeat_steel_clarification_after_answer() -> None:
    prompt = """
STEELWORKS
Multiple UKB beams (sizes provided)
UC columns
""".strip()

    response = normalize_intake_prompt(
        PromptNormalizationRequest(
            prompt=prompt,
            clarification_answers=[
                PromptNormalizationAnswer(
                    question_id="steel_scope_detail",
                    value="203x203x71 UC 7.0m with 2 splices, 152x152x30 UC 3.0m and 4.0m, plus 4 steel posts",
                )
            ],
        )
    )

    assert all(question.id != "steel_scope_detail" for question in response.questions)


def test_normalize_prompt_requests_scaffold_clarification_when_access_scope_is_mentioned() -> (
    None
):
    prompt = """
FULL SCOPE
Scaffolding to rear elevation for roof and render works
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.status == "needs_clarification"
    assert any(
        question.id == "scaffold_allowance_scope" for question in response.questions
    )
    assert any(
        blocker.code == "scaffold_allowance_scope" for blocker in response.blockers
    )


def test_normalize_prompt_requests_waste_access_clarification_for_restricted_access_scope() -> (
    None
):
    prompt = """
FULL SCOPE
Bag out demolition waste through the house with restricted access to the rear garden
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.status == "needs_clarification"
    assert any(question.id == "waste_access_scope" for question in response.questions)
    assert any(blocker.code == "waste_access_scope" for blocker in response.blockers)


def test_normalize_prompt_requests_permit_or_cdm_clarification_when_detected() -> None:
    prompt = """
PROPERTY
Type: House

FULL SCOPE
CDM principal contractor fee if required
Skip permit to be confirmed
""".strip()

    response = normalize_intake_prompt(PromptNormalizationRequest(prompt=prompt))

    assert response.status == "needs_clarification"
    assert any(question.id == "permit_cdm_scope" for question in response.questions)


def test_normalize_endpoint_accepts_valid_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    reload(app_config)
    reload(app_security)

    client = TestClient(app)
    response = client.post(
        "/v1/estimate/normalize",
        json={
            "prompt": "Property\nType: First floor flat\nAREAS\nKitchen: 4 x 3\nBathroom: 2 x 2",
        },
        headers={"x-api-key": "test-secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ready", "needs_clarification"}
    assert "normalized_prompt_markdown" in body
    assert "FLOOR AREAS & ROOMS:" in body["normalized_prompt_markdown"]
