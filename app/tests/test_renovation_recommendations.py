from __future__ import annotations

from app.services.renovation_recommendations_service import (
    RecommendationObject,
    RenovationRecommendationsRequest,
    RenovationRecommendationsService,
)


def _service_disabled() -> RenovationRecommendationsService:
    service = RenovationRecommendationsService()
    service.enabled = False
    service.client = None
    return service


def _simple_request(**overrides) -> RenovationRecommendationsRequest:
    defaults = dict(
        room_name="Kitchen",
        floor_area_sq_m=14.0,
        objects=[
            RecommendationObject(
                id="oven-1", category="Oven", condition="needs_replacement", condition_notes="Door seal worn through."
            ),
            RecommendationObject(id="cabinet-1", category="Cabinet", condition="worn"),
        ],
        existing_scope_items=[],
    )
    defaults.update(overrides)
    return RenovationRecommendationsRequest(**defaults)


def test_suggest_with_no_objects_returns_empty_recommendations_not_an_error():
    service = _service_disabled()
    response = service.suggest(_simple_request(objects=[]))

    assert response.recommendations == []
    assert "No objects" in response.warnings[0]


def test_suggest_falls_back_to_mock_when_llm_disabled():
    service = _service_disabled()
    response = service.suggest(_simple_request())

    assert response.mode == "mock"
    assert response.recommendations == []
    assert response.warnings


def test_mock_response_never_claims_llm_mode():
    service = _service_disabled()
    response = service._mock_response()

    assert response.mode == "mock"


def test_llm_response_drops_related_object_id_for_unknown_ids():
    from unittest.mock import MagicMock

    from app.services.renovation_recommendations_service import (
        _LLMRecommendation,
        _LLMRecommendationsOutput,
    )

    service = RenovationRecommendationsService()
    service.enabled = True

    fake_response = MagicMock()
    fake_response.output_parsed = _LLMRecommendationsOutput(
        recommendations=[
            _LLMRecommendation(
                title="Replace the oven",
                rationale="It's flagged as needing replacement.",
                priority="high",
                related_object_id="oven-1",
            ),
            _LLMRecommendation(
                title="Fix the made-up thing",
                rationale="The model invented this id.",
                priority="low",
                related_object_id="not-a-real-id",
            ),
        ]
    )
    service.client = MagicMock()
    service.client.responses.parse.return_value = fake_response

    response = service._llm_response(_simple_request())

    assert response.mode == "llm"
    assert len(response.recommendations) == 2
    assert response.recommendations[0].related_object_id == "oven-1"
    assert response.recommendations[1].related_object_id is None


def test_llm_response_corrects_invalid_priority_to_medium():
    from unittest.mock import MagicMock

    from app.services.renovation_recommendations_service import (
        _LLMRecommendation,
        _LLMRecommendationsOutput,
    )

    service = RenovationRecommendationsService()
    service.enabled = True

    fake_response = MagicMock()
    fake_response.output_parsed = _LLMRecommendationsOutput(
        recommendations=[
            _LLMRecommendation(
                title="Repaint the walls",
                rationale="Cosmetic refresh.",
                priority="urgent",  # not a valid value
                related_object_id=None,
            ),
        ]
    )
    service.client = MagicMock()
    service.client.responses.parse.return_value = fake_response

    response = service._llm_response(_simple_request())

    assert response.recommendations[0].priority == "medium"


def test_service_uses_the_main_estimation_model_not_the_fast_one(monkeypatch):
    """Unlike room-layout's simple spatial repositioning, renovation recommendations is an
    analytical/advisory task closer to the main estimation pipeline — should use the stronger
    reserved model, not the cheaper intake-fast one."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-strong")
    monkeypatch.setenv("OPENAI_INTAKE_FAST_MODEL", "gpt-5-mini")
    from importlib import reload
    import app.config as app_config
    import app.services.renovation_recommendations_service as recs_module

    reload(app_config)
    reload(recs_module)

    service = recs_module.RenovationRecommendationsService()
    assert service.model == "gpt-5-strong"


def test_suggest_falls_back_to_mock_with_warning_when_llm_raises():
    from unittest.mock import MagicMock

    service = RenovationRecommendationsService()
    service.enabled = True
    service.client = MagicMock()
    service.client.responses.parse.side_effect = RuntimeError("boom")

    response = service.suggest(_simple_request())

    assert response.mode == "mock"
    assert any("unavailable" in w.lower() for w in response.warnings)


def test_suggest_passes_existing_scope_items_through_to_user_message():
    """Confirms the existing scope items actually reach the prompt, so the LLM instruction not to
    duplicate them has something to act on."""
    from app.services.renovation_recommendations_service import _build_user_message

    request = _simple_request(existing_scope_items=["Replace kitchen flooring"])
    message = _build_user_message(request)

    assert "Replace kitchen flooring" in message
