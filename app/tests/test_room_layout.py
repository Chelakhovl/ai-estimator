from __future__ import annotations

from app.services.room_layout_service import (
    LayoutObject,
    LayoutSurface,
    RoomLayoutRequest,
    RoomLayoutService,
)


def _service_disabled() -> RoomLayoutService:
    service = RoomLayoutService()
    service.enabled = False
    service.client = None
    return service


def _simple_request(**overrides) -> RoomLayoutRequest:
    defaults = dict(
        room_name="Bedroom",
        floor_area_sq_m=12.0,
        surfaces=[
            LayoutSurface(kind="wall", width_m=4.0, x=0, z=0, rotation_radians=0),
            LayoutSurface(kind="door", width_m=0.8, x=0, z=0, rotation_radians=0),
        ],
        objects=[
            LayoutObject(id="bed-1", category="Bed", width_m=1.5, depth_m=2.0),
        ],
    )
    defaults.update(overrides)
    return RoomLayoutRequest(**defaults)


def test_suggest_with_no_objects_returns_empty_variants_not_an_error():
    service = _service_disabled()
    response = service.suggest(_simple_request(objects=[]))

    assert response.variants == []
    assert "No objects" in response.warnings[0]


def test_suggest_falls_back_to_mock_when_llm_disabled():
    service = _service_disabled()
    response = service.suggest(_simple_request())

    assert response.mode == "mock"
    assert response.variants == []
    assert response.warnings


def test_mock_response_never_claims_llm_mode():
    service = _service_disabled()
    response = service._mock_response(_simple_request())

    assert response.mode == "mock"


def test_request_rejects_more_than_five_variants():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _simple_request(variant_count=6)


def test_request_accepts_up_to_five_variants():
    request = _simple_request(variant_count=5)
    assert request.variant_count == 5


def test_llm_response_drops_placements_for_unknown_object_ids():
    from unittest.mock import MagicMock

    from app.services.room_layout_service import (
        _LLMLayoutOutput,
        _LLMLayoutVariant,
        _LLMPlacedObject,
    )

    service = RoomLayoutService()
    service.enabled = True

    fake_response = MagicMock()
    fake_response.output_parsed = _LLMLayoutOutput(
        variants=[
            _LLMLayoutVariant(
                label="Against the wall",
                rationale="Leaves a clear walkway.",
                placements=[
                    _LLMPlacedObject(id="bed-1", x=1.0, z=1.0, rotation_radians=0.0),
                    # The model invented an id that was never in the request — must be dropped,
                    # not trusted outright.
                    _LLMPlacedObject(id="made-up-wardrobe", x=2.0, z=2.0, rotation_radians=0.0),
                ],
            )
        ]
    )
    service.client = MagicMock()
    service.client.responses.parse.return_value = fake_response

    response = service._llm_response(_simple_request())

    assert response.mode == "llm"
    assert len(response.variants) == 1
    placed_ids = [p.id for p in response.variants[0].placements]
    assert placed_ids == ["bed-1"]


def test_llm_response_raises_when_all_placements_are_unknown_ids():
    from unittest.mock import MagicMock

    import pytest

    from app.services.room_layout_service import (
        RoomLayoutError,
        _LLMLayoutOutput,
        _LLMLayoutVariant,
        _LLMPlacedObject,
    )

    service = RoomLayoutService()
    service.enabled = True

    fake_response = MagicMock()
    fake_response.output_parsed = _LLMLayoutOutput(
        variants=[
            _LLMLayoutVariant(
                label="Bad variant",
                rationale="Invalid.",
                placements=[_LLMPlacedObject(id="not-real", x=1.0, z=1.0, rotation_radians=0.0)],
            )
        ]
    )
    service.client = MagicMock()
    service.client.responses.parse.return_value = fake_response

    with pytest.raises(RoomLayoutError):
        service._llm_response(_simple_request())


def test_service_uses_the_cheaper_fast_model_not_the_main_estimation_model(monkeypatch):
    """Rearranging a few objects within fixed walls is much simpler than the pricing/estimation
    pipeline (llm_client.py) this ai-service otherwise mostly serves — this should run on the cheaper
    intake-fast model, not the stronger one reserved for that harder task."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-strong")
    monkeypatch.setenv("OPENAI_INTAKE_FAST_MODEL", "gpt-5-mini")
    from importlib import reload
    import app.config as app_config
    import app.services.room_layout_service as room_layout_module

    reload(app_config)
    reload(room_layout_module)

    service = room_layout_module.RoomLayoutService()
    assert service.model == "gpt-5-mini"


def test_suggest_falls_back_to_mock_with_warning_when_llm_raises():
    from unittest.mock import MagicMock

    service = RoomLayoutService()
    service.enabled = True
    service.client = MagicMock()
    service.client.responses.parse.side_effect = RuntimeError("boom")

    response = service.suggest(_simple_request())

    assert response.mode == "mock"
    assert any("unavailable" in w.lower() for w in response.warnings)
