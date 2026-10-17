from __future__ import annotations

from app.services.object_condition_service import (
    ObjectConditionRequest,
    ObjectConditionService,
)


def _service_disabled() -> ObjectConditionService:
    service = ObjectConditionService()
    service.enabled = False
    return service


def _request(**overrides) -> ObjectConditionRequest:
    defaults = dict(
        category="Stove",
        width_m=0.6,
        height_m=0.9,
        depth_m=0.6,
        room_name="Kitchen",
        image_base64="ZmFrZS1pbWFnZS1ieXRlcw==",  # "fake-image-bytes"
    )
    defaults.update(overrides)
    return ObjectConditionRequest(**defaults)


def test_assess_falls_back_to_mock_when_disabled():
    service = _service_disabled()
    response = service.assess(_request())

    assert response.mode == "mock"
    assert response.condition == "unknown"
    assert response.category == "Stove"


def test_mock_response_echoes_the_same_category():
    service = _service_disabled()
    response = service._mock_response(_request(category="Bathtub"))

    assert response.category == "Bathtub"
    assert response.mode == "mock"


def test_assess_falls_back_to_mock_when_llm_raises(monkeypatch):
    service = ObjectConditionService()
    service.enabled = True

    def _boom(self, request):
        raise RuntimeError("network error")

    monkeypatch.setattr(ObjectConditionService, "_llm_response", _boom)

    response = service.assess(_request())
    assert response.mode == "mock"


def _patch_openai_client(monkeypatch, fake_client) -> None:
    """`_llm_response` does `from openai import OpenAI` locally (lazy import, same convention as
    `RoomLayoutService`/`LLMClient` elsewhere in this codebase) — patch the real `openai.OpenAI`
    symbol the import resolves to, not a module-level name that doesn't exist until that import runs."""
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **_: fake_client)


def test_llm_response_rejects_unknown_condition_values(monkeypatch):
    from unittest.mock import MagicMock
    import json as json_module

    service = ObjectConditionService()
    service.enabled = True

    fake_message = MagicMock()
    fake_message.content = json_module.dumps(
        {
            "condition": "pristine",  # not one of the allowed values
            "condition_notes": "Looks fine.",
            "renovation_flags": [],
        }
    )
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    _patch_openai_client(monkeypatch, fake_client)

    response = service._llm_response(_request())
    assert response.condition == "unknown"


def test_llm_response_caps_renovation_flags_at_three(monkeypatch):
    from unittest.mock import MagicMock
    import json as json_module

    service = ObjectConditionService()
    service.enabled = True

    fake_message = MagicMock()
    fake_message.content = json_module.dumps(
        {
            "condition": "worn",
            "condition_notes": "Visible rust on the base.",
            "renovation_flags": ["a", "b", "c", "d", "e"],
        }
    )
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    _patch_openai_client(monkeypatch, fake_client)

    response = service._llm_response(_request())
    assert len(response.renovation_flags) == 3
