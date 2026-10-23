from __future__ import annotations

import base64

from app.services.room_redesign_service import RoomRedesignRequest, RoomRedesignService

_FAKE_JPEG_BASE64 = base64.b64encode(b"not a real jpeg, just test bytes").decode("ascii")


def _service_disabled() -> RoomRedesignService:
    service = RoomRedesignService()
    service.enabled = False
    service.client = None
    return service


def _simple_request(**overrides) -> RoomRedesignRequest:
    defaults = dict(
        room_name="Living Room",
        style="scandinavian",
        image_base64=_FAKE_JPEG_BASE64,
    )
    defaults.update(overrides)
    return RoomRedesignRequest(**defaults)


def test_generate_with_unknown_style_returns_mock_not_an_error():
    service = _service_disabled()
    response = service.generate(_simple_request(style="brutalist"))

    assert response.mode == "mock"
    assert response.image_base64 is None
    assert "Unknown style" in response.warnings[0]


def test_generate_falls_back_to_mock_when_llm_disabled():
    service = _service_disabled()
    response = service.generate(_simple_request())

    assert response.mode == "mock"
    assert response.image_base64 is None
    assert response.warnings


def test_mock_response_never_claims_llm_mode():
    service = _service_disabled()
    response = service._mock_response()

    assert response.mode == "mock"


def test_all_four_documented_styles_are_accepted():
    service = _service_disabled()
    for style in ["scandinavian", "modern", "traditional", "minimalist"]:
        response = service.generate(_simple_request(style=style))
        # Disabled service still returns mock mode, but NOT the "Unknown style" warning — confirms
        # each of these four styles passed the validation step before falling through to mock-disabled.
        assert response.mode == "mock"
        assert not any("Unknown style" in w for w in response.warnings)


def test_llm_response_returns_image_from_api():
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True

    fake_image = MagicMock()
    fake_image.b64_json = "ZmFrZS1nZW5lcmF0ZWQtaW1hZ2U="
    fake_response = MagicMock()
    fake_response.data = [fake_image]
    service.client = MagicMock()
    service.client.images.edit.return_value = fake_response

    response = service._llm_response(_simple_request())

    assert response.mode == "llm"
    assert response.image_base64 == "ZmFrZS1nZW5lcmF0ZWQtaW1hZ2U="


def test_llm_response_raises_when_api_returns_no_image_data():
    from unittest.mock import MagicMock

    import pytest

    from app.services.room_redesign_service import RoomRedesignError

    service = RoomRedesignService()
    service.enabled = True

    fake_response = MagicMock()
    fake_response.data = []
    service.client = MagicMock()
    service.client.images.edit.return_value = fake_response

    with pytest.raises(RoomRedesignError):
        service._llm_response(_simple_request())


def test_suggest_falls_back_to_mock_with_warning_when_llm_raises():
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True
    service.client = MagicMock()
    service.client.images.edit.side_effect = RuntimeError("boom")

    response = service.generate(_simple_request())

    assert response.mode == "mock"
    assert any("unavailable" in w.lower() for w in response.warnings)


def test_llm_response_uses_gpt_image_1_model():
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True

    fake_image = MagicMock()
    fake_image.b64_json = "ZmFrZQ=="
    fake_response = MagicMock()
    fake_response.data = [fake_image]
    service.client = MagicMock()
    service.client.images.edit.return_value = fake_response

    service._llm_response(_simple_request())

    _, kwargs = service.client.images.edit.call_args
    assert kwargs["model"] == "gpt-image-1"


def test_prompt_includes_style_specific_and_shared_instructions():
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True

    fake_image = MagicMock()
    fake_image.b64_json = "ZmFrZQ=="
    fake_response = MagicMock()
    fake_response.data = [fake_image]
    service.client = MagicMock()
    service.client.images.edit.return_value = fake_response

    service._llm_response(_simple_request(style="minimalist"))

    _, kwargs = service.client.images.edit.call_args
    assert "minimalist" in kwargs["prompt"].lower()
    assert "same walls" in kwargs["prompt"].lower()
