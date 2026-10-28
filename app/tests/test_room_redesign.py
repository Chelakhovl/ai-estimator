from __future__ import annotations

import base64

from app.services.room_redesign_service import RoomRedesignRequest, RoomRedesignService

_FAKE_JPEG_BASE64 = base64.b64encode(b"not a real jpeg, just test bytes").decode("ascii")
_FAKE_JPEG_BASE64_2 = base64.b64encode(b"not a real jpeg, second frame").decode("ascii")
_FAKE_JPEG_BASE64_3 = base64.b64encode(b"not a real jpeg, third frame").decode("ascii")


def _service_disabled() -> RoomRedesignService:
    service = RoomRedesignService()
    service.enabled = False
    service.client = None
    return service


def _simple_request(**overrides) -> RoomRedesignRequest:
    defaults = dict(
        room_name="Living Room",
        style="scandinavian",
        images_base64=[_FAKE_JPEG_BASE64],
    )
    defaults.update(overrides)
    return RoomRedesignRequest(**defaults)


def test_generate_with_unknown_style_returns_mock_not_an_error():
    service = _service_disabled()
    response = service.generate(_simple_request(style="brutalist"))

    assert response.mode == "mock"
    assert response.images_base64 == []
    assert "Unknown style" in response.warnings[0]


def test_generate_falls_back_to_mock_when_llm_disabled():
    service = _service_disabled()
    response = service.generate(_simple_request())

    assert response.mode == "mock"
    assert response.images_base64 == []
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
    assert response.images_base64 == ["ZmFrZS1nZW5lcmF0ZWQtaW1hZ2U="]


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
    assert kwargs["model"] == "gpt-image-2.5-sunburst"


def test_llm_response_omits_mask_when_client_sent_none():
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True

    fake_image = MagicMock()
    fake_image.b64_json = "ZmFrZQ=="
    fake_response = MagicMock()
    fake_response.data = [fake_image]
    service.client = MagicMock()
    service.client.images.edit.return_value = fake_response

    # No masks_base64 — an older app install, or a scan where mask projection failed (see the iOS
    # client's RoomPhotoCapture.structureMaskPNGData) — must fall back to prompt-only behaviour
    # exactly as before masks existed, not send a missing/None mask to the API.
    service._llm_response(_simple_request())

    _, kwargs = service.client.images.edit.call_args
    assert "mask" not in kwargs


def test_llm_response_passes_mask_to_api_when_provided():
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True

    fake_image = MagicMock()
    fake_image.b64_json = "ZmFrZQ=="
    fake_response = MagicMock()
    fake_response.data = [fake_image]
    service.client = MagicMock()
    service.client.images.edit.return_value = fake_response

    fake_mask_base64 = base64.b64encode(b"not a real png, just test bytes").decode("ascii")
    service._llm_response(_simple_request(masks_base64=[fake_mask_base64]))

    _, kwargs = service.client.images.edit.call_args
    assert "mask" in kwargs


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


# ── Multi-image (up to 3 angles per scan) behaviour ──────────────────────────


def test_llm_response_calls_images_edit_once_per_image_with_matching_mask():
    """Each image gets its own images.edit call, paired with the mask at the SAME index — not a
    single shared mask, and not mismatched pairing (e.g. image 2 getting mask 1)."""
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True

    def _make_fake_response(tag: str) -> MagicMock:
        fake_image = MagicMock()
        fake_image.b64_json = tag
        fake_response = MagicMock()
        fake_response.data = [fake_image]
        return fake_response

    service.client = MagicMock()
    service.client.images.edit.side_effect = [
        _make_fake_response("frame-1-result"),
        _make_fake_response("frame-2-result"),
        _make_fake_response("frame-3-result"),
    ]

    mask_1 = base64.b64encode(b"mask for frame 1").decode("ascii")
    mask_3 = base64.b64encode(b"mask for frame 3").decode("ascii")

    response = service._llm_response(
        _simple_request(
            images_base64=[_FAKE_JPEG_BASE64, _FAKE_JPEG_BASE64_2, _FAKE_JPEG_BASE64_3],
            masks_base64=[mask_1, None, mask_3],
        )
    )

    assert response.mode == "llm"
    assert set(response.images_base64) == {"frame-1-result", "frame-2-result", "frame-3-result"}
    assert service.client.images.edit.call_count == 3

    # images.edit calls may complete in any order under asyncio.gather, so match each call back to
    # its frame via the image bytes rather than assuming call order.
    calls_by_image_name = {}
    for call in service.client.images.edit.call_args_list:
        _, kwargs = call
        image_bytes = kwargs["image"].getvalue()
        calls_by_image_name[image_bytes] = kwargs

    frame_1_bytes = base64.b64decode(_FAKE_JPEG_BASE64)
    frame_2_bytes = base64.b64decode(_FAKE_JPEG_BASE64_2)
    frame_3_bytes = base64.b64decode(_FAKE_JPEG_BASE64_3)

    assert "mask" in calls_by_image_name[frame_1_bytes]
    assert calls_by_image_name[frame_1_bytes]["mask"].getvalue() == base64.b64decode(mask_1)

    assert "mask" not in calls_by_image_name[frame_2_bytes]

    assert "mask" in calls_by_image_name[frame_3_bytes]
    assert calls_by_image_name[frame_3_bytes]["mask"].getvalue() == base64.b64decode(mask_3)


def test_llm_response_returns_successful_frames_when_one_frame_fails():
    """One frame raising (e.g. OpenAI returned no data, or the call itself errored) must not fail the
    whole request — the other frames' successful results still come back, per the contract that
    images_base64 in the response can be shorter than in the request."""
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True

    fake_image_ok = MagicMock()
    fake_image_ok.b64_json = "frame-ok-result"
    fake_response_ok = MagicMock()
    fake_response_ok.data = [fake_image_ok]

    service.client = MagicMock()
    service.client.images.edit.side_effect = [
        fake_response_ok,
        RuntimeError("OpenAI blew up on this one frame"),
        fake_response_ok,
    ]

    response = service._llm_response(
        _simple_request(
            images_base64=[_FAKE_JPEG_BASE64, _FAKE_JPEG_BASE64_2, _FAKE_JPEG_BASE64_3],
            masks_base64=[None, None, None],
        )
    )

    assert response.mode == "llm"
    assert response.images_base64 == ["frame-ok-result", "frame-ok-result"]
    assert service.client.images.edit.call_count == 3


def test_generate_falls_back_to_mock_only_when_every_frame_fails():
    """All frames failing (not just one) is the only case that should fall through to the mock
    response via generate()'s except-block."""
    from unittest.mock import MagicMock

    service = RoomRedesignService()
    service.enabled = True
    service.client = MagicMock()
    service.client.images.edit.side_effect = RuntimeError("boom")

    response = service.generate(
        _simple_request(
            images_base64=[_FAKE_JPEG_BASE64, _FAKE_JPEG_BASE64_2],
            masks_base64=[None, None],
        )
    )

    assert response.mode == "mock"
    assert response.images_base64 == []
    assert any("unavailable" in w.lower() for w in response.warnings)


def test_masks_base64_defaults_and_pads_when_shorter_than_images():
    """masks_base64 is optional and, if shorter than images_base64 (e.g. omitted entirely by an older
    client), is padded with None rather than raising."""
    request = _simple_request(
        images_base64=[_FAKE_JPEG_BASE64, _FAKE_JPEG_BASE64_2, _FAKE_JPEG_BASE64_3],
        masks_base64=[],
    )

    assert request.masks_base64 == [None, None, None]
