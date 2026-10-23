from __future__ import annotations

from unittest.mock import MagicMock
import json as json_module

from app.services.object_condition_service import (
    ObjectConditionBatchRequest,
    ObjectConditionItem,
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


def _item(**overrides) -> ObjectConditionItem:
    defaults = dict(
        id="item-1",
        category="Chair",
        width_m=0.5,
        height_m=0.9,
        depth_m=0.5,
        image_base64="ZmFrZS1pbWFnZS1ieXRlcw==",
    )
    defaults.update(overrides)
    return ObjectConditionItem(**defaults)


def _patch_openai_client(monkeypatch, fake_client) -> None:
    """`_llm_batch_response` does `from openai import OpenAI` locally (lazy import, same convention
    as `RoomLayoutService`/`LLMClient` elsewhere in this codebase) — patch the real `openai.OpenAI`
    symbol the import resolves to, not a module-level name that doesn't exist until that import runs."""
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **_: fake_client)


def _fake_chat_response(results: list[dict]) -> MagicMock:
    fake_message = MagicMock()
    fake_message.content = json_module.dumps({"results": results})
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = None
    return fake_response


# ── single-item path (now a thin wrapper over the batch path) ──────────────


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


def test_assess_falls_back_to_mock_when_batch_raises(monkeypatch):
    service = ObjectConditionService()
    service.enabled = True

    def _boom(self, request):
        raise RuntimeError("network error")

    monkeypatch.setattr(ObjectConditionService, "_llm_batch_response", _boom)

    response = service.assess(_request())
    assert response.mode == "mock"


def test_assess_single_item_uses_batch_path(monkeypatch):
    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [{"item_index": 0, "condition": "worn", "condition_notes": "Some scuffing.", "renovation_flags": []}]
    )
    _patch_openai_client(monkeypatch, fake_client)

    response = service.assess(_request())
    assert response.mode == "llm"
    assert response.condition == "worn"
    assert fake_client.chat.completions.create.call_count == 1


# ── batch path ───────────────────────────────────────────────────────────


def test_assess_batch_with_no_items_returns_empty_mock():
    service = _service_disabled()
    response = service.assess_batch(ObjectConditionBatchRequest(items=[]))

    assert response.results == []
    assert response.mode == "mock"


def test_assess_batch_falls_back_to_mock_when_disabled():
    service = _service_disabled()
    response = service.assess_batch(ObjectConditionBatchRequest(items=[_item(id="a"), _item(id="b")]))

    assert response.mode == "mock"
    assert [r.id for r in response.results] == ["a", "b"]
    assert all(r.condition == "unknown" for r in response.results)


def test_assess_batch_rejects_unknown_condition_values(monkeypatch):
    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [{"item_index": 0, "condition": "pristine", "condition_notes": "", "renovation_flags": []}]
    )
    _patch_openai_client(monkeypatch, fake_client)

    response = service.assess_batch(ObjectConditionBatchRequest(items=[_item()]))
    assert response.results[0].condition == "unknown"


def test_assess_batch_caps_renovation_flags_at_three(monkeypatch):
    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [{"item_index": 0, "condition": "worn", "condition_notes": "", "renovation_flags": ["a", "b", "c", "d"]}]
    )
    _patch_openai_client(monkeypatch, fake_client)

    response = service.assess_batch(ObjectConditionBatchRequest(items=[_item()]))
    assert len(response.results[0].renovation_flags) == 3


def test_assess_batch_sends_one_call_for_multiple_distinct_items(monkeypatch):
    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [
            {"item_index": 0, "condition": "good", "condition_notes": "", "renovation_flags": []},
            {"item_index": 1, "condition": "worn", "condition_notes": "", "renovation_flags": []},
        ]
    )
    _patch_openai_client(monkeypatch, fake_client)

    items = [_item(id="stove", category="Stove"), _item(id="sofa", category="Sofa", width_m=2.0)]
    response = service.assess_batch(ObjectConditionBatchRequest(items=items))

    assert fake_client.chat.completions.create.call_count == 1
    assert {r.id: r.condition for r in response.results} == {"stove": "good", "sofa": "worn"}


def test_assess_batch_deduplicates_near_identical_items(monkeypatch):
    """Two chairs with matching category and near-identical dimensions should be sent to the model
    ONCE, not twice — the second chair's result is copied from the first's, not a separate vision call."""
    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [{"item_index": 0, "condition": "worn", "condition_notes": "Scuffed legs.", "renovation_flags": []}]
    )
    _patch_openai_client(monkeypatch, fake_client)

    items = [
        _item(id="chair-1", category="Chair", width_m=0.50, height_m=0.90, depth_m=0.50),
        _item(id="chair-2", category="Chair", width_m=0.52, height_m=0.91, depth_m=0.49),  # same bucket
    ]
    response = service.assess_batch(ObjectConditionBatchRequest(items=items))

    # Only one item actually sent to the model (one image in the user content).
    call_args = fake_client.chat.completions.create.call_args
    user_message = call_args.kwargs["messages"][1]
    image_blocks = [block for block in user_message["content"] if block["type"] == "image_url"]
    assert len(image_blocks) == 1

    # But both original items get a result, both copied from the single assessment.
    assert len(response.results) == 2
    assert response.results[0].condition == "worn"
    assert response.results[1].condition == "worn"
    assert response.results[0].condition_notes == response.results[1].condition_notes


def test_assess_batch_defaults_to_unknown_when_model_omits_an_item(monkeypatch):
    """Partial-batch failure: the model returns fewer results than groups sent (e.g. truncated output,
    or it silently skips one item) — every original item must still get a result back, with the
    missing one defaulting to "unknown" rather than being dropped or crashing the whole batch."""
    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    # Only item_index 0 is present — item_index 1 (the sofa) is missing from the model's response.
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [{"item_index": 0, "condition": "good", "condition_notes": "", "renovation_flags": []}]
    )
    _patch_openai_client(monkeypatch, fake_client)

    items = [_item(id="stove", category="Stove"), _item(id="sofa", category="Sofa", width_m=2.0)]
    response = service.assess_batch(ObjectConditionBatchRequest(items=items))

    assert len(response.results) == 2
    results_by_id = {r.id: r for r in response.results}
    assert results_by_id["stove"].condition == "good"
    assert results_by_id["sofa"].condition == "unknown"
    assert results_by_id["sofa"].condition_notes == ""
    assert results_by_id["sofa"].renovation_flags == []
    # Mode stays "llm" — a partially-incomplete model response is still a real response, not a total
    # failure that should fall back to mock (that path is covered separately, see
    # test_assess_falls_back_to_mock_when_batch_raises).
    assert response.mode == "llm"


def test_assess_batch_does_not_deduplicate_different_categories(monkeypatch):
    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [
            {"item_index": 0, "condition": "good", "condition_notes": "", "renovation_flags": []},
            {"item_index": 1, "condition": "needs_replacement", "condition_notes": "", "renovation_flags": []},
        ]
    )
    _patch_openai_client(monkeypatch, fake_client)

    items = [
        _item(id="chair", category="Chair", width_m=0.5, height_m=0.9, depth_m=0.5),
        _item(id="table", category="Table", width_m=0.5, height_m=0.9, depth_m=0.5),
    ]
    response = service.assess_batch(ObjectConditionBatchRequest(items=items))

    call_args = fake_client.chat.completions.create.call_args
    user_message = call_args.kwargs["messages"][1]
    image_blocks = [block for block in user_message["content"] if block["type"] == "image_url"]
    assert len(image_blocks) == 2


def test_assess_batch_uses_system_prompt_with_no_dynamic_content(monkeypatch):
    """The system prompt must be identical across calls (no interpolated category/dimensions/room
    name) for OpenAI's automatic prompt caching to apply to it — this locks that invariant in."""
    from app.services.object_condition_service import _SYSTEM_PROMPT

    service = ObjectConditionService()
    service.enabled = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_chat_response(
        [{"item_index": 0, "condition": "good", "condition_notes": "", "renovation_flags": []}]
    )
    _patch_openai_client(monkeypatch, fake_client)

    service.assess_batch(ObjectConditionBatchRequest(room_name="Kitchen", items=[_item(category="Oven")]))

    call_args = fake_client.chat.completions.create.call_args
    system_message = call_args.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert system_message["content"] == _SYSTEM_PROMPT
    assert "Kitchen" not in system_message["content"]
    assert "Oven" not in system_message["content"]
