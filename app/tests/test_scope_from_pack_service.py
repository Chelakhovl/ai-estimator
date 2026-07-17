from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services import scope_from_pack_service
from app.services.scope_from_pack_service import (
    _build_user_content,
    _fallback_scope,
    generate_scope_from_context_pack,
)


class FakeSettings:
    """Stand-in for app.config.Settings (a frozen dataclass, so it can't be
    monkeypatched attribute-by-attribute)."""

    def __init__(self, openai_api_key: str = "", openai_intake_model: str = ""):
        self.openai_api_key = openai_api_key
        self.openai_intake_model = openai_intake_model
        self.openai_timeout_seconds = 45.0


SAMPLE_PACK = {
    "property_address": "12 Example Road, London",
    "rooms": [
        {"name": "Kitchen", "area_m2": 18.5, "floor_level": "Ground"},
        {"name": "Bathroom", "area_m2": 6.0, "floor_level": "First"},
    ],
    "work_items": [
        {"trade": "Electrics", "description": "Rewire kitchen sockets"},
        {"trade": "Electrics", "description": "New consumer unit"},
        {"trade": "Plumbing", "description": "First fix bathroom"},
    ],
    "structural_notes": [{"text": "Remove load-bearing wall between kitchen and dining room"}],
    "prelim_requirements": [{"category": "access", "text": "Scaffold to rear elevation"}],
    "electrical_fixtures": [{"type": "socket"}, {"type": "socket"}],
    "uncertainties": [{"text": f"Uncertainty {i}"} for i in range(8)],
}


class TestFallbackScope:
    def test_includes_room_areas_and_floor_levels(self):
        text = _fallback_scope(SAMPLE_PACK)

        assert "Kitchen (Ground) — 18.5 m²" in text
        assert "Bathroom (First) — 6.0 m²" in text

    def test_groups_work_items_by_trade(self):
        text = _fallback_scope(SAMPLE_PACK)

        assert "ELECTRICS:" in text
        assert "- Rewire kitchen sockets" in text
        assert "- New consumer unit" in text
        assert "PLUMBING:" in text
        assert "- First fix bathroom" in text

    def test_includes_structural_notes_and_prelims(self):
        text = _fallback_scope(SAMPLE_PACK)

        assert "STRUCTURAL:" in text
        assert "- Remove load-bearing wall between kitchen and dining room" in text
        assert "PRELIMS:" in text
        assert "- Scaffold to rear elevation" in text

    def test_empty_pack_uses_generic_trade_headers(self):
        text = _fallback_scope({})

        for heading in ("DEMOLITION:", "STRUCTURE:", "ELECTRICAL:", "PLUMBING:", "FLOORING:", "DECORATING:"):
            assert heading in text


class TestBuildUserContent:
    def test_includes_all_pack_sections(self):
        content = _build_user_content(SAMPLE_PACK)

        assert "Property: 12 Example Road, London" in content
        assert "ROOMS FROM DRAWINGS:" in content
        assert "Kitchen (Ground, 18.5 m²)" in content
        assert "WORK ITEMS FROM DRAWINGS:" in content
        assert "[Electrics] Rewire kitchen sockets" in content
        assert "STRUCTURAL NOTES:" in content
        assert "PRELIM REQUIREMENTS:" in content
        assert "[access] Scaffold to rear elevation" in content
        assert "ELECTRICAL FIXTURES: 2 items found in drawings" in content

    def test_truncates_uncertainties_to_five(self):
        content = _build_user_content(SAMPLE_PACK)

        assert content.count("Uncertainty ") == 5

    def test_empty_pack_produces_empty_content(self):
        assert _build_user_content({}) == ""


class TestGenerateScopeFromContextPack:
    def test_returns_fallback_when_openai_not_configured(self):
        with patch.object(scope_from_pack_service, "settings", FakeSettings(openai_api_key="")):
            result = generate_scope_from_context_pack(SAMPLE_PACK)

        assert result["model_name"] == "fallback"
        assert "ELECTRICS:" in result["scope_text"]
        assert "sections" not in result

    def test_returns_parsed_llm_output_on_success(self):
        fake_message = MagicMock()
        fake_message.content = (
            '{"scope_text": "Full scope summary", '
            '"sections": [{"key": "electrics", "title": "Electrics", "lines": ["Rewire kitchen"]}]}'
        )
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage.prompt_tokens = 120
        fake_response.usage.completion_tokens = 40

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            scope_from_pack_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = generate_scope_from_context_pack(SAMPLE_PACK)

        assert result["scope_text"] == "Full scope summary"
        assert result["sections"][0]["key"] == "electrics"
        assert result["model_name"] == "gpt-4o"
        assert result["usage"] == {"prompt_tokens": 120, "completion_tokens": 40}

    def test_falls_back_when_llm_call_raises(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("network unreachable")

        with patch.object(
            scope_from_pack_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = generate_scope_from_context_pack(SAMPLE_PACK)

        assert result["model_name"] == "fallback"
        assert result["sections"] == []
        assert "ELECTRICS:" in result["scope_text"]

    def test_falls_back_when_llm_returns_invalid_json(self):
        fake_message = MagicMock()
        fake_message.content = "not valid json"
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = None

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            scope_from_pack_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = generate_scope_from_context_pack(SAMPLE_PACK)

        assert result["model_name"] == "fallback"
        assert result["sections"] == []
