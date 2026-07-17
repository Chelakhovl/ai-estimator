from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.schemas import BatchCorrectionsRequest, ItemContext
from app.services import corrections_parser
from app.services.corrections_parser import _mock_parse, parse_batch_corrections


def _items() -> list[ItemContext]:
    return [
        ItemContext(id=1, ref="1.1.4", display_index=1, name="Strip out kitchen", area="120m²", quantity="120.00"),
        ItemContext(id=2, ref="1.1.6", display_index=2, name="Fix new kitchen units", area="", quantity="10.00"),
        ItemContext(id=3, ref="2.3.1", display_index=3, name="Paint walls", area="94m²", quantity="94.00"),
    ]


class FakeSettings:
    """Stand-in for app.config.Settings, since the real one is a frozen dataclass
    and cannot be monkeypatched attribute-by-attribute."""

    def __init__(self, openai_api_key: str = "", openai_model: str = "", openai_timeout_seconds: float = 45.0):
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_timeout_seconds = openai_timeout_seconds


class TestMockParse:
    def test_matches_item_by_ref_code_and_extracts_area(self):
        request = BatchCorrectionsRequest(text="1.1.4 area = 94m²", items=_items())
        result = _mock_parse(request)

        assert result.service_mode == "mock"
        assert result.unresolved == []
        assert len(result.patches) == 1
        patch_ = result.patches[0]
        assert patch_.item_id == 1
        assert patch_.ref == "1.1.4"
        assert patch_.field == "area"
        assert patch_.old_value == "120m²"
        assert patch_.new_value == "94m²"

    def test_matches_item_by_index_number(self):
        request = BatchCorrectionsRequest(text="item 2 quantity = 12.5", items=_items())
        result = _mock_parse(request)

        assert len(result.patches) == 1
        patch_ = result.patches[0]
        assert patch_.item_id == 2
        assert patch_.field == "quantity"
        assert patch_.old_value == "10.00"
        assert patch_.new_value == "12.50"

    def test_matches_russian_index_keyword(self):
        request = BatchCorrectionsRequest(text="позиция 3 площадь 100m²", items=_items())
        result = _mock_parse(request)

        assert len(result.patches) == 1
        assert result.patches[0].item_id == 3
        assert result.patches[0].field == "area"
        assert result.patches[0].new_value == "100m²"

    def test_one_clause_can_produce_two_patches(self):
        request = BatchCorrectionsRequest(text="1.1.4 area 94m² qty 45", items=_items())
        result = _mock_parse(request)

        fields = {p.field for p in result.patches}
        assert fields == {"area", "quantity"}
        assert all(p.item_id == 1 for p in result.patches)

    def test_clause_with_no_matching_item_is_unresolved(self):
        request = BatchCorrectionsRequest(text="9.9.9 change area to 50m²", items=_items())
        result = _mock_parse(request)

        assert result.patches == []
        assert result.unresolved == ["9.9.9 change area to 50m²"]

    def test_clause_matching_item_but_no_recognised_field_is_unresolved(self):
        request = BatchCorrectionsRequest(text="1.1.4 rename to Strip out utility room", items=_items())
        result = _mock_parse(request)

        assert result.patches == []
        assert result.unresolved == ["1.1.4 rename to Strip out utility room"]

    def test_splits_multiple_clauses_on_semicolon_and_newline(self):
        request = BatchCorrectionsRequest(
            text="1.1.4 area 94m²; item 2 qty 12\n2.3.1 quantity 88",
            items=_items(),
        )
        result = _mock_parse(request)

        assert len(result.patches) == 3
        assert {p.item_id for p in result.patches} == {1, 2, 3}

    def test_blank_text_produces_no_patches_or_unresolved(self):
        request = BatchCorrectionsRequest(text="   \n  ", items=_items())
        result = _mock_parse(request)

        assert result.patches == []
        assert result.unresolved == []


class TestParseBatchCorrectionsDispatch:
    def test_uses_mock_when_openai_not_configured(self):
        with patch.object(corrections_parser, "settings", FakeSettings(openai_api_key="", openai_model="")):
            request = BatchCorrectionsRequest(text="1.1.4 area 94m²", items=_items())
            result = parse_batch_corrections(request)

        assert result.service_mode == "mock"
        assert len(result.patches) == 1

    def test_falls_back_to_mock_when_llm_call_raises(self):
        fake_client = MagicMock()
        fake_client.responses = None
        fake_client.beta.chat.completions.parse.side_effect = RuntimeError("network unreachable")

        with patch.object(
            corrections_parser, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            request = BatchCorrectionsRequest(text="1.1.4 area 94m²", items=_items())
            result = parse_batch_corrections(request)

        assert result.service_mode == "mock-fallback"
        assert len(result.patches) == 1
        assert result.patches[0].field == "area"

    def test_uses_llm_result_when_structured_output_parses(self):
        parsed_message = MagicMock()
        parsed_message.patches = []
        parsed_message.unresolved = ["some unresolved clause"]

        fake_response = MagicMock()
        fake_response.output_parsed = parsed_message

        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response

        with patch.object(
            corrections_parser, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            request = BatchCorrectionsRequest(text="anything", items=_items())
            result = parse_batch_corrections(request)

        assert result.service_mode == "real"
        assert result.unresolved == ["some unresolved clause"]
        fake_client.responses.parse.assert_called_once()


class TestCorrectionsApi:
    def _client(self, monkeypatch) -> TestClient:
        monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_MODEL", "")

        from importlib import reload

        import app.config as app_config
        import app.security as app_security
        import app.services.corrections_parser as app_corrections_parser

        reload(app_config)
        reload(app_security)
        reload(app_corrections_parser)

        from app.main import app

        return TestClient(app)

    def test_parse_endpoint_requires_api_key(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/corrections/parse",
            json={"text": "1.1.4 area 94m²", "items": [item.model_dump() for item in _items()]},
        )

        assert response.status_code == 401

    def test_parse_endpoint_returns_mock_patches(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/corrections/parse",
            json={"text": "1.1.4 area 94m²", "items": [item.model_dump() for item in _items()]},
            headers={"x-api-key": "test-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["service_mode"] == "mock"
        assert len(body["patches"]) == 1
        assert body["patches"][0]["field"] == "area"
