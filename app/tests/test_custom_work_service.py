from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.schemas import (
    CatalogContext,
    CatalogLabourRateContext,
    CatalogWorkGroupContext,
    CustomPricedRow,
    CustomWorkChatMessage,
    CustomWorkChatRequest,
    CustomWorkProposal,
    LLMBatchPriceOutput,
    LLMBatchPricedItem,
    LLMCustomWorkOutput,
    PreviewUnmatchedItem,
)
from app.services import custom_work_service
from app.services.custom_work_service import (
    CustomWorkLLMClient,
    _build_batch_price_system_prompt,
    _build_messages,
    _build_system_prompt,
    _MOCK_RESPONSE,
    batch_price_unmatched,
    chat_custom_work,
)


def _catalog_context() -> CatalogContext:
    return CatalogContext(
        work_groups=[
            CatalogWorkGroupContext(id=1, name="Kitchen"),
            CatalogWorkGroupContext(id=2, name="Electrical"),
        ],
        labour_rates=[
            CatalogLabourRateContext(code="ELEC", name="Electrician", net_rate=38.0),
        ],
    )


class FakeSettings:
    """Stand-in for app.config.Settings, since the real one is a frozen dataclass
    and cannot be monkeypatched attribute-by-attribute."""

    def __init__(self, openai_api_key: str = "", openai_intake_model: str = "", openai_timeout_seconds: float = 45.0):
        self.openai_api_key = openai_api_key
        self.openai_intake_model = openai_intake_model
        self.openai_timeout_seconds = openai_timeout_seconds


class TestBuildSystemPrompt:
    def test_includes_work_groups_and_labour_rates(self):
        prompt = _build_system_prompt(_catalog_context())

        assert "1: Kitchen" in prompt
        assert "2: Electrical" in prompt
        assert "Electrician (ELEC): £38.00/hr" in prompt

    def test_batch_price_prompt_includes_work_groups_and_labour_rates(self):
        prompt = _build_batch_price_system_prompt(_catalog_context())

        assert "1: Kitchen" in prompt
        assert "2: Electrical" in prompt
        assert "Electrician (ELEC): £38.00/hr" in prompt


class TestBuildMessages:
    def test_prepends_system_prompt_and_maps_roles(self):
        messages = [
            CustomWorkChatMessage(role="user", content="I need a new kitchen island"),
            CustomWorkChatMessage(role="assistant", content="How big is the island?"),
        ]
        built = _build_messages(_catalog_context(), messages)

        assert built[0]["role"] == "system"
        assert "Kitchen" in built[0]["content"]
        assert built[1] == {"role": "user", "content": "I need a new kitchen island"}
        assert built[2] == {"role": "assistant", "content": "How big is the island?"}


class TestCustomWorkLLMClient:
    def test_disabled_when_no_api_key(self):
        with patch.object(custom_work_service, "settings", FakeSettings(openai_api_key="", openai_intake_model="")):
            client = CustomWorkLLMClient()

        assert client.is_enabled() is False
        assert client.client is None

    def test_disabled_when_no_intake_model(self):
        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="")
        ):
            client = CustomWorkLLMClient()

        assert client.is_enabled() is False
        assert client.client is None

    def test_enabled_constructs_openai_client(self):
        fake_client = MagicMock()
        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client) as mock_openai:
            client = CustomWorkLLMClient()

        assert client.is_enabled() is True
        assert client.client is fake_client
        mock_openai.assert_called_once()

    def test_chat_raises_when_not_enabled(self):
        with patch.object(custom_work_service, "settings", FakeSettings(openai_api_key="", openai_intake_model="")):
            client = CustomWorkLLMClient()

        try:
            client.chat([], _catalog_context())
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "not configured" in str(exc)

    def test_chat_returns_parsed_output_and_records_metadata(self):
        parsed = LLMCustomWorkOutput(message="hi", status="chatting")
        fake_response = MagicMock()
        fake_response.choices[0].message.parsed = parsed
        fake_client = MagicMock()
        fake_client.beta.chat.completions.parse.return_value = fake_response

        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = CustomWorkLLMClient()
            result = client.chat(
                [CustomWorkChatMessage(role="user", content="new kitchen island")],
                _catalog_context(),
            )

        assert result is parsed
        assert client.last_metadata["model_name"] == "gpt-4o"
        assert client.last_metadata["latency_ms"] >= 0
        fake_client.beta.chat.completions.parse.assert_called_once()
        _, kwargs = fake_client.beta.chat.completions.parse.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["response_format"] is LLMCustomWorkOutput

    def test_chat_raises_when_parsed_is_none(self):
        fake_response = MagicMock()
        fake_response.choices[0].message.parsed = None
        fake_client = MagicMock()
        fake_client.beta.chat.completions.parse.return_value = fake_response

        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = CustomWorkLLMClient()
            try:
                client.chat([CustomWorkChatMessage(role="user", content="hi")], _catalog_context())
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "no structured output" in str(exc)


class TestBatchPriceUnmatched:
    def test_returns_empty_list_for_no_items(self):
        result = batch_price_unmatched([], _catalog_context())
        assert result == []

    def test_returns_empty_list_when_llm_not_configured(self):
        items = [PreviewUnmatchedItem(source_text="Fit new bespoke shelving", reason="no match")]
        with patch.object(custom_work_service, "settings", FakeSettings(openai_api_key="", openai_intake_model="")):
            result = batch_price_unmatched(items, _catalog_context())

        assert result == []

    def test_maps_llm_output_to_custom_priced_rows(self):
        items = [PreviewUnmatchedItem(source_text="Fit new bespoke shelving", reason="no match")]
        priced_item = LLMBatchPricedItem(
            source_text="Fit new bespoke shelving",
            name="Bespoke shelving installation",
            unit="lm",
            labour_cost=120.0,
            material_cost=80.0,
            other_cost=0.0,
            work_days=1.0,
            qty_for_norm=1.0,
            suggested_work_group_id=1,
            suggested_work_group_name="Kitchen",
            market_justification="Standard joinery rate.",
            price_source_notes="Labour: Electrician (ELEC): £38.00/hr from Combit internal catalogue.",
            confidence_level=0.8,
        )
        fake_response = MagicMock()
        fake_response.choices[0].message.parsed = LLMBatchPriceOutput(items=[priced_item])
        fake_client = MagicMock()
        fake_client.beta.chat.completions.parse.return_value = fake_response

        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = batch_price_unmatched(items, _catalog_context())

        assert len(result) == 1
        row = result[0]
        assert isinstance(row, CustomPricedRow)
        assert row.source_text == "Fit new bespoke shelving"
        assert row.name == "Bespoke shelving installation"
        assert row.labour_cost == 120.0
        assert row.material_cost == 80.0
        assert row.confidence_level == 0.8

    def test_returns_empty_list_when_parsed_is_none(self):
        items = [PreviewUnmatchedItem(source_text="Fit new bespoke shelving", reason="no match")]
        fake_response = MagicMock()
        fake_response.choices[0].message.parsed = None
        fake_client = MagicMock()
        fake_client.beta.chat.completions.parse.return_value = fake_response

        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = batch_price_unmatched(items, _catalog_context())

        assert result == []

    def test_returns_empty_list_when_llm_call_raises(self):
        items = [PreviewUnmatchedItem(source_text="Fit new bespoke shelving", reason="no match")]
        fake_client = MagicMock()
        fake_client.beta.chat.completions.parse.side_effect = RuntimeError("network unreachable")

        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = batch_price_unmatched(items, _catalog_context())

        assert result == []


class TestChatCustomWork:
    def test_returns_mock_response_when_not_enabled(self):
        request = CustomWorkChatRequest(
            messages=[CustomWorkChatMessage(role="user", content="I need new shelving")],
            catalog_context=_catalog_context(),
        )
        with patch.object(custom_work_service, "settings", FakeSettings(openai_api_key="", openai_intake_model="")):
            result = chat_custom_work(request)

        assert result is _MOCK_RESPONSE
        assert result.model_name == "mock"

    def test_wraps_parsed_output_when_enabled(self):
        proposal = CustomWorkProposal(
            name="Bespoke shelving",
            description="Fit new bespoke shelving to alcove",
            unit="lm",
            labour_cost=120.0,
            material_cost=80.0,
            other_cost=0.0,
            work_days=1.0,
            qty_for_norm=1.0,
            suggested_work_group_id=1,
            suggested_work_group_name="Kitchen",
            market_justification="Standard joinery rate.",
            price_source_notes="Labour: Electrician (ELEC): £38.00/hr from Combit internal catalogue.",
            confidence_level=0.8,
        )
        parsed = LLMCustomWorkOutput(message="Here is your proposal.", status="proposing", work_proposal=proposal)
        fake_response = MagicMock()
        fake_response.choices[0].message.parsed = parsed
        fake_client = MagicMock()
        fake_client.beta.chat.completions.parse.return_value = fake_response

        request = CustomWorkChatRequest(
            messages=[CustomWorkChatMessage(role="user", content="I need new shelving")],
            catalog_context=_catalog_context(),
        )
        with patch.object(
            custom_work_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = chat_custom_work(request)

        assert result.message == "Here is your proposal."
        assert result.status == "proposing"
        assert result.work_proposal is proposal
        assert result.model_name == "gpt-4o"


class TestCustomWorkApi:
    def _client(self, monkeypatch) -> TestClient:
        monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_INTAKE_MODEL", "")

        from importlib import reload

        import app.config as app_config
        import app.security as app_security
        import app.services.custom_work_service as app_custom_work_service

        reload(app_config)
        reload(app_security)
        reload(app_custom_work_service)

        from app.main import app

        return TestClient(app)

    def test_endpoint_requires_api_key(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/estimate/custom-work/chat",
            json={
                "messages": [{"role": "user", "content": "I need new shelving"}],
                "catalog_context": {"work_groups": [], "labour_rates": []},
            },
        )

        assert response.status_code == 401

    def test_endpoint_returns_mock_response(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/estimate/custom-work/chat",
            json={
                "messages": [{"role": "user", "content": "I need new shelving"}],
                "catalog_context": {"work_groups": [], "labour_rates": []},
            },
            headers={"x-api-key": "test-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model_name"] == "mock"
        assert body["status"] == "chatting"
