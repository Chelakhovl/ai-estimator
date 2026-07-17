from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.schemas import (
    IntakeAssetExcerpt,
    IntakeChatRequest,
    IntakeFinalizeRequest,
    IntakeQuestion,
    IntakeTurn,
    LLMIntakeChatOutput,
    LLMIntakeFinalizeOutput,
    ProjectBriefFact,
    ProjectBriefSection,
    ProjectBriefV1,
)
from app.services import project_intake_llm_client, project_intake_service
from app.services.project_intake_knowledge import load_project_intake_knowledge
from app.services.project_intake_llm_client import IntakeLLMClient
from app.services.project_intake_service import (
    _build_missing_items,
    _empty_project_brief,
    _merge_current_brief,
    chat_project_intake,
    finalize_project_intake,
)


def _empty_brief() -> ProjectBriefV1:
    return _empty_project_brief()


def _asset(excerpt: str = "PROPERTY: Victorian house\nKitchen: 3.5 x 5.8") -> IntakeAssetExcerpt:
    return IntakeAssetExcerpt(
        asset_id=1,
        filename="drawings.pdf",
        asset_type="pdf",
        checksum="abc",
        extraction_status="completed",
        document_excerpt=excerpt,
    )


def _chat_request(message: str = "Client uploaded the kitchen drawings.") -> IntakeChatRequest:
    return IntakeChatRequest(
        session_id=1,
        quote_id=12,
        current_project_brief_json={},
        assets=[_asset()],
        turns=[],
        message=message,
    )


def _finalize_request() -> IntakeFinalizeRequest:
    return IntakeFinalizeRequest(
        session_id=1,
        quote_id=12,
        current_project_brief_json={},
        assets=[_asset()],
        turns=[],
    )


class FakeSettings:
    """Stand-in for app.config.Settings, since the real one is a frozen dataclass
    and cannot be monkeypatched attribute-by-attribute."""

    def __init__(
        self,
        openai_api_key: str = "",
        openai_intake_model: str = "",
        openai_intake_fast_model: str = "",
        openai_timeout_seconds: float = 45.0,
        allow_mock_fallback: bool = True,
        openai_model: str = "",
    ):
        self.openai_api_key = openai_api_key
        self.openai_intake_model = openai_intake_model
        self.openai_intake_fast_model = openai_intake_fast_model
        self.openai_timeout_seconds = openai_timeout_seconds
        self.allow_mock_fallback = allow_mock_fallback
        self.openai_model = openai_model


class TestLoadProjectIntakeKnowledge:
    def test_loads_instructions_and_knowledge_files(self, tmp_path):
        estimator_files = tmp_path / "estimator_files"
        estimator_files.mkdir()
        (estimator_files / "PROJECT_INSTRUCTIONS_PASTE_INTO_CLAUDE.md").write_text(
            "Main instructions.", encoding="utf-8"
        )
        knowledge_dir = estimator_files / "knowledge"
        knowledge_dir.mkdir()
        (knowledge_dir / "01_first.txt").write_text("First chunk.", encoding="utf-8")
        (knowledge_dir / "02_second.txt").write_text("Second chunk.", encoding="utf-8")

        load_project_intake_knowledge.cache_clear()
        with patch(
            "app.services.project_intake_knowledge._repo_root", return_value=tmp_path
        ):
            result = load_project_intake_knowledge()
        load_project_intake_knowledge.cache_clear()

        assert "Main instructions." in result
        assert "First chunk." in result
        assert "Second chunk." in result
        assert result.index("First chunk.") < result.index("Second chunk.")

    def test_missing_knowledge_dir_only_loads_instructions(self, tmp_path):
        estimator_files = tmp_path / "estimator_files"
        estimator_files.mkdir()
        (estimator_files / "PROJECT_INSTRUCTIONS_PASTE_INTO_CLAUDE.md").write_text(
            "Only instructions.", encoding="utf-8"
        )

        load_project_intake_knowledge.cache_clear()
        with patch(
            "app.services.project_intake_knowledge._repo_root", return_value=tmp_path
        ):
            result = load_project_intake_knowledge()
        load_project_intake_knowledge.cache_clear()

        assert result.strip() == f"# PROJECT_INSTRUCTIONS_PASTE_INTO_CLAUDE.md\nOnly instructions."


class TestIntakeLLMClient:
    def test_disabled_when_no_api_key(self):
        with patch.object(
            project_intake_llm_client, "settings", FakeSettings(openai_api_key="", openai_intake_model="")
        ):
            client = IntakeLLMClient()

        assert client.is_enabled() is False
        assert client.client is None

    def test_enabled_constructs_openai_client(self):
        fake_client = MagicMock()
        with patch.object(
            project_intake_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o", openai_intake_fast_model="gpt-4o-mini"),
        ), patch("openai.OpenAI", return_value=fake_client) as mock_openai:
            client = IntakeLLMClient()

        assert client.is_enabled() is True
        assert client.client is fake_client
        mock_openai.assert_called_once()

    def test_supports_responses_parse_false_when_not_enabled(self):
        with patch.object(
            project_intake_llm_client, "settings", FakeSettings(openai_api_key="", openai_intake_model="")
        ):
            client = IntakeLLMClient()

        assert client._supports_responses_parse() is False

    def test_chat_raises_when_not_enabled(self):
        with patch.object(
            project_intake_llm_client, "settings", FakeSettings(openai_api_key="", openai_intake_model="")
        ):
            client = IntakeLLMClient()

        try:
            client.chat(_chat_request())
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "not configured" in str(exc)

    def test_chat_returns_parsed_output_using_fast_model(self):
        parsed = LLMIntakeChatOutput(
            assistant_message="ok",
            updated_project_brief_json=_empty_brief(),
        )
        fake_response = MagicMock()
        fake_response.output_parsed = parsed
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response

        with patch.object(
            project_intake_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o", openai_intake_fast_model="gpt-4o-mini"),
        ), patch("openai.OpenAI", return_value=fake_client), patch(
            "app.services.project_intake_llm_client.load_project_intake_knowledge",
            return_value="knowledge",
        ):
            client = IntakeLLMClient()
            result = client.chat(_chat_request())

        assert result is parsed
        assert client.last_metadata["model_name"] == "gpt-4o-mini"
        fake_client.responses.parse.assert_called_once()
        _, kwargs = fake_client.responses.parse.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["text_format"] is LLMIntakeChatOutput

    def test_chat_raises_when_parsed_is_none(self):
        fake_response = MagicMock()
        fake_response.output_parsed = None
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response

        with patch.object(
            project_intake_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o", openai_intake_fast_model="gpt-4o-mini"),
        ), patch("openai.OpenAI", return_value=fake_client), patch(
            "app.services.project_intake_llm_client.load_project_intake_knowledge",
            return_value="knowledge",
        ):
            client = IntakeLLMClient()
            try:
                client.chat(_chat_request())
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "no structured output" in str(exc)

    def test_finalize_raises_when_not_enabled(self):
        with patch.object(
            project_intake_llm_client, "settings", FakeSettings(openai_api_key="", openai_intake_model="")
        ):
            client = IntakeLLMClient()

        try:
            client.finalize(_finalize_request())
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "not configured" in str(exc)

    def test_finalize_returns_parsed_output_using_main_model(self):
        parsed = LLMIntakeFinalizeOutput(
            project_brief_json=_empty_brief(),
            structured_scope_markdown="PROPERTY:\nVictorian house",
        )
        fake_response = MagicMock()
        fake_response.output_parsed = parsed
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response

        with patch.object(
            project_intake_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o", openai_intake_fast_model="gpt-4o-mini"),
        ), patch("openai.OpenAI", return_value=fake_client), patch(
            "app.services.project_intake_llm_client.load_project_intake_knowledge",
            return_value="knowledge",
        ):
            client = IntakeLLMClient()
            result = client.finalize(_finalize_request())

        assert result is parsed
        assert client.last_metadata["model_name"] == "gpt-4o"
        fake_client.responses.parse.assert_called_once()
        _, kwargs = fake_client.responses.parse.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["text_format"] is LLMIntakeFinalizeOutput

    def test_build_asset_content_includes_pdf_file_and_summary(self):
        with patch.object(
            project_intake_llm_client, "settings", FakeSettings(openai_api_key="", openai_intake_model="")
        ):
            client = IntakeLLMClient()

        asset = IntakeAssetExcerpt(
            asset_id=1,
            filename="drawings.pdf",
            asset_type="pdf",
            checksum="abc",
            extraction_status="completed",
            document_excerpt="Kitchen: 3.5 x 5.8",
            pdf_base64="base64data",
            warnings=["low confidence OCR"],
        )
        content = client._build_asset_content([asset])

        assert content[0] == {
            "type": "input_file",
            "filename": "drawings.pdf",
            "file_data": "base64data",
        }
        assert content[1]["type"] == "input_text"
        assert "Kitchen: 3.5 x 5.8" in content[1]["text"]
        assert "low confidence OCR" in content[1]["text"]


class TestMergeCurrentBrief:
    def test_returns_generated_brief_when_no_current_brief(self):
        generated = _empty_brief()
        result = _merge_current_brief({}, generated)
        assert result is generated

    def test_returns_generated_brief_when_current_brief_invalid(self):
        generated = _empty_brief()
        result = _merge_current_brief({"not": "a valid brief"}, generated)
        assert result is generated

    def test_merges_facts_and_keeps_summary_when_generated_is_blank(self):
        current = _empty_brief()
        current.project_overview.summary = "Victorian house"
        current.project_overview.facts.append(
            ProjectBriefFact(text="Victorian house", source="document", confidence="high")
        )
        current_json = current.model_dump(mode="json")

        generated = _empty_brief()
        result = _merge_current_brief(current_json, generated)

        assert result.project_overview.summary == "Victorian house"
        assert any(fact.text == "Victorian house" for fact in result.project_overview.facts)

    def test_merges_source_documents_and_confidence_summary(self):
        current = _empty_brief()
        current.source_documents = [{"filename": "old.pdf"}]
        current.confidence_summary = {"asset_count": 1}
        current_json = current.model_dump(mode="json")

        generated = _empty_brief()
        generated.confidence_summary = {"room_count": 2}
        result = _merge_current_brief(current_json, generated)

        assert result.source_documents == [{"filename": "old.pdf"}]
        assert result.confidence_summary == {"asset_count": 1, "room_count": 2}


class TestBuildMissingItems:
    def test_flags_missing_areas_site_conditions_and_supply_split(self):
        brief = _empty_brief()
        missing_items, questions, warnings = _build_missing_items(brief)

        assert "Room dimensions or area schedule are still missing." in missing_items
        assert any(q.id == "areas-dimensions" for q in questions)
        assert any(q.id == "site-conditions" for q in questions)
        assert any(q.id == "supply-split" for q in questions)
        assert any("MEP scope is still thin" in w for w in warnings)
        assert brief.missing_rfi.facts

    def test_no_missing_items_when_brief_is_complete(self):
        brief = _empty_brief()
        brief.areas_dimensions.facts.append(ProjectBriefFact(text="Kitchen: 3.5 x 5.8"))
        brief.site_conditions.facts.append(ProjectBriefFact(text="Vacant, no scaffold needed"))
        brief.client_supply_vs_combit_supply.facts.append(ProjectBriefFact(text="Combit supply throughout"))
        brief.mep.facts.append(ProjectBriefFact(text="12 downlights"))

        missing_items, questions, warnings = _build_missing_items(brief)

        assert missing_items == []
        assert questions == []

    def test_document_warning_count_adds_extraction_warning(self):
        brief = _empty_brief()
        brief.confidence_summary = {"document_warning_count": 2}

        _missing_items, _questions, warnings = _build_missing_items(brief)

        assert any("extraction-poor" in w for w in warnings)


class TestChatProjectIntake:
    def test_uses_fallback_when_llm_not_configured(self):
        with patch.object(
            project_intake_service, "settings", FakeSettings(openai_api_key="", openai_intake_model="")
        ):
            result = chat_project_intake(_chat_request())

        assert result.model_name == "fallback-intake"
        assert result.updated_project_brief_json.areas_dimensions.facts

    def test_uses_llm_result_when_enabled(self):
        llm_output = LLMIntakeChatOutput(
            assistant_message="What's the kitchen size?",
            updated_project_brief_json=_empty_brief(),
            questions=[IntakeQuestion(id="q1", label="Kitchen size?", reason="needed for takeoff")],
            ready_to_finalize=False,
        )
        fake_client = MagicMock()
        fake_client.is_enabled.return_value = True
        fake_client.chat.return_value = llm_output
        fake_client.last_metadata = {"model_name": "gpt-4o-mini"}

        with patch.object(
            project_intake_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch.object(project_intake_service, "IntakeLLMClient", return_value=fake_client):
            result = chat_project_intake(_chat_request())

        assert result.assistant_message == "What's the kitchen size?"
        assert result.model_name == "gpt-4o-mini"
        assert result.questions[0].id == "q1"

    def test_falls_back_when_llm_raises_and_fallback_allowed(self):
        fake_client = MagicMock()
        fake_client.is_enabled.return_value = True
        fake_client.chat.side_effect = RuntimeError("network unreachable")

        with patch.object(
            project_intake_service,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o", allow_mock_fallback=True),
        ), patch.object(project_intake_service, "IntakeLLMClient", return_value=fake_client):
            result = chat_project_intake(_chat_request())

        assert result.model_name == "fallback-intake"

    def test_reraises_when_llm_raises_and_fallback_disabled(self):
        fake_client = MagicMock()
        fake_client.is_enabled.return_value = True
        fake_client.chat.side_effect = RuntimeError("network unreachable")

        with patch.object(
            project_intake_service,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o", allow_mock_fallback=False),
        ), patch.object(project_intake_service, "IntakeLLMClient", return_value=fake_client):
            try:
                chat_project_intake(_chat_request())
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "network unreachable" in str(exc)


class TestFinalizeProjectIntake:
    def test_uses_fallback_when_llm_not_configured(self):
        with patch.object(
            project_intake_service, "settings", FakeSettings(openai_api_key="", openai_intake_model="")
        ):
            result = finalize_project_intake(_finalize_request())

        assert result.model_name == "fallback-intake"
        assert result.handoff_readiness is True
        assert "PROPERTY:" in result.structured_scope_markdown

    def test_uses_llm_result_when_enabled(self):
        llm_output = LLMIntakeFinalizeOutput(
            project_brief_json=_empty_brief(),
            structured_scope_markdown="PROPERTY:\nVictorian house",
            handoff_readiness=True,
        )
        fake_client = MagicMock()
        fake_client.is_enabled.return_value = True
        fake_client.finalize.return_value = llm_output
        fake_client.last_metadata = {"model_name": "gpt-4o"}

        with patch.object(
            project_intake_service, "settings", FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o")
        ), patch.object(project_intake_service, "IntakeLLMClient", return_value=fake_client):
            result = finalize_project_intake(_finalize_request())

        assert result.model_name == "gpt-4o"
        assert result.handoff_readiness is True
        assert result.structured_scope_markdown == "PROPERTY:\nVictorian house"

    def test_falls_back_when_llm_raises_and_fallback_allowed(self):
        fake_client = MagicMock()
        fake_client.is_enabled.return_value = True
        fake_client.finalize.side_effect = RuntimeError("network unreachable")

        with patch.object(
            project_intake_service,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o", allow_mock_fallback=True),
        ), patch.object(project_intake_service, "IntakeLLMClient", return_value=fake_client):
            result = finalize_project_intake(_finalize_request())

        assert result.model_name == "fallback-intake"
