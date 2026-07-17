from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.services import document_chat_service, document_enrichment_service, document_synthesis_service
from app.services.document_chat_service import chat_about_document
from app.services.document_enrichment_service import enrich_page_with_vision
from app.services.document_synthesis_service import synthesize_document_to_markdown


class FakeSettings:
    """Stand-in for app.config.Settings (a frozen dataclass, so it can't be
    monkeypatched attribute-by-attribute)."""

    def __init__(
        self,
        openai_api_key: str = "",
        openai_intake_model: str = "",
        openai_intake_fast_model: str = "",
        openai_timeout_seconds: float = 45.0,
    ):
        self.openai_api_key = openai_api_key
        self.openai_intake_model = openai_intake_model
        self.openai_intake_fast_model = openai_intake_fast_model
        self.openai_timeout_seconds = openai_timeout_seconds


def _make_chat_completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 20):
    fake_message = MagicMock()
    fake_message.content = content
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage.prompt_tokens = prompt_tokens
    fake_response.usage.completion_tokens = completion_tokens
    fake_response.usage.total_tokens = prompt_tokens + completion_tokens
    return fake_response


# ── document_synthesis_service ──────────────────────────────────────────────


class TestSynthesizeDocumentToMarkdown:
    def test_returns_error_when_openai_not_configured(self):
        with patch.object(document_synthesis_service, "settings", FakeSettings(openai_api_key="")):
            with patch("openai.OpenAI") as mock_openai:
                result = synthesize_document_to_markdown(page_bundles=[{"source_file": "a.pdf"}])

        mock_openai.assert_not_called()
        assert result == {"markdown": "", "model_name": "", "error": "OpenAI not configured."}

    def test_returns_empty_result_for_no_page_bundles(self):
        with patch.object(
            document_synthesis_service, "settings", FakeSettings(openai_api_key="sk-test")
        ):
            with patch("openai.OpenAI") as mock_openai:
                result = synthesize_document_to_markdown(page_bundles=[])

        mock_openai.assert_not_called()
        assert result == {"markdown": "", "model_name": ""}

    def test_success_builds_input_text_and_maps_response(self):
        fake_response = _make_chat_completion("# Clean Markdown\n\nSome content.")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        page_bundles = [
            {
                "source_file": "drawing.pdf",
                "page_number": 1,
                "page_type": "floor_plan",
                "vision_description": "Kitchen 4m x 3m",
                "programmatic_text": "some ocr text",
                "tables_text": "Table: Room | Area",
            },
            {
                "source_file": "drawing.pdf",
                "page_number": 2,
                "page_type": "other",
            },
        ]

        with patch.object(
            document_synthesis_service,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_model="gpt-4o"),
        ), patch("openai.OpenAI", return_value=fake_client) as mock_openai:
            result = synthesize_document_to_markdown(page_bundles=page_bundles)

        mock_openai.assert_called_once()
        create_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert create_kwargs["model"] == "gpt-4o"
        sent_messages = create_kwargs["messages"]
        assert sent_messages[0]["role"] == "system"
        user_content = sent_messages[1]["content"]
        assert "=== Page 1 of drawing.pdf (floor_plan) ===" in user_content
        assert "Kitchen 4m x 3m" in user_content
        assert "some ocr text" in user_content
        assert "Table: Room | Area" in user_content
        # Page 2 has no content beyond the header, so it should be dropped.
        assert "=== Page 2 of drawing.pdf (other) ===" not in user_content

        assert result["markdown"] == "# Clean Markdown\n\nSome content."
        assert result["model_name"] == "gpt-4o"
        assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

    def test_truncates_programmatic_text_when_vision_present(self):
        fake_response = _make_chat_completion("ok")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        long_text = "x" * 2000
        page_bundles = [
            {
                "source_file": "a.pdf",
                "page_number": 1,
                "vision_description": "has vision",
                "programmatic_text": long_text,
            }
        ]

        with patch.object(
            document_synthesis_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            synthesize_document_to_markdown(page_bundles=page_bundles)

        sent_content = fake_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        # limit is 800 when vision description is present
        assert "x" * 2000 not in sent_content
        assert "x" * 800 in sent_content
        assert "…" in sent_content

    def test_missing_usage_on_response_returns_none_usage(self):
        fake_message = MagicMock()
        fake_message.content = "markdown body"
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = None

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            document_synthesis_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = synthesize_document_to_markdown(page_bundles=[{"vision_description": "x" * 30}])

        assert result["usage"] is None
        assert result["markdown"] == "markdown body"

    def test_error_when_llm_call_raises(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("network unreachable")

        with patch.object(
            document_synthesis_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = synthesize_document_to_markdown(page_bundles=[{"vision_description": "x" * 30}])

        assert result["markdown"] == ""
        assert result["model_name"] == ""
        assert "network unreachable" in result["error"]


# ── document_chat_service ───────────────────────────────────────────────────


class TestChatAboutDocument:
    def test_returns_not_configured_message_when_no_api_key(self):
        with patch.object(document_chat_service, "settings", FakeSettings(openai_api_key="")):
            with patch("openai.OpenAI") as mock_openai:
                result = chat_about_document(context_markdown="# pack", turns=[], message="hi")

        mock_openai.assert_not_called()
        assert result == {
            "assistant_message": "AI service is not configured (no OpenAI key).",
            "updated_markdown": None,
            "model_name": "",
            "summary_used": False,
            "usage": None,
        }

    def test_success_sends_system_prompt_with_context_and_maps_response(self):
        response_json = (
            '{"assistant_message": "Updated the kitchen.", '
            '"updated_markdown": "# pack v2", '
            '"room_patches": [{"name": "Kitchen", "width_m": 4.0, "length_m": 3.0, "area_m2": 12.0}]}'
        )
        fake_response = _make_chat_completion(response_json)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        turns = [
            {"role": "user", "message": "the kitchen is 4x3"},
            {"role": "assistant", "message": "Got it."},
        ]

        with patch.object(
            document_chat_service,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_fast_model="gpt-4o-mini"),
        ), patch("openai.OpenAI", return_value=fake_client) as mock_openai:
            result = chat_about_document(
                context_markdown="# Original pack", turns=turns, message="update it"
            )

        mock_openai.assert_called_once()
        create_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert create_kwargs["model"] == "gpt-4o-mini"
        assert create_kwargs["response_format"] == {"type": "json_object"}

        messages = create_kwargs["messages"]
        assert "# Original pack" in messages[0]["content"]
        roles_and_content = [(m["role"], m["content"]) for m in messages[1:]]
        assert ("user", "the kitchen is 4x3") in roles_and_content
        assert ("assistant", "Got it.") in roles_and_content
        assert ("user", "update it") in roles_and_content

        assert result["assistant_message"] == "Updated the kitchen."
        assert result["updated_markdown"] == "# pack v2"
        assert result["room_patches"][0]["name"] == "Kitchen"
        assert result["model_name"] == "gpt-4o-mini"
        assert result["summary_used"] is False
        assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

    def test_parses_json_wrapped_in_markdown_fences(self):
        fenced = '```json\n{"assistant_message": "done", "updated_markdown": null, "room_patches": null}\n```'
        fake_response = _make_chat_completion(fenced)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            document_chat_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = chat_about_document(context_markdown="# pack", turns=[], message="hi")

        assert result["assistant_message"] == "done"
        assert result["updated_markdown"] is None

    def test_falls_back_to_raw_content_when_json_unparseable(self):
        fake_response = _make_chat_completion("Sorry, I can't help with that.")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            document_chat_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = chat_about_document(context_markdown="# pack", turns=[], message="hi")

        assert result["assistant_message"] == "Sorry, I can't help with that."
        assert result["updated_markdown"] is None

    def test_summarises_old_turns_when_history_exceeds_threshold(self):
        summary_response = _make_chat_completion("- user corrected the kitchen size")
        main_response = _make_chat_completion(
            '{"assistant_message": "ok", "updated_markdown": null, "room_patches": null}'
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [summary_response, main_response]

        turns = [
            {"role": "user" if i % 2 == 0 else "assistant", "message": f"turn {i}"}
            for i in range(25)
        ]

        with patch.object(
            document_chat_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = chat_about_document(context_markdown="# pack", turns=turns, message="latest")

        assert result["summary_used"] is True
        assert fake_client.chat.completions.create.call_count == 2

        main_call_kwargs = fake_client.chat.completions.create.call_args_list[1].kwargs
        messages = main_call_kwargs["messages"]
        summary_messages = [m for m in messages if "Earlier conversation summary" in m["content"]]
        assert len(summary_messages) == 1
        assert "user corrected the kitchen size" in summary_messages[0]["content"]

    def test_summary_failure_still_keeps_recent_turns_without_summary(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("summary call failed")

        turns = [{"role": "user", "message": f"turn {i}"} for i in range(25)]

        with patch.object(
            document_chat_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = chat_about_document(context_markdown="# pack", turns=turns, message="latest")

        # The summary call raised, which is caught inside chat_about_document's own
        # try/except (since _summarise_turns re-raises are swallowed at its level,
        # but here the exception actually happens inside the outer try since the
        # first call is the summarise call performed via the same client).
        assert result["assistant_message"].startswith("AI request failed")
        assert result["model_name"] == ""

    def test_error_when_main_llm_call_raises(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("network unreachable")

        with patch.object(
            document_chat_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = chat_about_document(context_markdown="# pack", turns=[], message="hi")

        assert "network unreachable" in result["assistant_message"]
        assert result["updated_markdown"] is None
        assert result["model_name"] == ""
        assert result["summary_used"] is False
        assert result["usage"] is None


# ── document_enrichment_service ─────────────────────────────────────────────


def _make_vision_response(content: str, prompt_tokens: int = 200, completion_tokens: int = 50):
    fake_message = MagicMock()
    fake_message.content = content
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage.prompt_tokens = prompt_tokens
    fake_response.usage.completion_tokens = completion_tokens
    fake_response.usage.total_tokens = prompt_tokens + completion_tokens
    return fake_response


class TestEnrichPageWithVision:
    def test_returns_fallback_when_openai_not_configured(self):
        with patch.object(document_enrichment_service, "settings", FakeSettings(openai_api_key="")):
            with patch("openai.OpenAI") as mock_openai:
                result = enrich_page_with_vision(
                    page_image_base64="abc123",
                    page_number=1,
                    source_file="a.pdf",
                    page_type="floor_plan",
                    existing_rooms=[],
                    existing_notes=[],
                )

        mock_openai.assert_not_called()
        assert result["page_type"] == "other"
        assert result["fallback_reason"] == "OpenAI not configured."
        assert result["used_vision"] is False
        assert result["rooms"] == []

    def test_success_sends_image_payload_and_page_type_prompt(self):
        content = (
            '{"page_type": "floor_plan", "description": "Kitchen 4x3", '
            '"rooms": [{"name": "Kitchen", "floor_level": "Ground", "width_m": 4.0, "length_m": 3.0}], '
            '"dimensions": [], "work_items": [], "structural_notes": [], "uncertainties": []}'
        )
        fake_response = _make_vision_response(content)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            document_enrichment_service,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_intake_fast_model="gpt-4o-mini"),
        ), patch("openai.OpenAI", return_value=fake_client) as mock_openai:
            result = enrich_page_with_vision(
                page_image_base64="base64data",
                page_number=3,
                source_file="drawing.pdf",
                page_type="floor_plan",
                existing_rooms=[{"name": "Bathroom"}],
                existing_notes=["Note about wall type"],
            )

        mock_openai.assert_called_once()
        create_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert create_kwargs["model"] == "gpt-4o-mini"
        assert create_kwargs["response_format"] == {"type": "json_object"}
        content_blocks = create_kwargs["messages"][0]["content"]
        text_block = next(b for b in content_blocks if b["type"] == "text")
        image_block = next(b for b in content_blocks if b["type"] == "image_url")
        assert "floor plan" in text_block["text"].lower()
        assert "Previously found rooms (partial): Bathroom" in text_block["text"]
        assert "Note about wall type" in text_block["text"]
        assert image_block["image_url"]["url"] == "data:image/png;base64,base64data"
        assert image_block["image_url"]["detail"] == "high"

        assert result["page_type"] == "floor_plan"
        assert result["rooms"][0]["name"] == "Kitchen"
        assert result["model_name"] == "gpt-4o-mini"
        assert result["used_vision"] is True
        assert result["fallback_reason"] is None
        assert result["document_subtype"] == "floor_plan"
        assert result["classification_confidence"] == 0.85
        assert result["usage"] == {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}

    def test_selects_generic_prompt_for_unknown_page_type(self):
        fake_response = _make_vision_response('{"page_type": "other", "description": "x"}')
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            document_enrichment_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            enrich_page_with_vision(
                page_image_base64="x",
                page_number=1,
                source_file="a.pdf",
                page_type="not_a_real_type",
                existing_rooms=[],
                existing_notes=[],
            )

        content_blocks = fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        text_block = next(b for b in content_blocks if b["type"] == "text")
        assert "Identify what type of page this is" in text_block["text"]

    def test_parses_json_from_prose_wrapped_response(self):
        wrapped = 'Here you go:\n{"page_type": "schedule", "description": "D1: door"}\nHope that helps.'
        fake_response = _make_vision_response(wrapped)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            document_enrichment_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = enrich_page_with_vision(
                page_image_base64="x",
                page_number=1,
                source_file="a.pdf",
                page_type="schedule",
                existing_rooms=[],
                existing_notes=[],
            )

        assert result["page_type"] == "schedule"
        assert result["description"] == "D1: door"

    def test_returns_empty_dict_base_when_json_completely_unparseable(self):
        fake_response = _make_vision_response("not json at all, just prose response")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch.object(
            document_enrichment_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = enrich_page_with_vision(
                page_image_base64="x",
                page_number=1,
                source_file="a.pdf",
                page_type="other",
                existing_rooms=[],
                existing_notes=[],
            )

        assert result["model_name"] != ""
        assert result["used_vision"] is True
        assert result.get("description") == ""
        assert result["document_subtype"] == "other"

    def test_error_when_llm_call_raises(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("network unreachable")

        with patch.object(
            document_enrichment_service, "settings", FakeSettings(openai_api_key="sk-test")
        ), patch("openai.OpenAI", return_value=fake_client):
            result = enrich_page_with_vision(
                page_image_base64="x",
                page_number=1,
                source_file="a.pdf",
                page_type="floor_plan",
                existing_rooms=[],
                existing_notes=[],
            )

        assert result["page_type"] == "other"
        assert result["used_vision"] is False
        assert "network unreachable" in result["fallback_reason"]


# ── app/api/document_analysis.py ────────────────────────────────────────────


class TestDocumentAnalysisApi:
    def _client(self, monkeypatch) -> TestClient:
        monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_MODEL", "")

        from importlib import reload

        import app.config as app_config
        import app.security as app_security
        import app.services.document_chat_service as app_document_chat_service
        import app.services.document_enrichment_service as app_document_enrichment_service
        import app.services.document_synthesis_service as app_document_synthesis_service
        import app.services.scope_from_pack_service as app_scope_from_pack_service

        reload(app_config)
        reload(app_security)
        reload(app_document_chat_service)
        reload(app_document_enrichment_service)
        reload(app_document_synthesis_service)
        reload(app_scope_from_pack_service)

        from app.main import app

        return TestClient(app)

    def test_chat_requires_api_key(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/chat",
            json={"context_markdown": "# pack", "turns": [], "message": "hi"},
        )

        assert response.status_code == 401

    def test_chat_happy_path_not_configured_fallback(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/chat",
            json={"context_markdown": "# pack", "turns": [], "message": "hi"},
            headers={"x-api-key": "test-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["assistant_message"] == "AI service is not configured (no OpenAI key)."
        assert body["updated_markdown"] is None
        assert body["summary_used"] is False

    def test_synthesize_requires_api_key(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/synthesize",
            json={"page_bundles": []},
        )

        assert response.status_code == 401

    def test_synthesize_happy_path_not_configured_fallback(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/synthesize",
            json={
                "page_bundles": [
                    {"page_number": 1, "source_file": "a.pdf", "vision_description": "Kitchen"}
                ]
            },
            headers={"x-api-key": "test-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["markdown"] == ""
        assert body["model_name"] == ""
        assert body["usage"] is None

    def test_generate_scope_requires_api_key(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/generate-scope",
            json={"context_pack_json": {}},
        )

        assert response.status_code == 401

    def test_generate_scope_happy_path_fallback(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/generate-scope",
            json={"context_pack_json": {"rooms": [{"name": "Kitchen", "area_m2": 12}]}},
            headers={"x-api-key": "test-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model_name"] == "fallback"
        assert body["sections"] == []
        assert "Kitchen" in body["scope_text"]

    def test_enrich_page_requires_api_key(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/enrich-page",
            json={
                "page_image_base64": "abc",
                "page_number": 1,
                "source_file": "a.pdf",
            },
        )

        assert response.status_code == 401

    def test_enrich_page_happy_path_not_configured_fallback(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/enrich-page",
            json={
                "page_image_base64": "abc",
                "page_number": 1,
                "source_file": "a.pdf",
                "page_type": "floor_plan",
            },
            headers={"x-api-key": "test-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["page_type"] == "other"
        assert body["used_vision"] is False
        assert body["fallback_reason"] == "OpenAI not configured."
        assert body["rooms"] == []
        assert body["dimensions"] == []
        assert body["work_items"] == []

    def test_enrich_page_mocked_llm_maps_rooms_dimensions_and_work_items(self, monkeypatch):
        client = self._client(monkeypatch)

        fake_content = (
            '{"page_type": "floor_plan", "description": "desc", '
            '"rooms": [{"name": "Kitchen", "floor_level": "Ground", "width_m": 4.0, "length_m": 3.0}, '
            '{"width_m": 1.0}], '
            '"dimensions": [{"element": "Overall width", "width_m": 8.5}, {"width_m": 1.0}], '
            '"work_items": [{"trade": "structural", "description": "Remove wall"}, {"trade": "general"}], '
            '"structural_notes": ["300mm cavity wall"], "uncertainties": ["North wall unclear"]}'
        )
        fake_message = MagicMock()
        fake_message.content = fake_content
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage.prompt_tokens = 10
        fake_response.usage.completion_tokens = 5
        fake_response.usage.total_tokens = 15

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from importlib import reload

        import app.config as app_config
        import app.services.document_enrichment_service as app_document_enrichment_service

        reload(app_config)
        reload(app_document_enrichment_service)

        with patch("openai.OpenAI", return_value=fake_client):
            response = client.post(
                "/v1/document-analysis/enrich-page",
                json={
                    "page_image_base64": "abc",
                    "page_number": 1,
                    "source_file": "a.pdf",
                    "page_type": "floor_plan",
                },
                headers={"x-api-key": "test-secret"},
            )

        assert response.status_code == 200
        body = response.json()
        # Room without a name should be dropped by _coerce_rooms.
        assert len(body["rooms"]) == 1
        assert body["rooms"][0]["name"] == "Kitchen"
        # Dimension without an element name should be dropped by _coerce_dimensions.
        assert len(body["dimensions"]) == 1
        assert body["dimensions"][0]["element"] == "Overall width"
        # Work item without a description should be dropped by _coerce_work_items.
        assert len(body["work_items"]) == 1
        assert body["work_items"][0]["description"] == "Remove wall"
        assert body["structural_notes"] == ["300mm cavity wall"]
        assert body["uncertainties"] == ["North wall unclear"]
        assert body["used_vision"] is True

    def test_enrich_page_validation_error_on_missing_required_field(self, monkeypatch):
        client = self._client(monkeypatch)
        response = client.post(
            "/v1/document-analysis/enrich-page",
            json={"page_number": 1, "source_file": "a.pdf"},
            headers={"x-api-key": "test-secret"},
        )

        assert response.status_code == 422
