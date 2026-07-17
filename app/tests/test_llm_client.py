from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.schemas import LLMPreviewOutput
from app.services import llm_client as app_llm_client

# LLMClient / LLMStructuredOutputError / LLMRequestError are looked up via
# app_llm_client.* (not imported by name) because other test modules in this
# suite call importlib.reload(app_llm_client), which rebinds these classes to
# new objects; a name captured at collection time would stop matching with
# `except` after such a reload runs.


class FakeSettings:
    """Stand-in for app.config.Settings, since the real one is a frozen dataclass
    and cannot be monkeypatched attribute-by-attribute."""

    def __init__(
        self,
        openai_api_key: str = "",
        openai_model: str = "",
        openai_timeout_seconds: float = 45.0,
        openai_request_max_attempts: int = 2,
    ):
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_timeout_seconds = openai_timeout_seconds
        self.openai_request_max_attempts = openai_request_max_attempts


def _empty_llm_output() -> LLMPreviewOutput:
    return LLMPreviewOutput(matched_rows=[], unmatched_items=[], assumptions=[])


class TestLLMClientInit:
    def test_disabled_when_no_api_key(self):
        with patch.object(app_llm_client, "settings", FakeSettings(openai_api_key="", openai_model="")):
            client = app_llm_client.LLMClient()

        assert client.is_enabled() is False
        assert client.client is None

    def test_enabled_constructs_openai_client(self):
        fake_client = MagicMock()
        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client) as mock_openai:
            client = app_llm_client.LLMClient()

        assert client.is_enabled() is True
        assert client.client is fake_client
        mock_openai.assert_called_once()


class TestSupportsStructuredOutputApis:
    def test_supports_responses_parse_false_when_client_none(self):
        with patch.object(app_llm_client, "settings", FakeSettings(openai_api_key="", openai_model="")):
            client = app_llm_client.LLMClient()

        assert client._supports_responses_parse() is False
        assert client._supports_chat_completions_parse() is False

    def test_supports_responses_parse_true_when_available(self):
        fake_client = MagicMock()
        fake_client.responses.parse = MagicMock()
        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()

        assert client._supports_responses_parse() is True

    def test_supports_chat_completions_parse_true_when_available(self):
        fake_client = MagicMock()
        fake_client.responses = None
        fake_client.beta.chat.completions.parse = MagicMock()
        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()

        assert client._supports_chat_completions_parse() is True


class TestRequestStructuredPreview:
    def test_raises_when_client_not_configured(self):
        with patch.object(app_llm_client, "settings", FakeSettings(openai_api_key="", openai_model="")):
            client = app_llm_client.LLMClient()

        try:
            client._request_structured_preview(
                prompt="Paint walls 20m2",
                candidate_rows=[],
                extracted_scope=None,
                accepted_examples=None,
                retry_mode=False,
            )
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "not configured" in str(exc)

    def test_uses_responses_parse_when_available(self):
        parsed = _empty_llm_output()
        fake_usage = MagicMock(input_tokens=100, output_tokens=50, total_tokens=150)
        fake_response = MagicMock(output_parsed=parsed, usage=fake_usage)
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response

        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            result, metadata = client._request_structured_preview(
                prompt="Paint walls 20m2",
                candidate_rows=[],
                extracted_scope=None,
                accepted_examples=None,
                retry_mode=False,
            )

        assert result is parsed
        assert metadata["structured_output_mode"] == "responses.parse"
        assert metadata["usage"] == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        fake_client.responses.parse.assert_called_once()
        _, kwargs = fake_client.responses.parse.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["text_format"] is LLMPreviewOutput

    def test_raises_structured_output_error_when_responses_parse_returns_none(self):
        fake_response = MagicMock(output_parsed=None)
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response

        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            try:
                client._request_structured_preview(
                    prompt="Paint walls 20m2",
                    candidate_rows=[],
                    extracted_scope=None,
                    accepted_examples=None,
                    retry_mode=False,
                )
                assert False, "expected LLMStructuredOutputError"
            except app_llm_client.LLMStructuredOutputError:
                pass

    def test_falls_back_to_chat_completions_parse_when_responses_unavailable(self):
        parsed = _empty_llm_output()
        fake_message = MagicMock(parsed=parsed)
        fake_choice = MagicMock(message=fake_message)
        fake_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        fake_response = MagicMock(choices=[fake_choice], usage=fake_usage)
        fake_client = MagicMock()
        fake_client.responses = None
        fake_client.beta.chat.completions.parse.return_value = fake_response

        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            result, metadata = client._request_structured_preview(
                prompt="Paint walls 20m2",
                candidate_rows=[],
                extracted_scope=None,
                accepted_examples=None,
                retry_mode=False,
            )

        assert result is parsed
        assert metadata["structured_output_mode"] == "chat.completions.parse"
        assert metadata["usage"]["prompt_tokens"] == 10

    def test_raises_structured_output_error_when_neither_api_available(self):
        fake_client = MagicMock()
        fake_client.responses = None
        fake_client.beta.chat.completions.parse = None

        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            try:
                client._request_structured_preview(
                    prompt="Paint walls 20m2",
                    candidate_rows=[],
                    extracted_scope=None,
                    accepted_examples=None,
                    retry_mode=False,
                )
                assert False, "expected LLMStructuredOutputError"
            except app_llm_client.LLMStructuredOutputError as exc:
                assert "does not support structured parsing" in str(exc)

    def test_wraps_unexpected_exception_in_llm_request_error(self):
        fake_client = MagicMock()
        fake_client.responses.parse.side_effect = RuntimeError("network unreachable")

        with patch.object(
            app_llm_client, "settings", FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o")
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            try:
                client._request_structured_preview(
                    prompt="Paint walls 20m2",
                    candidate_rows=[],
                    extracted_scope=None,
                    accepted_examples=None,
                    retry_mode=False,
                )
                assert False, "expected LLMRequestError"
            except app_llm_client.LLMRequestError as exc:
                assert "network unreachable" in str(exc)


class TestPreviewMatch:
    def test_raises_when_not_enabled(self):
        with patch.object(app_llm_client, "settings", FakeSettings(openai_api_key="", openai_model="")):
            client = app_llm_client.LLMClient()

        try:
            client.preview_match(prompt="Paint walls 20m2", candidate_rows=[])
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "not configured" in str(exc)

    def test_returns_parsed_output_on_first_attempt(self):
        parsed = _empty_llm_output()
        fake_response = MagicMock(output_parsed=parsed, usage=None)
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response

        with patch.object(
            app_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o", openai_request_max_attempts=2),
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            result = client.preview_match(prompt="Paint walls 20m2", candidate_rows=[])

        assert result is parsed
        assert client.last_preview_metadata["llm_attempt_count"] == 1
        assert client.last_preview_metadata["parse_retry_used"] is False
        fake_client.responses.parse.assert_called_once()

    def test_retries_with_retry_mode_after_structured_output_error(self):
        fake_response_bad = MagicMock(output_parsed=None, usage=None)
        parsed = _empty_llm_output()
        fake_response_good = MagicMock(output_parsed=parsed, usage=None)
        fake_client = MagicMock()
        fake_client.responses.parse.side_effect = [fake_response_bad, fake_response_good]

        with patch.object(
            app_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o", openai_request_max_attempts=1),
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            result = client.preview_match(prompt="Paint walls 20m2", candidate_rows=[])

        assert result is parsed
        assert fake_client.responses.parse.call_count == 2
        assert client.last_preview_metadata["parse_retry_used"] is True
        assert client.last_preview_metadata["llm_attempt_count"] == 2

    def test_retries_request_error_up_to_max_attempts_then_raises(self):
        fake_client = MagicMock()
        fake_client.responses.parse.side_effect = RuntimeError("network unreachable")

        with patch.object(
            app_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o", openai_request_max_attempts=2),
        ), patch("openai.OpenAI", return_value=fake_client), patch("app.services.llm_client.sleep"):
            client = app_llm_client.LLMClient()
            try:
                client.preview_match(prompt="Paint walls 20m2", candidate_rows=[])
                assert False, "expected LLMRequestError"
            except app_llm_client.LLMRequestError:
                pass

        # 2 request attempts per retry_mode pass (False, True) = 4 total calls.
        assert fake_client.responses.parse.call_count == 4

    def test_raises_structured_output_error_when_both_passes_fail_to_parse(self):
        fake_response_bad = MagicMock(output_parsed=None, usage=None)
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = fake_response_bad

        with patch.object(
            app_llm_client,
            "settings",
            FakeSettings(openai_api_key="sk-test", openai_model="gpt-4o", openai_request_max_attempts=1),
        ), patch("openai.OpenAI", return_value=fake_client):
            client = app_llm_client.LLMClient()
            try:
                client.preview_match(prompt="Paint walls 20m2", candidate_rows=[])
                assert False, "expected LLMStructuredOutputError"
            except app_llm_client.LLMStructuredOutputError:
                pass

        assert fake_client.responses.parse.call_count == 2
