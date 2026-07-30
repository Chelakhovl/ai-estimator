from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services import retry as retry_module
from app.services.retry import call_with_request_retries


class FakeSettings:
    def __init__(self, openai_request_max_attempts: int = 3):
        self.openai_request_max_attempts = openai_request_max_attempts


class TestCallWithRequestRetries:
    def test_returns_result_on_first_success_without_retrying(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        with patch.object(retry_module, "settings", FakeSettings(openai_request_max_attempts=3)):
            result = call_with_request_retries(fn, label="test-call")

        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_failure_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("transient")
            return "recovered"

        with patch.object(retry_module, "settings", FakeSettings(openai_request_max_attempts=3)), patch(
            "app.services.retry.sleep"
        ):
            result = call_with_request_retries(fn, label="test-call")

        assert result == "recovered"
        assert len(calls) == 2

    def test_raises_last_error_after_exhausting_all_attempts(self):
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError(f"failure {len(calls)}")

        with patch.object(retry_module, "settings", FakeSettings(openai_request_max_attempts=2)), patch(
            "app.services.retry.sleep"
        ):
            with pytest.raises(RuntimeError, match="failure 2"):
                call_with_request_retries(fn, label="test-call")

        assert len(calls) == 2
