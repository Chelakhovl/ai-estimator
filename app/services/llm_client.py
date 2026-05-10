from __future__ import annotations

import logging
from time import sleep
from time import perf_counter
from typing import TYPE_CHECKING

from app.config import settings
from app.schemas import CandidateRow
from app.schemas import LLMPreviewOutput
from app.schemas import PreviewAcceptedExample
from app.schemas import ScopeExtractionResponse
from app.services.prompt_builder import build_preview_system_prompt, build_preview_user_message

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMStructuredOutputError(ValueError):
    pass


class LLMRequestError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        self.model = settings.openai_model
        self.enabled = bool(settings.openai_api_key and settings.openai_model)
        self.client: OpenAI | None = None
        self.last_preview_metadata: dict[str, object] = {}
        if self.enabled:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=3,
            )

    def is_enabled(self) -> bool:
        return self.enabled

    def _supports_responses_parse(self) -> bool:
        if self.client is None:
            return False
        responses_api = getattr(self.client, "responses", None)
        return callable(getattr(responses_api, "parse", None))

    def _supports_chat_completions_parse(self) -> bool:
        if self.client is None:
            return False
        beta_api = getattr(self.client, "beta", None)
        chat_api = getattr(beta_api, "chat", None)
        completions_api = getattr(chat_api, "completions", None)
        return callable(getattr(completions_api, "parse", None))

    def _build_messages(
        self,
        *,
        prompt: str,
        candidate_rows: list[CandidateRow],
        extracted_scope: ScopeExtractionResponse | None,
        accepted_examples: list[PreviewAcceptedExample] | None,
        retry_mode: bool,
        document_context: str | None = None,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": build_preview_system_prompt(retry_mode=retry_mode)},
            {
                "role": "user",
                "content": build_preview_user_message(prompt, candidate_rows, extracted_scope, accepted_examples, document_context),
            },
        ]

    def _request_structured_preview(
        self,
        *,
        prompt: str,
        candidate_rows: list[CandidateRow],
        extracted_scope: ScopeExtractionResponse | None,
        accepted_examples: list[PreviewAcceptedExample] | None,
        retry_mode: bool,
        document_context: str | None = None,
    ) -> tuple[LLMPreviewOutput, dict[str, object]]:
        if self.client is None:
            raise RuntimeError("LLM client is not configured.")

        messages = self._build_messages(
            prompt=prompt,
            candidate_rows=candidate_rows,
            extracted_scope=extracted_scope,
            accepted_examples=accepted_examples,
            retry_mode=retry_mode,
            document_context=document_context,
        )

        try:
            if self._supports_responses_parse():
                started_at = perf_counter()
                response = self.client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=LLMPreviewOutput,
                    temperature=0,
                )
                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    raise LLMStructuredOutputError("Model returned no parsed structured output.")
                usage = getattr(response, "usage", None)
                return parsed, {
                    "llm_latency_ms": int((perf_counter() - started_at) * 1000),
                    "structured_output_mode": "responses.parse",
                    "usage": {
                        "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                        "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                    },
                }

            if not self._supports_chat_completions_parse():
                raise LLMStructuredOutputError("The installed OpenAI SDK does not support structured parsing APIs.")

            started_at = perf_counter()
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=LLMPreviewOutput,
                temperature=0,
            )
            parsed = response.choices[0].message.parsed if response.choices else None
            if parsed is None:
                raise LLMStructuredOutputError("Model returned no parsed structured output.")
            usage = getattr(response, "usage", None)
            return parsed, {
                "llm_latency_ms": int((perf_counter() - started_at) * 1000),
                "structured_output_mode": "chat.completions.parse",
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                },
            }
        except LLMStructuredOutputError:
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            raise LLMRequestError(
                f"LLM request failed before a structured preview could be produced: {exc}"
            ) from exc

    def preview_match(
        self,
        *,
        prompt: str,
        candidate_rows: list[CandidateRow],
        extracted_scope: ScopeExtractionResponse | None = None,
        accepted_examples: list[PreviewAcceptedExample] | None = None,
        document_context: str | None = None,
    ) -> LLMPreviewOutput:
        if not self.enabled or self.client is None:
            raise RuntimeError("LLM client is not configured.")

        last_error: Exception | None = None
        structured_output_mode = "responses.parse" if self._supports_responses_parse() else "chat.completions.parse"
        self.last_preview_metadata = {}
        for attempt_number, retry_mode in enumerate((False, True), start=1):
            for request_attempt in range(1, settings.openai_request_max_attempts + 1):
                try:
                    parsed, request_metadata = self._request_structured_preview(
                        prompt=prompt,
                        candidate_rows=candidate_rows,
                        extracted_scope=extracted_scope,
                        accepted_examples=accepted_examples,
                        retry_mode=retry_mode,
                        document_context=document_context,
                    )
                    latency_ms = int(request_metadata.get("llm_latency_ms", 0) or 0)
                    usage = request_metadata.get("usage", {}) or {}
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    logger.info(
                        "LLM preview completed model=%s latency=%dms "
                        "prompt_tokens=%s completion_tokens=%s total_tokens=%s attempt=%d",
                        self.model, latency_ms,
                        prompt_tokens, completion_tokens, total_tokens,
                        attempt_number,
                    )
                    self.last_preview_metadata = {
                        "llm_attempt_count": attempt_number,
                        "llm_latency_ms": latency_ms,
                        "llm_request_attempt_count": request_attempt,
                        "parse_retry_used": retry_mode,
                        "structured_output_mode": request_metadata.get("structured_output_mode", structured_output_mode) or structured_output_mode,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                    return parsed
                except LLMStructuredOutputError as exc:
                    last_error = exc
                    self.last_preview_metadata = {
                        "llm_attempt_count": attempt_number,
                        "llm_latency_ms": 0,
                        "llm_request_attempt_count": request_attempt,
                        "parse_retry_used": retry_mode,
                        "structured_output_mode": structured_output_mode,
                    }
                    break
                except LLMRequestError as exc:
                    last_error = exc
                    self.last_preview_metadata = {
                        "llm_attempt_count": attempt_number,
                        "llm_latency_ms": 0,
                        "llm_request_attempt_count": request_attempt,
                        "parse_retry_used": retry_mode,
                        "structured_output_mode": structured_output_mode,
                    }
                    if request_attempt >= settings.openai_request_max_attempts:
                        break
                    sleep(0.25 * request_attempt)

            if retry_mode:
                break

        raise last_error or LLMStructuredOutputError("Failed to get a structured preview response from the model.")
