from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.config import settings
from app.schemas import CandidateRow
from app.schemas import LLMPreviewOutput
from app.services.prompt_builder import build_preview_system_prompt, build_preview_user_message

if TYPE_CHECKING:
    from openai import OpenAI


class LLMClient:
    def __init__(self) -> None:
        self.model = settings.openai_model
        self.enabled = bool(settings.openai_api_key and settings.openai_model)
        self.client: OpenAI | None = None
        if self.enabled:
            from openai import OpenAI

            self.client = OpenAI(api_key=settings.openai_api_key)

    def is_enabled(self) -> bool:
        return self.enabled

    def preview_match(self, prompt: str, candidate_rows: list[CandidateRow]) -> LLMPreviewOutput:
        if not self.enabled or self.client is None:
            raise RuntimeError("LLM client is not configured.")

        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": build_preview_system_prompt()},
                    {
                        "role": "user",
                        "content": build_preview_user_message(prompt, candidate_rows),
                    },
                ],
            )
            output_text = getattr(response, "output_text", "") or ""
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": build_preview_system_prompt()},
                    {
                        "role": "user",
                        "content": build_preview_user_message(prompt, candidate_rows),
                    },
                ],
            )
            output_text = (
                response.choices[0].message.content
                if response.choices and response.choices[0].message.content
                else ""
            )

        if not output_text.strip():
            raise ValueError("Model returned empty output.")

        try:
            return LLMPreviewOutput.model_validate_json(output_text)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(
                "Failed to parse model output as LLMPreviewOutput. "
                f"Output: {json.dumps(output_text[:1000])}"
            ) from exc
