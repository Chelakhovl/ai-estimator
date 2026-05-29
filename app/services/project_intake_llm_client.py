from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from app.config import settings
from app.schemas import (
    IntakeAssetExcerpt,
    IntakeChatRequest,
    IntakeFinalizeRequest,
    LLMIntakeChatOutput,
    LLMIntakeFinalizeOutput,
)
from app.services.project_intake_knowledge import load_project_intake_knowledge

if TYPE_CHECKING:
    from openai import OpenAI


class IntakeLLMClient:
    def __init__(self) -> None:
        self.enabled = bool(settings.openai_api_key and settings.openai_intake_model)
        self.model = settings.openai_intake_model
        self.fast_model = settings.openai_intake_fast_model
        self.client: OpenAI | None = None
        self.last_metadata: dict[str, object] = {}
        if self.enabled:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=3,
            )

    def is_enabled(self) -> bool:
        return self.enabled and self.client is not None

    def _supports_responses_parse(self) -> bool:
        if self.client is None:
            return False
        responses_api = getattr(self.client, "responses", None)
        return callable(getattr(responses_api, "parse", None))

    def _build_asset_content(self, assets: list[IntakeAssetExcerpt]) -> list[dict[str, str]]:
        content: list[dict[str, str]] = []
        for asset in assets:
            if asset.asset_type == "pdf" and asset.pdf_base64:
                content.append(
                    {
                        "type": "input_file",
                        "filename": asset.filename,
                        "file_data": asset.pdf_base64,
                    }
                )
            asset_summary = [
                f"Filename: {asset.filename}",
                f"Type: {asset.asset_type}",
                f"Status: {asset.extraction_status}",
                f"Excerpt: {asset.document_excerpt}",
            ]
            if asset.table_excerpt:
                asset_summary.append(f"Table excerpt: {asset.table_excerpt}")
            if asset.warnings:
                asset_summary.append(f"Warnings: {'; '.join(asset.warnings)}")
            content.append(
                {
                    "type": "input_text",
                    "text": "\n".join(part for part in asset_summary if part),
                }
            )
        return content

    def _build_chat_input(self, payload: IntakeChatRequest) -> list[dict[str, object]]:
        transcript = "\n".join(f"{turn.role.upper()}: {turn.message}" for turn in payload.turns if turn.message.strip())
        current_brief = payload.current_project_brief_json
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": (
                    "You are Combit Project Intake AI. Your only job is to build and refine a structured "
                    "project brief from uploaded drawings, schedules, and estimator notes.\n\n"
                    "RULES:\n"
                    "- Extract facts into the correct brief section: project_overview, existing_layout, "
                    "proposed_layout_scope, areas_dimensions, structural, extension_roof_external, "
                    "internal_build, mep, finishes, site_conditions, client_supply_vs_combit_supply, missing_rfi\n"
                    "- areas_dimensions: capture every room as 'Room name: length x width m' — this is critical for AI Fill\n"
                    "- mep: capture plumbing/heating items AND electrical items together; tag each fact clearly\n"
                    "- finishes: capture plastering, tiling, painting, flooring items\n"
                    "- Flag gaps and unknowns in missing_rfi and raise them as targeted questions\n"
                    "- NEVER invent measurements — only record what is explicitly stated in documents or messages\n"
                    "- NEVER output costs, rates, or quote rows\n"
                    "- Set ready_to_finalize=true only when project_overview, proposed_layout_scope, and "
                    "areas_dimensions all have substantive facts\n\n"
                    f"Knowledge context:\n{load_project_intake_knowledge()}\n\n"
                    f"Current brief JSON:\n{current_brief}\n\n"
                    f"Conversation so far:\n{transcript or '(none)'}\n\n"
                    f"Latest estimator message:\n{payload.message}"
                ),
            }
        ]
        content.extend(self._build_asset_content(payload.assets))
        return [{"role": "user", "content": content}]

    def _build_finalize_input(self, payload: IntakeFinalizeRequest) -> list[dict[str, object]]:
        transcript = "\n".join(f"{turn.role.upper()}: {turn.message}" for turn in payload.turns if turn.message.strip())
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": (
                    "You are Combit Project Intake AI. Finalize this intake session into a production handoff artifact.\n\n"
                    "TASKS:\n"
                    "1. Consolidate the project_brief_json — remove duplicates, merge fragments, resolve contradictions\n"
                    "2. Write structured_scope_markdown using these section headings (omit any that are empty):\n"
                    "   PROPERTY, FLOOR AREAS & ROOMS, DEMOLITION, STRUCTURE, PLUMBING, ELECTRICAL,\n"
                    "   PLASTERING, TILING, DECORATING, FLOORING, PRELIMINARIES, FULL SCOPE\n"
                    "3. Each section: one fact per line, no sub-bullets, no costs, no markdown formatting\n"
                    "4. PLUMBING and ELECTRICAL must be separate sections even if they came from the same mep brief section\n"
                    "5. List remaining unknowns in final_missing_items\n"
                    "6. Set handoff_readiness=true if FLOOR AREAS & ROOMS and at least one trade section have content\n\n"
                    "RULES: No prices. No quote rows. No invented measurements.\n\n"
                    f"Knowledge context:\n{load_project_intake_knowledge()}\n\n"
                    f"Current brief JSON:\n{payload.current_project_brief_json}\n\n"
                    f"Conversation:\n{transcript or '(none)'}"
                ),
            }
        ]
        content.extend(self._build_asset_content(payload.assets))
        return [{"role": "user", "content": content}]

    def chat(self, payload: IntakeChatRequest) -> LLMIntakeChatOutput:
        if not self.is_enabled() or not self._supports_responses_parse():
            raise RuntimeError("Intake LLM is not configured.")
        started_at = perf_counter()
        response = self.client.responses.parse(
            model=self.fast_model or self.model,
            input=self._build_chat_input(payload),
            text_format=LLMIntakeChatOutput,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("Intake chat returned no structured output.")
        self.last_metadata = {
            "model_name": self.fast_model or self.model,
            "latency_ms": int((perf_counter() - started_at) * 1000),
        }
        return parsed

    def finalize(self, payload: IntakeFinalizeRequest) -> LLMIntakeFinalizeOutput:
        if not self.is_enabled() or not self._supports_responses_parse():
            raise RuntimeError("Intake LLM is not configured.")
        started_at = perf_counter()
        response = self.client.responses.parse(
            model=self.model,
            input=self._build_finalize_input(payload),
            text_format=LLMIntakeFinalizeOutput,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("Intake finalize returned no structured output.")
        self.last_metadata = {
            "model_name": self.model,
            "latency_ms": int((perf_counter() - started_at) * 1000),
        }
        return parsed

