from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from app.config import settings
from app.schemas import (
    CatalogContext,
    CustomWorkChatRequest,
    CustomWorkChatResponse,
    LLMCustomWorkOutput,
)

if TYPE_CHECKING:
    from openai import OpenAI

_MOCK_RESPONSE = CustomWorkChatResponse(
    message=(
        "(Mock mode — OPENAI_API_KEY not set) "
        "Tell me what work you need done and I'll help you build a cost estimate."
    ),
    status="chatting",
    work_proposal=None,
    model_name="mock",
)


def _build_system_prompt(catalog_context: CatalogContext) -> str:
    work_groups_text = "\n".join(
        f"  {wg.id}: {wg.name}" for wg in catalog_context.work_groups
    )
    labour_rates_text = "\n".join(
        f"  {lr.name} ({lr.code}): £{lr.net_rate:.2f}/hr"
        for lr in catalog_context.labour_rates
    )
    return f"""You are a UK construction estimating AI assistant for Combit Renovations Ltd, a London refurbishment contractor.

Your job: help the estimator define a new bespoke work item, then propose a priced breakdown.

AVAILABLE WORK GROUPS (use the integer id in suggested_work_group_id):
{work_groups_text}

LABOUR RATES (net cost to contractor, ex markup, ex VAT):
{labour_rates_text}

PRICING RULES:
- All costs are NET to contractor, ex VAT, ex profit/overhead markup.
- Use 2024-2025 London / South-East England market rates.
- labour_cost = sum of (men × hours × net_rate) for every trade required, normalised to qty_for_norm units.
- material_cost = net material cost for the qty_for_norm basis quantity.
- other_cost = subcontract, plant hire, specialist attendance, or fixed costs.
- qty_for_norm: the normalisation basis (e.g. 1.0 for a per-m2 rate, 1.0 for each/nr items).
- unit must be one of: m2, lm, m3, each, nr.
- work_days: elapsed calendar working days for a typical team to complete the item.
- Never include VAT, profit, or overhead markups in the cost fields.
- market_justification: 2-4 sentences citing trade rates, material market, and scope basis.
- price_source_notes: Attribute every cost line explicitly. For labour: state the exact rate name and £/hr from the LABOUR RATES list above (e.g. "Electrician (ELEC): £38.00/hr from Combit internal catalogue"). For materials: state they are estimated from 2024-2025 UK market rates based on AI training knowledge — no live internet data. For other/subcontract costs: describe the basis. Keep it factual and concise, one line per cost type.
- confidence_level: 0.9+ for well-defined scope, 0.7 for moderately complex, 0.5 for speculative.
- Pick the most appropriate suggested_work_group_id from the list above. Use the integer ID, not the name.

CONVERSATION RULES:
- Ask 2-4 focused clarifying questions before proposing. Do not propose on the first turn.
- Questions should resolve: area/quantity extent, material specification (grade, finish, size), access difficulty, client-supply vs. contractor-supply, whether removal/disposal is included.
- Once you have sufficient information, set status="proposing" and fully populate work_proposal.
- If the user asks you to revise the price or challenges an assumption, update the proposal and re-justify.
- Keep messages concise and professional. Use plain prose — no markdown headers or bullet lists in the message field.
- Never state costs in the conversational message before a formal proposal.
- Set status="chatting" while asking questions. Set status="proposing" only when you return a full work_proposal."""


def _build_messages(
    catalog_context: CatalogContext,
    messages: list,
) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": _build_system_prompt(catalog_context)},
        *[{"role": m.role, "content": m.content} for m in messages],
    ]


class CustomWorkLLMClient:
    def __init__(self) -> None:
        self.enabled = bool(settings.openai_api_key and settings.openai_intake_model)
        self.model = settings.openai_intake_model
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

    def chat(
        self,
        messages: list,
        catalog_context: CatalogContext,
    ) -> LLMCustomWorkOutput:
        if not self.is_enabled() or not self._supports_responses_parse():
            raise RuntimeError("Custom work LLM is not configured.")
        started_at = perf_counter()
        response = self.client.responses.parse(  # type: ignore[union-attr]
            model=self.model,
            input=_build_messages(catalog_context, messages),
            text_format=LLMCustomWorkOutput,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("Custom work chat returned no structured output.")
        self.last_metadata = {
            "model_name": self.model,
            "latency_ms": int((perf_counter() - started_at) * 1000),
        }
        return parsed


def chat_custom_work(payload: CustomWorkChatRequest) -> CustomWorkChatResponse:
    client = CustomWorkLLMClient()
    if not client.is_enabled():
        return _MOCK_RESPONSE
    parsed = client.chat(payload.messages, payload.catalog_context)
    return CustomWorkChatResponse(
        message=parsed.message,
        status=parsed.status,
        work_proposal=parsed.work_proposal,
        model_name=str(client.last_metadata.get("model_name", "")),
    )
