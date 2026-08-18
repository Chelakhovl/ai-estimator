from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING

from app.config import settings
from app.schemas import (
    CatalogContext,
    CustomPricedRow,
    CustomWorkChatRequest,
    CustomWorkChatResponse,
    LLMBatchPriceOutput,
    LLMCustomWorkOutput,
    PreviewUnmatchedItem,
)
from app.services.retry import call_with_request_retries

if TYPE_CHECKING:
    from openai import OpenAI


class CustomPricingUnavailable(Exception):
    """Raised when batch_price_unmatched could not get a usable priced result
    (timeout, malformed output, rate limit) — distinct from "nothing to price",
    which returns an empty list normally, not this exception."""


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
- Every message with status="chatting" MUST end with at least one concrete question
  the user can directly answer. Never send a "chatting" message that only states you
  need more information without asking for it — that wastes a turn and forces the
  user to ask you what you need before you actually ask it.
- Ask at most 1-2 clarifying questions per message, never more. If more ground needs
  covering, ask the next 1-2 questions in your following turn instead of stacking
  them all at once — this chat supports voice input/output, and a wall of questions
  read aloud is hard to hold in mind and answer in one go.
- Across the conversation, resolve: area/quantity extent, material specification
  (grade, finish, size), access difficulty, client-supply vs. contractor-supply,
  whether removal/disposal is included. Do not propose on the first turn.
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

    def chat(
        self,
        messages: list,
        catalog_context: CatalogContext,
    ) -> LLMCustomWorkOutput:
        if not self.is_enabled():
            raise RuntimeError("Custom work LLM is not configured.")
        started_at = perf_counter()
        response = self.client.beta.chat.completions.parse(  # type: ignore[union-attr]
            model=self.model,
            messages=_build_messages(catalog_context, messages),
            response_format=LLMCustomWorkOutput,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("Custom work chat returned no structured output.")
        self.last_metadata = {
            "model_name": self.model,
            "latency_ms": int((perf_counter() - started_at) * 1000),
        }
        return parsed


def _build_batch_price_system_prompt(catalog_context: CatalogContext) -> str:
    work_groups_text = "\n".join(f"  {wg.id}: {wg.name}" for wg in catalog_context.work_groups)
    labour_rates_text = "\n".join(
        f"  {lr.name} ({lr.code}): £{lr.net_rate:.2f}/hr" for lr in catalog_context.labour_rates
    )
    return f"""You are a UK construction estimating AI for Combit Renovations Ltd, a London refurbishment contractor.

You will receive a list of work items that could not be matched to the existing catalogue. Price each one immediately — do NOT ask clarifying questions.

AVAILABLE WORK GROUPS (use the integer id in suggested_work_group_id):
{work_groups_text}

LABOUR RATES (net cost to contractor, ex markup, ex VAT):
{labour_rates_text}

PRICING RULES:
- All costs are NET to contractor, ex VAT, ex profit/overhead markup.
- Use 2024-2025 London / South-East England market rates.
- labour_cost = labour net cost for qty_for_norm units.
- material_cost = net material cost for qty_for_norm units.
- other_cost = subcontract, plant hire, or fixed costs.
- qty_for_norm: normalisation basis (1.0 for per-unit rates).
- unit must be one of: m2, lm, m3, each, nr.
- work_days: elapsed working days for a typical team.
- Never include VAT, profit, or overhead markups.
- confidence_level: 0.85+ for standard items, 0.65 for specialist/complex, 0.5 for speculative.
- source_text: copy the original item description exactly as given.
- Pick the most appropriate suggested_work_group_id from the list above.

Return a JSON object with an "items" array containing one priced entry per input item."""


def batch_price_unmatched(
    items: list[PreviewUnmatchedItem],
    catalog_context: CatalogContext,
) -> list[CustomPricedRow]:
    """Single GPT call to price all unmatched items.

    Returns [] when there's legitimately nothing to price (no items, or custom
    pricing disabled). Raises CustomPricingUnavailable on a real failure
    (timeout, malformed output, rate limit) so the caller can tell the two
    apart and surface a warning instead of silently returning no rows."""
    if not items:
        return []
    client = CustomWorkLLMClient()
    if not client.is_enabled():
        return []
    items_text = "\n".join(f"{i + 1}. {item.source_text}" for i, item in enumerate(items))
    messages = [
        {"role": "system", "content": _build_batch_price_system_prompt(catalog_context)},
        {"role": "user", "content": f"Price the following work items:\n\n{items_text}"},
    ]
    try:
        started_at = perf_counter()
        response = call_with_request_retries(
            lambda: client.client.beta.chat.completions.parse(  # type: ignore[union-attr]
                model=client.model,
                messages=messages,
                response_format=LLMBatchPriceOutput,
            ),
            label="batch_price_unmatched",
        )
        parsed: LLMBatchPriceOutput | None = response.choices[0].message.parsed
        if parsed is None:
            raise CustomPricingUnavailable("Model returned no structured output.")
        elapsed = int((perf_counter() - started_at) * 1000)
        logging.getLogger(__name__).info(
            "batch_price_unmatched: priced %d items in %dms", len(parsed.items), elapsed
        )
        return [
            CustomPricedRow(
                source_text=item.source_text,
                name=item.name,
                unit=item.unit,
                labour_cost=item.labour_cost,
                material_cost=item.material_cost,
                other_cost=item.other_cost,
                work_days=item.work_days,
                qty_for_norm=item.qty_for_norm,
                suggested_work_group_id=item.suggested_work_group_id,
                suggested_work_group_name=item.suggested_work_group_name,
                market_justification=item.market_justification,
                price_source_notes=item.price_source_notes,
                confidence_level=item.confidence_level,
            )
            for item in parsed.items
        ]
    except CustomPricingUnavailable:
        logging.getLogger(__name__).exception("batch_price_unmatched failed")
        raise
    except Exception as exc:
        logging.getLogger(__name__).exception("batch_price_unmatched failed")
        raise CustomPricingUnavailable(str(exc)) from exc


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
