from __future__ import annotations

from dataclasses import dataclass
import logging
import re

from app.config import settings
from app.schemas import (
    CandidateRow,
    PreviewAssumption,
    PreviewCoveragePrompt,
    PreviewCoverageSummary,
    PreviewMatchedRow,
    PreviewRequest,
    PreviewReviewQueueItem,
    PreviewReviewSummary,
    PreviewResponse,
    PreviewTelemetry,
    PreviewUnmatchedItem,
)
from app.services.calculator import calculate_client_cost_per_unit, calculate_client_total_cost
from app.services.candidate_shortlist import _extract_active_sections, shortlist_candidates
from app.services.llm_client import LLMClient
from app.services.normalizer import (
    detect_area,
    expand_tokens,
    is_scope_descriptor_segment,
    normalize_prompt,
    normalize_unit,
    parse_quantity_and_unit,
    split_prompt_segments,
    tokenize,
)
from app.services.prompt_builder import build_preview_user_message
from app.services.scope_coverage import build_scope_coverage_assumptions
from app.services.scope_extractor import extract_scope
from app.services.scope_matching import (
    build_scope_matching_context,
    is_extracted_context_segment,
    match_row_to_extracted_section,
    suggest_quantity_from_takeoff,
)
from app.services.scope_takeoff import TAKEOFF_HEURISTICS_SOURCE, TAKEOFF_HEURISTICS_VERSION
from app.services.preview_validator import validate_and_materialize_match


@dataclass(frozen=True)
class MatchCandidate:
    row: CandidateRow
    score: float


logger = logging.getLogger(__name__)


_GUARDRAIL_STATUS_TO_CODE = {
    "Allowance required": "allowance_required",
    "Client allowance": "client_allowance_review_required",
    "Do not include": "do_not_include",
    "External pricing": "external_pricing_needed",
    "Pricing review": "review_required_specialist_scope",
    "User amount required": "user_amount_required",
    "specialist_required": "review_required_specialist_scope",
}
_REVIEW_GUARDRAIL_CODES = {
    "allowance_required",
    "client_allowance_review_required",
    "external_pricing_needed",
    "review_required_removal_scope",
    "review_required_small_scope",
    "review_required_specialist_scope",
    "user_amount_required",
}
_DIRECT_SUPPORT_STOP_TOKENS = {
    "all",
    "allowance",
    "build",
    "convert",
    "existing",
    "fit",
    "fix",
    "general",
    "install",
    "item",
    "items",
    "new",
    "old",
    "preliminaries",
    "project",
    "property",
    "refurb",
    "remove",
    "replace",
    "room",
    "rooms",
    "service",
    "services",
    "strip",
    "structure",
    "supply",
    "throughout",
    "work",
    "works",
}
_GUARDRAIL_DEFAULT_REASONS = {
    "allowance_required": "This row should stay as an allowance and needs estimator review before pricing.",
    "client_allowance_review_required": "This row depends on a client or architect allowance and needs manual review.",
    "external_pricing_needed": "This row needs external pricing confirmation before it can be finalized.",
    "review_required_removal_scope": "Pricing review due to guardrail status.",
    "review_required_small_scope": "Pricing review due to guardrail status.",
    "review_required_specialist_scope": "Pricing review due to guardrail status.",
    "user_amount_required": "This row needs a user-provided amount and should not be finalized automatically.",
}
_DYNAMIC_SHORTLIST_BASE_LIMIT = 150
_DYNAMIC_SHORTLIST_SECTION_BONUS = 10
_DYNAMIC_SHORTLIST_CANDIDATE_BONUS_DIVISOR = 10
_DYNAMIC_SHORTLIST_CANDIDATE_BONUS_MAX = 100
_DYNAMIC_SHORTLIST_MAX_LIMIT = 320
_PREVIEW_ROUGH_TOKEN_BUDGET = 21000
_PREVIEW_ROUGH_CHAR_BUDGET = _PREVIEW_ROUGH_TOKEN_BUDGET * 4
_PREVIEW_SHORTLIST_TRIM_STEP = 20
_PREVIEW_HARD_MIN_SHORTLIST_SIZE = 80
_PREVIEW_ACCEPTED_EXAMPLE_PRESSURE_RATIO = 0.88
_PREVIEW_ACCEPTED_EXAMPLE_COVERAGE_THRESHOLD = 140
_MEASURED_QUANTITY_REQUIRED_UNITS = {"m", "m2", "m3", "lm"}
_GENERIC_HOURLY_NAME_HINTS = (
    "general electrics",
    "general plumbing",
    "building control fee",
    "temporary floor protection",
)


def _allows_profit_override(prompt: str) -> bool:
    return bool(re.search(r"\b(profit|margin)\b", prompt, re.IGNORECASE))


def _allows_labour_markup_override(prompt: str) -> bool:
    return bool(
        re.search(r"\b(labou?r markup|labou?r margin|worker markup)\b", prompt, re.IGNORECASE)
    )


def _allows_material_markup_override(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(material markup|material margin|materials markup|materials margin)\b",
            prompt,
            re.IGNORECASE,
        )
    )


def _score_segment_against_row(segment: str, row: CandidateRow, requested_unit: str | None) -> float:
    segment_tokens = expand_tokens(tokenize(segment))
    work_tokens = expand_tokens(tokenize(row.WorkName))
    context_tokens = expand_tokens(tokenize(
        " ".join(
            filter(
                None,
                [
                    row.WorkName,
                    row.WorkGroupName or "",
                    row.WorkItemCode or "",
                    row.GuardrailStatus or "",
                ],
            )
        )
    ))
    overlap = len(segment_tokens & work_tokens) * 2 + len(segment_tokens & context_tokens)

    score = float(overlap)
    normalized_row_unit = normalize_unit(row.Unit)
    if overlap > 0 and requested_unit and normalized_row_unit == requested_unit:
        score += 1.5
    if overlap > 0 and requested_unit and normalized_row_unit and normalized_row_unit != requested_unit:
        score -= 0.75
    return score


def _best_match(
    segment: str,
    rows: list[CandidateRow],
    used_ids: set[str],
    requested_unit: str | None,
) -> MatchCandidate | None:
    best: MatchCandidate | None = None
    for row in rows:
        if row.INSIDEQUOTESGUID in used_ids:
            continue
        score = _score_segment_against_row(segment, row, requested_unit)
        if best is None or score > best.score:
            best = MatchCandidate(row=row, score=score)
    return best


def _append_review_reason(existing: str | None, extra: str) -> str:
    existing = (existing or "").strip()
    if not existing:
        return extra
    if extra in existing:
        return existing
    return f"{existing} {extra}".strip()


def _normalized_guardrail_code(row: CandidateRow) -> str:
    code = (row.GuardrailCode or "").strip().lower()
    if code:
        return code
    return _GUARDRAIL_STATUS_TO_CODE.get((row.GuardrailStatus or "").strip(), "")


def _guardrail_reason_text(row: CandidateRow) -> str:
    code = _normalized_guardrail_code(row)
    detail = (row.GuardrailDetail or "").strip()
    if detail:
        return detail
    return _GUARDRAIL_DEFAULT_REASONS.get(code, (row.GuardrailStatus or "").strip())


def _significant_support_tokens(value: str) -> set[str]:
    return {
        token
        for token in expand_tokens(tokenize(value))
        if token and token not in _DIRECT_SUPPORT_STOP_TOKENS and not token.isdigit()
    }


def _build_scope_support_tokens(scope_text: str, extracted_scope) -> set[str]:
    scope_parts = [scope_text]
    if extracted_scope is not None:
        property_context = extracted_scope.property_context
        scope_parts.append(property_context.property_type)
        scope_parts.append(property_context.location)
        scope_parts.extend(property_context.project_scopes)
        scope_parts.extend(room.name for room in extracted_scope.rooms)
        for section in extracted_scope.sections:
            scope_parts.append(section.title)
            scope_parts.extend(section.lines[:24])
            scope_parts.extend(item.name for item in section.count_items[:8])
            scope_parts.extend(item.name for item in section.measure_items[:8])
            scope_parts.extend(item.name for item in section.dimension_items[:8])
    return _significant_support_tokens(" ".join(part for part in scope_parts if part))


def _has_direct_scope_support(
    row: CandidateRow,
    *,
    scope_text: str,
    extracted_scope,
    takeoff_suggestion=None,
) -> bool:
    if takeoff_suggestion and getattr(takeoff_suggestion, "quantity", None):
        return True

    row_tokens = _significant_support_tokens(row.WorkName)
    if not row_tokens:
        row_tokens = _significant_support_tokens(row.WorkGroupName or "")
    if not row_tokens:
        return True

    scope_tokens = _build_scope_support_tokens(scope_text, extracted_scope)
    return bool(scope_tokens & row_tokens)


def _resolve_shortlist_limit(raw_prompt: str, extracted_scope, *, candidate_count: int = 0) -> int:
    active_section_count = len(_extract_active_sections(raw_prompt))
    if active_section_count == 0 and extracted_scope is not None:
        active_section_count = len(
            [
                section
                for section in extracted_scope.sections
                if section.lines or section.count_items or section.measure_items or section.dimension_items
            ]
        )
    candidate_scale_bonus = 0
    if candidate_count > _DYNAMIC_SHORTLIST_BASE_LIMIT:
        candidate_scale_bonus = min(
            (candidate_count - _DYNAMIC_SHORTLIST_BASE_LIMIT) // _DYNAMIC_SHORTLIST_CANDIDATE_BONUS_DIVISOR,
            _DYNAMIC_SHORTLIST_CANDIDATE_BONUS_MAX,
        )
    return min(
        _DYNAMIC_SHORTLIST_BASE_LIMIT
        + active_section_count * _DYNAMIC_SHORTLIST_SECTION_BONUS
        + candidate_scale_bonus,
        _DYNAMIC_SHORTLIST_MAX_LIMIT,
    )


def _count_actionable_scope_sections(extracted_scope) -> int:
    if extracted_scope is None:
        return 0
    return len(
        [
            section
            for section in extracted_scope.sections
            if section.key not in {"property", "full_scope", "floor_areas_rooms"}
            and (section.lines or section.count_items or section.measure_items or section.dimension_items)
        ]
    )


def _minimum_budget_shortlist_size(extracted_scope) -> int:
    actionable_sections = _count_actionable_scope_sections(extracted_scope)
    if actionable_sections <= 0:
        return _PREVIEW_HARD_MIN_SHORTLIST_SIZE
    return min(max(_PREVIEW_HARD_MIN_SHORTLIST_SIZE, actionable_sections * 8), 160)


def _estimate_preview_message_chars(
    *,
    prompt: str,
    candidate_rows: list[CandidateRow],
    extracted_scope,
    accepted_examples,
) -> int:
    return len(
        build_preview_user_message(
            prompt=prompt,
            candidate_rows=candidate_rows,
            extracted_scope=extracted_scope,
            accepted_examples=accepted_examples,
        )
    )


def _build_shortlist_budget_assumption(
    *,
    original_size: int,
    final_size: int,
    rough_tokens: int,
) -> PreviewAssumption | None:
    if final_size >= original_size:
        return None
    return PreviewAssumption(
        text=(
            f"Candidate shortlist was reduced from {original_size} to {final_size} rows "
            f"to keep the LLM request within the current token budget (~{rough_tokens} input tokens)."
        ),
        kind="token_budget_trim",
        severity="info",
    )


def _build_examples_budget_assumption(
    *,
    original_count: int,
    final_count: int,
    rough_tokens: int,
) -> PreviewAssumption | None:
    if final_count >= original_count:
        return None
    action = "omitted" if final_count == 0 else f"reduced from {original_count} to {final_count}"
    return PreviewAssumption(
        text=(
            f"Accepted examples were {action} to preserve candidate coverage within the current token budget "
            f"(~{rough_tokens} input tokens)."
        ),
        kind="token_budget_trim",
        severity="info",
    )


def _fit_accepted_examples_to_budget(
    *,
    prompt: str,
    shortlist: list[CandidateRow],
    extracted_scope,
    accepted_examples,
) -> tuple[list, PreviewAssumption | None]:
    if not accepted_examples:
        return [], None

    baseline_chars = _estimate_preview_message_chars(
        prompt=prompt,
        candidate_rows=shortlist,
        extracted_scope=extracted_scope,
        accepted_examples=accepted_examples,
    )
    if baseline_chars <= _PREVIEW_ROUGH_CHAR_BUDGET and not (
        len(shortlist) >= _PREVIEW_ACCEPTED_EXAMPLE_COVERAGE_THRESHOLD
        and baseline_chars > int(_PREVIEW_ROUGH_CHAR_BUDGET * _PREVIEW_ACCEPTED_EXAMPLE_PRESSURE_RATIO)
    ):
        return list(accepted_examples), None

    example_variants: list[list] = []
    if len(accepted_examples) > 1:
        example_variants.append(list(accepted_examples[:1]))
    example_variants.append([])

    best_variant = list(accepted_examples)
    best_chars = baseline_chars
    chosen_assumption: PreviewAssumption | None = None

    for variant in example_variants:
        variant_chars = _estimate_preview_message_chars(
            prompt=prompt,
            candidate_rows=shortlist,
            extracted_scope=extracted_scope,
            accepted_examples=variant,
        )
        if variant_chars < best_chars:
            best_variant = list(variant)
            best_chars = variant_chars
            chosen_assumption = _build_examples_budget_assumption(
                original_count=len(accepted_examples),
                final_count=len(variant),
                rough_tokens=max(variant_chars // 4, 1),
            )
        if variant_chars <= _PREVIEW_ROUGH_CHAR_BUDGET:
            break

    return best_variant, chosen_assumption


def _fit_shortlist_to_budget(
    *,
    prompt: str,
    shortlist: list[CandidateRow],
    extracted_scope,
    accepted_examples,
) -> tuple[list[CandidateRow], PreviewAssumption | None]:
    if len(shortlist) <= 1:
        return shortlist, None

    current = shortlist
    message_chars = _estimate_preview_message_chars(
        prompt=prompt,
        candidate_rows=current,
        extracted_scope=extracted_scope,
        accepted_examples=accepted_examples,
    )
    if message_chars <= _PREVIEW_ROUGH_CHAR_BUDGET:
        return current, None

    minimum_size = min(_minimum_budget_shortlist_size(extracted_scope), len(shortlist))
    while len(current) > minimum_size and message_chars > _PREVIEW_ROUGH_CHAR_BUDGET:
        next_size = max(minimum_size, len(current) - _PREVIEW_SHORTLIST_TRIM_STEP)
        current = current[:next_size]
        message_chars = _estimate_preview_message_chars(
            prompt=prompt,
            candidate_rows=current,
            extracted_scope=extracted_scope,
            accepted_examples=accepted_examples,
        )

    while len(current) > _PREVIEW_HARD_MIN_SHORTLIST_SIZE and message_chars > _PREVIEW_ROUGH_CHAR_BUDGET:
        next_size = max(_PREVIEW_HARD_MIN_SHORTLIST_SIZE, len(current) - _PREVIEW_SHORTLIST_TRIM_STEP)
        if next_size >= len(current):
            break
        current = current[:next_size]
        message_chars = _estimate_preview_message_chars(
            prompt=prompt,
            candidate_rows=current,
            extracted_scope=extracted_scope,
            accepted_examples=accepted_examples,
        )

    return current, _build_shortlist_budget_assumption(
        original_size=len(shortlist),
        final_size=len(current),
        rough_tokens=max(message_chars // 4, 1),
    )


def _looks_like_token_budget_error(exc: Exception) -> bool:
    fragments: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        text = str(current).strip()
        if text:
            fragments.append(text.lower())
        current = current.__cause__
    haystack = " ".join(fragments)
    return any(
        marker in haystack
        for marker in (
            "request too large",
            "tokens per min",
            "rate_limit_exceeded",
            "input or output tokens must be reduced",
        )
    )


def _retry_trimmed_shortlist(
    *,
    prompt: str,
    shortlist: list[CandidateRow],
    extracted_scope,
    accepted_examples,
) -> tuple[list[CandidateRow], PreviewAssumption | None]:
    if len(shortlist) <= _PREVIEW_HARD_MIN_SHORTLIST_SIZE:
        return shortlist, None

    next_size = max(_PREVIEW_HARD_MIN_SHORTLIST_SIZE, len(shortlist) - _PREVIEW_SHORTLIST_TRIM_STEP)
    trimmed = shortlist[:next_size]
    message_chars = _estimate_preview_message_chars(
        prompt=prompt,
        candidate_rows=trimmed,
        extracted_scope=extracted_scope,
        accepted_examples=accepted_examples,
    )
    return trimmed, _build_shortlist_budget_assumption(
        original_size=len(shortlist),
        final_size=len(trimmed),
        rough_tokens=max(message_chars // 4, 1),
    )


def _apply_guardrail_policy(
    matched_row: PreviewMatchedRow,
    *,
    row: CandidateRow,
) -> tuple[bool, list[PreviewAssumption], PreviewUnmatchedItem | None]:
    code = _normalized_guardrail_code(row)
    if not code:
        return False, [], None

    reason = _guardrail_reason_text(row)
    assumptions: list[PreviewAssumption] = []
    if code == "do_not_include":
        assumptions.append(
            PreviewAssumption(
                text=f"Guardrail excluded '{row.WorkName}' from the AI draft because it is marked do_not_include.",
                kind="guardrail_excluded",
                severity="warning",
            )
        )
        return True, assumptions, PreviewUnmatchedItem(
            source_text=row.WorkName,
            reason=reason or "This item is marked do_not_include and should not be auto-added to the quote.",
        )

    if code in _REVIEW_GUARDRAIL_CODES:
        matched_row.NeedsReview = True
        matched_row.Confidence = min(matched_row.Confidence, 0.74)
        matched_row.ReviewReason = _append_review_reason(
            matched_row.ReviewReason,
            reason or "Pricing review due to guardrail status.",
        )
        assumptions.append(
            PreviewAssumption(
                text=f"Guardrail review retained for '{row.WorkName}': {reason or 'Pricing review due to guardrail status.'}",
                kind="guardrail_review",
                severity="warning",
            )
        )

    return False, assumptions, None


def _apply_specialist_trade_review(
    matched_row: PreviewMatchedRow,
    *,
    row: CandidateRow,
    segment: str,
) -> list[PreviewAssumption]:
    assumptions: list[PreviewAssumption] = []
    tokens = tokenize(
        " ".join(
            filter(
                None,
                [
                    row.WorkName,
                    row.WorkItemCode or "",
                    row.WorkGroupName or "",
                    segment,
                ],
            )
        )
    )

    specialist_reasons: list[str] = []
    specialist_trade_tags: list[str] = []
    if ("consumer" in tokens and "unit" in tokens) or ("distribution" in tokens and "board" in tokens):
        specialist_reasons.append(
            "Consumer unit / board work should stay on specialist review because board spec, protection type, and testing scope still need confirmation."
        )
        specialist_trade_tags.append("electrical")
    if "rewire" in tokens:
        specialist_reasons.append(
            "Rewire scope should stay on specialist review because circuit count, containment, and testing scope usually need a manual electrical decision."
        )
        specialist_trade_tags.append("electrical")
    if "circuit" in tokens:
        specialist_reasons.append(
            "Circuit-based electrical scope should stay on specialist review because circuit schedule, load assumptions, and certification requirements need confirmation."
        )
        specialist_trade_tags.append("electrical")
    if any(token in tokens for token in {"boiler", "cylinder", "megaflo"}):
        specialist_reasons.append(
            "Boiler / cylinder work should stay on specialist review because appliance spec, flue or discharge routing, and commissioning scope still need confirmation."
        )
        specialist_trade_tags.append("plumbing-heating")
    if "ufh" in tokens or ("underfloor" in tokens and "heating" in tokens):
        specialist_reasons.append(
            "Underfloor heating scope should stay on specialist review because zoning, manifold layout, controls, and floor build-up coordination need confirmation."
        )
        specialist_trade_tags.append("plumbing-heating")
    if "gas" in tokens and any(token in tokens for token in {"line", "pipe", "pipework"}):
        specialist_reasons.append(
            "Gas line scope should stay on specialist review because route, capacity, and compliance requirements need confirmation."
        )
        specialist_trade_tags.append("plumbing-heating")
    if any(token in tokens for token in {"manifold", "soil", "drain", "stack"}):
        specialist_reasons.append(
            "Drainage / manifold / soil-stack work should stay on specialist review because routing, access, and connection strategy still need confirmation."
        )
        specialist_trade_tags.append("plumbing-heating")

    if specialist_reasons:
        matched_row.NeedsReview = True
        matched_row.Confidence = min(matched_row.Confidence, 0.74)
        for reason in specialist_reasons:
            matched_row.ReviewReason = _append_review_reason(matched_row.ReviewReason, reason)
        scope_label = "specialist trade"
        if "electrical" in specialist_trade_tags and "plumbing-heating" in specialist_trade_tags:
            scope_label = "specialist electrical and plumbing/heating"
        elif "electrical" in specialist_trade_tags:
            scope_label = "specialist electrical"
        elif "plumbing-heating" in specialist_trade_tags:
            scope_label = "specialist plumbing/heating"
        assumptions.append(
            PreviewAssumption(
                text=f"{scope_label.capitalize()} review retained for '{row.WorkName}' because the matched scope includes high-risk engineering decisions that still need manual confirmation.",
                kind="specialist_review",
                severity="warning",
            )
        )

    return assumptions


def _apply_hourly_unit_review(
    matched_row: PreviewMatchedRow,
    *,
    row: CandidateRow,
    scope_text: str,
    matched_section_key: str | None,
) -> list[PreviewAssumption]:
    normalized_unit = normalize_unit(row.Unit)
    if normalized_unit != "hour":
        return []

    section_key = matched_section_key or ""
    lowered_scope = scope_text.lower()
    lowered_name = (row.WorkName or "").strip().lower()
    assumptions: list[PreviewAssumption] = []
    review_reason: str | None = None

    if any(hint in lowered_name for hint in _GENERIC_HOURLY_NAME_HINTS):
        review_reason = (
            "This match uses a generic hourly row. Prefer a more specific counted or measured row if one exists in the catalog."
        )
    elif section_key == "electrical" and any(
        keyword in lowered_scope for keyword in ("switch", "socket", "spotlight", "downlight", "extractor", "appliance", "thermostat")
    ):
        review_reason = (
            "Electrical point scope matched to an hourly row. Confirm whether a point-based electrical row should be used instead."
        )
    elif section_key in {"plumbing", "heating"} and any(
        keyword in lowered_scope for keyword in ("radiator", "basin", "bath", "shower", "toilet", "wc", "boiler", "ufh")
    ):
        review_reason = (
            "Plumbing / heating scope matched to an hourly row. Confirm whether a point-based or measured row should be used instead."
        )
    elif section_key in {"flooring", "tiling", "plastering", "ceilings_and_insulation"} and matched_row.QUANTITY >= 8:
        review_reason = (
            "Area-driven finish scope matched to an hourly row. Confirm whether an m2-based row should be used instead."
        )
    elif section_key in {"joinery", "carpentry", "woodwork_painting"} and any(
        keyword in lowered_scope for keyword in ("skirting", "architrave", "window board")
    ):
        review_reason = (
            "Joinery finish scope matched to an hourly row. Confirm whether a length- or count-based row should be used instead."
        )
    elif section_key == "ceilings_and_insulation" and any(
        keyword in lowered_scope for keyword in ("insulation", "rockwool", "pir", "acoustic")
    ):
        review_reason = (
            "Insulation scope matched to an hourly row. Confirm whether an m2-based insulation row should be used instead."
        )

    if not review_reason:
        return assumptions

    matched_row.NeedsReview = True
    matched_row.Confidence = min(matched_row.Confidence, 0.72)
    matched_row.ReviewReason = _append_review_reason(matched_row.ReviewReason, review_reason)
    assumptions.append(
        PreviewAssumption(
            text=f"Hourly review retained for '{row.WorkName}' because the current scope looks count- or area-driven rather than open-ended labour time.",
            kind="hourly_row_review",
            severity="warning",
        )
    )
    return assumptions


def _build_preview_row_from_candidate(
    row: CandidateRow,
    quantity: float,
    area: str,
    confidence: float,
    needs_review: bool,
    review_reason: str | None = None,
    matched_section_key: str | None = None,
    matched_section_title: str | None = None,
    matched_section_order: int | None = None,
) -> PreviewMatchedRow:
    client_cost_per_unit = calculate_client_cost_per_unit(
        work_labour_cost=row.WorkLabourCost,
        work_mat_cost=row.WorkMatCost,
        work_other_cost=row.WorkOtherCost,
        labour_markup=row.LabourMarkup,
        material_markup=row.MaterialMarkup,
        profit=row.PROFIT,
        work_qty_for_norm=row.WorkQTYforNorm,
    )
    client_total_cost = calculate_client_total_cost(quantity, client_cost_per_unit)

    return PreviewMatchedRow(
        INSIDEQUOTESGUID=row.INSIDEQUOTESGUID,
        WORKGUID=row.WORKGUID,
        WorkName=row.WorkName,
        Unit=row.Unit,
        AREA=area,
        QUANTITY=quantity,
        PROFIT=row.PROFIT,
        LabourMarkup=row.LabourMarkup,
        MaterialMarkup=row.MaterialMarkup,
        WorkLabourCost=row.WorkLabourCost,
        WorkMatCost=row.WorkMatCost,
        WorkOtherCost=row.WorkOtherCost,
        WorkQTYforNorm=row.WorkQTYforNorm,
        ClientCostPerUnit=client_cost_per_unit,
        ClientTotalCost=client_total_cost,
        Confidence=round(confidence, 2),
        NeedsReview=needs_review,
        ReviewReason=review_reason,
        MatchedSectionKey=matched_section_key,
        MatchedSectionTitle=matched_section_title,
        MatchedSectionOrder=matched_section_order,
    )


def _build_review_summary(
    matched_rows: list[PreviewMatchedRow],
    unmatched_items: list[PreviewUnmatchedItem],
    assumptions: list[PreviewAssumption],
) -> PreviewReviewSummary:
    review_count = sum(1 for row in matched_rows if row.NeedsReview)
    return PreviewReviewSummary(
        matched_count=len(matched_rows),
        ready_count=len(matched_rows) - review_count,
        review_count=review_count,
        unmatched_count=len(unmatched_items),
        assumption_count=len(assumptions),
    )


def _truncate_fallback_reason(reason: str, *, limit: int = 240) -> str:
    normalized = " ".join((reason or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def _build_preview_telemetry(
    request: PreviewRequest,
    *,
    shortlist_size: int,
    extracted_scope,
    llm_metadata: dict[str, object] | None = None,
    accepted_example_count: int | None = None,
    fallback_reason: str = "",
) -> PreviewTelemetry:
    room_count = 0
    if extracted_scope is not None:
        room_count = (
            getattr(extracted_scope.takeoff_summary, "room_count", 0)
            or len(getattr(extracted_scope, "room_takeoff", []) or [])
            or len(getattr(extracted_scope, "rooms", []) or [])
        )

    llm_metadata = llm_metadata or {}
    rules_context = request.rules_context
    return PreviewTelemetry(
        candidate_row_count=len(request.candidate_rows),
        shortlist_size=shortlist_size,
        accepted_example_count=len(request.accepted_examples) if accepted_example_count is None else accepted_example_count,
        room_count=room_count,
        llm_attempt_count=int(llm_metadata.get("llm_attempt_count", 0) or 0),
        llm_latency_ms=int(llm_metadata.get("llm_latency_ms", 0) or 0),
        parse_retry_used=bool(llm_metadata.get("parse_retry_used", False)),
        structured_output_mode=str(llm_metadata.get("structured_output_mode", "") or ""),
        fallback_reason=_truncate_fallback_reason(fallback_reason),
        takeoff_heuristics_version=TAKEOFF_HEURISTICS_VERSION if extracted_scope is not None else "",
        takeoff_heuristics_source=TAKEOFF_HEURISTICS_SOURCE if extracted_scope is not None else "",
        rule_version=rules_context.rule_version if rules_context is not None else "",
        canonical_source=rules_context.canonical_source if rules_context is not None else "",
        runtime_source=rules_context.runtime_source if rules_context is not None else "",
        published_version=rules_context.published_version if rules_context is not None else "",
        published_at=rules_context.published_at or "" if rules_context is not None else "",
    )


def _build_extracted_scope(prompt: str):
    try:
        return extract_scope(prompt)
    except Exception:
        logger.exception("Structured scope extraction failed during preview generation.")
        return None


def _has_normalized_input(request: PreviewRequest) -> bool:
    return bool(request.normalized_prompt_markdown or request.normalized_scope)


def _resolve_preview_input(request: PreviewRequest) -> tuple[str, object | None, str]:
    if _has_normalized_input(request):
        normalized_prompt = (request.normalized_prompt_markdown or request.prompt).strip()
        extracted_scope = request.normalized_scope or _build_extracted_scope(normalized_prompt)
        # When scope comes from document analysis (sections=[]), supplement sections from the
        # user's scope prompt so shortlisting and active_scope_sections work correctly.
        if extracted_scope is not None and not extracted_scope.sections:
            prompt_scope = _build_extracted_scope(normalized_prompt)
            if prompt_scope and prompt_scope.sections:
                extracted_scope = extracted_scope.model_copy(update={"sections": prompt_scope.sections})
        return normalized_prompt, extracted_scope, "normalized"
    extracted_scope = _build_extracted_scope(request.prompt)
    return request.prompt, extracted_scope, "raw"


def _build_response_warnings(assumptions: list[PreviewAssumption], *, extra_messages: list[str] | None = None) -> list[str]:
    warnings: list[str] = []
    for assumption in assumptions:
        if assumption.severity != "warning":
            continue
        text = (assumption.text or "").strip()
        if text and text not in warnings:
            warnings.append(text)
    for message in extra_messages or []:
        normalized = " ".join(message.split()).strip()
        if normalized and normalized not in warnings:
            warnings.append(normalized)
    return warnings[:8]


def _build_coverage_summary(
    matched_rows: list[PreviewMatchedRow],
    unmatched_items: list[PreviewUnmatchedItem],
    extracted_scope,
    *,
    normalized_scope_used: bool,
) -> PreviewCoverageSummary:
    matched_section_count = len({row.MatchedSectionKey for row in matched_rows if row.MatchedSectionKey})
    section_count = 0
    room_count = 0
    if extracted_scope is not None:
        section_count = len(
            [
                section
                for section in getattr(extracted_scope, "sections", [])
                if section.lines or section.count_items or section.measure_items or section.dimension_items
            ]
        )
        room_count = (
            getattr(extracted_scope.takeoff_summary, "room_count", 0)
            or len(getattr(extracted_scope, "room_takeoff", []) or [])
            or len(getattr(extracted_scope, "rooms", []) or [])
        )
    return PreviewCoverageSummary(
        room_count=room_count,
        section_count=section_count,
        matched_section_count=matched_section_count,
        unmatched_item_count=len(unmatched_items),
        normalized_scope_used=normalized_scope_used,
    )


def _normalize_queue_code(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or fallback


def _build_review_queue(
    matched_rows: list[PreviewMatchedRow],
    unmatched_items: list[PreviewUnmatchedItem],
    assumptions: list[PreviewAssumption],
) -> list[PreviewReviewQueueItem]:
    queue: list[PreviewReviewQueueItem] = []
    seen_codes: set[str] = set()

    def register(item: PreviewReviewQueueItem) -> None:
        if item.code in seen_codes:
            return
        seen_codes.add(item.code)
        queue.append(item)

    for row in matched_rows:
        if not row.NeedsReview:
            continue
        title = row.WorkName or "Review matched row"
        reason = (row.ReviewReason or "").strip() or "This matched row still needs estimator review before it is safe to apply."
        lowered_reason = reason.lower()
        kind = "manual_review"
        recommended_action = "Check the matched row, confirm the quantity or unit basis, and only apply it if the estimator agrees."
        if any(token in lowered_reason for token in ("specialist", "boiler", "consumer unit", "rewire", "circuit", "ufh", "gas line", "drainage")):
            kind = "specialist_review"
            recommended_action = "Hold this on estimator review and confirm the specialist scope before it is priced or applied."
        elif "hourly row" in lowered_reason or "generic hourly" in lowered_reason:
            recommended_action = "Swap this to a counted or measured row if a better catalog match exists."
        register(
            PreviewReviewQueueItem(
                code=_normalize_queue_code(f"row-{row.WORKGUID or row.WorkName}", fallback="row-review"),
                title=title,
                message=reason,
                severity="warning",
                kind=kind,
                scope_label=row.MatchedSectionTitle,
                recommended_action=recommended_action,
                suggested_prompt=f"Clarify '{title}' with room/level, quantity, and the exact scope basis if it still belongs in the quote.",
            )
        )

    for item in unmatched_items:
        source_text = (item.source_text or "").strip()
        if not source_text:
            continue
        register(
            PreviewReviewQueueItem(
                code=_normalize_queue_code(f"unmatched-{source_text}", fallback="unmatched-scope"),
                title=source_text,
                message=item.reason or "This scope line was not matched into a priced row.",
                severity="warning",
                kind="missing_scope",
                recommended_action="Either add clearer scope detail and rerun AI or add the row manually in the quote workspace.",
                suggested_prompt=f"Add explicit scope for '{source_text}' with room/level, quantity, unit, and whether it should be included.",
            )
        )

    for assumption in assumptions:
        kind = (assumption.kind or "").strip()
        if kind not in {"guardrail_review", "guardrail_excluded", "scope_support_missing"}:
            continue
        text = (assumption.text or "").strip()
        if not text:
            continue
        register(
            PreviewReviewQueueItem(
                code=_normalize_queue_code(f"{kind}-{text[:48]}", fallback=kind or "review"),
                title="Pricing / scope guardrail",
                message=text,
                severity=assumption.severity or "warning",
                kind="manual_review",
                recommended_action="Keep this on estimator review and confirm whether it should be carried, priced externally, or left out.",
            )
        )

    return queue[:10]


def _build_coverage_prompts(
    unmatched_items: list[PreviewUnmatchedItem],
    assumptions: list[PreviewAssumption],
) -> list[PreviewCoveragePrompt]:
    prompts: list[PreviewCoveragePrompt] = []
    seen_codes: set[str] = set()

    def register(prompt: PreviewCoveragePrompt) -> None:
        if prompt.code in seen_codes:
            return
        seen_codes.add(prompt.code)
        prompts.append(prompt)

    for item in unmatched_items:
        source_text = (item.source_text or "").strip()
        if not source_text:
            continue
        register(
            PreviewCoveragePrompt(
                code=_normalize_queue_code(f"coverage-{source_text}", fallback="coverage-item"),
                title=source_text,
                message=item.reason or "AI could not place this scope into the draft.",
                severity="warning",
                suggested_prompt=f"Add explicit scope for '{source_text}' with room or level, quantity, unit, and whether it should still be included.",
            )
        )

    for assumption in assumptions:
        kind = (assumption.kind or "").strip()
        if kind not in {
            "coverage_gap",
            "section_coverage_gap",
            "section_detail_coverage_gap",
            "room_finish_coverage_gap",
            "level_joinery_coverage_gap",
        }:
            continue
        text = (assumption.text or "").strip()
        if not text:
            continue
        register(
            PreviewCoveragePrompt(
                code=_normalize_queue_code(f"{kind}-{text[:48]}", fallback=kind or "coverage"),
                title="Coverage gap to confirm",
                message=text,
                severity=assumption.severity or "warning",
                suggested_prompt="Add the missing scope explicitly with room or level, plus counts, areas, or lengths, then rerun the AI draft.",
            )
        )

    return prompts[:8]


def _generate_mock_preview(request: PreviewRequest) -> PreviewResponse:
    source_prompt, extracted_scope, input_mode = _resolve_preview_input(request)
    prompt = normalize_prompt(source_prompt)
    segments = split_prompt_segments(prompt)
    shortlist_prompt = f"{prompt}, {build_scope_matching_context(extracted_scope)}" if extracted_scope else prompt
    shortlist_limit = _resolve_shortlist_limit(
        source_prompt,
        extracted_scope,
        candidate_count=len(request.candidate_rows),
    )
    shortlist = shortlist_candidates(shortlist_prompt, request.candidate_rows, limit=shortlist_limit)

    matched_rows: list[PreviewMatchedRow] = []
    unmatched_items: list[PreviewUnmatchedItem] = []
    assumptions: list[PreviewAssumption] = []
    used_ids: set[str] = set()

    for segment in segments:
        if extracted_scope and is_extracted_context_segment(segment, extracted_scope):
            assumptions.append(
                PreviewAssumption(
                    text=f"Structured scope label noted: '{segment}'. Matching was focused on actionable work rows instead of section headings.",
                    kind="scope_context",
                    severity="info",
                )
            )
            continue

        if ":" not in segment and is_scope_descriptor_segment(segment):
            assumptions.append(
                PreviewAssumption(
                    text=f"Scope descriptor noted: '{segment}'. Matching was focused on priced work rows instead of the high-level project label.",
                    kind="scope_descriptor",
                    severity="info",
                )
            )
            continue

        quantity, requested_unit = parse_quantity_and_unit(segment)
        best = _best_match(segment, shortlist, used_ids, requested_unit)

        if not best or best.score <= 0:
            unmatched_items.append(
                PreviewUnmatchedItem(
                    source_text=segment,
                    reason="No confident match found in candidate quote rows.",
                )
            )
            continue

        row = best.row
        used_ids.add(row.INSIDEQUOTESGUID)

        resolved_quantity = quantity or (row.QUANTITY if row.QUANTITY and row.QUANTITY > 0 else 1)
        resolved_area = detect_area(segment) or (row.AREA or "")
        confidence = min(0.99, round(0.45 + best.score / 4, 2))
        needs_review = confidence < 0.75
        review_reasons: list[str] = []
        normalized_row_unit = normalize_unit(row.Unit)

        takeoff_suggestion = suggest_quantity_from_takeoff(row, extracted_scope, segment) if extracted_scope else None

        if quantity is None:
            if takeoff_suggestion and takeoff_suggestion.quantity and takeoff_suggestion.quantity > 0:
                resolved_quantity = takeoff_suggestion.quantity
                if takeoff_suggestion.area:
                    resolved_area = takeoff_suggestion.area
                if takeoff_suggestion.assumption:
                    assumptions.append(takeoff_suggestion.assumption)
                if takeoff_suggestion.needs_review:
                    needs_review = True
                confidence = max(confidence, 0.78 if not takeoff_suggestion.needs_review else 0.7)
                if takeoff_suggestion.review_reason:
                    review_reasons.append(takeoff_suggestion.review_reason)
            else:
                if normalized_row_unit in _MEASURED_QUANTITY_REQUIRED_UNITS and not (row.QUANTITY and row.QUANTITY > 0):
                    assumptions.append(
                        PreviewAssumption(
                            text=f"Skipped '{row.WorkName}' because its {normalized_row_unit} quantity was not stated and no defensible takeoff could be recovered.",
                            kind="quantity_missing",
                            severity="warning",
                        )
                    )
                    unmatched_items.append(
                        PreviewUnmatchedItem(
                            source_text=segment,
                            reason=f"Matched '{row.WorkName}' by scope, but {normalized_row_unit} quantity was missing and could not be safely inferred.",
                        )
                    )
                    continue
                assumptions.append(
                    PreviewAssumption(
                        text=f"Quantity not found for '{row.WorkName}', defaulted to {resolved_quantity}.",
                        kind="quantity_missing",
                        severity="warning",
                    )
                )
                needs_review = True
                review_reasons.append("Quantity was not found in the prompt, so a default value was used.")

        if requested_unit and normalized_row_unit and requested_unit != normalized_row_unit:
            assumptions.append(
                PreviewAssumption(
                    text=(
                        f"Unit mismatch for '{row.WorkName}': prompt used {requested_unit}, "
                        f"candidate row uses {normalized_row_unit}."
                    ),
                    kind="unit_mismatch",
                    severity="warning",
                )
            )
            needs_review = True
            review_reasons.append(f"Prompt unit {requested_unit} does not match the row unit {normalized_row_unit}.")

        if needs_review and not review_reasons:
            review_reasons.append("Low-confidence match. Please review before applying.")

        section_match = match_row_to_extracted_section(row, extracted_scope, segment) if extracted_scope else None
        if not _has_direct_scope_support(
            row,
            scope_text=segment,
            extracted_scope=extracted_scope,
            takeoff_suggestion=takeoff_suggestion,
        ):
            unmatched_items.append(
                PreviewUnmatchedItem(
                    source_text=segment,
                    reason=f"Matched row '{row.WorkName}' was rejected because the prompt does not contain direct scope evidence for it.",
                )
            )
            assumptions.append(
                PreviewAssumption(
                    text=f"Rejected '{row.WorkName}' because the segment did not contain direct scope evidence for that row.",
                    kind="scope_support_missing",
                    severity="warning",
                )
            )
            continue

        matched_row = _build_preview_row_from_candidate(
            row=row,
            quantity=resolved_quantity,
            area=resolved_area,
            confidence=confidence,
            needs_review=needs_review,
            review_reason=" ".join(review_reasons).strip() or None,
            matched_section_key=section_match.key if section_match else None,
            matched_section_title=section_match.title if section_match else None,
            matched_section_order=section_match.order if section_match else None,
        )
        assumptions.extend(
            _apply_hourly_unit_review(
                matched_row,
                row=row,
                scope_text=source_prompt,
                matched_section_key=matched_row.MatchedSectionKey,
            )
        )
        should_skip, guardrail_assumptions, guardrail_unmatched = _apply_guardrail_policy(
            matched_row,
            row=row,
        )
        assumptions.extend(guardrail_assumptions)
        if should_skip:
            if guardrail_unmatched is not None:
                unmatched_items.append(guardrail_unmatched)
            continue
        matched_rows.append(matched_row)
        specialist_assumptions = _apply_specialist_trade_review(
            matched_rows[-1],
            row=row,
            segment=segment,
        )
        assumptions.extend(specialist_assumptions)

    summary_text = (
        f"Matched {len(matched_rows)} works, {len(unmatched_items)} items need review."
        if matched_rows or unmatched_items
        else "No matches were produced."
    )

    assumptions.extend(build_scope_coverage_assumptions(source_prompt, matched_rows, extracted_scope))
    review_queue = _build_review_queue(matched_rows, unmatched_items, assumptions)
    coverage_prompts = _build_coverage_prompts(unmatched_items, assumptions)

    return PreviewResponse(
        summary_text=summary_text,
        service_mode="mock",
        input_mode=input_mode,
        review_summary=_build_review_summary(matched_rows, unmatched_items, assumptions),
        matched_rows=matched_rows,
        unmatched_items=unmatched_items,
        assumptions=assumptions,
        review_queue=review_queue,
        coverage_prompts=coverage_prompts,
        telemetry=_build_preview_telemetry(
            request,
            shortlist_size=len(shortlist),
            extracted_scope=extracted_scope,
        ),
        warnings=_build_response_warnings(assumptions),
        coverage_summary=_build_coverage_summary(
            matched_rows,
            unmatched_items,
            extracted_scope,
            normalized_scope_used=input_mode == "normalized",
        ),
        error_text="",
    )


def _generate_llm_preview(request: PreviewRequest, client: LLMClient) -> PreviewResponse:
    source_prompt, extracted_scope, input_mode = _resolve_preview_input(request)
    prompt = normalize_prompt(source_prompt)
    shortlist_prompt = f"{prompt}, {build_scope_matching_context(extracted_scope)}" if extracted_scope else prompt
    shortlist_limit = _resolve_shortlist_limit(
        source_prompt,
        extracted_scope,
        candidate_count=len(request.candidate_rows),
    )
    shortlist = shortlist_candidates(shortlist_prompt, request.candidate_rows, limit=shortlist_limit)
    budget_assumptions: list[PreviewAssumption] = []
    accepted_examples, examples_budget_assumption = _fit_accepted_examples_to_budget(
        prompt=source_prompt,
        shortlist=shortlist,
        extracted_scope=extracted_scope,
        accepted_examples=request.accepted_examples,
    )
    if examples_budget_assumption:
        budget_assumptions.append(examples_budget_assumption)
    shortlist, budget_assumption = _fit_shortlist_to_budget(
        prompt=source_prompt,
        shortlist=shortlist,
        extracted_scope=extracted_scope,
        accepted_examples=accepted_examples,
    )
    if budget_assumption:
        budget_assumptions.append(budget_assumption)
    candidate_lookup = {row.INSIDEQUOTESGUID: row for row in shortlist}
    allow_profit_override = _allows_profit_override(prompt)
    allow_labour_markup_override = _allows_labour_markup_override(prompt)
    allow_material_markup_override = _allows_material_markup_override(prompt)

    # Pass the structured source prompt so the LLM sees headings and room dimensions intact.
    while True:
        try:
            llm_output = client.preview_match(
                prompt=source_prompt,
                candidate_rows=shortlist,
                extracted_scope=extracted_scope,
                accepted_examples=accepted_examples,
                document_context=request.document_context or None,
            )
            break
        except TypeError as exc:
            error_text = str(exc)
            if "extracted_scope" in error_text or "accepted_examples" in error_text:
                llm_output = client.preview_match(prompt=source_prompt, candidate_rows=shortlist)
                break
            raise
        except Exception as exc:
            if not _looks_like_token_budget_error(exc):
                raise
            if accepted_examples:
                accepted_examples = []
                retry_examples_assumption = _build_examples_budget_assumption(
                    original_count=len(request.accepted_examples),
                    final_count=0,
                    rough_tokens=max(
                        _estimate_preview_message_chars(
                            prompt=source_prompt,
                            candidate_rows=shortlist,
                            extracted_scope=extracted_scope,
                            accepted_examples=accepted_examples,
                        )
                        // 4,
                        1,
                    ),
                )
                if retry_examples_assumption:
                    budget_assumptions.append(retry_examples_assumption)
                continue
            retried_shortlist, retry_budget_assumption = _retry_trimmed_shortlist(
                prompt=source_prompt,
                shortlist=shortlist,
                extracted_scope=extracted_scope,
                accepted_examples=accepted_examples,
            )
            if len(retried_shortlist) >= len(shortlist):
                raise
            shortlist = retried_shortlist
            candidate_lookup = {row.INSIDEQUOTESGUID: row for row in shortlist}
            if retry_budget_assumption:
                budget_assumptions.append(retry_budget_assumption)

    llm_metadata = getattr(client, "last_preview_metadata", None) or {"llm_attempt_count": 1}
    matched_rows: list[PreviewMatchedRow] = []
    unmatched_items = list(llm_output.unmatched_items)
    assumptions = [*budget_assumptions, *list(llm_output.assumptions)]
    used_ids: set[str] = set()

    for llm_row in llm_output.matched_rows:
        if llm_row.INSIDEQUOTESGUID in used_ids:
            assumptions.append(
                PreviewAssumption(
                    text=f"Duplicate AI selection ignored for row {llm_row.INSIDEQUOTESGUID}."
                )
            )
            continue

        candidate = candidate_lookup.get(llm_row.INSIDEQUOTESGUID)
        takeoff_suggestion = None
        effective_llm_row = llm_row
        if (llm_row.QUANTITY is None or llm_row.QUANTITY <= 0) and extracted_scope and candidate:
            # Try to recover missing quantities from the structured takeoff before strict validation.
            segment_hint = " ".join(filter(None, [llm_row.AREA, candidate.WorkName]))
            takeoff_suggestion = suggest_quantity_from_takeoff(candidate, extracted_scope, segment_hint)
            if takeoff_suggestion.quantity and takeoff_suggestion.quantity > 0:
                effective_llm_row = llm_row.model_copy(
                    update={
                        "QUANTITY": takeoff_suggestion.quantity,
                        "AREA": takeoff_suggestion.area or llm_row.AREA,
                    }
                )

        try:
            matched_row, validation_assumptions = validate_and_materialize_match(
                llm_row=effective_llm_row,
                candidate_lookup=candidate_lookup,
                allow_profit_override=allow_profit_override,
                allow_labour_markup_override=allow_labour_markup_override,
                allow_material_markup_override=allow_material_markup_override,
            )
        except ValueError as exc:
            quantity_missing = llm_row.QUANTITY is None or llm_row.QUANTITY <= 0
            source_text = candidate.WorkName if candidate else llm_row.INSIDEQUOTESGUID
            reason = f"AI returned an invalid row and it was skipped: {exc}"
            if quantity_missing:
                reason = "AI matched this row but could not produce a usable quantity after takeoff recovery."
                assumptions.append(
                    PreviewAssumption(
                        text=f"Skipped '{source_text}' because no defensible quantity could be recovered from the prompt, scope summary, or candidate defaults.",
                        kind="quantity_missing",
                        severity="warning",
                    )
                )
            unmatched_items.append(
                PreviewUnmatchedItem(
                    source_text=source_text,
                    reason=reason,
                )
            )
            continue

        if takeoff_suggestion and takeoff_suggestion.quantity and takeoff_suggestion.quantity > 0:
            matched_row.QUANTITY = takeoff_suggestion.quantity
            if takeoff_suggestion.area:
                matched_row.AREA = takeoff_suggestion.area
            matched_row.ClientTotalCost = calculate_client_total_cost(
                matched_row.QUANTITY,
                matched_row.ClientCostPerUnit,
            )
            matched_row.Confidence = max(matched_row.Confidence, 0.78 if not takeoff_suggestion.needs_review else 0.7)
            if takeoff_suggestion.needs_review:
                matched_row.NeedsReview = True
            if takeoff_suggestion.review_reason and not matched_row.ReviewReason:
                matched_row.ReviewReason = takeoff_suggestion.review_reason
            if takeoff_suggestion.assumption:
                assumptions.append(takeoff_suggestion.assumption)
        if extracted_scope:
            if candidate:
                # Use AREA hint for section matching too.
                segment_hint = " ".join(filter(None, [llm_row.AREA, candidate.WorkName]))
                section_match = match_row_to_extracted_section(candidate, extracted_scope, segment_hint)
                if section_match:
                    matched_row.MatchedSectionKey = section_match.key
                    matched_row.MatchedSectionTitle = section_match.title
                    matched_row.MatchedSectionOrder = section_match.order
        if candidate and not _has_direct_scope_support(
            candidate,
            scope_text=source_prompt,
            extracted_scope=extracted_scope,
            takeoff_suggestion=takeoff_suggestion,
        ):
            unmatched_items.append(
                PreviewUnmatchedItem(
                    source_text=matched_row.WorkName,
                    reason="AI selected a row that is not directly supported by the prompt or structured scope.",
                )
            )
            assumptions.append(
                PreviewAssumption(
                    text=f"Rejected '{matched_row.WorkName}' because the prompt and structured scope did not contain direct evidence for that row.",
                    kind="scope_support_missing",
                    severity="warning",
                )
            )
            continue
        if candidate:
            should_skip, guardrail_assumptions, guardrail_unmatched = _apply_guardrail_policy(
                matched_row,
                row=candidate,
            )
            assumptions.extend(guardrail_assumptions)
            if should_skip:
                if guardrail_unmatched is not None:
                    unmatched_items.append(guardrail_unmatched)
                continue
        if candidate := candidate_lookup.get(llm_row.INSIDEQUOTESGUID):
            segment_hint = " ".join(filter(None, [llm_row.AREA, candidate.WorkName]))
            assumptions.extend(
                _apply_hourly_unit_review(
                    matched_row,
                    row=candidate,
                    scope_text=source_prompt,
                    matched_section_key=matched_row.MatchedSectionKey,
                )
            )
            assumptions.extend(
                _apply_specialist_trade_review(
                    matched_row,
                    row=candidate,
                    segment=segment_hint,
                )
            )
        matched_rows.append(matched_row)
        assumptions.extend(validation_assumptions)
        used_ids.add(llm_row.INSIDEQUOTESGUID)

    summary_text = (
        f"Matched {len(matched_rows)} works, {len(unmatched_items)} items need review."
        if matched_rows or unmatched_items
        else "No matches were produced."
    )

    assumptions.extend(build_scope_coverage_assumptions(source_prompt, matched_rows, extracted_scope))
    review_queue = _build_review_queue(matched_rows, unmatched_items, assumptions)
    coverage_prompts = _build_coverage_prompts(unmatched_items, assumptions)

    return PreviewResponse(
        summary_text=summary_text,
        service_mode="llm",
        input_mode=input_mode,
        review_summary=_build_review_summary(matched_rows, unmatched_items, assumptions),
        matched_rows=matched_rows,
        unmatched_items=unmatched_items,
        assumptions=assumptions,
        review_queue=review_queue,
        coverage_prompts=coverage_prompts,
        telemetry=_build_preview_telemetry(
            request,
            shortlist_size=len(shortlist),
            extracted_scope=extracted_scope,
            llm_metadata=llm_metadata,
            accepted_example_count=len(accepted_examples),
        ),
        warnings=_build_response_warnings(assumptions),
        coverage_summary=_build_coverage_summary(
            matched_rows,
            unmatched_items,
            extracted_scope,
            normalized_scope_used=input_mode == "normalized",
        ),
        error_text="",
    )


def generate_preview(request: PreviewRequest) -> PreviewResponse:
    client = LLMClient()
    if client.is_enabled():
        try:
            return _generate_llm_preview(request, client)
        except Exception as exc:
            if not settings.allow_mock_fallback:
                raise RuntimeError("LLM preview failed and mock fallback is disabled in this environment.") from exc
            fallback = _generate_mock_preview(request)
            fallback_kind = "normalized_service_fallback" if _has_normalized_input(request) else "service_fallback"
            fallback_message = (
                f"Normalized AI parsing failed, so heuristic fallback mode was used instead: {str(exc)}"
                if _has_normalized_input(request)
                else f"LLM preview failed, mock fallback was used instead: {str(exc)}"
            )
            fallback.assumptions.append(
                PreviewAssumption(
                    text=fallback_message,
                    kind=fallback_kind,
                    severity="warning",
                )
            )
            fallback.summary_text = (
                f"{fallback.summary_text} Fallback heuristic mode was used."
            )
            fallback.service_mode = "mock-fallback"
            llm_metadata = getattr(client, "last_preview_metadata", None) or {}
            fallback.telemetry.llm_attempt_count = max(
                fallback.telemetry.llm_attempt_count,
                int(llm_metadata.get("llm_attempt_count", 0) or 0),
            )
            fallback.telemetry.llm_latency_ms = max(
                fallback.telemetry.llm_latency_ms,
                int(llm_metadata.get("llm_latency_ms", 0) or 0),
            )
            fallback.telemetry.parse_retry_used = bool(
                llm_metadata.get("parse_retry_used", fallback.telemetry.parse_retry_used)
            )
            fallback.telemetry.structured_output_mode = str(
                llm_metadata.get("structured_output_mode", fallback.telemetry.structured_output_mode)
                or fallback.telemetry.structured_output_mode
            )
            fallback.telemetry.fallback_reason = _truncate_fallback_reason(f"{type(exc).__name__}: {exc}")
            if fallback.telemetry.room_count > 0:
                fallback.telemetry.takeoff_heuristics_version = TAKEOFF_HEURISTICS_VERSION
                fallback.telemetry.takeoff_heuristics_source = TAKEOFF_HEURISTICS_SOURCE
            fallback.review_summary = _build_review_summary(
                fallback.matched_rows,
                fallback.unmatched_items,
                fallback.assumptions,
            )
            fallback.warnings = _build_response_warnings(fallback.assumptions)
            return fallback

    if not settings.allow_mock_fallback:
        raise RuntimeError("LLM preview is not configured and mock mode is disabled in this environment.")

    fallback = _generate_mock_preview(request)
    fallback.assumptions.append(
        PreviewAssumption(
            text="Service is running in mock mode because OPENAI_API_KEY or OPENAI_MODEL is not configured.",
            kind="service_mode",
            severity="info",
        )
    )
    fallback.telemetry.fallback_reason = "llm_not_configured"
    fallback.review_summary = _build_review_summary(
        fallback.matched_rows,
        fallback.unmatched_items,
        fallback.assumptions,
    )
    fallback.warnings = _build_response_warnings(fallback.assumptions)
    return fallback
