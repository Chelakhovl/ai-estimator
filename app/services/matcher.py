from __future__ import annotations

from dataclasses import dataclass
import re

from app.schemas import (
    CandidateRow,
    PreviewAssumption,
    PreviewMatchedRow,
    PreviewRequest,
    PreviewResponse,
    PreviewUnmatchedItem,
)
from app.services.calculator import calculate_client_cost_per_unit, calculate_client_total_cost
from app.services.candidate_shortlist import shortlist_candidates
from app.services.llm_client import LLMClient
from app.services.normalizer import (
    detect_area,
    normalize_prompt,
    normalize_unit,
    parse_quantity_and_unit,
    split_prompt_segments,
    tokenize,
)
from app.services.preview_validator import validate_and_materialize_match


@dataclass(frozen=True)
class MatchCandidate:
    row: CandidateRow
    score: float


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
    segment_tokens = tokenize(segment)
    work_tokens = tokenize(row.WorkName)
    overlap = len(segment_tokens & work_tokens)

    score = float(overlap)
    normalized_row_unit = normalize_unit(row.Unit)
    if requested_unit and normalized_row_unit == requested_unit:
        score += 1.5
    if requested_unit and normalized_row_unit and normalized_row_unit != requested_unit:
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


def _build_preview_row_from_candidate(
    row: CandidateRow,
    quantity: float,
    area: str,
    confidence: float,
    needs_review: bool,
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
    )


def _generate_mock_preview(request: PreviewRequest) -> PreviewResponse:
    prompt = normalize_prompt(request.prompt)
    segments = split_prompt_segments(prompt)
    shortlist = shortlist_candidates(prompt, request.candidate_rows)

    matched_rows: list[PreviewMatchedRow] = []
    unmatched_items: list[PreviewUnmatchedItem] = []
    assumptions: list[PreviewAssumption] = []
    used_ids: set[str] = set()

    for segment in segments:
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

        if quantity is None:
            assumptions.append(
                PreviewAssumption(
                    text=f"Quantity not found for '{row.WorkName}', defaulted to {resolved_quantity}."
                )
            )
            needs_review = True

        normalized_row_unit = normalize_unit(row.Unit)
        if requested_unit and normalized_row_unit and requested_unit != normalized_row_unit:
            assumptions.append(
                PreviewAssumption(
                    text=(
                        f"Unit mismatch for '{row.WorkName}': prompt used {requested_unit}, "
                        f"candidate row uses {normalized_row_unit}."
                    )
                )
            )
            needs_review = True

        matched_rows.append(
            _build_preview_row_from_candidate(
                row=row,
                quantity=resolved_quantity,
                area=resolved_area,
                confidence=confidence,
                needs_review=needs_review,
            )
        )

    summary_text = (
        f"Matched {len(matched_rows)} works, {len(unmatched_items)} items need review."
        if matched_rows or unmatched_items
        else "No matches were produced."
    )

    return PreviewResponse(
        summary_text=summary_text,
        matched_rows=matched_rows,
        unmatched_items=unmatched_items,
        assumptions=assumptions,
        error_text="",
    )


def _generate_llm_preview(request: PreviewRequest, client: LLMClient) -> PreviewResponse:
    prompt = normalize_prompt(request.prompt)
    shortlist = shortlist_candidates(prompt, request.candidate_rows)
    candidate_lookup = {row.INSIDEQUOTESGUID: row for row in shortlist}
    allow_profit_override = _allows_profit_override(prompt)
    allow_labour_markup_override = _allows_labour_markup_override(prompt)
    allow_material_markup_override = _allows_material_markup_override(prompt)

    llm_output = client.preview_match(prompt=prompt, candidate_rows=shortlist)

    matched_rows: list[PreviewMatchedRow] = []
    unmatched_items = list(llm_output.unmatched_items)
    assumptions = list(llm_output.assumptions)
    used_ids: set[str] = set()

    for llm_row in llm_output.matched_rows:
        if llm_row.INSIDEQUOTESGUID in used_ids:
            assumptions.append(
                PreviewAssumption(
                    text=f"Duplicate AI selection ignored for row {llm_row.INSIDEQUOTESGUID}."
                )
            )
            continue

        matched_row, validation_assumptions = validate_and_materialize_match(
            llm_row=llm_row,
            candidate_lookup=candidate_lookup,
            allow_profit_override=allow_profit_override,
            allow_labour_markup_override=allow_labour_markup_override,
            allow_material_markup_override=allow_material_markup_override,
        )
        matched_rows.append(matched_row)
        assumptions.extend(validation_assumptions)
        used_ids.add(llm_row.INSIDEQUOTESGUID)

    summary_text = (
        f"Matched {len(matched_rows)} works, {len(unmatched_items)} items need review."
        if matched_rows or unmatched_items
        else "No matches were produced."
    )

    return PreviewResponse(
        summary_text=summary_text,
        matched_rows=matched_rows,
        unmatched_items=unmatched_items,
        assumptions=assumptions,
        error_text="",
    )


def generate_preview(request: PreviewRequest) -> PreviewResponse:
    client = LLMClient()
    if client.is_enabled():
        try:
            return _generate_llm_preview(request, client)
        except Exception as exc:
            fallback = _generate_mock_preview(request)
            fallback.assumptions.append(
                PreviewAssumption(
                    text=f"LLM preview failed, mock fallback was used instead: {str(exc)}"
                )
            )
            fallback.summary_text = (
                f"{fallback.summary_text} Fallback heuristic mode was used."
            )
            return fallback

    fallback = _generate_mock_preview(request)
    fallback.assumptions.append(
        PreviewAssumption(
            text="Service is running in mock mode because OPENAI_API_KEY or OPENAI_MODEL is not configured."
        )
    )
    return fallback
