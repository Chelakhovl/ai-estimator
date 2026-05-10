from __future__ import annotations

from app.schemas import CandidateRow, LLMPreviewMatchedRow, PreviewAssumption, PreviewMatchedRow
from app.services.calculator import calculate_client_cost_per_unit, calculate_client_total_cost


def _resolve_optional_numeric_override(
    llm_value: float | None,
    candidate_value: float,
    allow_zero_override: bool,
) -> float:
    if llm_value is None:
        return candidate_value
    if llm_value == 0 and candidate_value != 0 and not allow_zero_override:
        return candidate_value
    return llm_value


def validate_and_materialize_match(
    llm_row: LLMPreviewMatchedRow,
    candidate_lookup: dict[str, CandidateRow],
    allow_profit_override: bool = False,
    allow_labour_markup_override: bool = False,
    allow_material_markup_override: bool = False,
) -> tuple[PreviewMatchedRow, list[PreviewAssumption]]:
    candidate = candidate_lookup.get(llm_row.INSIDEQUOTESGUID)
    if candidate is None:
        raise ValueError(f"Unknown INSIDEQUOTESGUID returned by model: {llm_row.INSIDEQUOTESGUID}")

    assumptions: list[PreviewAssumption] = []

    quantity = llm_row.QUANTITY if llm_row.QUANTITY is not None else candidate.QUANTITY
    if quantity is None or quantity <= 0:
        raise ValueError(f"Invalid quantity for row {candidate.INSIDEQUOTESGUID}")

    profit = _resolve_optional_numeric_override(
        llm_value=llm_row.PROFIT,
        candidate_value=candidate.PROFIT,
        allow_zero_override=allow_profit_override,
    )
    labour_markup = _resolve_optional_numeric_override(
        llm_value=llm_row.LabourMarkup,
        candidate_value=candidate.LabourMarkup,
        allow_zero_override=allow_labour_markup_override,
    )
    material_markup = _resolve_optional_numeric_override(
        llm_value=llm_row.MaterialMarkup,
        candidate_value=candidate.MaterialMarkup,
        allow_zero_override=allow_material_markup_override,
    )

    if not 0 <= profit <= 200:
        raise ValueError(f"Invalid PROFIT for row {candidate.INSIDEQUOTESGUID}")
    if not 0 <= labour_markup <= 200:
        raise ValueError(f"Invalid LabourMarkup for row {candidate.INSIDEQUOTESGUID}")
    if not 0 <= material_markup <= 100:
        raise ValueError(f"Invalid MaterialMarkup for row {candidate.INSIDEQUOTESGUID}")

    area = (llm_row.AREA if llm_row.AREA is not None else candidate.AREA) or ""
    if len(area) > 50:
        area = area[:50]
        assumptions.append(
            PreviewAssumption(
                text=f"AREA was truncated to 50 characters for row '{candidate.WorkName}'.",
                kind="area_truncated",
                severity="info",
            )
        )

    confidence = llm_row.Confidence if llm_row.Confidence is not None else 0.7
    confidence = min(1, max(0, confidence))
    needs_review = llm_row.NeedsReview if llm_row.NeedsReview is not None else confidence < 0.75
    review_reason = llm_row.ReviewReason
    if needs_review and not review_reason:
        review_reason = "Model flagged this row for manual review."

    client_cost_per_unit = calculate_client_cost_per_unit(
        work_labour_cost=candidate.WorkLabourCost,
        work_mat_cost=candidate.WorkMatCost,
        work_other_cost=candidate.WorkOtherCost,
        labour_markup=labour_markup,
        material_markup=material_markup,
        profit=profit,
        work_qty_for_norm=candidate.WorkQTYforNorm,
    )
    client_total_cost = calculate_client_total_cost(quantity, client_cost_per_unit)

    matched_row = PreviewMatchedRow(
        INSIDEQUOTESGUID=candidate.INSIDEQUOTESGUID,
        WORKGUID=candidate.WORKGUID,
        WorkName=candidate.WorkName,
        Unit=candidate.Unit,
        AREA=area,
        QUANTITY=quantity,
        PROFIT=profit,
        LabourMarkup=labour_markup,
        MaterialMarkup=material_markup,
        WorkLabourCost=candidate.WorkLabourCost,
        WorkMatCost=candidate.WorkMatCost,
        WorkOtherCost=candidate.WorkOtherCost,
        WorkQTYforNorm=candidate.WorkQTYforNorm,
        ClientCostPerUnit=client_cost_per_unit,
        ClientTotalCost=client_total_cost,
        Confidence=round(confidence, 2),
        NeedsReview=needs_review,
        ReviewReason=review_reason,
    )
    return matched_row, assumptions
