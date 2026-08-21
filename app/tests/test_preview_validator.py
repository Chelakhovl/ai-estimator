from app.schemas import CandidateRow, LLMPreviewMatchedRow
from app.services.preview_validator import validate_and_materialize_match


def test_validator_preserves_candidate_defaults_when_zero_not_explicitly_allowed() -> (
    None
):
    candidate = CandidateRow(
        INSIDEQUOTESGUID="inside-1",
        WORKGUID="work-1",
        WorkName="Paint walls",
        Unit="m2",
        WorkLabourCost=10,
        WorkMatCost=4,
        WorkOtherCost=1,
        WorkQTYforNorm=1,
        PROFIT=20,
        LabourMarkup=15,
        MaterialMarkup=10,
    )
    llm_row = LLMPreviewMatchedRow(
        INSIDEQUOTESGUID="inside-1",
        AREA="Kitchen",
        QUANTITY=120,
        PROFIT=0,
        LabourMarkup=0,
        MaterialMarkup=0,
        Confidence=1,
        NeedsReview=False,
    )

    matched_row, assumptions = validate_and_materialize_match(
        llm_row=llm_row,
        candidate_lookup={"inside-1": candidate},
        allow_profit_override=False,
        allow_labour_markup_override=False,
        allow_material_markup_override=False,
    )

    assert matched_row.PROFIT == 20
    assert matched_row.LabourMarkup == 15
    assert matched_row.MaterialMarkup == 10
    assert matched_row.ClientCostPerUnit == 20.4
    assert matched_row.ClientTotalCost == 2448.0
    assert matched_row.ReviewReason is None
    assert assumptions == []
    assert matched_row.MarkupOverridden is False


def test_validator_allows_zero_override_when_explicitly_permitted() -> None:
    candidate = CandidateRow(
        INSIDEQUOTESGUID="inside-1",
        WORKGUID="work-1",
        WorkName="Paint walls",
        Unit="m2",
        WorkLabourCost=10,
        WorkMatCost=4,
        WorkOtherCost=1,
        WorkQTYforNorm=1,
        PROFIT=20,
        LabourMarkup=15,
        MaterialMarkup=10,
    )
    llm_row = LLMPreviewMatchedRow(
        INSIDEQUOTESGUID="inside-1",
        AREA="Kitchen",
        QUANTITY=120,
        PROFIT=0,
        LabourMarkup=0,
        MaterialMarkup=0,
        Confidence=1,
        NeedsReview=False,
    )

    matched_row, _ = validate_and_materialize_match(
        llm_row=llm_row,
        candidate_lookup={"inside-1": candidate},
        allow_profit_override=True,
        allow_labour_markup_override=True,
        allow_material_markup_override=True,
    )

    assert matched_row.PROFIT == 0
    assert matched_row.LabourMarkup == 0
    assert matched_row.MaterialMarkup == 0
    assert matched_row.ClientCostPerUnit == 15.0
    assert matched_row.ClientTotalCost == 1800.0
    assert matched_row.MarkupOverridden is True


def test_validator_flags_markup_overridden_when_llm_returns_inflated_nonzero_values() -> (
    None
):
    # Regression: a nonzero PROFIT/markup from the model is used unconditionally by
    # _resolve_optional_numeric_override (the allow_*_override gate only guards a
    # zero value replacing a nonzero default) - so an inflated nonzero value from
    # the model silently passes straight through into the computed price with no
    # signal to the estimator that it differs from the catalog's own default.
    candidate = CandidateRow(
        INSIDEQUOTESGUID="inside-1",
        WORKGUID="work-1",
        WorkName="Floor and walls tiling",
        Unit="m2",
        WorkLabourCost=10,
        WorkMatCost=4,
        WorkOtherCost=1,
        WorkQTYforNorm=1,
        PROFIT=20,
        LabourMarkup=15,
        MaterialMarkup=10,
    )
    llm_row = LLMPreviewMatchedRow(
        INSIDEQUOTESGUID="inside-1",
        AREA="Bathroom",
        QUANTITY=12.94,
        PROFIT=180,
        LabourMarkup=15,
        MaterialMarkup=10,
        Confidence=0.9,
        NeedsReview=False,
    )

    matched_row, _ = validate_and_materialize_match(
        llm_row=llm_row,
        candidate_lookup={"inside-1": candidate},
    )

    assert matched_row.PROFIT == 180
    assert matched_row.MarkupOverridden is True


def test_validator_does_not_flag_markup_overridden_when_llm_omits_fields() -> None:
    candidate = CandidateRow(
        INSIDEQUOTESGUID="inside-1",
        WORKGUID="work-1",
        WorkName="Paint walls",
        Unit="m2",
        WorkLabourCost=10,
        WorkMatCost=4,
        WorkOtherCost=1,
        WorkQTYforNorm=1,
        PROFIT=20,
        LabourMarkup=15,
        MaterialMarkup=10,
    )
    llm_row = LLMPreviewMatchedRow(
        INSIDEQUOTESGUID="inside-1",
        AREA="Kitchen",
        QUANTITY=120,
        Confidence=1,
        NeedsReview=False,
    )

    matched_row, _ = validate_and_materialize_match(
        llm_row=llm_row,
        candidate_lookup={"inside-1": candidate},
    )

    assert matched_row.PROFIT == 20
    assert matched_row.MarkupOverridden is False
