from __future__ import annotations

from app.schemas import CustomPricedRow, PreviewMatchedRow, PreviewUnmatchedItem
from app.services.matcher import _build_indicative_range


def _matched_row(
    *,
    guid: str = "row-1",
    work_name: str = "Paint walls",
    total_cost: float = 1000.0,
    confidence: float = 0.9,
    needs_review: bool = False,
) -> PreviewMatchedRow:
    return PreviewMatchedRow(
        INSIDEQUOTESGUID=guid,
        WorkName=work_name,
        Unit="m2",
        AREA="Kitchen",
        QUANTITY=10,
        PROFIT=20,
        LabourMarkup=15,
        MaterialMarkup=10,
        WorkQTYforNorm=1,
        ClientCostPerUnit=total_cost / 10,
        ClientTotalCost=total_cost,
        Confidence=confidence,
        NeedsReview=needs_review,
    )


def _custom_row(*, total_cost: float = 500.0) -> CustomPricedRow:
    return CustomPricedRow(
        source_text="bespoke item",
        name="Bespoke item",
        unit="item",
        labour_cost=total_cost * 0.6,
        material_cost=total_cost * 0.4,
        other_cost=0,
        work_days=1,
        qty_for_norm=1,
    )


class TestBuildIndicativeRange:
    def test_returns_none_when_nothing_matched_or_priced(self):
        result = _build_indicative_range([], [], [])

        assert result is None

    def test_confident_rows_get_a_tight_ten_percent_band(self):
        rows = [_matched_row(total_cost=1000.0, confidence=0.95, needs_review=False)]

        result = _build_indicative_range(rows, [], [])

        assert result is not None
        assert result.matched_total == 1000.0
        assert result.low == 900.0
        assert result.high == 1100.0

    def test_needs_review_rows_get_a_wide_thirty_five_percent_band(self):
        rows = [_matched_row(total_cost=1000.0, confidence=0.5, needs_review=True)]

        result = _build_indicative_range(rows, [], [])

        assert result is not None
        assert result.low == 650.0
        assert result.high == 1350.0

    def test_low_confidence_row_gets_wide_band_even_when_not_flagged(self):
        # Confidence below the threshold is treated as uncertain even if
        # NeedsReview itself wasn't set - both conditions must hold for the tight band.
        rows = [_matched_row(total_cost=1000.0, confidence=0.6, needs_review=False)]

        result = _build_indicative_range(rows, [], [])

        assert result is not None
        assert result.low == 650.0
        assert result.high == 1350.0

    def test_mixed_confident_and_uncertain_rows_blend_the_bands(self):
        rows = [
            _matched_row(
                guid="row-1", total_cost=1000.0, confidence=0.95, needs_review=False
            ),
            _matched_row(
                guid="row-2", total_cost=1000.0, confidence=0.5, needs_review=True
            ),
        ]

        result = _build_indicative_range(rows, [], [])

        assert result is not None
        assert result.matched_total == 2000.0
        # confident half tight (900-1100), uncertain half wide (650-1350)
        assert result.low == 1550.0
        assert result.high == 2450.0

    def test_unmatched_items_only_widen_the_high_end(self):
        rows = [_matched_row(total_cost=1000.0, confidence=0.95, needs_review=False)]
        without_gap = _build_indicative_range(rows, [], [])
        with_gap = _build_indicative_range(
            rows,
            [
                PreviewUnmatchedItem(
                    source_text="loft insulation", reason="no match found"
                )
            ],
            [],
        )

        assert without_gap is not None and with_gap is not None
        assert with_gap.low == without_gap.low
        assert with_gap.high > without_gap.high

    def test_custom_priced_rows_are_included_in_matched_total_with_a_wide_band(self):
        result = _build_indicative_range([], [], [_custom_row(total_cost=500.0)])

        assert result is not None
        assert result.matched_total == 500.0
        assert result.low == 325.0
        assert result.high == 675.0

    def test_basis_text_describes_the_row_counts(self):
        rows = [
            _matched_row(guid="row-1", needs_review=False, confidence=0.9),
            _matched_row(guid="row-2", needs_review=True, confidence=0.4),
        ]

        result = _build_indicative_range(
            rows,
            [PreviewUnmatchedItem(source_text="x", reason="y")],
            [],
        )

        assert result is not None
        assert "2 rows matched" in result.basis
        assert "1 flagged for review" in result.basis
        assert "1 scope item unmatched" in result.basis

    def test_low_never_goes_negative(self):
        rows = [_matched_row(total_cost=10.0, confidence=0.3, needs_review=True)]

        result = _build_indicative_range(rows, [], [])

        assert result is not None
        assert result.low >= 0
