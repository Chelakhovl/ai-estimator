from __future__ import annotations

from unittest.mock import MagicMock

from app.schemas import CandidateRow, LLMPreviewOutput, PreviewMatchedRow
from app.services.matcher import _run_double_read, _should_recommend_double_read


def _candidate_row(guid: str = "row-1", work_name: str = "Paint walls") -> CandidateRow:
    return CandidateRow(INSIDEQUOTESGUID=guid, WorkName=work_name, Unit="m2")


def _matched_row(
    *,
    guid: str = "row-1",
    quantity: float = 10.0,
    total_cost: float = 100.0,
    confidence: float = 0.9,
    needs_review: bool = False,
) -> PreviewMatchedRow:
    return PreviewMatchedRow(
        INSIDEQUOTESGUID=guid,
        WorkName="Paint walls",
        Unit="m2",
        AREA="Kitchen",
        QUANTITY=quantity,
        PROFIT=20,
        LabourMarkup=15,
        MaterialMarkup=10,
        WorkQTYforNorm=1,
        ClientCostPerUnit=total_cost / quantity,
        ClientTotalCost=total_cost,
        Confidence=confidence,
        NeedsReview=needs_review,
    )


def _llm_output_with_rows(rows: list[tuple[str, float]]) -> LLMPreviewOutput:
    from app.schemas import LLMPreviewMatchedRow

    return LLMPreviewOutput(
        matched_rows=[
            LLMPreviewMatchedRow(INSIDEQUOTESGUID=guid, QUANTITY=quantity)
            for guid, quantity in rows
        ],
        unmatched_items=[],
        assumptions=[],
    )


class TestShouldRecommendDoubleRead:
    def test_false_when_no_matched_rows(self):
        assert _should_recommend_double_read([]) is False

    def test_true_when_total_exceeds_threshold(self):
        rows = [_matched_row(total_cost=20000.0, confidence=0.95, needs_review=False)]

        assert _should_recommend_double_read(rows) is True

    def test_true_when_review_ratio_exceeds_threshold(self):
        rows = [
            _matched_row(guid="row-1", total_cost=100.0, needs_review=True),
            _matched_row(guid="row-2", total_cost=100.0, needs_review=False),
        ]

        assert _should_recommend_double_read(rows) is True

    def test_false_for_a_small_confident_draft(self):
        rows = [_matched_row(total_cost=100.0, confidence=0.95, needs_review=False)]

        assert _should_recommend_double_read(rows) is False


class TestRunDoubleRead:
    def test_shuffles_candidate_order_and_uses_nonzero_temperature(self):
        shortlist = [_candidate_row(guid=f"row-{i}") for i in range(20)]
        matched_rows = [_matched_row(guid="row-1", quantity=10.0)]
        client = MagicMock()
        client.preview_match.return_value = _llm_output_with_rows([("row-1", 10.0)])

        _run_double_read(
            client,
            source_prompt="Paint walls 10m2",
            shortlist=shortlist,
            extracted_scope=None,
            document_context=None,
            matched_rows=matched_rows,
        )

        client.preview_match.assert_called_once()
        _, kwargs = client.preview_match.call_args
        assert kwargs["temperature"] == 0.4
        assert kwargs["accepted_examples"] is None
        # Shuffled order differs from the original (deterministically seeded on
        # the prompt, but not equal to input order for a 20-item list).
        called_guids = [row.INSIDEQUOTESGUID for row in kwargs["candidate_rows"]]
        original_guids = [row.INSIDEQUOTESGUID for row in shortlist]
        assert called_guids != original_guids
        assert sorted(called_guids) == sorted(original_guids)

    def test_corroborates_a_row_with_matching_quantity(self):
        matched_rows = [_matched_row(guid="row-1", quantity=10.0)]
        client = MagicMock()
        client.preview_match.return_value = _llm_output_with_rows([("row-1", 10.5)])

        summary = _run_double_read(
            client,
            source_prompt="Paint walls 10m2",
            shortlist=[_candidate_row(guid="row-1")],
            extracted_scope=None,
            document_context=None,
            matched_rows=matched_rows,
        )

        assert matched_rows[0].Corroborated is True
        assert summary is not None
        assert summary.corroborated_count == 1
        assert summary.diverged_count == 0
        assert summary.only_in_first_pass_count == 0

    def test_flags_diverging_quantity_as_not_corroborated(self):
        matched_rows = [_matched_row(guid="row-1", quantity=10.0)]
        client = MagicMock()
        # 50% off - well outside the 15% tolerance.
        client.preview_match.return_value = _llm_output_with_rows([("row-1", 15.0)])

        summary = _run_double_read(
            client,
            source_prompt="Paint walls 10m2",
            shortlist=[_candidate_row(guid="row-1")],
            extracted_scope=None,
            document_context=None,
            matched_rows=matched_rows,
        )

        assert matched_rows[0].Corroborated is False
        assert summary is not None
        assert summary.diverged_count == 1
        assert summary.corroborated_count == 0

    def test_flags_row_absent_from_second_pass_as_only_in_first_pass(self):
        matched_rows = [_matched_row(guid="row-1", quantity=10.0)]
        client = MagicMock()
        client.preview_match.return_value = _llm_output_with_rows([])

        summary = _run_double_read(
            client,
            source_prompt="Paint walls 10m2",
            shortlist=[_candidate_row(guid="row-1")],
            extracted_scope=None,
            document_context=None,
            matched_rows=matched_rows,
        )

        assert matched_rows[0].Corroborated is False
        assert summary is not None
        assert summary.only_in_first_pass_count == 1
        assert summary.corroborated_count == 0
        assert summary.diverged_count == 0

    def test_counts_a_row_only_the_second_pass_found(self):
        matched_rows = [_matched_row(guid="row-1", quantity=10.0)]
        client = MagicMock()
        client.preview_match.return_value = _llm_output_with_rows(
            [("row-1", 10.0), ("row-2", 5.0)]
        )

        summary = _run_double_read(
            client,
            source_prompt="Paint walls 10m2",
            shortlist=[_candidate_row(guid="row-1"), _candidate_row(guid="row-2")],
            extracted_scope=None,
            document_context=None,
            matched_rows=matched_rows,
        )

        assert summary is not None
        assert summary.only_in_second_pass_count == 1

    def test_returns_none_when_second_pass_fails(self):
        matched_rows = [_matched_row(guid="row-1", quantity=10.0)]
        client = MagicMock()
        client.preview_match.side_effect = RuntimeError("network unreachable")

        summary = _run_double_read(
            client,
            source_prompt="Paint walls 10m2",
            shortlist=[_candidate_row(guid="row-1")],
            extracted_scope=None,
            document_context=None,
            matched_rows=matched_rows,
        )

        assert summary is None
        # Corroborated must not be forced to some value on failure - untouched.
        assert matched_rows[0].Corroborated is None
