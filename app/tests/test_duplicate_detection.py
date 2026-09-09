from __future__ import annotations

from app.schemas import PreviewMatchedRow
from app.services.matcher import _build_duplicate_flags


def _matched_row(
    *,
    guid: str = "row-1",
    work_name: str = "Paint walls",
    area: str = "Kitchen",
    matched_section_key: str | None = None,
) -> PreviewMatchedRow:
    return PreviewMatchedRow(
        INSIDEQUOTESGUID=guid,
        WorkName=work_name,
        Unit="m2",
        AREA=area,
        QUANTITY=10,
        PROFIT=20,
        LabourMarkup=15,
        MaterialMarkup=10,
        WorkQTYforNorm=1,
        ClientCostPerUnit=10,
        ClientTotalCost=100,
        Confidence=0.9,
        MatchedSectionKey=matched_section_key,
    )


class TestBuildDuplicateFlags:
    def test_no_flags_for_a_single_row(self):
        rows = [_matched_row()]

        assert _build_duplicate_flags(rows) == []

    def test_flags_near_identical_names_in_the_same_room(self):
        rows = [
            _matched_row(
                guid="row-1",
                work_name="Strip out existing kitchen units",
                area="Kitchen",
            ),
            _matched_row(
                guid="row-2", work_name="Strip out kitchen units", area="Kitchen"
            ),
        ]

        flags = _build_duplicate_flags(rows)

        assert len(flags) == 1
        assert {flags[0].row_a_guid, flags[0].row_b_guid} == {"row-1", "row-2"}
        assert flags[0].scope_label == "kitchen"

    def test_does_not_flag_different_rooms(self):
        rows = [
            _matched_row(
                guid="row-1",
                work_name="Strip out existing kitchen units",
                area="Kitchen",
            ),
            _matched_row(
                guid="row-2", work_name="Strip out kitchen units", area="Bathroom"
            ),
        ]

        assert _build_duplicate_flags(rows) == []

    def test_does_not_flag_distinct_scope_in_the_same_room(self):
        rows = [
            _matched_row(guid="row-1", work_name="Tile floor", area="Bathroom"),
            _matched_row(guid="row-2", work_name="Tile walls", area="Bathroom"),
        ]

        assert _build_duplicate_flags(rows) == []

    def test_does_not_flag_short_generic_names_with_only_one_shared_token(self):
        rows = [
            _matched_row(guid="row-1", work_name="Paint walls", area="Kitchen"),
            _matched_row(guid="row-2", work_name="Paint ceiling", area="Kitchen"),
        ]

        assert _build_duplicate_flags(rows) == []

    def test_groups_by_matched_section_key_over_area_when_present(self):
        rows = [
            _matched_row(
                guid="row-1",
                work_name="Wet underfloor heating, screeded",
                area="12.8 m2",
                matched_section_key="heating",
            ),
            _matched_row(
                guid="row-2",
                work_name="Wet underfloor heating screeded",
                area="42.8 m2",
                matched_section_key="heating",
            ),
        ]

        flags = _build_duplicate_flags(rows)

        assert len(flags) == 1
        assert flags[0].scope_label == "heating"

    def test_three_rows_in_one_group_only_flags_the_similar_pair(self):
        rows = [
            _matched_row(
                guid="row-1", work_name="Strip out kitchen units", area="Kitchen"
            ),
            _matched_row(
                guid="row-2",
                work_name="Strip out existing kitchen units",
                area="Kitchen",
            ),
            _matched_row(
                guid="row-3", work_name="Install new kitchen tap", area="Kitchen"
            ),
        ]

        flags = _build_duplicate_flags(rows)

        assert len(flags) == 1
        assert {flags[0].row_a_guid, flags[0].row_b_guid} == {"row-1", "row-2"}

    def test_ungrouped_rows_with_no_area_or_section_still_compare_within_that_bucket(
        self,
    ):
        rows = [
            _matched_row(guid="row-1", work_name="Strip out kitchen units", area=""),
            _matched_row(
                guid="row-2", work_name="Strip out existing kitchen units", area=""
            ),
        ]

        flags = _build_duplicate_flags(rows)

        assert len(flags) == 1
        assert flags[0].scope_label == ""
