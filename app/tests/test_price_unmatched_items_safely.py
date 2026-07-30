from __future__ import annotations

from unittest.mock import patch

from app.schemas import (
    CatalogContext,
    CatalogLabourRateContext,
    CatalogWorkGroupContext,
    CustomPricedRow,
    PreviewAssumption,
    PreviewUnmatchedItem,
)
from app.services.matcher import _price_unmatched_items_safely


def _catalog_context() -> CatalogContext:
    return CatalogContext(
        work_groups=[CatalogWorkGroupContext(id=1, name="Kitchen")],
        labour_rates=[CatalogLabourRateContext(name="Electrician", code="ELEC", net_rate=38.0)],
    )


def _unmatched_item() -> PreviewUnmatchedItem:
    return PreviewUnmatchedItem(source_text="Fit new bespoke shelving", reason="no match")


class TestPriceUnmatchedItemsSafely:
    def test_returns_empty_list_with_no_assumption_when_nothing_to_price(self):
        assumptions: list[PreviewAssumption] = []
        result = _price_unmatched_items_safely([], _catalog_context(), assumptions)

        assert result == []
        assert assumptions == []

    def test_returns_priced_rows_with_no_assumption_on_success(self):
        priced_row = CustomPricedRow(
            source_text="Fit new bespoke shelving",
            name="Bespoke shelving installation",
            unit="lm",
            labour_cost=120.0,
            material_cost=80.0,
            other_cost=0.0,
            work_days=1.0,
            qty_for_norm=1.0,
            suggested_work_group_id=1,
            suggested_work_group_name="Kitchen",
            market_justification="Standard joinery rate.",
            price_source_notes="",
            confidence_level=0.8,
        )
        assumptions: list[PreviewAssumption] = []
        with patch("app.services.custom_work_service.batch_price_unmatched", return_value=[priced_row]):
            result = _price_unmatched_items_safely([_unmatched_item()], _catalog_context(), assumptions)

        assert result == [priced_row]
        assert assumptions == []

    def test_appends_warning_assumption_and_returns_empty_list_on_failure(self):
        # Some sibling test modules reload app.services.custom_work_service (to
        # re-check env-driven settings), which mints a fresh exception class —
        # raise via the module's *current* class rather than a class captured
        # at this file's import time, so isinstance still matches regardless
        # of test execution order.
        def _raise_pricing_unavailable(*_args, **_kwargs):
            from app.services.custom_work_service import CustomPricingUnavailable as CurrentException

            raise CurrentException("network unreachable")

        assumptions: list[PreviewAssumption] = []
        with patch(
            "app.services.custom_work_service.batch_price_unmatched",
            side_effect=_raise_pricing_unavailable,
        ):
            result = _price_unmatched_items_safely([_unmatched_item()], _catalog_context(), assumptions)

        assert result == []
        assert len(assumptions) == 1
        assert assumptions[0].kind == "custom_pricing_failed"
        assert assumptions[0].severity == "warning"
        assert "1" in assumptions[0].text
