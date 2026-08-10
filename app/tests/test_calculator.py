from app.services.calculator import (
    calculate_client_cost_per_unit,
    calculate_client_total_cost,
    resolve_client_cost_per_unit,
)


def test_calculate_client_cost_per_unit() -> None:
    result = calculate_client_cost_per_unit(
        work_labour_cost=10,
        work_mat_cost=4,
        work_other_cost=1,
        labour_markup=15,
        material_markup=10,
        profit=20,
        work_qty_for_norm=1,
    )
    assert result == 20.4


def test_calculate_client_total_cost() -> None:
    assert (
        calculate_client_total_cost(quantity=120, client_cost_per_unit=20.4) == 2448.0
    )


def test_resolve_client_cost_per_unit_uses_fixed_rate_when_flagged() -> None:
    # A workbook-rate catalog item: labour/material breakdown is reference-only,
    # the client-facing rate is the fixed catalog total, not the markup formula.
    result = resolve_client_cost_per_unit(
        is_fixed_rate=True,
        fixed_client_rate=76.53,
        work_labour_cost=521,
        work_mat_cost=91.27,
        work_other_cost=0,
        labour_markup=15,
        material_markup=10,
        profit=20,
        work_qty_for_norm=1,
    )
    assert result == 76.53


def test_resolve_client_cost_per_unit_falls_back_to_formula_when_not_fixed() -> None:
    result = resolve_client_cost_per_unit(
        is_fixed_rate=False,
        fixed_client_rate=None,
        work_labour_cost=10,
        work_mat_cost=4,
        work_other_cost=1,
        labour_markup=15,
        material_markup=10,
        profit=20,
        work_qty_for_norm=1,
    )
    assert result == 20.4


def test_resolve_client_cost_per_unit_falls_back_when_flagged_but_rate_missing() -> (
    None
):
    result = resolve_client_cost_per_unit(
        is_fixed_rate=True,
        fixed_client_rate=None,
        work_labour_cost=10,
        work_mat_cost=4,
        work_other_cost=1,
        labour_markup=15,
        material_markup=10,
        profit=20,
        work_qty_for_norm=1,
    )
    assert result == 20.4
