from __future__ import annotations


def calculate_client_cost_per_unit(
    work_labour_cost: float,
    work_mat_cost: float,
    work_other_cost: float,
    labour_markup: float,
    material_markup: float,
    profit: float,
    work_qty_for_norm: float,
) -> float:
    if work_qty_for_norm <= 0:
        raise ValueError("work_qty_for_norm must be greater than zero")

    return round(
        (
            (
                work_labour_cost * (1 + labour_markup / 100)
                + (work_mat_cost + work_other_cost) * (1 + material_markup / 100)
            )
            * (1 + profit / 100)
        )
        / work_qty_for_norm,
        2,
    )


def calculate_client_total_cost(quantity: float, client_cost_per_unit: float) -> float:
    if quantity <= 0:
        raise ValueError("quantity must be greater than zero")
    return round(quantity * client_cost_per_unit, 2)


def resolve_client_cost_per_unit(
    *,
    is_fixed_rate: bool,
    fixed_client_rate: float | None,
    work_labour_cost: float,
    work_mat_cost: float,
    work_other_cost: float,
    labour_markup: float,
    material_markup: float,
    profit: float,
    work_qty_for_norm: float,
) -> float:
    # Fixed-rate rows (workbook rate / custom rate override) bypass the labour+material
    # markup/profit formula entirely, mirroring the Django backend's
    # resolve_quote_item_pricing. Skipping this branch inflates the preview total for
    # any master-estimator catalog item priced this way (the majority of the catalog).
    if is_fixed_rate and fixed_client_rate is not None:
        return round(fixed_client_rate, 2)
    return calculate_client_cost_per_unit(
        work_labour_cost=work_labour_cost,
        work_mat_cost=work_mat_cost,
        work_other_cost=work_other_cost,
        labour_markup=labour_markup,
        material_markup=material_markup,
        profit=profit,
        work_qty_for_norm=work_qty_for_norm,
    )
