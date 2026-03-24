from app.services.calculator import calculate_client_cost_per_unit, calculate_client_total_cost



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
    assert result == 18.45



def test_calculate_client_total_cost() -> None:
    assert calculate_client_total_cost(quantity=120, client_cost_per_unit=18.45) == 2214.0
