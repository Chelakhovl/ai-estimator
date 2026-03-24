from app.schemas import PreviewRequest
from app.services.matcher import generate_preview



def test_preview_contract_returns_expected_shape() -> None:
    request = PreviewRequest(
        quote_guid="quote-1",
        prompt="Kitchen renovation, paint walls 120 m2",
        candidate_rows=[
            {
                "INSIDEQUOTESGUID": "inside-1",
                "WORKGUID": "work-1",
                "WorkName": "Paint walls",
                "Unit": "m2",
                "WorkLabourCost": 10,
                "WorkMatCost": 4,
                "WorkOtherCost": 1,
                "WorkQTYforNorm": 1,
                "PROFIT": 20,
                "LabourMarkup": 15,
                "MaterialMarkup": 10,
            }
        ],
    )

    response = generate_preview(request)

    assert response.error_text == ""
    assert len(response.matched_rows) == 1
    assert response.matched_rows[0].WorkName == "Paint walls"
    assert response.matched_rows[0].QUANTITY == 120
