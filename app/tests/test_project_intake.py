from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _payload(
    message: str = "Client uploaded the kitchen drawings and room schedule.",
) -> dict:
    return {
        "session_id": 1,
        "quote_id": 12,
        "knowledge_version": "demo",
        "current_project_brief_json": {},
        "assets": [
            {
                "asset_id": 101,
                "filename": "drawings.pdf",
                "asset_type": "pdf",
                "checksum": "abc",
                "extraction_status": "completed",
                "document_excerpt": (
                    "PROPERTY: Victorian house\n"
                    "Kitchen: 3.5 x 5.8\n"
                    "Bathroom: 2.2 x 2.8\n"
                    "Existing layout: retain current staircase\n"
                    "Electrical: 12 downlights, 8 sockets\n"
                    "Decorating: paint walls and ceilings\n"
                ),
                "warnings": [],
                "page_count": 2,
                "sheet_count": 0,
                "page_previews": [],
                "sheets": [],
                "table_excerpt": "",
            },
            {
                "asset_id": 102,
                "filename": "room-schedule.xlsx",
                "asset_type": "xlsx",
                "checksum": "def",
                "extraction_status": "completed",
                "document_excerpt": "Room | Length | Width | Area\nKitchen | 3.5 | 5.8 | 20.3",
                "warnings": [],
                "page_count": 0,
                "sheet_count": 1,
                "page_previews": [],
                "sheets": [
                    {
                        "name": "Room Schedule",
                        "row_count_excerpt": 2,
                        "recognized_schedule": True,
                        "rows": [
                            ["Room", "Length", "Width", "Area"],
                            ["Kitchen", "3.5", "5.8", "20.3"],
                        ],
                        "excerpt": "Room | Length | Width | Area\nKitchen | 3.5 | 5.8 | 20.3",
                    }
                ],
                "table_excerpt": "",
            },
        ],
        "turns": [],
        "message": message,
    }


def test_project_intake_chat_requires_valid_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security
    import app.services.project_intake_llm_client as intake_llm_client
    import app.services.project_intake_service as intake_service

    reload(app_config)
    reload(app_security)
    reload(intake_llm_client)
    reload(intake_service)

    client = TestClient(app)
    response = client.post("/v1/project-intake/chat", json=_payload())

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_project_intake_chat_returns_structured_brief_in_fallback_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security
    import app.services.project_intake_llm_client as intake_llm_client
    import app.services.project_intake_service as intake_service

    reload(app_config)
    reload(app_security)
    reload(intake_llm_client)
    reload(intake_service)

    client = TestClient(app)
    response = client.post(
        "/v1/project-intake/chat", json=_payload(), headers={"x-api-key": "test-secret"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]
    assert body["updated_project_brief_json"]["areas_dimensions"]["facts"]
    assert body["updated_project_brief_json"]["mep"]["facts"]
    assert body["model_name"] == "fallback-intake"


def test_project_intake_finalize_returns_structured_scope_markdown(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security
    import app.services.project_intake_llm_client as intake_llm_client
    import app.services.project_intake_service as intake_service

    reload(app_config)
    reload(app_security)
    reload(intake_llm_client)
    reload(intake_service)

    client = TestClient(app)
    response = client.post(
        "/v1/project-intake/finalize",
        json={k: v for k, v in _payload().items() if k != "message"},
        headers={"x-api-key": "test-secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["handoff_readiness"] is True
    assert "PROPERTY:" in body["structured_scope_markdown"]
    assert "FLOOR AREAS & ROOMS:" in body["structured_scope_markdown"]
