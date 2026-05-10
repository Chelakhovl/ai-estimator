from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _payload() -> dict:
    return {
        "quote_guid": "quote-1",
        "prompt": "Kitchen renovation, paint walls 120 m2",
        "candidate_rows": [
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
    }


def test_preview_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security
    import app.services.llm_client as app_llm_client
    import app.services.matcher as app_matcher

    reload(app_config)
    reload(app_security)
    reload(app_llm_client)
    reload(app_matcher)

    client = TestClient(app)
    response = client.post("/v1/estimate/preview", json=_payload())

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_preview_accepts_valid_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security
    import app.services.llm_client as app_llm_client
    import app.services.matcher as app_matcher

    reload(app_config)
    reload(app_security)
    reload(app_llm_client)
    reload(app_matcher)

    client = TestClient(app)
    response = client.post(
        "/v1/estimate/preview",
        json=_payload(),
        headers={"x-api-key": "test-secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["error_text"] == ""
    assert body["service_mode"] == "mock"
    assert len(body["matched_rows"]) == 1


def test_prompt_templates_requires_valid_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    from importlib import reload
    import app.config as app_config
    import app.security as app_security
    import app.services.llm_client as app_llm_client
    import app.services.matcher as app_matcher

    reload(app_config)
    reload(app_security)
    reload(app_llm_client)
    reload(app_matcher)

    client = TestClient(app)
    response = client.get("/v1/estimate/prompt-templates", headers={"x-api-key": "test-secret"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["templates"]) >= 1
    assert body["templates"][0]["prompt"]
