from __future__ import annotations

from importlib import reload

from fastapi.testclient import TestClient

from app.main import app
from app.services.calculator_ai import (
    CalculatorExplainRequest,
    _clean_bullet,
    _explain_mock,
    _parse_mock,
    _sanitise_explain,
)


def _mock_env(monkeypatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    import app.config as app_config
    import app.security as app_security
    import app.services.calculator_ai as calc_ai

    reload(app_config)
    reload(app_security)
    reload(calc_ai)


# ---- guardrails -------------------------------------------------------------


def test_clean_bullet_strips_money_and_hours():
    assert _clean_bullet("Structure and first fix.") == "Structure and first fix."
    assert _clean_bullet("Plumbing runs £120 per point") is None
    assert _clean_bullet("Allow 8 hrs for the electrician") is None
    assert _clean_bullet("   ") is None


def test_sanitise_explain_dedupes_and_caps():
    got = _sanitise_explain(["a", "a", "b £5", "c", "d", "e", "f", "g"], limit=3)
    assert got == ["a", "c"] or got == ["a", "c", "d"]
    assert len(got) <= 3


# ---- mock parse ----------------------------------------------------------------


def test_parse_mock_recognises_types_and_options():
    r = _parse_mock(
        "Rear extension about 25m2, new kitchen and underfloor heating, conservation area"
    )
    assert "Back extension" in r.project_types
    assert r.options.get("is_kitchen_fitting") is True
    assert r.options.get("is_ufh_water") is True
    assert r.location.get("area") == "Conservation"
    assert r.understood is True
    assert r.service_mode == "mock"


def test_parse_mock_off_topic_marks_not_understood():
    r = _parse_mock("hello, what's the weather today?")
    assert r.project_types == []
    assert r.understood is False
    assert r.summary == ""
    assert r.uncertain == []


def test_parse_mock_location_only_is_understood():
    r = _parse_mock("listed building in a conservation area, Inner London")
    assert r.project_types == []
    assert r.understood is True


# ---- mock explain ----------------------------------------------------------------


def test_explain_mock_shape_and_guardrails():
    req = CalculatorExplainRequest(
        project_types=["Back extension"],
        location={"location": "Inner London", "listed_building": "Yes"},
        options={"is_live_in_during_the_project": True},
        totals={"total": 95000.0},
        floor_areas={"back_extension_area": 25.0},
    )
    r = _explain_mock(req)
    assert 3 <= len(r.included) <= 6
    assert r.excluded
    assert r.cost_drivers
    assert r.disclaimer
    blob = " ".join(r.included + r.excluded + r.cost_drivers)
    assert "£" not in blob
    assert "hrs" not in blob
    # driver context reflected
    assert any("listed" in d.lower() for d in r.cost_drivers)


# ---- endpoints (mock mode) --------------------------------------------------------


def test_parse_endpoint_requires_api_key(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/parse-project", json={"text": "loft conversion"}
    )
    assert resp.status_code == 401


def test_parse_endpoint_mock(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/parse-project",
        json={"text": "Dormer loft conversion plus a new bathroom, listed building"},
        headers={"x-api-key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "Dormer loft conversion" in body["project_types"]
    assert body["options"].get("is_bathrooms") is True
    assert body["location"].get("listed_building") == "Yes"
    assert body["understood"] is True
    assert body["service_mode"] == "mock"


def test_parse_endpoint_off_topic(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/parse-project",
        json={"text": "can you tell me a joke"},
        headers={"x-api-key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["understood"] is False
    assert body["project_types"] == []


def test_explain_endpoint_mock(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/explain-estimate",
        json={
            "project_types": ["House refurbishment"],
            "location": {"location": "Outer London"},
            "options": {},
            "totals": {"total": 120000.0},
            "floor_areas": {"hr_ground_floor_area": 90.0},
        },
        headers={"x-api-key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["included"] and body["excluded"] and body["disclaimer"]
    assert body["service_mode"] == "mock"
    assert "£" not in " ".join(
        body["included"] + body["excluded"] + body["cost_drivers"]
    )
