from __future__ import annotations

from importlib import reload

from fastapi.testclient import TestClient

from app.main import app
from app.services.calculator_ai import (
    CalculatorAskRequest,
    CalculatorExplainRequest,
    _ask_mock,
    _clean_bullet,
    _explain_mock,
    _parse_mock,
    _sanitise_answer,
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


# ---- ask (guardrails + mock) -------------------------------------------------


def test_sanitise_answer_rejects_price_hours_and_worklists():
    assert _sanitise_answer("Planning is usually straightforward here.") == (
        "Planning is usually straightforward here."
    )
    assert _sanitise_answer("Budget about £2,000 for that.") == ""
    assert _sanitise_answer("Allow 12 hours for the electrician.") == ""
    assert _sanitise_answer("Your project needs:\n1. strip out\n2. new roof") == ""
    assert _sanitise_answer("- demolish\n- rebuild\n- fit out") == ""
    assert _sanitise_answer("   ") == ""


def test_sanitise_answer_caps_length():
    long = " ".join(["word"] * 120)
    out = _sanitise_answer(long)
    assert out.endswith("…")
    assert len(out.split()) <= 71


def test_ask_mock_answers_planning_question():
    r = _ask_mock(
        CalculatorAskRequest(
            question="Do I need planning permission for a rear extension?",
            project_types=["Back extension"],
        )
    )
    assert r.deflected is False
    assert r.service_mode == "mock"
    assert "£" not in r.answer
    assert "combit" in r.answer.lower()


def test_ask_mock_deflects_detailed_quote_request():
    r = _ask_mock(
        CalculatorAskRequest(
            question="Can you give me an itemised breakdown of every cost?"
        )
    )
    assert r.deflected is True
    assert "£" not in r.answer


# ---- ask endpoint (mock mode) ---------------------------------------------------


def test_ask_endpoint_requires_api_key(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/ask", json={"question": "how long does a loft take?"}
    )
    assert resp.status_code == 401


def test_ask_endpoint_mock(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/ask",
        json={
            "question": "How long will a dormer loft take?",
            "project_types": ["Dormer loft conversion"],
            "location": {"location": "Inner London"},
            "options": {},
            "totals": {"total": 78000.0},
        },
        headers={"x-api-key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["deflected"] is False
    assert body["service_mode"] == "mock"
    assert "£" not in body["answer"]


def test_ask_endpoint_deflects_scope_request(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/ask",
        json={"question": "Please write a full scope of works for my project"},
        headers={"x-api-key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deflected"] is True


# ---- fit-check + extra notes (mock parse) --------------------------------------


def test_parse_mock_flags_too_small_job():
    r = _parse_mock("Just need to repaint one room, nothing structural")
    assert r.project_types == []
    assert r.fit == "too_small"
    assert r.fit_message
    assert "£" not in r.fit_message
    # a too-small verdict is itself real signal — must not collapse to "didn't understand"
    assert r.understood is True


def test_parse_mock_matched_type_is_fit():
    r = _parse_mock("Dormer loft conversion, roughly 30 m2")
    assert r.fit == "fit"
    assert r.fit_message == ""


def test_parse_mock_unsure_when_nothing_matches():
    r = _parse_mock("listed building in a conservation area, Inner London")
    assert r.project_types == []
    assert r.fit == "unsure"
    assert r.fit_message == ""


# ---- voice parse (mock mode, no OPENAI_API_KEY) --------------------------------


def test_voice_parse_without_key_is_unavailable(monkeypatch):
    _mock_env(monkeypatch)
    from app.services.calculator_ai import (
        CalculatorVoiceParseRequest,
        voice_parse_project,
    )

    r = voice_parse_project(
        CalculatorVoiceParseRequest(audio_base64="AAAA", mime_type="audio/webm")
    )
    assert r.understood is False
    assert r.transcript == ""
    assert r.service_mode == "mock"


def test_voice_parse_endpoint_requires_api_key(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/voice-parse",
        json={"audio_base64": "AAAA", "mime_type": "audio/webm"},
    )
    assert resp.status_code == 401


def test_voice_parse_endpoint_mock_without_key(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/voice-parse",
        json={"audio_base64": "AAAA", "mime_type": "audio/webm"},
        headers={"x-api-key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["understood"] is False
    assert body["service_mode"] == "mock"


def test_voice_parse_endpoint_rejects_empty_audio(monkeypatch):
    _mock_env(monkeypatch)
    resp = TestClient(app).post(
        "/v1/calculator/voice-parse",
        json={"audio_base64": "", "mime_type": "audio/webm"},
        headers={"x-api-key": "test-secret"},
    )
    assert resp.status_code == 422


def test_voice_parse_stitches_transcript_into_parsed_result(monkeypatch):
    _mock_env(monkeypatch)
    import dataclasses

    import app.services.calculator_ai as calc_ai
    from app.services.calculator_ai import CalculatorVoiceParseRequest

    monkeypatch.setattr(
        calc_ai,
        "settings",
        dataclasses.replace(calc_ai.settings, openai_api_key="sk-test"),
    )
    monkeypatch.setattr(
        calc_ai, "_transcribe_audio", lambda req: "Dormer loft conversion, 30 m2"
    )

    r = calc_ai.voice_parse_project(
        CalculatorVoiceParseRequest(audio_base64="AAAA", mime_type="audio/webm")
    )
    assert r.transcript == "Dormer loft conversion, 30 m2"
    assert "Dormer loft conversion" in r.project_types
    assert r.fit == "fit"


def test_parse_project_real_path_keeps_too_small_as_understood(monkeypatch):
    _mock_env(monkeypatch)
    import dataclasses

    import app.services.calculator_ai as calc_ai
    from app.services.calculator_ai import LLMCalculatorParseOutput

    monkeypatch.setattr(
        calc_ai,
        "settings",
        dataclasses.replace(
            calc_ai.settings, openai_api_key="sk-test", openai_model="gpt-4o"
        ),
    )

    parsed = LLMCalculatorParseOutput(
        understood=True,
        fit="too_small",
        fit_message="Combit mainly takes on larger refurbishment projects.",
    )

    class _Msg:
        def __init__(self, parsed):
            self.parsed = parsed

    class _Choice:
        def __init__(self, parsed):
            self.message = _Msg(parsed)

    class _Completion:
        def __init__(self, parsed):
            self.choices = [_Choice(parsed)]

    class _Completions:
        def parse(self, **kwargs):
            return _Completion(parsed)

    class _Chat:
        completions = _Completions()

    class _Beta:
        chat = _Chat()

    class _FakeClient:
        beta = _Beta()

    monkeypatch.setattr(calc_ai, "_openai_client", lambda: _FakeClient())

    r = calc_ai.parse_project("Just want to repaint one room, nothing structural")
    assert r.service_mode == "real"
    assert r.understood is True
    assert r.fit == "too_small"
    assert r.fit_message
    assert r.project_types == []
