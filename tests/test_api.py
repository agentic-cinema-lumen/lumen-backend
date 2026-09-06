"""Contract smoke check: both endpoints answer in the shapes the frontend expects."""
import base64
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
from api import app

client = TestClient(app)

PAYLOAD = {
    "story": "A disgraced Stockholm detective is locked in a night-long interrogation with the "
             "one suspect who knows what happened to her missing daughter.",
    "medium": "Feature film",
    "targetGeography": "Nordics",
    "classifications": {"genre": "Thriller", "title": "The Long Night"},
    "materials": [],
}


def test_prediction():
    r = client.post("/v1/predictions", json=PAYLOAD)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["outcome"] in ("hit", "miss", "inconclusive")
    assert 0 <= b["score"] <= 100
    assert 0.0 <= b["confidence"] <= 1.0
    assert {a["agentId"] for a in b["agents"]} == {"story", "audience", "market", "visual"}
    assert b["evidence"]


def test_rejects_short_story():
    """Contract says 400 "Invalid submission", not FastAPI's default 422."""
    r = client.post("/v1/predictions", json={**PAYLOAD, "story": "too short"})
    assert r.status_code == 400, r.text
    assert r.json()["detail"]


def test_diagnostics():
    r = client.get("/v1/diagnostics/model")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["trainerStatus"] in ("healthy", "running", "degraded", "failed")
    assert b["activeModel"]


# ------------------------------------------------------- data: URI materials

def _movie_script(name):
    return (Path("data/movies") / name / "script.txt").read_text(encoding="utf-8", errors="replace")


# a real screenplay, so it clears the validation gate in run_engine
SCRIPT_TEXT = _movie_script("fight_club")
# alien/script.txt is a plot synopsis, not a screenplay: the gate must reject it
NOT_A_SCREENPLAY = _movie_script("alien")
# 1x1 transparent PNG
PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
           "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

SCRIPT_REPORT = {
    "mode": "script_premortem",
    "projected_baseline_rating": 7.4,
    "historical_show_mean": 6.6,
    "craft_metrics": {"estimated_runtime_min": 112, "total_scenes": 84,
                      "words_per_minute": 91, "climax_acceleration": 1.2},
    "vision_summary": "Low-key, high contrast.",
    "audience_trope_intelligence": {"audience_consensus_claims": [], "queries_executed": []},
    "detected_craft_flaws": [],
    "recommendations": ["Tighten act two."],
}
PREMISE_REPORT = {
    "mode": "premise_premortem",
    "projected_rating_potential": {"median_target": 7.0, "floor": 6.2, "ceiling": 7.8},
    "empirical_genre_baseline": 6.6,
    "genre": "Thriller",
    "detected_tone": "bleak",
    "style_premise_cohesion": {"rating": "STRONG", "evaluation": "Aligned."},
    "audience_fatigue_radar": {"audience_consensus_claims": [], "queries_executed": []},
    "greenlight_verdict": "Proceed with care.",
    "make_or_break_dependencies": ["Cast the detective well."],
}


@pytest.fixture
def engine(monkeypatch):
    """Replace the engine so no LLM or live search runs; record what reached it."""
    seen = {}

    def script(script_path_or_text, keyframes_path_or_dir=None, title=None, **kw):
        seen["script"] = script_path_or_text
        seen["keyframes"] = keyframes_path_or_dir
        seen["frames"] = sorted(os.listdir(keyframes_path_or_dir)) if keyframes_path_or_dir else []
        return SCRIPT_REPORT

    def premise(logline, keyframes_path_or_dir=None, title=None, genre=None, **kw):
        seen["logline"] = logline
        return PREMISE_REPORT

    monkeypatch.setattr(api._agent, "run_script_premortem", script)
    monkeypatch.setattr(api._agent, "run_premise_premortem", premise)
    return seen


def _data_uri(mime, raw_b64):
    return f"data:{mime};base64,{raw_b64}"


def test_script_data_uri_reaches_engine(engine):
    b64 = base64.b64encode(SCRIPT_TEXT.encode()).decode()
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": _data_uri("text/plain", b64), "mimeType": "text/plain"}]})
    assert r.status_code == 200, r.text
    assert engine["script"] == SCRIPT_TEXT


def test_frame_data_uri_lands_in_keyframes_dir(engine):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": _data_uri("text/plain",
                                            base64.b64encode(SCRIPT_TEXT.encode()).decode()),
         "mimeType": "text/plain"},
        {"kind": "frame", "uri": _data_uri("image/png", PNG_B64), "mimeType": "image/png"}]})
    assert r.status_code == 200, r.text
    assert any(f.endswith(".png") for f in engine["frames"]), engine["frames"]


def test_oversized_data_uri_is_413(engine, monkeypatch):
    # ponytail: shrink the cap instead of posting a real 20 MB payload
    monkeypatch.setattr(api, "MAX_MATERIAL_BYTES", 16)
    b64 = base64.b64encode(b"x" * 64).decode()
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": _data_uri("text/plain", b64), "mimeType": "text/plain"}]})
    assert r.status_code == 413, r.text


def test_malformed_data_uri_degrades(engine):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": "data:text/plain;base64,!!!not base64!!!",
         "mimeType": "text/plain"}]})
    assert r.status_code == 200, r.text
    assert "logline" in engine  # fell through to premise mode


def test_rejects_unknown_uri_scheme():
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": "ftp://example.com/s.txt", "mimeType": "text/plain"}]})
    assert r.status_code == 400, r.text


if __name__ == "__main__":
    test_prediction(); test_rejects_short_story(); test_diagnostics()
    print("ok")


# ------------------------------------------------------- screenplay validation gate

def test_non_screenplay_script_is_rejected(engine):
    b64 = base64.b64encode(NOT_A_SCREENPLAY.encode()).decode()
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": _data_uri("text/plain", b64), "mimeType": "text/plain"}]})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "not_a_screenplay"
    assert detail["reasons"]
    assert "script" not in engine  # gate ran before the engine


def test_real_screenplay_passes_the_gate(engine):
    b64 = base64.b64encode(SCRIPT_TEXT.encode()).decode()
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": _data_uri("text/plain", b64), "mimeType": "text/plain"}]})
    assert r.status_code == 200, r.text
    assert engine["script"] == SCRIPT_TEXT
