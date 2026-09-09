"""Seam 1 — POST /v1/predictions, the contract smoke check.

Asserts externally observable behaviour and structural invariants: research
evidence carries a live source URL, no number in a model signal is absent from
the deterministic layer, an all-inside-the-noise-floor submission is not sold as
a verdict, a run with no screenplay or a degraded run returns no verdict at all,
concept mode returns the same schema with its craft slots not complete and no
craft residual narrated, and every degradation reason the run recorded reaches
both the summary and the failed panels. Nothing here asserts prose or call order.

Both LLM subagents are recorded fixtures. LIVE_AGENTS=1 adds a live run.
"""
import base64
import json
import os
import re
import unittest

try:
    import pytest
except ImportError:
    pytest = None

if pytest is None:
    raise unittest.SkipTest("pytest is required to run test_api (run with pytest)")

from fastapi.testclient import TestClient

import api
from api import PredictionRequest, app
from src.agents import response_mapping as M
from tests.stub_agents import (
    StubResearchAgent,
    StubSynthesisAgent,
    stub_orchestrator,
)

client = TestClient(app)

PAYLOAD = {
    "story": "A disgraced Stockholm detective is locked in a night-long interrogation with the "
             "one suspect who knows what happened to her missing daughter.",
    "medium": "Feature film",
    "targetGeography": "Nordics",
    "classifications": {"genre": "Thriller", "title": "The Long Night"},
    "materials": [],
}


def _movie_script(name):
    return (Path("data/movies") / name / "script.txt").read_text(encoding="utf-8", errors="replace")


# a real screenplay, so it clears the validation gate in run_engine
SCRIPT_TEXT = _movie_script("fight_club")
# alien/script.txt is a plot synopsis, not a screenplay: the gate must reject it
NOT_A_SCREENPLAY = _movie_script("alien")
# 1x1 transparent PNG
PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
           "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def _data_uri(mime, raw_b64):
    return f"data:{mime};base64,{raw_b64}"


def _script_material():
    return {"kind": "script", "mimeType": "text/plain",
            "uri": _data_uri("text/plain", base64.b64encode(SCRIPT_TEXT.encode()).decode())}


@pytest.fixture
def stubbed(monkeypatch):
    """Swap the engine's subagents for fixtures; the deterministic half runs for real."""
    research = StubResearchAgent()
    synthesis = StubSynthesisAgent()
    monkeypatch.setattr(api._agent, "orchestrator", stub_orchestrator(research, synthesis))
    return research, synthesis


# ---------------------------------------------------------------- contract shape

def test_prediction(stubbed):
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
    """trainerStatus is unchanged by phase 2."""
    r = client.get("/v1/diagnostics/model")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["trainerStatus"] in ("healthy", "running", "degraded", "failed")
    assert b["activeModel"]


# ---------------------------------------------------------------- provenance

def test_every_research_evidence_carries_a_source_url(stubbed):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [_script_material()]})
    assert r.status_code == 200, r.text
    found = [e for e in r.json()["evidence"] if e["sourceType"] == "parallel_search"]
    assert found
    assert all(e["sourceUrl"] for e in found)


def test_research_slots_only_list_urls_that_were_retrieved(stubbed):
    research, _ = stubbed
    r = client.post("/v1/predictions", json=PAYLOAD)
    retrieved = set(research.run("x")["retrieved_urls"])
    for agent in r.json()["agents"]:
        assert set(agent["sources"]) <= retrieved


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def test_no_model_signal_invents_a_number(stubbed):
    """Every figure in a model signal must be present in the deterministic report."""
    report = api._agent.run_premortem(
        story=PAYLOAD["story"], script_text=SCRIPT_TEXT,
        keyframes_dir="data/movies/fight_club/keyframes", genre="Drama", title="Fight Club",
    )
    response = api._to_response(report, PredictionRequest(**PAYLOAD))
    deterministic = json.dumps(report, default=str).replace(",", "")

    for evidence in response.evidence:
        if evidence.sourceType != "model_signal":
            continue
        for token in _NUMBER.findall(evidence.statement):
            assert (token in deterministic
                    or token.rstrip("0").rstrip(".") in deterministic), \
                f"{token!r} in {evidence.statement!r} is absent from the deterministic layer"


def test_noise_floor_rows_are_labelled_and_lead_the_model_signals(stubbed):
    report = api._agent.run_premortem(
        story=PAYLOAD["story"], script_text=SCRIPT_TEXT,
        keyframes_dir="data/movies/fight_club/keyframes", genre="Drama", title="Fight Club",
    )
    signals = M.model_signals(report)
    inside = [s for s in signals if "inside the noise floor" in s[1]]
    assert inside, signals
    # the studio renders four cards, so the sub-threshold rows have to be in them
    assert "inside the noise floor" in signals[0][1]


# ---------------------------------------------------------------- outcome honesty

def test_all_inside_the_noise_floor_is_not_sold_as_a_verdict(stubbed):
    """A sweep entirely inside the error bar must not read as a confident call."""
    report = api._agent.run_premortem(
        story=PAYLOAD["story"], script_text=SCRIPT_TEXT,
        keyframes_dir="data/movies/fight_club/keyframes", genre="Drama", title="Fight Club",
    )
    if not all(row["inside_noise_floor"] for row in report["sweep"]):
        # a retrained model may put an effect outside the bar; the invariant is
        # about what happens when none of them do
        pytest.skip("this submission has a counterfactual outside the error bar")
    response = api._to_response(report, PredictionRequest(**PAYLOAD))
    assert (response.outcome == "inconclusive"
            or "inside the noise floor" in response.summary
            or "noise" in response.summary.lower()), response.summary


def test_outcome_is_inconclusive_across_a_straddled_boundary():
    """cv_mae widens the band, so 68 and 72 are not opposite verdicts."""
    assert M.outcome_for(68, 0.441) == "inconclusive"
    assert M.outcome_for(72, 0.441) == "inconclusive"
    assert M.outcome_for(90, 0.441) == "hit"
    assert M.outcome_for(20, 0.441) == "miss"


def test_a_verdict_needs_something_behind_it():
    """No screenplay and no evidence is not a `hit`, wherever the score lands."""
    assert M.outcome_for(90, 0.441) == "hit"
    assert M.outcome_for(90, 0.441, has_screenplay=False) == "inconclusive"
    assert M.outcome_for(90, 0.441, degraded=True) == "inconclusive"
    assert M.outcome_for(20, 0.441, has_screenplay=False) == "inconclusive"


def test_concept_mode_has_no_verdict_and_says_what_the_score_is(stubbed):
    """The observed defect: four failed slots, `outcome: hit`, `score: 80`."""
    b = client.post("/v1/predictions", json=PAYLOAD).json()
    assert b["outcome"] == "inconclusive"
    assert b["score"] > 0  # the genre prior is still worth reporting
    assert "prior" in b["summary"].lower()
    assert "no screenplay" in b["summary"].lower()


def test_a_degraded_run_has_no_verdict(monkeypatch):
    monkeypatch.setattr(api._agent, "orchestrator", stub_orchestrator(
        StubResearchAgent(claims=[], degraded=True, reasons=["no GEMINI_API_KEY"]),
        StubSynthesisAgent(),
    ))
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [_script_material()]})
    assert r.status_code == 200, r.text
    assert r.json()["outcome"] == "inconclusive"


def test_confidence_ignores_distance_to_the_threshold():
    """Same score, more evidence and a better model: confidence must move."""
    thin = M.confidence_for(0, 0.441, True, False)
    thick = M.confidence_for(6, 0.441, True, False)
    accurate = M.confidence_for(0, 0.10, True, False)
    assert thick > thin
    assert accurate > thin
    assert M.confidence_for(6, 0.441, True, True) < thick


# ---------------------------------------------------------------- concept mode

def test_concept_mode_returns_the_same_schema_with_craft_slots_not_complete(stubbed):
    r = client.post("/v1/predictions", json=PAYLOAD)  # no materials
    assert r.status_code == 200, r.text
    b = r.json()
    assert {a["agentId"] for a in b["agents"]} == {"story", "audience", "market", "visual"}
    status = {a["agentId"]: a["status"] for a in b["agents"]}
    assert status["story"] != "complete"
    assert status["visual"] != "complete"
    story = next(a for a in b["agents"] if a["agentId"] == "story")
    assert "screenplay" in story["finding"].lower()


def test_concept_mode_narrates_no_craft_residual_or_counterfactual(stubbed):
    """With no screenplay the features are corpus medians, so there is nothing to sweep."""
    b = client.post("/v1/predictions", json=PAYLOAD).json()
    signals = [e for e in b["evidence"] if e["sourceType"] == "model_signal"]
    assert not [e for e in signals if e["title"] == "Counterfactual"], signals
    statement = next(e["statement"] for e in signals if e["title"] == "Model signal")
    assert "no screenplay" in statement.lower(), statement
    assert "residual" not in " ".join(e["statement"] for e in signals).lower()


def test_concept_mode_skips_the_sweep_and_the_vulnerabilities(stubbed):
    report = api._agent.run_premortem(story=PAYLOAD["story"], title="The Long Night")
    assert report["sweep"] == []
    assert report["risk_flags"] == []


def test_script_mode_completes_the_craft_slots(stubbed):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        _script_material(),
        {"kind": "frame", "uri": _data_uri("image/png", PNG_B64), "mimeType": "image/png"}]})
    assert r.status_code == 200, r.text
    status = {a["agentId"]: a["status"] for a in r.json()["agents"]}
    assert status["story"] == "complete"
    assert status["visual"] == "complete"


# ---------------------------------------------------------------- loud mock

def test_a_mocked_run_is_labelled_degraded(monkeypatch):
    monkeypatch.setattr(api._agent, "orchestrator", stub_orchestrator(
        StubResearchAgent(degraded=True, reasons=["search results are mock output"]),
        StubSynthesisAgent(),
    ))
    r = client.post("/v1/predictions", json=PAYLOAD)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["summary"].startswith("DEGRADED RUN")
    assert "mock" in b["summary"]


def test_a_failed_synthesis_call_degrades_rather_than_500s(monkeypatch):
    monkeypatch.setattr(api._agent, "orchestrator", stub_orchestrator(
        StubResearchAgent(),
        StubSynthesisAgent(report=None, degraded=True,
                           reasons=["synthesis agent call failed: 429"]),
    ))
    r = client.post("/v1/predictions", json=PAYLOAD)
    assert r.status_code == 200, r.text
    assert "DEGRADED RUN" in r.json()["summary"]


def test_research_failure_does_not_complete_the_research_slots(monkeypatch):
    monkeypatch.setattr(api._agent, "orchestrator", stub_orchestrator(
        StubResearchAgent(claims=[], degraded=True, reasons=["no GEMINI_API_KEY"]),
        StubSynthesisAgent(),
    ))
    b = client.post("/v1/predictions", json=PAYLOAD).json()
    status = {a["agentId"]: a["status"] for a in b["agents"]}
    assert status["audience"] == "failed"
    assert status["market"] == "failed"


def test_every_degradation_reason_reaches_the_summary_and_the_failed_slots(monkeypatch):
    """A reason the run recorded but the producer never sees is a silent failure."""
    reasons = [
        "search results are mock output, not live retrieval",
        "research agent call failed: ServerError: 503 UNAVAILABLE",
        "4 of 4 claim(s) dropped by the grounding filter; none survived",
    ]
    monkeypatch.setattr(api._agent, "orchestrator", stub_orchestrator(
        StubResearchAgent(claims=[], degraded=True, reasons=reasons),
        StubSynthesisAgent(),
    ))
    b = client.post("/v1/predictions", json=PAYLOAD).json()
    assert b["summary"].startswith("DEGRADED RUN")
    for reason in reasons:
        assert reason in b["summary"], reason
    for slot in ("audience", "market"):
        agent = next(a for a in b["agents"] if a["agentId"] == slot)
        assert agent["status"] == "failed"
        for reason in reasons:
            assert reason in agent["finding"], (slot, reason)


# ---------------------------------------------------------------- materials

def test_script_data_uri_reaches_the_engine(stubbed):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [_script_material()]})
    assert r.status_code == 200, r.text
    assert r.json()["agents"][0]["status"] == "complete"


def test_frame_data_uri_lands_in_the_visual_slot(stubbed):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "frame", "uri": _data_uri("image/png", PNG_B64), "mimeType": "image/png"}]})
    assert r.status_code == 200, r.text
    visual = next(a for a in r.json()["agents"] if a["agentId"] == "visual")
    assert visual["status"] == "complete"
    assert "1 frames measured" in visual["finding"]


def test_oversized_data_uri_is_413(stubbed, monkeypatch):
    # ponytail: shrink the cap instead of posting a real 20 MB payload
    monkeypatch.setattr(api, "MAX_MATERIAL_BYTES", 16)
    b64 = base64.b64encode(b"x" * 64).decode()
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": _data_uri("text/plain", b64), "mimeType": "text/plain"}]})
    assert r.status_code == 413, r.text


def test_malformed_data_uri_degrades_to_concept_mode(stubbed):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": "data:text/plain;base64,!!!not base64!!!",
         "mimeType": "text/plain"}]})
    assert r.status_code == 200, r.text
    status = {a["agentId"]: a["status"] for a in r.json()["agents"]}
    assert status["story"] == "failed"


def test_unreachable_material_degrades_rather_than_failing(stubbed):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": "https://127.0.0.1:1/nope.txt", "mimeType": "text/plain"}]})
    assert r.status_code == 200, r.text


def test_rejects_unknown_uri_scheme():
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": "ftp://example.com/s.txt", "mimeType": "text/plain"}]})
    assert r.status_code == 400, r.text


# ---------------------------------------------------------------- validation gate

def test_non_screenplay_script_is_rejected(stubbed):
    b64 = base64.b64encode(NOT_A_SCREENPLAY.encode()).decode()
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [
        {"kind": "script", "uri": _data_uri("text/plain", b64), "mimeType": "text/plain"}]})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "not_a_screenplay"
    assert detail["reasons"]


def test_real_screenplay_passes_the_gate(stubbed):
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [_script_material()]})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------- live

@pytest.mark.skipif(os.environ.get("LIVE_AGENTS") != "1", reason="set LIVE_AGENTS=1")
def test_live_prediction():
    r = client.post("/v1/predictions", json={**PAYLOAD, "materials": [_script_material()]})
    assert r.status_code == 200, r.text
    b = r.json()
    for e in b["evidence"]:
        if e["sourceType"] == "parallel_search":
            assert e["sourceUrl"]
