"""Contract smoke check: both endpoints answer in the shapes the frontend expects."""
from fastapi.testclient import TestClient

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
    assert client.post("/v1/predictions", json={**PAYLOAD, "story": "too short"}).status_code == 422


def test_diagnostics():
    r = client.get("/v1/diagnostics/model")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["trainerStatus"] in ("healthy", "running", "degraded", "failed")
    assert b["activeModel"]


if __name__ == "__main__":
    test_prediction(); test_rejects_short_story(); test_diagnostics()
    print("ok")
