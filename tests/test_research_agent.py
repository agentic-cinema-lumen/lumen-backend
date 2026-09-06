"""Seam 4 — ResearchAgent structured output.

Asserts the invariants that must hold for *any* valid agent output: a claim
traces to a span in the text actually retrieved for the URL it cites, claims
without a retrieved source are dropped, an empty search yields no claims, and
the research loop is bounded. Nothing here asserts prose, query strings, or
call order.

Runs against recorded fixtures by default. LIVE_AGENTS=1 adds a live Gemini run.
"""

import json
import os
from pathlib import Path

import pytest

from src.agents.research_agent import (
    RETRIEVAL_MODES,
    ResearchAgent,
    ResearchOutput,
    filter_claims,
)
from src.search.parallel_search_client import ParallelSearchClient

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH_RESULTS = json.loads((FIXTURES / "search_results.json").read_text())
AGENT_OUTPUT = json.loads((FIXTURES / "research_agent_output.json").read_text())


def _retrieved():
    """url -> retrieved text, as the tool wrapper records it."""
    return {r["url"]: r["text"] for r in SEARCH_RESULTS["results"]}


# ------------------------------------------------------------------ post-filter

def test_recorded_output_survives_the_filter():
    """The recorded fixture is a valid output: every claim is grounded."""
    kept, dropped = filter_claims(ResearchOutput(**AGENT_OUTPUT).claims, _retrieved())
    assert kept, dropped
    assert not dropped


def test_claim_without_a_retrieved_source_is_dropped():
    claims = ResearchOutput(**AGENT_OUTPUT).claims
    claims[0].source_url = "https://invented.example.com/never-retrieved"
    kept, dropped = filter_claims(claims, _retrieved())
    assert claims[0] not in kept
    assert any("not among retrieved" in d for d in dropped)


def test_evidence_must_be_a_substring_of_the_retrieved_text():
    claims = ResearchOutput(**AGENT_OUTPUT).claims
    claims[0].evidence = "the DP said the darkness was a mistake and apologised"
    kept, dropped = filter_claims(claims, _retrieved())
    assert claims[0] not in kept
    assert any("not a span" in d for d in dropped)


def test_evidence_span_tolerates_whitespace_reflow():
    """An LLM re-wrapping a quoted span is grounding, not fabrication."""
    claims = ResearchOutput(**AGENT_OUTPUT).claims
    original = claims[0].evidence
    claims[0].evidence = "\n   ".join(original.split())
    kept, dropped = filter_claims(claims, _retrieved())
    assert claims[0] in kept, dropped


def test_every_surviving_claim_carries_a_source_url():
    kept, _ = filter_claims(ResearchOutput(**AGENT_OUTPUT).claims, _retrieved())
    assert all(c.source_url for c in kept)


# ------------------------------------------------------------------ agent run

class _StubSearch:
    """A search client with no results at all."""

    def __init__(self, results=None):
        self.results = results if results is not None else []
        self.force_mock = False

    def search(self, query, num_results=5):
        return {"query": query, "results": self.results, "_source": "parallel_api"}


def _agent(search_client, invoke):
    a = ResearchAgent(search_client=search_client)
    a._invoke = invoke  # ponytail: swap the ADK call, keep the rest of the pipeline
    return a


def test_empty_search_yields_zero_claims():
    """No retrieved text means nothing can be grounded, so nothing survives."""
    def invoke(prompt, tool):
        tool("anything at all")
        return [ResearchOutput(**AGENT_OUTPUT)]

    out = _agent(_StubSearch([]), invoke).run(
        premise="A detective interrogates a suspect all night.", risk_flags=[]
    )
    assert out["claims"] == []
    assert out["retrieved_urls"] == []


def test_grounded_claims_survive_a_real_retrieval():
    def invoke(prompt, tool):
        tool("dark cinematography backlash")
        return [ResearchOutput(**AGENT_OUTPUT)]

    out = _agent(_StubSearch(SEARCH_RESULTS["results"]), invoke).run(
        premise="A detective interrogates a suspect all night.", risk_flags=[]
    )
    assert out["claims"]
    assert all(c["source_url"] in out["retrieved_urls"] for c in out["claims"])
    assert out["queries_executed"] == ["dark cinematography backlash"]


def test_genre_is_inferred_when_none_is_given():
    def invoke(prompt, tool):
        tool("q")
        return [ResearchOutput(**AGENT_OUTPUT)]

    out = _agent(_StubSearch(SEARCH_RESULTS["results"]), invoke).run(
        premise="A detective interrogates a suspect all night.", risk_flags=[]
    )
    assert out["inferred_genre"]


def test_given_genre_is_not_overridden():
    def invoke(prompt, tool):
        return [ResearchOutput(**AGENT_OUTPUT)]

    out = _agent(_StubSearch([]), invoke).run(
        premise="x", genre="Nordic Noir", risk_flags=[]
    )
    assert out["inferred_genre"] == "Nordic Noir"


def test_mocked_search_is_reported_as_degraded():
    client = ParallelSearchClient(force_mock=True)

    def invoke(prompt, tool):
        tool("dark cinematography")
        return [ResearchOutput(**AGENT_OUTPUT)]

    out = _agent(client, invoke).run(premise="x", risk_flags=[])
    assert out["degraded"]
    assert any("mock" in r.lower() for r in out["degradation_reasons"])


def test_a_cached_mock_still_reports_as_mocked():
    """Replaying the cache must not launder canned output into a clean run."""
    class _CachedMock(_StubSearch):
        def search(self, query, num_results=5):
            return {"query": query, "results": SEARCH_RESULTS["results"],
                    "_source": "disk_cache", "_mock": True}

    def invoke(prompt, tool):
        tool("q")
        return [ResearchOutput(**AGENT_OUTPUT)]

    out = _agent(_CachedMock(), invoke).run(premise="x", risk_flags=[])
    assert out["degraded"]
    assert any("mock" in r.lower() for r in out["degradation_reasons"])


def test_failed_llm_call_degrades_instead_of_raising():
    def invoke(prompt, tool):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    out = _agent(_StubSearch(SEARCH_RESULTS["results"]), invoke).run(
        premise="x", risk_flags=[]
    )
    assert out["claims"] == []
    assert out["degraded"]
    assert any("429" in r for r in out["degradation_reasons"])


def test_claims_from_every_loop_iteration_are_collected():
    """The loop may run more than once; claims accumulate rather than overwrite."""
    def invoke(prompt, tool):
        tool("q")
        return [ResearchOutput(**AGENT_OUTPUT), ResearchOutput(**AGENT_OUTPUT)]

    out = _agent(_StubSearch(SEARCH_RESULTS["results"]), invoke).run(
        premise="x", risk_flags=[]
    )
    assert len(out["claims"]) == 2 * len(AGENT_OUTPUT["claims"])


# ------------------------------------------------------------------ loop bound

def test_research_loop_is_bounded():
    agent = ResearchAgent(search_client=_StubSearch([]), max_iterations=3)
    loop = agent.build_loop(lambda q: {"results": []})
    assert loop.max_iterations == 3
    assert len(loop.sub_agents) == 1
    researcher = loop.sub_agents[0]
    assert researcher.output_schema is ResearchOutput
    # tool-call events are collected off ADK's own callbacks
    assert researcher.before_tool_callback and researcher.after_tool_callback


def test_all_three_research_modes_are_requested():
    """Craft precedent, trope fatigue and comparable reception, per the contract."""
    assert set(RETRIEVAL_MODES) == {
        "craft_precedent", "trope_fatigue", "comparable_reception"
    }


# ------------------------------------------------------------------ live

@pytest.mark.skipif(os.environ.get("LIVE_AGENTS") != "1", reason="set LIVE_AGENTS=1")
def test_live_research_run_is_grounded():
    out = ResearchAgent().run(
        premise="A disgraced Stockholm detective is locked in a night-long "
                "interrogation with the one suspect who knows what happened "
                "to her missing daughter.",
        risk_flags=[{"category": "Cinematography", "issue": "Extreme darkness",
                     "detail": "70.0% of keyframes are sub-40 luminance."}],
    )
    assert out["inferred_genre"]
    for c in out["claims"]:
        assert c["source_url"] in out["retrieved_urls"]
