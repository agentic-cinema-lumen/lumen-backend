"""Recorded-fixture stand-ins for the two LLM subagents.

The same dependency-injection seam the old tests used for `LLMClient(force_mock=True)`
and `ParallelSearchClient(force_mock=True)`. No Gemini call, no live search, no
metered credits — and the deterministic half of the orchestrator runs for real.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.orchestrator import Orchestrator

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH_RESULTS = json.loads((FIXTURES / "search_results.json").read_text())
RESEARCH_OUTPUT = json.loads((FIXTURES / "research_agent_output.json").read_text())
SYNTHESIS_REPORT = json.loads((FIXTURES / "synthesis_report.json").read_text())


class StubResearchAgent:
    def __init__(
        self,
        claims: Optional[List[Dict[str, Any]]] = None,
        inferred_genre: str = "Psychological Thriller",
        degraded: bool = False,
        reasons: Optional[List[str]] = None,
    ):
        self.claims = RESEARCH_OUTPUT["claims"] if claims is None else claims
        self.inferred_genre = inferred_genre
        self.degraded = degraded
        self.reasons = reasons or (["search results are mock output"] if degraded else [])
        self.seen: Dict[str, Any] = {}

    def run(self, premise, risk_flags=None, genre=None, title="Untitled Submission"):
        self.seen = {"premise": premise, "risk_flags": risk_flags, "genre": genre}
        return {
            "claims": self.claims,
            "dropped": [],
            "queries_executed": RESEARCH_OUTPUT["queries_executed"],
            "retrieved_urls": [r["url"] for r in SEARCH_RESULTS["results"]],
            "inferred_genre": genre or self.inferred_genre,
            "tool_events": [{"phase": "tool_start", "tool": "parallel_search",
                             "args": {"query": RESEARCH_OUTPUT["queries_executed"][0]}}],
            "degraded": self.degraded,
            "degradation_reasons": self.reasons,
        }


class StubSynthesisAgent:
    def __init__(self, report: Optional[Dict[str, Any]] = None, degraded: bool = False,
                 reasons: Optional[List[str]] = None):
        self.report = SYNTHESIS_REPORT if report is None else report
        self.degraded = degraded
        self.reasons = reasons or (["synthesis agent call failed: 429"] if degraded else [])
        self.seen: Dict[str, Any] = {}

    def run(self, inputs):
        self.seen = inputs
        return {
            "report": self.report,
            "degraded": self.degraded,
            "degradation_reasons": self.reasons,
        }


def stub_orchestrator(research=None, synthesis=None) -> Orchestrator:
    return Orchestrator(
        research_agent=research or StubResearchAgent(),
        synthesis_agent=synthesis or StubSynthesisAgent(),
    )
