"""The orchestrated pre-mortem engine, in script mode and concept mode.

Both LLM subagents are recorded fixtures; the deterministic half — parser,
ConceptInspector, oracle, sweep — runs for real. LIVE_AGENTS=1 adds a live run.
"""

import os
import unittest

import pytest

from src.premortem.premortem_agent import PreMortemAgent
from tests.stub_agents import StubResearchAgent, StubSynthesisAgent, stub_orchestrator

SCRIPT = "data/movies/fight_club/script.txt"
KEYFRAMES = "data/movies/fight_club/keyframes"
LOGLINE = (
    "An insomniac office worker and a soap salesman start an underground fight club "
    "that grows into something neither of them can stop."
)


def _script_text():
    with open(SCRIPT, encoding="utf-8", errors="replace") as f:
        return f.read()


class TestOrchestratedPreMortem(unittest.TestCase):

    def setUp(self):
        self.research = StubResearchAgent()
        self.synthesis = StubSynthesisAgent()
        self.agent = PreMortemAgent(
            orchestrator=stub_orchestrator(self.research, self.synthesis)
        )

    def test_script_mode(self):
        report = self.agent.run_premortem(
            story=LOGLINE, script_text=_script_text(), keyframes_dir=KEYFRAMES,
            title="Fight Club", genre="Drama",
        )
        self.assertTrue(report["has_screenplay"])
        self.assertTrue(report["has_concept_frames"])
        self.assertGreater(report["prediction"]["expected_rating"], 5.0)
        self.assertTrue(report["sweep"])
        self.assertTrue(report["claims"])
        self.assertFalse(report["degraded"])

    def test_concept_mode_has_the_same_report_shape(self):
        script = self.agent.run_premortem(
            story=LOGLINE, script_text=_script_text(), keyframes_dir=KEYFRAMES,
            title="Fight Club", genre="Drama",
        )
        concept = self.agent.run_premortem(story=LOGLINE, title="Fight Club")
        self.assertEqual(set(script), set(concept))
        self.assertFalse(concept["has_screenplay"])
        self.assertFalse(concept["has_concept_frames"])
        self.assertIsNone(concept["script_metrics"])

    def test_concept_mode_genre_comes_from_the_agent(self):
        """The parse_premise_logline keyword heuristic is gone."""
        report = self.agent.run_premortem(story=LOGLINE, title="Fight Club")
        self.assertEqual(report["genre"], self.research.inferred_genre)
        self.assertIsNone(self.research.seen["genre"])

    def test_the_research_agent_receives_the_deterministic_risk_flags(self):
        self.agent.run_premortem(
            story=LOGLINE, script_text=_script_text(), keyframes_dir=KEYFRAMES,
            title="Fight Club", genre="Drama",
        )
        flags = self.research.seen["risk_flags"]
        self.assertIsInstance(flags, list)
        # Fight Club's keyframes are measurably dark; the flag must reach research
        self.assertTrue(any(f["category"] == "Cinematography" for f in flags), flags)

    def test_the_synthesis_agent_never_sees_a_tool(self):
        from src.agents.synthesis_agent import SynthesisAgent
        built = SynthesisAgent().build_agent()
        self.assertFalse(built.tools)
        self.assertIsNotNone(built.output_schema)

    def test_a_degraded_subagent_degrades_the_report(self):
        agent = PreMortemAgent(orchestrator=stub_orchestrator(
            StubResearchAgent(degraded=True), StubSynthesisAgent()
        ))
        report = agent.run_premortem(story=LOGLINE, title="Fight Club")
        self.assertTrue(report["degraded"])
        self.assertTrue(report["degradation_reasons"])


@pytest.mark.skipif(os.environ.get("LIVE_AGENTS") != "1", reason="set LIVE_AGENTS=1")
def test_live_end_to_end():
    report = PreMortemAgent().run_premortem(
        story=LOGLINE, script_text=_script_text(), keyframes_dir=KEYFRAMES,
        title="Fight Club",
    )
    assert report["prediction"]["expected_rating"] > 0
    for claim in report["claims"]:
        assert claim["source_url"] in report["retrieved_urls"]


if __name__ == "__main__":
    unittest.main()
