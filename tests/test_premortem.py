"""Integration tests for PreMortemAgent (Script mode and Premise mode)."""

import unittest
from pathlib import Path

from src.premortem.premortem_agent import PreMortemAgent
from src.search.parallel_search_client import ParallelSearchClient
from src.utils.llm_client import LLMClient


class TestPreMortemAgent(unittest.TestCase):

    def setUp(self):
        llm = LLMClient(force_mock=True)
        search_client = ParallelSearchClient(force_mock=True)
        self.agent = PreMortemAgent(llm_client=llm)
        self.agent.trope_sleuth.client = search_client

    def test_run_script_premortem(self):
        script_file = "data/movies/alien/script.txt"
        keyframes_dir = "data/movies/alien/keyframes"

        report = self.agent.run_script_premortem(
            script_path_or_text=script_file,
            keyframes_path_or_dir=keyframes_dir,
            title="Alien",
            show_name="Alien Franchise",
            show_historical_mean=8.4
        )

        self.assertEqual(report["mode"], "script_premortem")
        self.assertIn("projected_baseline_rating", report)
        self.assertGreater(report["projected_baseline_rating"], 5.0)
        self.assertIn("craft_metrics", report)
        self.assertIn("audience_trope_intelligence", report)
        self.assertIn("recommendations", report)

    def test_run_premise_premortem(self):
        logline = "In a subterranean lunar courtroom, an accused rebel officer faces execution while the jury oxygen levels deplete."
        keyframes_dir = "data/movies/alien/keyframes"

        report = self.agent.run_premise_premortem(
            logline=logline,
            keyframes_path_or_dir=keyframes_dir,
            title="Lunar Justice Pitch"
        )

        self.assertEqual(report["mode"], "premise_greenlight_compass")
        self.assertIn("projected_rating_potential", report)
        self.assertIn("style_premise_cohesion", report)
        self.assertIn("make_or_break_dependencies", report)
        self.assertIn("greenlight_verdict", report)


if __name__ == "__main__":
    unittest.main()
