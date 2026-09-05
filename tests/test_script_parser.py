"""Unit tests for script_parser.py."""

import unittest
from src.ingestion.script_parser import ScriptParser, parse_premise_logline


class TestScriptParser(unittest.TestCase):

    def setUp(self):
        self.parser = ScriptParser()
        self.sample_script = """
INT. CONTROL ROOM - DAY

COMMANDER
We have lost contact with the perimeter team.

LIEUTENANT
Sending reinforcements now, sir.

The alarm lights flash red across the steel consoles.

EXT. LUNAR BASE - DOCKING BAY - NIGHT

Heavy boots crunch against the regolith.

COMMANDER
Do you see anything out there?

LIEUTENANT
Nothing yet. Just dust and silence.
"""

    def test_parse_script_text_scenes(self):
        metrics = self.parser.parse_script_text(self.sample_script, title="Lunar Base")
        self.assertEqual(metrics["title"], "Lunar Base")
        self.assertGreaterEqual(metrics["total_scenes"], 2)
        self.assertGreater(metrics["dialogue_words"], 0)
        self.assertGreater(metrics["words_per_minute"], 0)
        self.assertIn("COMMANDER", metrics["characters"])
        self.assertIn("LIEUTENANT", metrics["characters"])

    def test_to_episode_features_structure(self):
        metrics = self.parser.parse_script_text(self.sample_script, title="Lunar Base")
        features = self.parser.to_episode_features(metrics)
        self.assertIn("shots", features)
        self.assertIn("total_shots", features)
        self.assertIn("total_duration_sec", features)
        self.assertGreater(len(features["shots"]), 0)
        self.assertIn("dialogue", features["shots"][0])

    def test_parse_premise_logline(self):
        logline = "A claustrophobic courtroom trial on an isolated space station tests an idealistic lawyer."
        info = parse_premise_logline(logline)
        self.assertIn("genre", info)
        self.assertIn("expected_pacing", info)
        self.assertGreater(info["expected_pacing"]["words_per_minute"], 50.0)


if __name__ == "__main__":
    unittest.main()
