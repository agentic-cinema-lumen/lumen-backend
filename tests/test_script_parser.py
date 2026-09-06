"""Unit tests for script_parser.py."""

import json
import statistics
import unittest
from functools import lru_cache
from pathlib import Path

from src.ingestion.script_parser import (
    ScriptParser,
    parse_premise_logline,
    validate_screenplay,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MOVIES_DIR = REPO_ROOT / "data" / "movies"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@lru_cache(maxsize=None)
def _parse_movie(slug: str):
    return ScriptParser().parse_script_file(str(MOVIES_DIR / slug / "script.txt"))


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


class TestSceneHeaderDetection(unittest.TestCase):
    """Task 1: dash / em-dash / slash separated headings must count as scenes."""

    def test_dash_separated_headings_fixture(self):
        metrics = ScriptParser().parse_script_file(str(FIXTURES / "dash_headings.txt"))
        self.assertGreater(metrics["total_scenes"], 4)

    def test_django_unchained_parses_many_scenes(self):
        # Regression: `EXT - BENNETT MANOR- DAY` used to yield 2 scenes for a 165 min film.
        metrics = _parse_movie("django_unchained")
        self.assertGreater(metrics["total_scenes"], 4)


class TestCharacterCueClassification(unittest.TestCase):
    """Task 2: scene headings and camera/transition directions are not characters."""

    def test_django_character_count_is_plausible(self):
        # Regression: used to report 168 "characters" incl. CUT TO, WE SEE, A BIG TREE.
        metrics = _parse_movie("django_unchained")
        self.assertLess(metrics["characters_count"], 120)
        self.assertGreaterEqual(metrics["characters_count"], 3)

    def test_transition_lines_are_not_characters(self):
        metrics = ScriptParser().parse_script_file(str(FIXTURES / "dash_headings.txt"))
        for banned in ("CUT TO", "WE SEE", "ANGLE ON", "FADE OUT"):
            self.assertNotIn(banned, metrics["characters"])
        self.assertIn("DJANGO", metrics["characters"])


class TestScreenplayValidationGate(unittest.TestCase):
    """Task 3: reject documents whose derived metrics are structurally impossible."""

    def test_synopsis_is_rejected(self):
        # data/movies/alien/script.txt is a 501-word synopsis, not a screenplay.
        reasons = validate_screenplay(_parse_movie("alien"))
        self.assertTrue(reasons, "alien synopsis should be rejected")

    def test_real_screenplay_is_accepted(self):
        self.assertEqual(validate_screenplay(_parse_movie("fight_club")), [])


class TestDurationModel(unittest.TestCase):
    """Task 4: page-rate runtime model, checked as an aggregate bound over the corpus.

    Achieved with words_per_page = 186.0 over the 100 manifest films:
    median absolute runtime error 11.0%, 37/100 films off by more than 15%.
    (Before: median 32.6%, 81/100 off by more than 15%.)
    """

    def test_corpus_runtime_error_bounds(self):
        manifest = json.load(open(MOVIES_DIR / "movies_manifest.json"))
        errors = []
        for movie in manifest:
            script = MOVIES_DIR / movie["slug"] / "script.txt"
            if not script.exists() or not movie.get("duration"):
                continue
            metrics = _parse_movie(movie["slug"])
            errors.append(
                abs(metrics["estimated_duration_min"] - movie["duration"]) / movie["duration"]
            )
        self.assertGreaterEqual(len(errors), 90)
        self.assertLess(statistics.median(errors), 0.20)
        self.assertLessEqual(sum(1 for e in errors if e > 0.15), 40)

    def test_corpus_scene_detection(self):
        """Films collapsing to <4 scenes: 11/100 before, 6/100 after.

        The remainder are genuinely header-less prose transcripts (alien, saw,
        evil_dead, gravity, aladdin, finding_nemo) which the validation gate rejects.
        """
        manifest = json.load(open(MOVIES_DIR / "movies_manifest.json"))
        collapsed = [
            m["slug"]
            for m in manifest
            if (MOVIES_DIR / m["slug"] / "script.txt").exists()
            and _parse_movie(m["slug"])["total_scenes"] < 4
        ]
        self.assertLessEqual(len(collapsed), 6, f"collapsed: {collapsed}")


class TestPacingNotHardcoded(unittest.TestCase):
    """Task 5: climax/pacing acceleration must be measured, not defaulted to 1.0."""

    def test_fight_club_climax_acceleration_is_measured(self):
        metrics = _parse_movie("fight_club")
        self.assertGreaterEqual(metrics["total_scenes"], 4)
        self.assertNotEqual(metrics["climax_acceleration"], 1.0)


if __name__ == "__main__":
    unittest.main()
