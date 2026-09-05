"""Unit tests for parallel search client and trope sleuth."""

import unittest
import shutil
from pathlib import Path
from src.search.parallel_search_client import ParallelSearchClient
from src.search.trope_sleuth import TropeSleuth


class TestParallelSearch(unittest.TestCase):

    def setUp(self):
        self.test_cache = "data/cache/test_search"
        self.client = ParallelSearchClient(cache_dir=self.test_cache, force_mock=True)
        self.sleuth = TropeSleuth(search_client=self.client)

    def tearDown(self):
        p = Path(self.test_cache)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    def test_mock_search_results(self):
        res = self.client.search("lighting dark cinematography", num_results=2)
        self.assertIn("results", res)
        self.assertGreaterEqual(len(res["results"]), 1)
        text = (res["results"][0]["snippet"] + " " + res["results"][0]["title"]).lower()
        self.assertTrue(any(w in text for w in ["dark", "underexposed", "cinematography", "lighting", "black"]))

    def test_cache_persistence(self):
        # First call creates cache
        res1 = self.client.search("pacing climax rushed", num_results=2)
        # Second call reads cache
        res2 = self.client.search("pacing climax rushed", num_results=2)
        self.assertEqual(res2.get("_source"), "disk_cache")

    def test_trope_sleuth_script_mode(self):
        script_metrics = {"climax_acceleration": 2.1, "words_per_minute": 115.0}
        vision_metrics = {"dark_frame_ratio": 0.45}
        investigation = self.sleuth.investigate_script_craft(script_metrics, vision_metrics)
        self.assertEqual(investigation["mode"], "script_investigation")
        self.assertGreaterEqual(investigation["claims_count"], 1)
        self.assertTrue(any(c["category"] == "Cinematography" for c in investigation["claims"]))


if __name__ == "__main__":
    unittest.main()
