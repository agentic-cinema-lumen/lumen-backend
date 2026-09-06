"""Unit tests for parallel search client and trope sleuth."""

import unittest
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
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

    def test_live_request_shape_and_excerpt_normalisation(self):
        """The live call must send the documented Search API request and read excerpts."""
        client = ParallelSearchClient(api_key="test-key", cache_dir=self.test_cache)
        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = {
            "search_id": "srch_1",
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "A",
                    "publish_date": "2024-01-01",
                    "excerpts": ["first excerpt", "second excerpt"],
                }
            ],
        }
        with patch("src.search.parallel_search_client.requests.post", return_value=resp) as post:
            out = client.search("dark cinematography backlash", num_results=3)

        _, kwargs = post.call_args
        body = kwargs["json"]
        self.assertEqual(body["search_queries"], ["dark cinematography backlash"])
        self.assertEqual(body["objective"], "dark cinematography backlash")
        self.assertEqual(body["mode"], "advanced")
        self.assertIn("max_chars_total", body)
        self.assertEqual(body["advanced_settings"], {"max_results": 3})
        self.assertNotIn("query", body)
        self.assertEqual(kwargs["headers"]["x-api-key"], "test-key")
        self.assertNotIn("Authorization", kwargs["headers"])

        self.assertEqual(out["_source"], "parallel_api")
        r = out["results"][0]
        self.assertEqual(r["url"], "https://example.com/a")
        self.assertEqual(r["snippet"], "first excerpt second excerpt")
        self.assertEqual(r["published_date"], "2024-01-01")

    def test_trope_sleuth_script_mode(self):
        script_metrics = {"climax_acceleration": 2.1, "words_per_minute": 115.0}
        vision_metrics = {"dark_frame_ratio": 0.45}
        investigation = self.sleuth.investigate_script_craft(script_metrics, vision_metrics)
        self.assertEqual(investigation["mode"], "script_investigation")
        self.assertGreaterEqual(investigation["claims_count"], 1)
        self.assertTrue(any(c["category"] == "Cinematography" for c in investigation["claims"]))


if __name__ == "__main__":
    unittest.main()
