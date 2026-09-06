"""Unit tests for Benchmark Showcase Provider."""
import unittest
from src.quant.benchmarks import list_benchmark_movies, get_benchmark_movie


class TestBenchmarkShowcase(unittest.TestCase):

    def test_list_all_benchmarks(self):
        movies = list_benchmark_movies(featured_only=False)
        self.assertGreaterEqual(len(movies), 90)
        first = movies[0]
        self.assertIn("title", first)
        self.assertIn("slug", first)
        self.assertIn("year", first)
        self.assertIn("genre", first)
        self.assertIn("imdbRating", first)
        self.assertIn("isFeatured", first)

    def test_list_featured_benchmarks(self):
        featured = list_benchmark_movies(featured_only=True)
        self.assertGreater(len(featured), 0)
        self.assertTrue(all(m["isFeatured"] for m in featured))

    def test_get_benchmark_fight_club(self):
        fc = get_benchmark_movie("fight_club")
        self.assertIsNotNone(fc)
        self.assertEqual(fc["title"], "Fight Club")
        self.assertEqual(fc["slug"], "fight_club")
        self.assertGreater(fc["predictedRating"], 7.0)
        self.assertLess(fc["predictedRating"], 9.5)
        self.assertGreater(fc["score100"], 70)
        self.assertIn("counterfactualSweep", fc)
        sweep = fc["counterfactualSweep"]
        self.assertIn("baselineRating", sweep)
        self.assertIn("rows", sweep)
        self.assertEqual(len(sweep["rows"]), 3)
        self.assertIn("craftVulnerabilities", fc)

    def test_get_benchmark_alien(self):
        alien = get_benchmark_movie("alien")
        self.assertIsNotNone(alien)
        self.assertEqual(alien["title"], "Alien")
        self.assertGreater(alien["predictedRating"], 7.0)

    def test_get_nonexistent_movie(self):
        result = get_benchmark_movie("non_existent_blockbuster_xyz")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
