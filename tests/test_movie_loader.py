"""Unit tests for movie_dataset_loader.py."""

import unittest
from pathlib import Path
from src.quant.movie_dataset_loader import MovieDatasetLoader


class TestMovieDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.loader = MovieDatasetLoader.get_instance(min_votes=1000)

    def test_dataset_loaded(self):
        self.assertIsNotNone(self.loader.df)
        self.assertGreater(len(self.loader.df), 10000)
        self.assertGreater(len(self.loader.genre_baselines), 5)

    def test_genre_baselines(self):
        horror_mean = self.loader.get_genre_expectation("Horror")
        drama_mean = self.loader.get_genre_expectation("Drama")
        # In IMDb, Drama typically rates higher than Horror
        self.assertGreater(drama_mean, horror_mean)

    def test_find_comparable_movies(self):
        pitch = "A young defense lawyer uncovers a vast government conspiracy during a murder trial."
        comps = self.loader.find_comparable_movies(pitch, genre="Crime / Drama", top_k=3)
        self.assertEqual(len(comps), 3)
        self.assertIn("title", comps[0])
        self.assertIn("imdb_rating", comps[0])
        self.assertIn("logline", comps[0])


if __name__ == "__main__":
    unittest.main()
