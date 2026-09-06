"""
Unit tests for Deterministic Counterfactual Sweep, Safe Oracle Wrapper, and Model Diagnostics.
"""

import unittest
from src.quant.oracle import QuantOracle
from src.quant.sweep import SafeOracleWrapper, run_counterfactual_sweep
from src.quant.diagnostics import get_model_diagnostics


class TestQuantSweep(unittest.TestCase):
    def setUp(self):
        self.oracle = QuantOracle.get_instance()
        self.sample_features = {
            "total_duration_min": 181.3,
            "cuts_per_minute": 15.49,
            "pacing_acceleration": 0.901,
            "average_shot_length": 3.87,
            "words_per_minute": 51.65,
            "lines_per_minute": 4.30,
            "dialogue_shot_ratio": 0.36,
            "mean_luminance": 60.55,
            "luminance_std": 31.91,
            "dark_frame_ratio": 0.70,
            "genre": "Drama"
        }

    def test_corpus_genre_priors_recentered(self):
        """Verify that genre priors are centered on the 100-film corpus."""
        drama_base = self.oracle.get_genre_baseline("Drama")
        self.assertEqual(drama_base, 8.13)
        self.assertEqual(self.oracle.get_genre_baseline(None), 7.80)
        self.assertEqual(self.oracle.get_genre_baseline("UnknownGenreXYZ"), 7.80)

    def test_safe_oracle_wrapper_feature_retention(self):
        """SafeOracleWrapper must maintain base features and only modify targeted override."""
        wrapper = SafeOracleWrapper(self.sample_features, oracle=self.oracle)
        self.assertAlmostEqual(wrapper.baseline_rating, 8.33, delta=0.1)

        # Probe dark_frame_ratio override
        probe_res = wrapper.predict_override({"dark_frame_ratio": 0.10})
        # Base features like duration should remain 181.3, not default 120.0
        self.assertEqual(probe_res["input_craft_features"]["total_duration_min"], 181.3)
        self.assertEqual(probe_res["input_craft_features"]["dark_frame_ratio"], 0.10)

        # Single variable delta probe
        new_rating, delta = wrapper.probe_delta("dark_frame_ratio", 0.10)
        self.assertIsInstance(delta, float)
        self.assertGreater(new_rating, 8.0)

    def test_counterfactual_sweep_structure(self):
        """Ensure the sweep generates a 15-point deterministic table across 3 levers."""
        sweep = run_counterfactual_sweep(self.sample_features, oracle=self.oracle)

        self.assertIn("baseline_rating", sweep)
        self.assertIn("sweep_table", sweep)
        self.assertIn("noise_floor_mae", sweep)
        self.assertIn("outcome", sweep)
        self.assertIn("formatted_table", sweep)

        table = sweep["sweep_table"]
        self.assertEqual(len(table), 3)

        features_swept = [row["feature"] for row in table]
        self.assertIn("dark_frame_ratio", features_swept)
        self.assertIn("total_duration_min", features_swept)
        self.assertIn("pacing_acceleration", features_swept)

        # Total 15 probes
        total_probes = sum(len(row["probes"]) for row in table)
        self.assertEqual(total_probes, 15)

    def test_noise_floor_downgrades_to_inconclusive(self):
        """When all craft deltas are within the MAE noise floor, outcome must be 'inconclusive'."""
        sweep = run_counterfactual_sweep(self.sample_features, oracle=self.oracle)
        # For Fight Club, max craft delta is ~0.08, while MAE is ~0.42
        self.assertFalse(sweep["clears_noise_floor"])
        self.assertEqual(sweep["outcome"], "inconclusive")
        self.assertIn("noise floor", sweep["confidence_summary"].lower())

    def test_model_diagnostics_contract(self):
        """Ensure get_model_diagnostics adheres to OpenAPI ModelDiagnostics schema."""
        diag = get_model_diagnostics(oracle=self.oracle)

        self.assertIn(diag["trainerStatus"], ["healthy", "running", "degraded", "failed"])
        self.assertIsInstance(diag["activeModel"], str)
        self.assertIsInstance(diag["lastTrainingRun"], str)
        self.assertEqual(diag["featureCount"], 11)
        self.assertIn("hitPrecision", diag["evaluation"])
        self.assertIn("calibrationError", diag["evaluation"])
        self.assertGreater(diag["evaluation"]["calibrationError"], 0.0)


if __name__ == "__main__":
    unittest.main()
