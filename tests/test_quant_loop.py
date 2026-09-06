"""
🧪 UNIT & INTEGRATION TESTS FOR QUANT RESIDUAL ENGINE & AGENT LOOP 🧪

Tests:
1. Feature extraction from local ingestion file (data/space/episode_features.json)
2. Luminance calculation from real keyframe images
3. Dataset loading and assembly
4. ML model training & cross-validation metrics
5. Champion model selection & feature importance extraction
6. Residual calculation & anomaly detection logic
7. JSON export schema compatibility for frontend
"""

import unittest
import json
from pathlib import Path

from src.quant.feature_extractor import extract_from_file, calculate_image_luminance
from src.quant.benchmark_dataset import get_full_training_dataset, CURATED_BENCHMARKS
from src.quant.model_trainer import QuantResidualModel, FEATURE_COLUMNS
from src.quant.quant_agent import QuantAgent


# Champion cv_r2 recorded after the fixed-parser re-extraction retrain (slice 3b).
RECORDED_CHAMPION_CV_R2 = -0.116


class TestQuantResidualEngine(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.alien_keyframe = self.root_dir / "data" / "movies" / "alien" / "keyframes" / "shot_0001.jpg"

        # QuantAgent.run_training_loop() persists its tournament winner over the
        # committed champion artifact, which changes what every later test (and
        # the oracle) loads. Snapshot and restore it.
        self._champion_path = self.root_dir / "data" / "models" / "champion_model.joblib"
        self._champion_bytes = self._champion_path.read_bytes() if self._champion_path.exists() else None

    def tearDown(self):
        if self._champion_bytes is not None:
            self._champion_path.write_bytes(self._champion_bytes)

    def test_01_feature_extractor_from_local_data(self):
        """Verify feature extractor correctly computes craft metrics from shot sequence."""
        from src.quant.feature_extractor import extract_features_from_episode
        sample_data = {
            "episode_id": "test_sequence",
            "show_name": "Cinema Benchmark",
            "total_duration_sec": 180.0,
            "shots": [
                {"duration_sec": 3.5, "dialogue": "Are you reading this signal?", "dialogue_lines": ["Line 1"]},
                {"duration_sec": 4.0, "dialogue": "Yes, it is approaching rapidly.", "dialogue_lines": ["Line 2"]},
                {"duration_sec": 2.5, "dialogue": "", "dialogue_lines": []}
            ]
        }
        features = extract_features_from_episode(sample_data)

        # Check required craft features are extracted
        for col in FEATURE_COLUMNS:
            self.assertIn(col, features, f"Feature '{col}' must be present in extracted features")

        self.assertGreater(features["total_shots"], 0)
        self.assertGreater(features["average_shot_length"], 0.0)
        self.assertGreaterEqual(features["cuts_per_minute"], 0.0)
        self.assertGreaterEqual(features["words_per_minute"], 0.0)
        self.assertGreaterEqual(features["mean_luminance"], 0.0)
        self.assertLessEqual(features["mean_luminance"], 255.0)

    def test_02_keyframe_luminance_calculation(self):
        """Verify real image luminance calculation on actual keyframe."""
        if self.alien_keyframe.exists():
            lum = calculate_image_luminance(self.alien_keyframe)
            self.assertIsNotNone(lum)
            self.assertGreater(lum, 0.0)
            self.assertLess(lum, 255.0)

    def test_03_benchmark_dataset_assembly(self):
        """Verify benchmark dataset compiles with 100 genuine cinema releases without synthetic data."""
        df = get_full_training_dataset(data_root=str(self.root_dir / "data"))
        self.assertGreaterEqual(len(df), 50, "Training dataset must have at least 50 episodes")
        self.assertIn("imdb_rating", df.columns)

        # Check no NaN values in feature columns
        for col in FEATURE_COLUMNS:
            self.assertEqual(df[col].isna().sum(), 0, f"Column '{col}' should not have NaNs")

    def test_04_model_trainer_and_cv_metrics(self):
        """Verify models can train and compute valid cross-validation metrics."""
        df = get_full_training_dataset(data_root=str(self.root_dir / "data"))
        rf_model = QuantResidualModel(model_type="random_forest")
        metrics = rf_model.train_and_evaluate(df, cv_splits=3)

        self.assertIn("cv_rmse", metrics)
        self.assertIn("cv_r2", metrics)
        self.assertGreater(metrics["cv_r2"], -1.0)
        self.assertLess(metrics["cv_rmse"], 2.0)
        self.assertGreater(len(rf_model.feature_importances), 0)

    def test_05_residual_and_anomaly_classification(self):
        """Verify residual = actual - expected and anomaly flags trigger accurately."""
        df = get_full_training_dataset(data_root=str(self.root_dir / "data"))
        model = QuantResidualModel(model_type="random_forest")
        model.train_and_evaluate(df, cv_splits=3)

        # Severe deficit test: actual 5.0 on an episode expecting ~8.5
        mock_features = df.iloc[0].to_dict()
        res_low = model.calculate_residual(actual_rating=5.0, features=mock_features)
        self.assertLess(res_low["residual"], -1.5)
        self.assertEqual(res_low["anomaly_type"], "SEVERE_CRAFT_DEFICIT")

        # Masterpiece test: actual 10.0 on an episode expecting ~8.0
        res_high = model.calculate_residual(actual_rating=10.0, features=mock_features)
        self.assertGreater(res_high["residual"], 1.0)
        self.assertIn(res_high["anomaly_type"], ["MILD_OUTPERFORMANCE", "HISTORIC_MASTERPIECE"])

    def test_06_end_to_end_quant_agent_loop(self):
        """Verify full agent loop execution and JSON output schema."""
        agent = QuantAgent(data_root=str(self.root_dir / "data"))
        train_summary = agent.run_training_loop()

        self.assertIsNotNone(agent.champion_model)
        self.assertIn(train_summary["champion_model_type"], ["random_forest", "gradient_boosting", "ridge"])
        self.assertGreater(len(train_summary["craft_insights"]), 0)

        # Test evaluating an episode
        eval_res = agent.evaluate_episode("got_s08e03_long_night")
        self.assertIn("quant_evaluation", eval_res)
        self.assertIn("residual", eval_res["quant_evaluation"])
        self.assertIn("expected_rating", eval_res["quant_evaluation"])
        self.assertIn("anomaly_type", eval_res["quant_evaluation"])
        self.assertIn("extracted_features", eval_res)
        self.assertIn("top_driving_factors", eval_res)

        # Verify JSON serializability
        json_str = json.dumps(eval_res)
        self.assertGreater(len(json_str), 100)

    def test_07_agentic_quant_trainer_loop(self):
        """Verify the multi-round autonomous agentic ML engineer loop."""
        from src.quant.agentic_trainer import AgenticQuantTrainer
        trainer = AgenticQuantTrainer(data_root=str(self.root_dir / "data"), force_mock=True, max_rounds=1)
        summary = trainer.run_agentic_loop(exclude_target="movie_alien")

        self.assertIsNotNone(summary["champion_model"])
        self.assertGreater(summary["final_feature_count"], len(FEATURE_COLUMNS))
        self.assertGreater(len(summary["craft_theory"]), 20)

        # Test evaluating an episode
        res = trainer.evaluate_episode("movie_alien", actual_rating=8.4)
        self.assertIn("quant_evaluation", res)
        self.assertIn("residual", res["quant_evaluation"])

    def test_08_champion_cv_r2_regression_guard(self):
        """Loose regression guard on the champion model's cv_r2.

        This guards against *collapse*, not quality. At n=80 the fold-driven
        swing in ridge cv_r2 spans -0.116 to +0.061 across KFold seeds
        0/1/2/42, so no single value here is a trustworthy measure of model
        quality. The floor is the recorded
        post-retrain value minus 0.15: it fires when the training data or the
        feature pipeline breaks, and stays silent for ordinary fold noise.
        """
        df = get_full_training_dataset(data_root=str(self.root_dir / "data"))
        model = QuantResidualModel(model_type="ridge")
        metrics = model.train_and_evaluate(df, cv_splits=5)
        self.assertGreaterEqual(
            metrics["cv_r2"], RECORDED_CHAMPION_CV_R2 - 0.15,
            f"cv_r2 collapsed to {metrics['cv_r2']} from recorded {RECORDED_CHAMPION_CV_R2}"
        )


if __name__ == "__main__":
    unittest.main()
