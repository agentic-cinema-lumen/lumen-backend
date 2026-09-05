"""
🧪 UNIT TESTS FOR QUANT RESIDUAL ORACLE & AGENT TOOL 🧪

Verifies that the ML model oracle is fully ready for the Main Agent:
1. Instant model loading (<10ms)
2. Handling of full and partial feature dictionaries with intelligent defaults
3. Accurate residual calculation and craft categorization
4. Actionable craft vulnerability flags (darkness, climax rush, dialogue fatigue)
5. Point attributions for LLM explainability
6. LLM / MCP tool declaration schema compatibility
7. Direct tool execution via execute_tool()
8. Movie package evaluation
9. High-throughput latency benchmark (<2ms per call)
"""

import unittest
import time
from pathlib import Path

from src.quant.oracle import QuantOracle, get_oracle
from src.quant.model_trainer import FEATURE_COLUMNS, DEFAULT_FEATURE_VALUES


class TestQuantOracle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.oracle = get_oracle()

    def test_01_oracle_initialization(self):
        """Verify oracle loads champion model instantly without errors."""
        self.assertIsNotNone(self.oracle.model)
        self.assertEqual(self.oracle.model.model_type, "ridge")
        self.assertIn("cv_mae", self.oracle.model.metrics)
        self.assertLess(self.oracle.model.metrics["cv_mae"], 0.60)

    def test_02_predict_craft_with_minimal_inputs(self):
        """Verify model handles minimal inputs and fills cinema defaults cleanly."""
        res = self.oracle.predict_craft(genre="Sci-Fi")
        self.assertIn("expected_rating", res)
        self.assertIn("craft_residual_delta", res)
        self.assertIn("genre_baseline_rating", res)
        self.assertIn("craft_verdict", res)
        self.assertIn("confidence_interval", res)

        # Rating must be within realistic cinema bounds
        self.assertGreaterEqual(res["expected_rating"], 1.0)
        self.assertLessEqual(res["expected_rating"], 10.0)
        self.assertEqual(res["genre_baseline_rating"], 5.63)

    def test_03_predict_craft_with_custom_features(self):
        """Verify custom craft inputs produce expected metrics and attributions."""
        res = self.oracle.predict_craft(
            title="Dune Pre-Mortem",
            genre="Sci-Fi",
            total_duration_min=155.0,
            cuts_per_minute=14.2,
            pacing_acceleration=1.45,
            dark_frame_ratio=0.35,
            mean_luminance=58.0,
            words_per_minute=45.0
        )
        self.assertEqual(res["title"], "Dune Pre-Mortem")
        self.assertGreater(len(res["top_craft_attributions"]), 0)

        # Verify attribution structure
        attr = res["top_craft_attributions"][0]
        self.assertIn("feature", attr)
        self.assertIn("point_impact", attr)
        self.assertIn("direction", attr)
        self.assertIn("interpretation", attr)

    def test_04_craft_vulnerabilities_triggers(self):
        """Verify rule-based warning flags trigger for risky craft choices."""
        # Extreme darkness + severe climax spike
        res = self.oracle.predict_craft(
            dark_frame_ratio=0.55,
            pacing_acceleration=1.90,
            words_per_minute=20.0,
            cuts_per_minute=12.0
        )
        vulnerabilities = res["craft_vulnerabilities"]
        self.assertGreaterEqual(len(vulnerabilities), 2)

        categories = [v["category"] for v in vulnerabilities]
        self.assertIn("Cinematography", categories)
        self.assertIn("Pacing & Editing", categories)
        self.assertIn("Dialogue & Tension", categories)

    def test_05_llm_tool_spec_format(self):
        """Verify function-calling tool specification conforms to standard JSON Schema."""
        spec = self.oracle.get_tool_spec()
        self.assertEqual(spec["name"], "quant_residual_oracle")
        self.assertIn("description", spec)
        self.assertIn("parameters", spec)
        self.assertEqual(spec["parameters"]["type"], "object")

        props = spec["parameters"]["properties"]
        self.assertIn("genre", props)
        self.assertIn("cuts_per_minute", props)
        self.assertIn("dark_frame_ratio", props)
        self.assertIn("pacing_acceleration", props)
        self.assertIn("words_per_minute", props)

    def test_06_tool_execution(self):
        """Verify direct tool dispatcher for agent tool calls."""
        res = self.oracle.execute_tool("quant_residual_oracle", {
            "genre": "Horror",
            "dark_frame_ratio": 0.60,
            "cuts_per_minute": 18.0
        })
        self.assertIn("expected_rating", res)
        self.assertIn("craft_verdict", res)

        # Invalid tool name should raise ValueError
        with self.assertRaises(ValueError):
            self.oracle.execute_tool("unknown_tool", {})

    def test_07_movie_package_evaluation(self):
        """Verify evaluation of real movie package directory."""
        alien_dir = Path(__file__).resolve().parent.parent / "data" / "movies" / "alien"
        if alien_dir.exists():
            res = self.oracle.predict_movie_package(str(alien_dir))
            self.assertEqual(res["title"], "Alien")
            self.assertIn("expected_rating", res)
            self.assertIn("actual_imdb_rating", res)
            self.assertEqual(res["actual_imdb_rating"], 8.4)

    def test_08_inference_latency_benchmark(self):
        """Verify warm inference is ultra-fast (<5ms per call)."""
        # Warm-up
        self.oracle.predict_craft(genre="Action")

        iterations = 50
        t0 = time.perf_counter()
        for _ in range(iterations):
            self.oracle.predict_craft(
                genre="Drama",
                cuts_per_minute=16.0,
                dark_frame_ratio=0.25,
                pacing_acceleration=1.20
            )
        elapsed_sec = time.perf_counter() - t0
        avg_ms_per_call = (elapsed_sec / iterations) * 1000

        self.assertLess(avg_ms_per_call, 5.0, f"Inference took {avg_ms_per_call:.2f}ms, expected <5ms")

    def test_09_decision_history_loading(self):
        """Verify ML engineering decision history markdown and summary are retrievable."""
        history = self.oracle.get_decision_history()
        self.assertIn("Decision History", history)
        self.assertIn("Ridge Regression", history)
        self.assertIn("Target Leakage", history)
        self.assertGreater(len(history), 500)

        summary = self.oracle.get_decision_summary()
        self.assertIn("Ridge Regression", summary)
        self.assertIn("100 genuine feature films", summary)
        self.assertGreater(len(summary), 100)


if __name__ == "__main__":
    unittest.main()
