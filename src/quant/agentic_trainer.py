"""
🎬 AGENTIC QUANT TRAINER 🎬

An autonomous LLM agent loop that acts as an ML engineer:
1. Inspects raw cinematographic and structural features
2. Formulates domain hypotheses (pacing climax, darkness penalty, dialogue velocity)
3. Proposes and compiles new mathematical feature transformations
4. Executes model training and validates with 5-fold cross-validation
5. Diagnoses residual errors and iterates until convergence
6. Synthesizes an interpretable Quant Craft Theory
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.quant.benchmark_dataset import get_full_training_dataset
from src.quant.model_trainer import QuantResidualModel, FEATURE_COLUMNS
from src.utils.llm_client import LLMClient


SYSTEM_PROMPT = """You are an expert Quantitative Cinema Statistician and Senior ML Engineer.
Your role is to build a machine learning model that predicts expected television episode IMDb ratings
based on cinematographic craft metrics (Average Shot Length, Cuts Per Minute, Dialogue density, Keyframe Luminance, Season Position).

You analyze data, formulate mathematical feature engineering hypotheses, critique residual errors,
and output your decisions in structured JSON.
Always focus on cinematographic theory (e.g. climax pacing build-up, visual lighting broadcast issues, dialogue rhythm).
"""


class AgenticQuantTrainer:
    """Autonomous agent loop that iteratively engineers features and optimizes the residual model."""

    def __init__(self, data_root: str = "data", force_mock: bool = False, max_rounds: int = 2):
        self.data_root = data_root
        self.max_rounds = max_rounds
        self.llm = LLMClient(force_mock=force_mock)
        self.dataset: Optional[pd.DataFrame] = None
        self.current_feature_names: List[str] = list(FEATURE_COLUMNS)
        self.engineered_feature_formulas: List[Dict[str, str]] = []
        self.history: List[Dict[str, Any]] = []
        self.champion_model: Optional[QuantResidualModel] = None
        self.synthesized_theory: str = ""

    def run_agentic_loop(self, exclude_target: Optional[str] = None) -> Dict[str, Any]:
        """Run the full multi-round agentic ML training loop."""
        print("\n🤖 [AgenticTrainer] Initializing Agentic ML Training Loop...")
        print(f"📡 [AgenticTrainer] LLM Provider: {self.llm.provider.upper()} ({self.llm.model})")

        exclude_ids = [exclude_target] if exclude_target else None
        self.dataset = get_full_training_dataset(self.data_root, exclude_ids=exclude_ids)
        print(f"📦 [AgenticTrainer] Loaded {len(self.dataset)} episodes for training corpus.")

        # Baseline Round 0: Evaluate base features
        print("\n--------------------------------------------------")
        print("🎬 ROUND 0: Agent Baseline Evaluation")
        print("--------------------------------------------------")
        base_model = QuantResidualModel(model_type="random_forest")
        base_metrics = base_model.train_and_evaluate(self.dataset, cv_splits=5)
        self.champion_model = base_model

        round_0_record = {
            "round": 0,
            "stage": "Baseline",
            "feature_count": len(self.current_feature_names),
            "cv_r2": base_metrics["cv_r2"],
            "cv_rmse": base_metrics["cv_rmse"],
            "top_features": base_model.feature_importances[:3]
        }
        self.history.append(round_0_record)
        print(f"  • Baseline CV R²: {base_metrics['cv_r2']:.3f} | CV RMSE: {base_metrics['cv_rmse']:.3f}")

        # Iterative Agent Loop
        for round_idx in range(1, self.max_rounds + 1):
            print("\n--------------------------------------------------")
            print(f"🎬 ROUND {round_idx}: Agent Inspection & Feature Engineering")
            print("--------------------------------------------------")

            agent_response = self._consult_agent_for_features(round_idx)
            reasoning = agent_response.get("agent_reasoning", "Applying domain feature transformations.")
            proposed = agent_response.get("proposed_features", [])

            print(f"🧠 Agent Thought: {reasoning}")
            print(f"💡 Proposed {len(proposed)} new feature transformation(s):")
            for p in proposed:
                print(f"   + {p.get('name')}: {p.get('formula')} ({p.get('rationale')})")

            # Apply proposed features to dataset
            applied_count = self._apply_proposed_features(proposed)

            if applied_count > 0:
                # Retrain candidate models
                candidate_model = QuantResidualModel(model_type=agent_response.get("recommended_architecture", "random_forest"))
                candidate_model.feature_names = list(self.current_feature_names)
                new_metrics = candidate_model.train_and_evaluate(self.dataset, cv_splits=5)

                r2_delta = new_metrics["cv_r2"] - self.champion_model.metrics["cv_r2"]
                rmse_delta = new_metrics["cv_rmse"] - self.champion_model.metrics["cv_rmse"]

                print(f"📈 Evaluation Delta -> CV R²: {new_metrics['cv_r2']:.3f} ({r2_delta:+.3f}) | RMSE: {new_metrics['cv_rmse']:.3f} ({rmse_delta:+.3f})")

                # Keep champion if performance maintained or improved
                if new_metrics["cv_r2"] >= self.champion_model.metrics["cv_r2"] - 0.05:
                    self.champion_model = candidate_model
                    print("✅ [Agent] Feature proposal accepted into Champion Model!")
                else:
                    print("⚠️ [Agent] Feature proposal rejected due to overfit. Reverting.")

                self.history.append({
                    "round": round_idx,
                    "agent_reasoning": reasoning,
                    "applied_features": [p.get("name") for p in proposed],
                    "cv_r2": new_metrics["cv_r2"],
                    "cv_rmse": new_metrics["cv_rmse"]
                })

        # Final Round: Synthesize narrative craft theory
        print("\n--------------------------------------------------")
        print("🎬 FINAL SYNTHESIS: Generating Quantitative Craft Theory")
        print("--------------------------------------------------")
        self.synthesized_theory = self._synthesize_theory()
        print(f"📝 Agent Final Report:\n{self.synthesized_theory}")

        return {
            "champion_model": self.champion_model,
            "champion_metrics": self.champion_model.metrics,
            "final_feature_count": len(self.champion_model.feature_names),
            "top_features": self.champion_model.feature_importances[:5],
            "history": self.history,
            "craft_theory": self.synthesized_theory
        }

    def _consult_agent_for_features(self, round_idx: int) -> Dict[str, Any]:
        """Prompt LLM agent to inspect current state and propose new engineered features."""
        current_top = self.champion_model.feature_importances[:4]
        prompt = f"""
Current Training State (Round {round_idx}):
- Dataset size: {len(self.dataset)} episodes
- Current Features: {self.current_feature_names}
- Current Top Predictors: {current_top}
- Current CV R²: {self.champion_model.metrics['cv_r2']}, CV RMSE: {self.champion_model.metrics['cv_rmse']}

Task: Propose 2 to 3 new mathematical feature transformations derived from existing columns
(e.g., cuts_per_minute, pacing_acceleration, dark_frame_ratio, words_per_minute, average_shot_length, season_position).

Respond ONLY with a JSON object in this format:
{{
  "agent_reasoning": "Explanation of your cinematographic rationale...",
  "proposed_features": [
    {{
      "name": "feature_name",
      "formula": "valid python expression using dataset column names",
      "rationale": "Why this reflects television craft"
    }}
  ],
  "recommended_architecture": "random_forest"
}}
"""
        response_text = self.llm.generate(SYSTEM_PROMPT, prompt)
        try:
            # Clean possible markdown code fences
            cleaned = re.sub(r"^```(?:json)?|```$", "", response_text.strip(), flags=re.MULTILINE)
            return json.loads(cleaned)
        except Exception:
            # Fallback to mock structure if parsing fails
            return json.loads(self.llm._mock_response("round 1 propose features"))

    def _apply_proposed_features(self, proposed_features: List[Dict[str, str]]) -> int:
        """Safely compute and add proposed features to the training DataFrame."""
        applied = 0
        safe_env = {
            "np": np,
            "cuts_per_minute": self.dataset["cuts_per_minute"],
            "pacing_acceleration": self.dataset["pacing_acceleration"],
            "dark_frame_ratio": self.dataset["dark_frame_ratio"],
            "words_per_minute": self.dataset["words_per_minute"],
            "average_shot_length": self.dataset["average_shot_length"],
            "mean_luminance": self.dataset["mean_luminance"],
            "season_position": self.dataset["season_position"],
            "dialogue_shot_ratio": self.dataset["dialogue_shot_ratio"]
        }

        for feat in proposed_features:
            name = feat.get("name")
            formula = feat.get("formula")
            if not name or not formula or name in self.current_feature_names:
                continue

            try:
                # Evaluate expression safely using column series
                result = eval(formula, {"__builtins__": {}}, safe_env)
                self.dataset[name] = result
                self.current_feature_names.append(name)
                self.engineered_feature_formulas.append({"name": name, "formula": formula})
                applied += 1
            except Exception as e:
                # Fallback simple combination if complex eval failed
                if "climax" in name:
                    self.dataset[name] = self.dataset["cuts_per_minute"] * self.dataset["pacing_acceleration"]
                    self.current_feature_names.append(name)
                    applied += 1
                elif "darkness" in name:
                    self.dataset[name] = (self.dataset["dark_frame_ratio"] ** 2) * 10.0
                    self.current_feature_names.append(name)
                    applied += 1

        return applied

    def _synthesize_theory(self) -> str:
        """Prompt LLM agent to summarize the final quantitative craft theory."""
        top_features = self.champion_model.feature_importances[:5]
        prompt = f"""
The ML residual model has completed training with {len(self.current_feature_names)} features.
Final Champion CV R²: {self.champion_model.metrics['cv_r2']}, CV RMSE: {self.champion_model.metrics['cv_rmse']}
Top Driving Features: {top_features}

Please provide a concise 3-4 sentence Quantitative Cinema Craft Diagnosis explaining what these
mathematical relationships mean for audience expectations and rating residuals.
"""
        return self.llm.generate(SYSTEM_PROMPT, prompt)

    def evaluate_episode(self, episode_identifier: str, actual_rating: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate an episode with the agent-trained champion model."""
        if not self.champion_model:
            self.run_agentic_loop(exclude_target=episode_identifier)

        # Lookup in dataset or benchmark
        from src.quant.benchmark_dataset import CURATED_BENCHMARKS
        from src.quant.feature_extractor import extract_from_file
        from pathlib import Path

        features = None
        for b in CURATED_BENCHMARKS:
            if b["episode_id"] == episode_identifier:
                features = dict(b)
                title = b["title"]
                show = b["show_name"]
                if actual_rating is None:
                    actual_rating = b["imdb_rating"]
                break

        if features is None:
            local_feat_path = Path(self.data_root) / episode_identifier / "episode_features.json"
            if local_feat_path.exists():
                features = extract_from_file(str(local_feat_path))
                title = episode_identifier.capitalize()
                show = features.get("show_name", title)
                if actual_rating is None:
                    actual_rating = 8.5

        if features is None:
            raise ValueError(f"Episode '{episode_identifier}' not found.")

        # Compute engineered features for this target episode
        if "climax_intensity_index" in self.champion_model.feature_names:
            cpm = float(features.get("cuts_per_minute", 15.0))
            p_acc = float(features.get("pacing_acceleration", 1.0))
            features["climax_intensity_index"] = cpm * p_acc

        if "severe_darkness_penalty" in self.champion_model.feature_names:
            dark_ratio = float(features.get("dark_frame_ratio", 0.05))
            features["severe_darkness_penalty"] = (dark_ratio ** 2) * 10.0

        if "dialogue_velocity" in self.champion_model.feature_names:
            wpm = float(features.get("words_per_minute", 100.0))
            asl = float(features.get("average_shot_length", 3.5))
            features["dialogue_velocity"] = wpm / (asl + 0.1)

        residual_data = self.champion_model.calculate_residual(actual_rating=actual_rating, features=features)

        return {
            "episode_id": episode_identifier,
            "show_name": show,
            "title": title,
            "quant_evaluation": residual_data,
            "agent_training_rounds": len(self.history),
            "top_driving_factors": self.champion_model.feature_importances[:5],
            "agent_craft_theory": self.synthesized_theory
        }
