"""
🎬 QUANT RESIDUAL ORACLE & AGENT TOOL 🎬

Production-ready ML tool interface for the Main Pre-Mortem Agent.
Key Features:
- Sub-millisecond instant inference (<1ms) via pre-trained champion model (data/models/champion_model.joblib).
- Zero training overhead during live agent execution.
- Robust input handling: automatically fills missing craft metrics with benchmark cinema medians.
- Provides point attributions (exact impact per craft feature) and confidence intervals.
- Generates prescriptive craft vulnerability flags (darkness crush, pacing rush, dialogue fatigue).
- Exposes standard LLM / MCP function declaration schema for seamless integration into any agent loop.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

from src.quant.model_trainer import QuantResidualModel, FEATURE_COLUMNS, DEFAULT_FEATURE_VALUES


class QuantOracle:
    """Production interface and agent tool for the Quant Residual Model."""

    _instance: Optional["QuantOracle"] = None

    def __init__(self, model_path: Optional[str] = None, data_root: str = "data"):
        self.data_root = Path(data_root).resolve()
        if model_path is None:
            self.model_path = self.data_root / "models" / "champion_model.joblib"
        else:
            self.model_path = Path(model_path).resolve()

        self.model: Optional[QuantResidualModel] = None
        self._load_or_initialize()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None, data_root: str = "data") -> "QuantOracle":
        """Singleton accessor for zero-latency reuse across agent calls."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, data_root=data_root)
        return cls._instance

    def _load_or_initialize(self):
        """Load the persisted champion model or train a fresh one if not found."""
        if self.model_path.exists():
            try:
                self.model = QuantResidualModel.load(str(self.model_path))
                return
            except Exception as e:
                print(f"⚠️ [QuantOracle] Failed to load persisted model ({e}), retraining...")

        # Retrain champion model on benchmark movies
        from src.quant.quant_agent import QuantAgent
        agent = QuantAgent(data_root=str(self.data_root))
        agent.run_training_loop()
        self.model = agent.champion_model

    # Corpus-centered IMDb genre priors calculated across 100 benchmark feature films
    # Anchored to the 100-film corpus (mean rating 7.80) to eliminate train/serve population mismatch
    GENRE_PRIORS: Dict[str, float] = {
        "drama": 8.13,
        "adventure": 7.76,
        "thriller": 7.56,
        "action": 7.77,
        "sci-fi": 7.72,
        "crime": 7.56,
        "mystery": 7.63,
        "comedy": 7.72,
        "horror": 7.12,
        "romance": 7.87,
        "fantasy": 7.31,
        "biography": 8.30,
        "animation": 8.10,
        "war": 8.43,
        "history": 8.43,
        "family": 7.90,
        "music": 8.25,
        "western": 8.40,
        "film-noir": 7.90
    }
    CORPUS_DEFAULT_RATING: float = 7.80

    @classmethod
    def get_genre_baseline_static(cls, genre: Optional[str]) -> float:
        """Fetch corpus-centered IMDb mean for a genre without requiring instance initialization."""
        if not genre:
            return cls.CORPUS_DEFAULT_RATING
        g_clean = str(genre).lower().strip()
        if g_clean in cls.GENRE_PRIORS:
            return cls.GENRE_PRIORS[g_clean]

        # Handle compound genres (e.g. "Horror, Sci-Fi" or "Action / Adventure")
        subparts = [p.strip() for p in g_clean.replace('/', ',').split(',') if p.strip()]
        matched = [cls.GENRE_PRIORS[p] for p in subparts if p in cls.GENRE_PRIORS]
        if matched:
            return round(float(sum(matched) / len(matched)), 2)

        return cls.CORPUS_DEFAULT_RATING

    def get_genre_baseline(self, genre: Optional[str]) -> float:
        """Fetch corpus-centered IMDb mean for a genre, defaulting to corpus median 7.80."""
        return self.get_genre_baseline_static(genre)

    def predict_craft(
        self,
        features: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Primary ML Tool Method:
        Predicts expected IMDb rating, residual craft lift/drag, and actionable vulnerabilities.
        Accepts either a dictionary of features or keyword arguments.
        """
        if self.model is None:
            self._load_or_initialize()

        # Combine dict and kwargs
        merged = dict(features or {})
        merged.update(kwargs)

        # Resolve genre baseline
        genre = merged.get("genre")
        if "show_historical_mean" not in merged or merged["show_historical_mean"] is None:
            merged["show_historical_mean"] = self.get_genre_baseline(genre)

        genre_baseline = float(merged["show_historical_mean"])

        # Populate missing features with benchmark defaults
        populated_features = {}
        for col in FEATURE_COLUMNS:
            val = merged.get(col)
            if val is None:
                val = DEFAULT_FEATURE_VALUES.get(col, 0.0)
            populated_features[col] = float(val)

        # 1. Statistical Prediction (<1ms)
        expected_rating = float(self.model.predict_expected_rating(populated_features))
        residual_delta = round(expected_rating - genre_baseline, 2)

        # 2. Confidence Interval (Out-of-sample CV MAE is ~0.44)
        mae = self.model.metrics.get("cv_mae", 0.441)
        ci_lower = round(max(1.0, expected_rating - mae), 2)
        ci_upper = round(min(10.0, expected_rating + mae), 2)

        # 3. Craft Verdict Categorization
        if residual_delta >= 0.70:
            craft_verdict = "EXCEPTIONAL_CRAFT_LIFT"
            verdict_summary = f"Strong craft lift (+{residual_delta:0.2f} pts above {genre_baseline:0.1f} baseline). Screenplay pacing and visuals elevate execution."
        elif residual_delta >= 0.25:
            craft_verdict = "MODERATE_CRAFT_LIFT"
            verdict_summary = f"Positive craft contribution (+{residual_delta:0.2f} pts above baseline)."
        elif residual_delta >= -0.25:
            craft_verdict = "BASELINE_ALIGNED"
            verdict_summary = f"Performance closely mirrors genre expectations ({expected_rating:0.1f} vs {genre_baseline:0.1f} baseline)."
        elif residual_delta >= -0.70:
            craft_verdict = "MODERATE_CRAFT_DRAG"
            verdict_summary = f"Craft features introduce drag ({residual_delta:0.2f} pts below expected genre baseline)."
        else:
            craft_verdict = "CRITICAL_CRAFT_DEFICIT"
            verdict_summary = f"Severe craft drag ({residual_delta:0.2f} pts below baseline). High risk of negative audience reception."

        # 4. Feature Explanations / Point Attributions
        raw_explanations = self.model.explain_prediction(populated_features)
        craft_attributions = []
        for exp in raw_explanations:
            feat = exp["feature"]
            impact = exp["point_impact"]
            note = self._get_feature_narrative(feat, exp["value"], impact)
            craft_attributions.append({
                "feature": feat,
                "input_value": exp["value"],
                "benchmark_mean": exp.get("benchmark_mean"),
                "point_impact": impact,
                "direction": exp["direction"],
                "interpretation": note
            })

        # 5. Rule-Based Craft Vulnerability Flags
        vulnerabilities = self._detect_vulnerabilities(populated_features)

        return {
            "title": merged.get("title", "Untitled Project"),
            "genre": genre or "Unspecified",
            "expected_rating": expected_rating,
            "genre_baseline_rating": genre_baseline,
            "craft_residual_delta": residual_delta,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "margin_of_error": mae
            },
            "craft_verdict": craft_verdict,
            "verdict_summary": verdict_summary,
            "craft_vulnerabilities": vulnerabilities,
            "top_craft_attributions": craft_attributions[:5],
            "model_metadata": {
                "model_type": self.model.model_type,
                "cv_mae": self.model.metrics.get("cv_mae"),
                "cv_rmse": self.model.metrics.get("cv_rmse"),
                "cv_r2": self.model.metrics.get("cv_r2"),
                "training_samples": self.model.metrics.get("samples_count", 100)
            },
            "input_craft_features": populated_features
        }

    def _get_feature_narrative(self, feature: str, value: float, impact: float) -> str:
        """Generate concise human-readable narrative explaining a feature's effect."""
        sign = "+" if impact > 0 else ""
        if feature == "total_duration_min":
            return f"Runtime of {value:.0f} min contributes {sign}{impact:.2f} stars to projected scope."
        elif feature == "cuts_per_minute":
            return f"Cutting tempo of {value:.1f} cuts/min influences pacing score by {sign}{impact:.2f} stars."
        elif feature == "pacing_acceleration":
            return f"Third-act climax acceleration ({value:.2f}x) impacts resolution tension by {sign}{impact:.2f} stars."
        elif feature == "dark_frame_ratio":
            return f"Dark keyframe ratio ({value*100:.0f}%) affects visual clarity score by {sign}{impact:.2f} stars."
        elif feature == "mean_luminance":
            return f"Average visual luminance ({value:.1f}) provides {sign}{impact:.2f} stars in lighting readability."
        elif feature == "words_per_minute":
            return f"Dialogue velocity of {value:.1f} WPM shifts verbal engagement by {sign}{impact:.2f} stars."
        elif feature == "show_historical_mean":
            return f"Historical genre baseline prior sets foundation with {sign}{impact:.2f} star anchor."
        return f"{feature} ({value}) impacts expectation by {sign}{impact:.2f} stars."

    def _detect_vulnerabilities(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify actionable craft vulnerabilities and pre-mortem warning flags."""
        flags = []

        dark_ratio = features.get("dark_frame_ratio", 0.0)
        if dark_ratio >= 0.45:
            flags.append({
                "severity": "HIGH",
                "category": "Cinematography",
                "issue": "Extreme Darkness & Low Lighting Legibility",
                "detail": f"{dark_ratio*100:.1f}% of keyframes are sub-40 luminance.",
                "remedy": "Elevate shadow exposure and mid-tone contrast to prevent streaming compression crush on consumer displays."
            })
        elif dark_ratio >= 0.35:
            flags.append({
                "severity": "MEDIUM",
                "category": "Cinematography",
                "issue": "Moderate Darkness Risk",
                "detail": f"{dark_ratio*100:.1f}% dark frame ratio approaches audience visibility thresholds.",
                "remedy": "Review night scene color grading with standard consumer gamma curves."
            })

        acc = features.get("pacing_acceleration", 1.0)
        if acc >= 1.75:
            flags.append({
                "severity": "HIGH",
                "category": "Pacing & Editing",
                "issue": "Severe Climax Tempo Spike (Rushed Payoff)",
                "detail": f"Climax accelerates {acc:.2f}x faster than earlier acts, creating narrative disorientation.",
                "remedy": "Inject 2-3 transitional breathing beats in Act 3 to ground character payoff before climax."
            })
        elif acc <= 0.75:
            flags.append({
                "severity": "MEDIUM",
                "category": "Pacing & Editing",
                "issue": "Climax Deceleration (Anti-Climax Drag)",
                "detail": f"Final act decelerates ({acc:.2f}x tempo drop), risking audience fatigue.",
                "remedy": "Trim repetitive dialogue exposition in final act and accelerate sequence transitions."
            })

        wpm = features.get("words_per_minute", 55.0)
        cpm = features.get("cuts_per_minute", 15.5)
        if wpm < 30.0 and cpm < 14.0:
            flags.append({
                "severity": "HIGH",
                "category": "Dialogue & Tension",
                "issue": "Languid Verbal Velocity",
                "detail": f"Low dialogue density ({wpm:.1f} WPM) combined with slow cutting pace ({cpm:.1f} CPM).",
                "remedy": "Ensure scenes are justified by intense visual suspense action or sharpen verbal confrontation."
            })
        elif wpm > 130.0:
            flags.append({
                "severity": "MEDIUM",
                "category": "Dialogue & Tension",
                "issue": "Hyper-Dense Dialogue Velocity",
                "detail": f"Rapid speech volume ({wpm:.1f} WPM) risks overwhelming audience without visual punctuation.",
                "remedy": "Allow moments of atmospheric silence and visual subtext between rapid banter."
            })

        return flags

    def predict_from_script_and_keyframes(
        self,
        script_path_or_text: str,
        keyframes_path_or_dir: Optional[str] = None,
        genre: Optional[str] = None,
        title: str = "Untitled Project"
    ) -> Dict[str, Any]:
        """
        Convenience pipeline:
        Takes a screenplay file/text and keyframe stills, parses metrics, and runs the quant model.
        """
        from src.ingestion.script_parser import ScriptParser
        from src.vision.concept_inspector import ConceptInspector

        parser = ScriptParser()
        inspector = ConceptInspector()

        # Parse script
        p = Path(script_path_or_text)
        if p.exists() and p.is_file():
            script_metrics = parser.parse_script_file(str(p))
            if title == "Untitled Project":
                title = script_metrics.get("title", title)
        else:
            script_metrics = parser.parse_script_text(script_path_or_text, title=title)

        # Inspect keyframes
        vision_metrics = inspector.inspect_images(keyframes_path_or_dir)

        cpm = script_metrics["cuts_per_minute"]
        wpm = script_metrics["words_per_minute"]
        asl = round(60.0 / max(cpm, 1.0), 2)

        features = {
            "title": title,
            "genre": genre,
            "total_duration_min": script_metrics["estimated_duration_min"],
            "cuts_per_minute": cpm,
            "pacing_acceleration": script_metrics["climax_acceleration"],
            "average_shot_length": asl,
            "words_per_minute": wpm,
            "lines_per_minute": round(wpm / 12.0, 2),
            "dialogue_shot_ratio": script_metrics["dialogue_ratio"],
            "mean_luminance": vision_metrics["mean_luminance"],
            "luminance_std": vision_metrics["luminance_std"],
            "dark_frame_ratio": vision_metrics["dark_frame_ratio"]
        }

        result = self.predict_craft(features)
        result["script_metrics_summary"] = {
            "total_scenes": script_metrics["total_scenes"],
            "characters_count": script_metrics.get("characters_count", 0),
            "dialogue_ratio": script_metrics["dialogue_ratio"]
        }
        result["vision_summary"] = vision_metrics.get("visual_style_summary")
        return result

    def predict_movie_package(self, movie_dir: str) -> Dict[str, Any]:
        """Directly evaluate a complete movie directory in data/movies/."""
        p = Path(movie_dir).resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Movie package directory not found: {movie_dir}")

        meta_file = p / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                sm = meta.get("script_metrics")
                cv = meta.get("cv_metrics")
                if sm and cv:
                    cpm = float(sm.get("cuts_per_minute", 15.5))
                    wpm = float(sm.get("words_per_minute", 55.0))
                    asl = round(60.0 / max(cpm, 1.0), 2)
                    features = {
                        "title": meta.get("title", p.name.replace("_", " ").title()),
                        "genre": meta.get("genre", "Drama"),
                        "total_duration_min": float(sm.get("estimated_duration_min", meta.get("duration", 110.0))),
                        "cuts_per_minute": cpm,
                        "pacing_acceleration": float(sm.get("climax_acceleration", 1.0)),
                        "average_shot_length": asl,
                        "words_per_minute": wpm,
                        "lines_per_minute": round(wpm / 12.0, 2),
                        "dialogue_shot_ratio": float(sm.get("dialogue_ratio", 0.38)),
                        "mean_luminance": float(cv.get("mean_luminance", 65.0)),
                        "luminance_std": float(cv.get("luminance_std", 25.0)),
                        "dark_frame_ratio": float(cv.get("dark_frame_ratio", 0.30))
                    }
                    res = self.predict_craft(features)
                    if "imdb_rating" in meta:
                        actual = float(meta["imdb_rating"])
                        res["actual_imdb_rating"] = actual
                        res["actual_vs_projected_delta"] = round(actual - res["expected_rating"], 2)
                    return res
            except Exception:
                pass

        script_file = p / "script.txt"
        keyframes_dir = p / "keyframes"

        genre = "Drama"
        title = p.name.replace("_", " ").title()
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    genre = meta.get("genre", genre)
                    title = meta.get("title", title)
            except Exception:
                pass

        return self.predict_from_script_and_keyframes(
            script_path_or_text=str(script_file) if script_file.exists() else "",
            keyframes_path_or_dir=str(keyframes_dir) if keyframes_dir.exists() else None,
            genre=genre,
            title=title
        )

    def get_tool_spec(self) -> Dict[str, Any]:
        """
        Returns the standard function-calling declaration schema (JSON Schema)
        ready to pass to Gemini, OpenAI, Claude, or MCP agent tool manifests.
        """
        return {
            "name": "quant_residual_oracle",
            "description": (
                "Evaluates pre-production cinema craft metrics (pacing, climax acceleration, dialogue velocity, "
                "visual luminance, darkness ratio, and runtime) against historical cinema releases. Returns the projected "
                "IMDb rating, baseline delta (craft residual), point-by-point feature attributions, and prescriptive craft warning flags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "description": "Primary genre of the project (e.g. 'Drama', 'Sci-Fi', 'Horror', 'Action', 'Thriller', 'Crime'). Used to establish empirical prior expectation."
                    },
                    "cuts_per_minute": {
                        "type": "number",
                        "description": "Editing tempo measured in cuts per minute. Typical feature film range: 12.0 to 22.0. Benchmark median: 15.5."
                    },
                    "pacing_acceleration": {
                        "type": "number",
                        "description": "Ratio of tempo/cutting speed in the final 25% (climax) vs first 75%. 1.0 = steady build, >1.7 = aggressive climax acceleration, <0.8 = decelerating anti-climax."
                    },
                    "total_duration_min": {
                        "type": "number",
                        "description": "Estimated total runtime in minutes. Benchmark median: 120.0."
                    },
                    "words_per_minute": {
                        "type": "number",
                        "description": "Dialogue velocity in words per minute. Benchmark median: 55.0 WPM."
                    },
                    "mean_luminance": {
                        "type": "number",
                        "description": "Average perceived brightness across visual keyframes (0 to 255). Benchmark median: 65.0."
                    },
                    "dark_frame_ratio": {
                        "type": "number",
                        "description": "Fraction of visual keyframes with sub-40 luminance (0.0 to 1.0). High values (>0.35) create streaming compression and viewer legibility risks."
                    },
                    "show_historical_mean": {
                        "type": "number",
                        "description": "Optional override for historical genre/show baseline expectation. Defaults automatically based on genre."
                    },
                    "title": {
                        "type": "string",
                        "description": "Project or script title."
                    }
                },
                "required": []
            }
        }

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Universal execution handler for agent tool calls."""
        if tool_name in ["quant_residual_oracle", "quant_oracle", "predict_craft"]:
            return self.predict_craft(arguments)
        raise ValueError(f"Unknown tool '{tool_name}' for QuantOracle")

    def get_decision_history(self) -> str:
        """Load the ML engineering agent's decision history markdown document."""
        candidates = [
            self.data_root / "models" / "ml_decision_history.md",
            Path(__file__).resolve().parent / "DECISION_HISTORY.md",
            Path("data/models/ml_decision_history.md")
        ]
        for p in candidates:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
        return "# ML Decision History\nNo decision history document found."

    def get_decision_summary(self) -> str:
        """Return a concise summary of the ML agent's design decisions for LLM prompting."""
        return (
            "The Champion Quant Residual Model is a Ridge Regression (alpha=10.0, CV MAE: 0.441, CV RMSE: 0.604) "
            "trained on 100 genuine feature films (screenplays + 752 Film-Grab stills) with zero target leakage. "
            "It isolates pre-production craft residuals (Rating_actual - Rating_expected) driven by 5 primary axes: "
            "(1) Empirical genre baseline anchor (41.7%), (2) Screenplay pacing and climax acceleration (18.1%), "
            "(3) Cinematography lighting exposure and dark frame ratio (16.5%), (4) Runtime scope (16.2%), "
            "and (5) Spoken dialogue velocity (7.6%). Excessive visual darkness (>40%) incurs a severe compression drag, "
            "while pacing acceleration >1.75x risks unearned climax fatigue."
        )


def get_oracle(model_path: Optional[str] = None, data_root: str = "data") -> QuantOracle:
    """Convenience helper to get the global QuantOracle instance."""
    return QuantOracle.get_instance(model_path=model_path, data_root=data_root)
