"""
🎬 PRE-MORTEM CINEMA ENGINE & ADVERSARIAL JUDGE 🎬

Orchestrates the end-to-end Pre-Mortem evaluation:
1. Ingests Script or Pitch Logline + Keyframes
2. Extracts numerical craft metrics (Pacing, Dialogue, Vision)
3. Computes statistical expected ratings via Quant Residual Model
4. Deploys Parallel Search Agent to investigate live audience consensus, tropes, and fatigue
5. Synthesizes an Adversarial Pre-Mortem Report & Greenlight Compass
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from src.ingestion.script_parser import ScriptParser, parse_premise_logline
from src.vision.concept_inspector import ConceptInspector
from src.search.trope_sleuth import TropeSleuth
from src.quant.quant_agent import QuantAgent
from src.utils.llm_client import LLMClient


class PreMortemAgent:
    """Autonomous cinema pre-mortem investigator and greenlight judge."""

    def __init__(
        self,
        quant_agent: Optional[QuantAgent] = None,
        trope_sleuth: Optional[TropeSleuth] = None,
        concept_inspector: Optional[ConceptInspector] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.script_parser = ScriptParser()
        self.concept_inspector = concept_inspector or ConceptInspector()
        self.trope_sleuth = trope_sleuth or TropeSleuth()
        self.quant_agent = quant_agent or QuantAgent()
        self.llm = llm_client or LLMClient()

    def run_script_premortem(
        self,
        script_path_or_text: str,
        keyframes_path_or_dir: Optional[str] = None,
        title: str = "Untitled Script",
        show_name: str = "Prestige Series",
        show_historical_mean: float = 8.5,
        season_position: float = 0.8,
        is_finale: bool = False
    ) -> Dict[str, Any]:
        """
        Mode 1: Evaluate a full script (or draft) + keyframe visuals.
        Predicts reception, residual risk, and cross-examines against audience consensus.
        """
        # 1. Parse screenplay
        p = Path(script_path_or_text)
        if p.exists() and p.is_file():
            script_metrics = self.script_parser.parse_script_file(str(p))
            if title == "Untitled Script":
                title = script_metrics["title"]
        else:
            script_metrics = self.script_parser.parse_script_text(script_path_or_text, title=title)

        # 2. Inspect keyframes
        vision_metrics = self.concept_inspector.inspect_images(keyframes_path_or_dir)

        # 3. Assemble complete feature vector for Quant residual model
        # Calculate derived metrics matching feature_extractor.py
        total_shots = script_metrics["estimated_total_shots"]
        duration_min = script_metrics["estimated_duration_min"]
        cuts_per_minute = script_metrics["cuts_per_minute"]
        wpm = script_metrics["words_per_minute"]
        asl = round(60.0 / max(cuts_per_minute, 1.0), 2)
        climax_acc = script_metrics["climax_acceleration"]

        combined_features = {
            "title": title,
            "show_name": show_name,
            "average_shot_length": asl,
            "cuts_per_minute": cuts_per_minute,
            "pacing_acceleration": climax_acc,
            "words_per_minute": wpm,
            "dialogue_shot_ratio": script_metrics["dialogue_ratio"],
            "lines_per_minute": round(wpm / 12.0, 2),
            "silence_ratio": round(max(0.05, 1.0 - script_metrics["dialogue_ratio"]), 3),
            "max_silence_streak_sec": 45.0,
            "mean_luminance": vision_metrics["mean_luminance"],
            "luminance_std": vision_metrics["luminance_std"],
            "dark_frame_ratio": vision_metrics["dark_frame_ratio"],
            "show_historical_mean": show_historical_mean,
            "season_position": season_position,
            "is_premiere": 1.0 if season_position <= 0.1 else 0.0,
            "is_finale": 1.0 if is_finale or season_position >= 0.9 else 0.0,
            "log_votes": 10.5
        }

        # 4. Quant Residual Prediction
        if not self.quant_agent.champion_model:
            self.quant_agent.run_training_loop()

        expected_rating = float(self.quant_agent.champion_model.predict_expected_rating(combined_features))
        
        # Calculate craft impact breakdown
        top_importances = self.quant_agent.champion_model.feature_importances[:5]

        # 5. Parallel Search Audience Sleuth
        search_results = self.trope_sleuth.investigate_script_craft(script_metrics, vision_metrics)

        # 6. Synthesize Adversarial Pre-Mortem Verdict
        craft_flaws = []
        recommendations = []

        if vision_metrics["dark_frame_ratio"] >= 0.35:
            craft_flaws.append({
                "category": "Cinematography",
                "severity": "HIGH",
                "finding": f"Excessive darkness: {vision_metrics['dark_frame_ratio']*100:.1f}% of keyframes are sub-40 luminance.",
                "predicted_penalty": "-0.6 to -1.2 rating drag on consumer displays."
            })
            recommendations.append("Elevate shadow detail exposure and mid-tone contrast to prevent streaming compression crush.")

        if climax_acc >= 1.7:
            craft_flaws.append({
                "category": "Pacing",
                "severity": "MODERATE",
                "finding": f"Hyper-accelerated climax tempo ({climax_acc}x faster than early acts).",
                "predicted_penalty": "Viewer disorientation and feeling of unearned, rushed resolution."
            })
            recommendations.append("Add 2-3 transitional character breathing beats in Act 3 to ground narrative payoff.")
        elif climax_acc <= 0.8:
            craft_flaws.append({
                "category": "Pacing",
                "severity": "MODERATE",
                "finding": f"Climax deceleration ({climax_acc}x tempo drop in final act).",
                "predicted_penalty": "Anticlimactic fatigue; episode feels like it drags to an inconclusive stop."
            })
            recommendations.append("Tighten late-stage scene transitions and trim repetitive dialogue blocks before the final twist.")

        if wpm < 80.0 and cuts_per_minute < 14.0:
            craft_flaws.append({
                "category": "Dialogue & Tension",
                "severity": "HIGH",
                "finding": f"Critically low dialogue velocity ({wpm:.1f} WPM) with languid cutting tempo.",
                "predicted_penalty": "High risk of audience boredom unless justified by intense suspense action."
            })
            recommendations.append("Inject sharper subtext or quicken verbal exchanges between key protagonists.")

        report = {
            "mode": "script_premortem",
            "title": title,
            "show_name": show_name,
            "projected_baseline_rating": round(expected_rating, 2),
            "historical_show_mean": show_historical_mean,
            "projected_residual_delta": round(expected_rating - show_historical_mean, 2),
            "craft_metrics": {
                "estimated_runtime_min": script_metrics["estimated_duration_min"],
                "total_scenes": script_metrics["total_scenes"],
                "cuts_per_minute": cuts_per_minute,
                "words_per_minute": wpm,
                "climax_acceleration": climax_acc,
                "mean_luminance": vision_metrics["mean_luminance"],
                "dark_frame_ratio": vision_metrics["dark_frame_ratio"]
            },
            "vision_summary": vision_metrics["visual_style_summary"],
            "detected_craft_flaws": craft_flaws,
            "audience_trope_intelligence": {
                "queries_executed": search_results["queries_executed"],
                "audience_consensus_claims": search_results["claims"][:4]
            },
            "recommendations": recommendations
        }

        return report

    def run_premise_premortem(
        self,
        logline: str,
        keyframes_path_or_dir: Optional[str] = None,
        title: str = "Untitled Pitch",
        genre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mode 2: Evaluate a pitch logline (2-4 sentences) + keyframe moodboard.
        Produces the 'Greenlight Compass' identifying audience traps and craft dependencies.
        """
        # 1. Parse logline
        premise_info = parse_premise_logline(logline, genre=genre)

        # 2. Inspect moodboard keyframes
        vision_metrics = self.concept_inspector.inspect_images(keyframes_path_or_dir)

        # 3. Parallel Search Market Sleuth
        search_results = self.trope_sleuth.investigate_premise_market(premise_info, vision_metrics)

        # 4. Fetch Real Comparable Movie Precedents from IMDb dataset (if available)
        comparables = []
        genre_mean = 6.8
        try:
            from src.quant.movie_dataset_loader import MovieDatasetLoader
            loader = MovieDatasetLoader.get_instance()
            if loader.df is not None:
                comparables = loader.find_comparable_movies(logline, genre=premise_info["genre"], top_k=3)
                genre_mean = loader.get_genre_expectation(premise_info["genre"])
        except Exception:
            pass

        # 5. Synthesize Greenlight Compass
        expected_pacing = premise_info["expected_pacing"]
        
        # Cohesion check: does visual style fit premise tone?
        dark_ratio = vision_metrics["dark_frame_ratio"]
        is_dark_premise = "dark" in premise_info["tone"].lower() or "claustrophobic" in premise_info["tone"].lower()

        if is_dark_premise and dark_ratio >= 0.25:
            cohesion_rating = "HIGH_HARMONY"
            cohesion_note = "Visual atmosphere closely mirrors the psychological weight of the premise."
        elif not is_dark_premise and dark_ratio >= 0.40:
            cohesion_rating = "TONAL_CLASH"
            cohesion_note = "Visual darkness is excessively heavy for a drama requiring narrative clarity."
        else:
            cohesion_rating = "BALANCED"
            cohesion_note = "Solid baseline aesthetic alignment."

        # Dependencies
        dependencies = [
            f"Requires dialogue velocity > {expected_pacing['words_per_minute']:.0f} WPM to maintain dramatic tension.",
            "Visual composition must prioritize character face readability over shadow obscurity.",
            "Avoid common third-act trope fatigue flagged in community audience discussions."
        ]

        report = {
            "mode": "premise_greenlight_compass",
            "title": title,
            "logline": logline,
            "genre": premise_info["genre"],
            "detected_tone": premise_info["tone"],
            "empirical_genre_baseline": genre_mean,
            "projected_rating_potential": {
                "ceiling": round(min(9.2, genre_mean + 1.4), 1),
                "floor": round(max(4.5, genre_mean - 1.2), 1),
                "median_target": round(genre_mean, 1)
            },
            "style_premise_cohesion": {
                "rating": cohesion_rating,
                "evaluation": cohesion_note,
                "mean_luminance": vision_metrics["mean_luminance"],
                "dark_frame_ratio": vision_metrics["dark_frame_ratio"]
            },
            "comparable_movie_precedents": comparables,
            "make_or_break_dependencies": dependencies,
            "audience_fatigue_radar": {
                "queries_executed": search_results["queries_executed"],
                "audience_consensus_claims": search_results["claims"][:4]
            },
            "greenlight_verdict": (
                f"GREENLIGHT CONDITIONAL: High potential in {premise_info['genre']} (empirical baseline {genre_mean:.1f}/10), "
                f"provided screenwriting adheres to high-tempo verbal confrontation to offset claustrophobic visual styling."
            )
        }

        return report
