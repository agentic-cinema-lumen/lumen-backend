"""Deterministic orchestrator. The LLM never touches the oracle.

Code owns every number: the features, the prediction, the risk flags and the
counterfactual sweep are arithmetic and are computed here. The two LLM
subagents own only language — the ResearchAgent decides what to investigate and
reads real pages, the SynthesisAgent writes the prose.

A prototype that exposed the oracle to a single LLM agent invented its own
baseline (`dark_frame_ratio 0.42 / pacing 1.35 / runtime 139 / rating 8.13`
against actual `0.70 / 0.901 / 181.3 / 8.21`) and computed every delta against
the fabrication. That is why this half is plain Python.

ponytail: the deterministic half is plain Python rather than an ADK
SequentialAgent. The PRD allows it explicitly, and wrapping arithmetic in an
agent shell buys nothing — ADK is used where it earns its place, on the two
LLM subagents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.agents.research_agent import ResearchAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.ingestion.script_parser import ScriptParser
from src.quant.model_trainer import DEFAULT_FEATURE_VALUES
from src.quant.oracle import get_oracle
from src.quant.sweep import SubmissionOracle, run_sweep
from src.utils.llm_client import LLMClient
from src.vision.concept_inspector import ConceptInspector

MODE = "orchestrated_premortem"

# The only craft features a concept-mode submission actually measures. Everything
# else in the vector is a `DEFAULT_FEATURE_VALUES` corpus median.
VISION_FEATURES = ("mean_luminance", "luminance_std", "dark_frame_ratio")


@dataclass
class Submission:
    story: str
    script_text: Optional[str] = None
    keyframes_dir: Optional[str] = None
    genre: Optional[str] = None
    title: str = "Untitled Submission"
    medium: str = "Feature film"
    target_geography: str = "Global streaming"


class Orchestrator:
    """Computes the numbers, then calls the ResearchAgent and the SynthesisAgent."""

    def __init__(
        self,
        research_agent: Optional[ResearchAgent] = None,
        synthesis_agent: Optional[SynthesisAgent] = None,
        concept_inspector: Optional[ConceptInspector] = None,
        oracle=None,
    ):
        self.research_agent = research_agent or ResearchAgent()
        self.synthesis_agent = synthesis_agent or SynthesisAgent()
        # ponytail: the visual slot is measurements only — ConceptInspector's LLM
        # prose is never read (it receives three scalars and never the frames,
        # which is out of scope here), so its client is mocked deliberately
        # rather than burning a Gemini call on output we discard.
        self.concept_inspector = concept_inspector or ConceptInspector(
            llm_client=LLMClient(force_mock=True)
        )
        self.parser = ScriptParser()
        self.oracle = oracle

    # ------------------------------------------------------- deterministic half

    def _features(self, sub: Submission, vision: Dict[str, Any]) -> Dict[str, Any]:
        """The submission's full feature vector. Never call the oracle with less."""
        features = dict(DEFAULT_FEATURE_VALUES)
        features.pop("show_historical_mean", None)
        features.update({
            "mean_luminance": vision["mean_luminance"],
            "luminance_std": vision["luminance_std"],
            "dark_frame_ratio": vision["dark_frame_ratio"],
        })
        if sub.script_text:
            m = self.parser.parse_script_text(sub.script_text, title=sub.title)
            cpm = m["cuts_per_minute"]
            wpm = m["words_per_minute"]
            features.update({
                "total_duration_min": m["estimated_duration_min"],
                "cuts_per_minute": cpm,
                "pacing_acceleration": m["climax_acceleration"],
                "average_shot_length": round(60.0 / max(cpm, 1.0), 2),
                "words_per_minute": wpm,
                "lines_per_minute": round(wpm / 12.0, 2),
                "dialogue_shot_ratio": m["dialogue_ratio"],
            })
            self._script_metrics = m
        else:
            self._script_metrics = None
        features["title"] = sub.title
        return features

    # ------------------------------------------------------------------- run

    def run(
        self, sub: Submission, on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        def _emit(event_type: str, data: Dict[str, Any]):
            if on_event:
                try:
                    on_event(event_type, data)
                except Exception:
                    pass

        _emit("stage", {"stage": "ingestion", "label": "Extracting screenplay metrics & concept frame measurements...", "progress": 0.15})
        vision = self.concept_inspector.inspect_images(sub.keyframes_dir)
        has_frames = vision.get("image_count", 0) > 0
        has_screenplay = bool(sub.script_text)
        features = self._features(sub, vision)

        # Risk flags come off the feature vector and need no genre, so research
        # can run before the genre is resolved. Without a screenplay every
        # non-vision feature is a corpus median, so flagging the full vector
        # would flag the median film rather than the submission.
        oracle = self.oracle or get_oracle()
        measured = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        if not has_screenplay:
            measured = {k: v for k, v in measured.items()
                        if has_frames and k in VISION_FEATURES}
        risk_flags = oracle._detect_vulnerabilities(measured)

        _emit("stage", {"stage": "research", "label": "Executing Parallel Search for audience reception & trope precedents...", "progress": 0.35})
        try:
            research = self.research_agent.run(
                premise=sub.story,
                risk_flags=risk_flags,
                genre=sub.genre,
                title=sub.title,
                on_event=on_event,
            )
        except TypeError:
            research = self.research_agent.run(
                premise=sub.story,
                risk_flags=risk_flags,
                genre=sub.genre,
                title=sub.title,
            )
        genre = sub.genre or research.get("inferred_genre")

        _emit("stage", {"stage": "quant_oracle", "label": "Calculating QuantOracle craft residual & 15-point counterfactual sweep...", "progress": 0.65})
        submission = SubmissionOracle({**features, "genre": genre}, oracle=oracle)
        prediction = submission.baseline()
        # A sweep of a median feature vector measures the median film, not this
        # submission, so concept mode has no counterfactuals to report.
        sweep = run_sweep(submission) if has_screenplay else []

        prediction_view = {
            "expected_rating": prediction["expected_rating"],
            "genre_baseline_rating": prediction["genre_baseline_rating"],
            "craft_residual_delta": prediction["craft_residual_delta"],
            "cv_mae": prediction["model_metadata"]["cv_mae"],
            "cv_r2": prediction["model_metadata"]["cv_r2"],
        }
        # The residual and the projection are craft measurements. Without a
        # screenplay they describe the corpus median, so the synthesis agent is
        # given the genre prior alone and cannot narrate them.
        synthesis_prediction = prediction_view if has_screenplay else {
            k: prediction_view[k] for k in ("genre_baseline_rating", "cv_mae", "cv_r2")
        }

        claims = research["claims"]
        synthesis_inputs = {
            "title": sub.title,
            "genre": genre,
            "logline": sub.story,
            "has_screenplay": has_screenplay,
            "has_concept_frames": has_frames,
            "script_metrics": self._script_metrics and {
                k: self._script_metrics[k] for k in (
                    "total_scenes", "characters_count", "dialogue_ratio",
                    "estimated_duration_min", "cuts_per_minute", "words_per_minute",
                    "climax_acceleration",
                )
            },
            "vision_measurements": {
                k: vision[k] for k in ("image_count", "mean_luminance", "luminance_std",
                                       "dark_frame_ratio") if k in vision
            },
            "prediction": synthesis_prediction,
            "risk_flags": risk_flags,
            "counterfactual_sweep": sweep,
            "research_claims": claims,
        }
        _emit("stage", {"stage": "synthesis", "label": "Synthesizing adversarial pre-mortem report...", "progress": 0.85})
        synthesis = self.synthesis_agent.run(synthesis_inputs)
        _emit("stage", {"stage": "finalizing", "label": "Finalizing pre-mortem report...", "progress": 0.95})

        reasons = list(research["degradation_reasons"]) + list(synthesis["degradation_reasons"])
        return {
            "mode": MODE,
            "title": sub.title,
            "genre": genre,
            "inferred_genre": research.get("inferred_genre"),
            "logline": sub.story,
            "has_screenplay": has_screenplay,
            "has_concept_frames": has_frames,
            "script_metrics": synthesis_inputs["script_metrics"],
            "vision": vision,
            "features": features,
            "prediction": prediction_view,
            "confidence_interval": prediction["confidence_interval"],
            "model_metadata": prediction["model_metadata"],
            "risk_flags": risk_flags,
            "sweep": sweep,
            "claims": claims,
            "dropped_claims": research["dropped"],
            "queries_executed": research["queries_executed"],
            "retrieved_urls": research["retrieved_urls"],
            "synthesis": synthesis["report"],
            "tool_events": research["tool_events"],
            "degraded": bool(research["degraded"] or synthesis["degraded"]),
            "degradation_reasons": reasons,
        }
