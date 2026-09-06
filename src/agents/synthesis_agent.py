"""SynthesisAgent — weighs the model's warnings against what people actually said.

No tools, no arithmetic. It receives the features, the prediction, the risk
flags, the sweep table and the research claims, and writes the producer-facing
prose. It cannot fabricate a baseline it was never asked to compute, and any
number it writes that is not present in its inputs is caught by
`unsourced_numbers` and marks the run degraded rather than reaching the producer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.adk_support import agent_model, has_gemini_key, run_agent

OUTPUT_KEY = "synthesis"

SLOTS = ("story", "visual", "audience", "market")


class SlotFinding(BaseModel):
    slot: str = Field(description="One of: story, visual, audience, market")
    finding: str = Field(description="Two or three sentences for this panel.")


class Recommendation(BaseModel):
    text: str = Field(description="One action the production can take.")
    feature: str = Field(
        description="The sweep feature this rests on (dark_frame_ratio, "
                    "total_duration_min, pacing_acceleration), or 'none' when it "
                    "rests on the research rather than the model."
    )


class SynthesisReport(BaseModel):
    summary: str = Field(
        description="Lead with the direction and the evidence, not the verdict."
    )
    findings: List[SlotFinding] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)


INSTRUCTION = """You are writing a pre-mortem for a film producer who will make a real
decision on it. Everything you receive was computed or retrieved for you.

You perform NO arithmetic and you invent NO numbers. Every figure you write must appear
verbatim in the JSON you were given. If you want to say something a number would express
and the number is not in your inputs, say it in words instead.

The model behind these predictions has a cross-validated mean absolute error of roughly
half a star on a ten-point scale, so any effect smaller than that error bar is not a
finding. The sweep rows tell you which effects are inside that noise floor. Where an
effect is inside it, say so plainly and do not recommend acting on the magnitude. The
direction of an effect can still be defensible when the magnitude is not.

Your real job is reconciliation: weigh each craft warning against what the research
claims actually say. A trait the model flags as a defect may be celebrated as authorship
by the sources — when the research says so, say so, and withdraw the flag.

When `has_screenplay` is false, no screenplay was submitted, so nothing about this
project's craft was measured: there is no projected rating, no craft residual and no
counterfactual sweep, and you will not find any of them in your inputs. Do not write the
words residual or counterfactual, and do not describe a projection. The only model input
you have is the genre prior for the training corpus. Say plainly that it is a prior for
the genre and not a measurement of this submission.

Write:
- summary: lead with direction and evidence. Do not lead with a hit/miss verdict; the
  corpus this model trained on cannot support one.
- findings: one entry per slot — story (screenplay craft and the prediction), visual
  (what the frames measure), audience (what craft precedent says), market (trope fatigue
  and comparable reception). If a slot has no inputs, say that it has none.
- recommendations: each with the sweep feature it rests on, or 'none'.
"""


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def unsourced_numbers(text: str, allowed: str) -> List[str]:
    """Numbers in `text` that do not appear in the serialised deterministic inputs.

    ponytail: a substring check against the serialised inputs, not a parse. It
    catches the fabrication case (a plausible figure the agent computed itself)
    without needing to model how numbers get formatted in prose.
    """
    haystack = allowed.replace(",", "")
    bad = []
    for match in _NUMBER.findall(text or ""):
        token = match.replace(",", "")
        if token in haystack:
            continue
        # a trailing zero is formatting, not a different quantity
        if token.rstrip("0").rstrip(".") in haystack:
            continue
        bad.append(match)
    return bad


class SynthesisAgent:
    """Gemini via ADK, no tools, output_schema forces the typed report."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or agent_model()

    def build_agent(self):
        from google.adk.agents import LlmAgent

        return LlmAgent(
            name="synthesis_agent",
            model=self.model,
            description="Reconciles the craft model against the research and writes the report.",
            instruction=INSTRUCTION,
            output_schema=SynthesisReport,
            output_key=OUTPUT_KEY,
        )

    def _invoke(self, prompt: str) -> Optional[SynthesisReport]:
        outputs = run_agent(self.build_agent(), prompt, OUTPUT_KEY)
        if not outputs:
            return None
        raw = outputs[-1]
        return SynthesisReport(**raw) if isinstance(raw, dict) else raw

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Returns {report, degraded, degradation_reasons}. Never raises."""
        payload = json.dumps(inputs, indent=2, default=str)
        reasons: List[str] = []
        report: Optional[SynthesisReport] = None

        if not has_gemini_key():
            reasons.append("no GEMINI_API_KEY: the synthesis agent did not run")
        else:
            try:
                report = self._invoke(payload)
                if report is None:
                    reasons.append("synthesis agent produced no structured output")
            except Exception as exc:
                reasons.append(f"synthesis agent call failed: {type(exc).__name__}: {exc}".strip(": "))

        if report is not None:
            prose = " ".join(
                [report.summary]
                + [f.finding for f in report.findings]
                + [r.text for r in report.recommendations]
            )
            invented = unsourced_numbers(prose, payload)
            if invented:
                reasons.append(
                    "synthesis quoted numbers absent from the deterministic inputs: "
                    + ", ".join(sorted(set(invented))[:6])
                )

        return {
            "report": report.model_dump() if report else None,
            "degraded": bool(reasons),
            "degradation_reasons": reasons,
        }
