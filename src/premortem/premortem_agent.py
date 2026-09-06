"""Pre-mortem engine facade.

This module used to hold the "adversarial judge": three `if` statements over
the vision metrics, a hardcoded penalty string per branch, and an `LLMClient`
that was constructed and never called. The reconciliation between the model's
warnings and audience reality — the stated purpose of the system — did not
happen.

It is now a thin adapter over `src.agents.orchestrator.Orchestrator`, which
computes every number in plain Python and delegates language to two ADK
subagents. The adapter exists so `api.py` and the CLI share one seam that tests
can substitute.
"""

from typing import Any, Dict, Optional

from src.agents.orchestrator import Orchestrator, Submission


class PreMortemAgent:
    """Single entry point for a pre-mortem run, in script mode or concept mode."""

    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        self.orchestrator = orchestrator or Orchestrator()

    def run_premortem(
        self,
        story: str,
        script_text: Optional[str] = None,
        keyframes_dir: Optional[str] = None,
        genre: Optional[str] = None,
        title: str = "Untitled Submission",
        medium: str = "Feature film",
        target_geography: str = "Global streaming",
    ) -> Dict[str, Any]:
        """Script mode when `script_text` is given, concept mode otherwise."""
        return self.orchestrator.run(Submission(
            story=story,
            script_text=script_text,
            keyframes_dir=keyframes_dir,
            genre=genre,
            title=title,
            medium=medium,
            target_geography=target_geography,
        ))
