"""Report -> contract mapping. Every number here is copied, never computed anew.

The four `agentId` values are presentation slots, not a claim about how many
agents exist:

    story    ScriptParser metrics + oracle prediction    deterministic
    visual   ConceptInspector measurements               deterministic
    audience ResearchAgent — craft precedent             LLM + live search
    market   ResearchAgent — trope fatigue + comparables LLM + live search

Statistical significance is expressed inside the existing schema, because
`Evidence.sourceType` is a closed enum and the contract is authoritative:
effects inside the model's error bar say so in `Evidence.statement`, and the
inconclusive band comes from the model's measured error rather than a fixed cut.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

CRAFT_NEEDS_SCREENPLAY = "Craft analysis requires a screenplay; none was submitted."
NO_FRAMES = "No concept frames submitted, so nothing was measured."

CRAFT_PRECEDENT = "craft_precedent"
MARKET_CATEGORIES = ("trope_fatigue", "comparable_reception")


def score_from_rating(rating: float) -> int:
    return max(0, min(100, round(float(rating) * 10)))


def outcome_for(score: int, cv_mae: Optional[float]) -> str:
    """`inconclusive` whenever the uncertainty band straddles a boundary.

    92/100 corpus films rate >= 7.0 and 1/100 rates <= 5.0, so `miss` is close
    to unreachable. The band comes from cv_mae on the 0-100 scale, not from a
    fixed 50/70 cut a 0.1-star difference could flip.
    """
    margin = float(cv_mae or 0.441) * 10.0
    lo, hi = score - margin, score + margin
    for boundary in (50, 70):
        if lo <= boundary <= hi:
            return "inconclusive"
    return "hit" if score >= 70 else "miss" if score <= 50 else "inconclusive"


def confidence_for(
    claim_count: int, cv_mae: Optional[float], has_screenplay: bool, degraded: bool
) -> float:
    """From evidence volume and the model's own error, not distance to a threshold."""
    model_term = max(0.0, 1.0 - float(cv_mae or 0.441)) * 0.4
    evidence_term = min(0.35, 0.05 * claim_count)
    value = model_term + evidence_term
    if not has_screenplay:
        value *= 0.7
    if degraded:
        value *= 0.5
    return round(max(0.05, min(0.95, value)), 2)


def sweep_statement(row: Dict[str, Any], cv_mae: Optional[float]) -> str:
    """One counterfactual, with its error bar attached when it sits inside it."""
    margin = float(cv_mae or 0.441)
    head = (
        f"{row['feature']} at {row['value']}: {row['delta']:+.2f} stars "
        f"against a ±{margin:.2f} margin"
    )
    return head + (" — inside the noise floor" if row["inside_noise_floor"] else "")


def best_rows_per_feature(sweep: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The largest-magnitude counterfactual per feature, noise-floor rows first.

    ponytail: the sweep produces 15 rows and the studio renders four evidence
    cards, so the model signals lead with the one row per feature that carries
    the most movement. The full sweep stays in the report.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for row in sweep:
        current = best.get(row["feature"])
        if current is None or abs(row["delta"]) > abs(current["delta"]):
            best[row["feature"]] = row
    rows = [r for r in best.values() if r["delta"] != 0.0] or list(best.values())
    return sorted(rows, key=lambda r: (not r["inside_noise_floor"], -abs(r["delta"])))


def story_finding(report: Dict[str, Any]) -> str:
    p = report["prediction"]
    if not report["has_screenplay"]:
        return CRAFT_NEEDS_SCREENPLAY
    m = report["script_metrics"] or {}
    return (
        f"{m.get('total_scenes')} scenes over an estimated "
        f"{m.get('estimated_duration_min')} min, {m.get('words_per_minute')} words per "
        f"minute, dialogue ratio {m.get('dialogue_ratio')}, climax acceleration "
        f"{m.get('climax_acceleration')}x. Projected {p['expected_rating']}/10 against a "
        f"{p['genre_baseline_rating']}/10 genre prior, inside a ±{p['cv_mae']} error bar."
    )


def visual_finding(report: Dict[str, Any]) -> str:
    if not report["has_concept_frames"]:
        return NO_FRAMES
    v = report["vision"]
    return (
        f"{v['image_count']} frames measured: mean luminance {v['mean_luminance']}/255, "
        f"spread {v['luminance_std']}, {v['dark_frame_ratio']} of frames below 40 luminance."
    )


def slot_findings(report: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic slots from the measurements; research slots from synthesis."""
    synthesis = report.get("synthesis") or {}
    written = {f["slot"]: f["finding"] for f in synthesis.get("findings", [])}
    audience_claims = claims_for("audience", report)
    market_claims = claims_for("market", report)
    return {
        "story": story_finding(report),
        "visual": visual_finding(report),
        "audience": written.get("audience") or _claims_fallback(audience_claims,
                                                               "craft precedent"),
        "market": written.get("market") or _claims_fallback(market_claims,
                                                            "trope fatigue and comparables"),
    }


def _claims_fallback(claims: List[Dict[str, Any]], label: str) -> str:
    if not claims:
        return f"No sourced {label} research survived grounding."
    return f"{len(claims)} sourced claim(s) on {label}: " + " ".join(
        c["claim"] for c in claims[:2]
    )


def claims_for(slot: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = report.get("claims") or []
    if slot == "audience":
        return [c for c in claims if c.get("category") == CRAFT_PRECEDENT]
    return [c for c in claims if c.get("category") in MARKET_CATEGORIES]


def slot_status(slot: str, report: Dict[str, Any]) -> str:
    """`failed` when a slot's inputs were unavailable, `partial` when evidence is thin."""
    if slot == "story":
        return "complete" if report["has_screenplay"] else "failed"
    if slot == "visual":
        return "complete" if report["has_concept_frames"] else "failed"
    if report.get("degraded") and not report.get("claims"):
        return "failed"
    return "complete" if claims_for(slot, report) else "partial"


def model_signals(report: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(title, statement) pairs; noise-floor counterfactuals lead."""
    p = report["prediction"]
    cv_mae = p.get("cv_mae")
    leading = best_rows_per_feature(report.get("sweep") or [])
    signals = [("Counterfactual", sweep_statement(r, cv_mae)) for r in leading]
    signals.append((
        "Model signal",
        f"Projected {p['expected_rating']}/10 against a {p['genre_baseline_rating']}/10 "
        f"genre prior, a craft residual of {p['craft_residual_delta']:+.2f} stars, "
        f"with cross-validated MAE {cv_mae} and R² {p.get('cv_r2')}.",
    ))
    remaining = [r for r in (report.get("sweep") or []) if r not in leading]
    signals += [("Counterfactual", sweep_statement(r, cv_mae)) for r in remaining]

    sweep_by_feature = {r["feature"]: r for r in leading}
    for rec in (report.get("synthesis") or {}).get("recommendations", []):
        row = sweep_by_feature.get(rec.get("feature"))
        tag = ""
        if row is not None:
            tag = (" — the sweep row this rests on is inside the noise floor"
                   if row["inside_noise_floor"]
                   else " — the sweep row this rests on clears the noise floor")
        signals.append(("Recommendation", rec["text"] + tag))
    return signals


def degradation_notice(report: Dict[str, Any]) -> str:
    reasons = report.get("degradation_reasons") or ["unspecified"]
    return "DEGRADED RUN — " + "; ".join(reasons) + ". "


def summary_for(report: Dict[str, Any]) -> str:
    """Leads with direction and evidence. The verdict is not the headline."""
    synthesis = report.get("synthesis") or {}
    if synthesis.get("summary"):
        body = synthesis["summary"]
    else:
        p = report["prediction"]
        body = (
            f"{len(report.get('claims') or [])} sourced research claim(s); model projects "
            f"{p['expected_rating']}/10 against a {p['genre_baseline_rating']}/10 genre "
            f"prior, within its own ±{p['cv_mae']} error bar."
        )
    return (degradation_notice(report) if report.get("degraded") else "") + body
