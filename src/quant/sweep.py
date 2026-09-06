"""
🎬 DETERMINISTIC COUNTERFACTUAL SWEEP & SAFE ORACLE WRAPPER 🎬

Provides safe, deterministic sensitivity analysis for cinema craft metrics:
1. SafeOracleWrapper: structurally prevents the partial-feature sign-inversion bug
   by anchoring to the submission's real feature vector and merging single-variable overrides.
2. 15-Point Sensitivity Sweep: systematically varies the top 3 craft levers
   (dark_frame_ratio, total_duration_min, pacing_acceleration) across 5 steps each.
3. Noise-Floor Guard: compares empirical deltas against the model's out-of-sample MAE error bar
   (±0.42 to ±0.44 pts), downgrading outcome to 'inconclusive' when all craft adjustments
   remain inside the noise floor.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
from src.quant.oracle import QuantOracle


class SafeOracleWrapper:
    """
    Wraps QuantOracle to prevent evaluating partial feature sets against generic defaults.
    Ensures every probe holds the submission's actual measured features constant,
    modifying only the explicitly targeted counterfactual lever.
    """

    def __init__(self, base_features: Dict[str, Any], oracle: Optional[QuantOracle] = None):
        self.oracle = oracle or QuantOracle.get_instance()
        self.base_features = dict(base_features or {})

        # Compute baseline prediction with complete feature set
        self.baseline_result = self.oracle.predict_craft(self.base_features)
        self.baseline_rating = float(self.baseline_result["expected_rating"])
        self.genre_baseline = float(self.baseline_result["genre_baseline_rating"])
        self.craft_residual = float(self.baseline_result["craft_residual_delta"])
        self.mae = float(self.baseline_result.get("model_metadata", {}).get("cv_mae", 0.42))

    def predict_override(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single probe by copying the base features and applying overrides."""
        probe = dict(self.base_features)
        probe.update(overrides)
        return self.oracle.predict_craft(probe)

    def probe_delta(self, feature_name: str, new_value: float) -> Tuple[float, float]:
        """
        Return (new_prediction, delta_from_baseline) for modifying a single feature.
        """
        res = self.predict_override({feature_name: new_value})
        new_rating = float(res["expected_rating"])
        delta = round(new_rating - self.baseline_rating, 3)
        return new_rating, delta


def generate_probe_values(feature_name: str, baseline_val: float) -> List[float]:
    """Generate 5 target probe values for a craft feature, anchored by the baseline."""
    if feature_name == "dark_frame_ratio":
        # Target levels from bright (0.10) to dark (0.70)
        candidates = [baseline_val, 0.55, 0.40, 0.25, 0.10]
        # De-duplicate preserving order
        unique = []
        for c in candidates:
            c_round = round(c, 2)
            if c_round not in unique and 0.0 <= c_round <= 1.0:
                unique.append(c_round)
        # Ensure 5 entries
        while len(unique) < 5:
            extra = round(0.10 + len(unique) * 0.15, 2)
            if extra not in unique:
                unique.append(extra)
        return sorted(unique, reverse=True)[:5]

    elif feature_name == "total_duration_min":
        # Target levels in minutes (e.g. 180 -> 165 -> 150 -> 135 -> 120)
        candidates = [baseline_val, 165.0, 150.0, 135.0, 120.0]
        unique = []
        for c in candidates:
            c_round = round(c, 1)
            if c_round not in unique and c_round >= 60.0:
                unique.append(c_round)
        while len(unique) < 5:
            extra = round(120.0 + (len(unique) - 1) * 15.0, 1)
            if extra not in unique:
                unique.append(extra)
        return sorted(unique, reverse=True)[:5]

    elif feature_name == "pacing_acceleration":
        # Target climax acceleration levels (0.9 to 1.75)
        candidates = [baseline_val, 1.10, 1.30, 1.50, 1.75]
        unique = []
        for c in candidates:
            c_round = round(c, 3)
            if c_round not in unique and c_round > 0.1:
                unique.append(c_round)
        while len(unique) < 5:
            extra = round(0.80 + len(unique) * 0.25, 2)
            if extra not in unique:
                unique.append(extra)
        return sorted(unique)[:5]

    else:
        # Generic ±10%, ±20% perturbations
        return [
            round(baseline_val * 0.8, 2),
            round(baseline_val * 0.9, 2),
            round(baseline_val, 2),
            round(baseline_val * 1.1, 2),
            round(baseline_val * 1.2, 2),
        ]


def run_counterfactual_sweep(
    features: Dict[str, Any],
    oracle: Optional[QuantOracle] = None,
    features_to_sweep: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Execute a deterministic 15-point sensitivity sweep across the top craft levers:
    - dark_frame_ratio (5 steps)
    - total_duration_min (5 steps)
    - pacing_acceleration (5 steps)

    Returns:
      - baseline: expected rating & craft residual
      - sweep_table: structured list of rows per feature with probe values & deltas
      - noise_floor_mae: float
      - clears_noise_floor: bool
      - max_delta: float
      - outcome_recommendation: 'hit' | 'miss' | 'inconclusive'
      - formatted_table: plain-text ASCII table
    """
    wrapper = SafeOracleWrapper(features, oracle=oracle)
    mae = wrapper.mae
    baseline_rating = wrapper.baseline_rating

    if features_to_sweep is None:
        features_to_sweep = ["dark_frame_ratio", "total_duration_min", "pacing_acceleration"]

    sweep_rows = []
    all_deltas = []

    for feat in features_to_sweep:
        base_val = float(features.get(feat, wrapper.baseline_result["input_craft_features"].get(feat, 0.0)))
        probe_vals = generate_probe_values(feat, base_val)

        row = {
            "feature": feat,
            "baseline_value": base_val,
            "probes": []
        }

        for p_val in probe_vals:
            pred, delta = wrapper.probe_delta(feat, p_val)
            all_deltas.append(delta)
            clears = abs(delta) > mae
            row["probes"].append({
                "value": p_val,
                "predicted_rating": pred,
                "delta": delta,
                "clears_noise_floor": clears
            })

        sweep_rows.append(row)

    max_delta = max((abs(d) for d in all_deltas), default=0.0)
    clears_noise_floor = max_delta > mae

    # Determine confidence outcome:
    # A submission whose craft effects all fall inside the error bar returns 'inconclusive'
    if not clears_noise_floor:
        outcome = "inconclusive"
        confidence_summary = (
            f"All available craft modifications remain within the model's noise floor "
            f"(max delta {max_delta:+.2f} pts vs ±{mae:.2f} MAE error bar). "
            f"Directional tendencies (e.g. lighting correlation) hold, but specific point gains cannot be guaranteed."
        )
    else:
        if baseline_rating >= 7.0:
            outcome = "hit"
        elif baseline_rating <= 5.0:
            outcome = "miss"
        else:
            outcome = "inconclusive"

        confidence_summary = (
            f"Craft sensitivity clears the ±{mae:.2f} noise floor "
            f"(max delta {max_delta:+.2f} pts). Clear actionable leverage identified."
        )

    # Generate ASCII presentation table
    ascii_table = format_ascii_sweep_table(baseline_rating, sweep_rows)

    return {
        "baseline_rating": baseline_rating,
        "genre_baseline": wrapper.genre_baseline,
        "craft_residual_delta": wrapper.craft_residual,
        "noise_floor_mae": mae,
        "clears_noise_floor": clears_noise_floor,
        "max_delta": round(max_delta, 3),
        "outcome": outcome,
        "confidence_summary": confidence_summary,
        "sweep_table": sweep_rows,
        "formatted_table": ascii_table
    }


def format_ascii_sweep_table(baseline_rating: float, sweep_rows: List[Dict[str, Any]]) -> str:
    """Format the 15-point counterfactual table into clean text."""
    lines = [f"BASELINE = {baseline_rating:.3f}"]
    for row in sweep_rows:
        feat = row["feature"]
        cols = []
        for p in row["probes"]:
            v = p["value"]
            d = p["delta"]
            d_str = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
            if isinstance(v, float) and v >= 10:
                cols.append(f"{v:5.1f} {d_str}")
            else:
                cols.append(f"{v:4.2f} {d_str}")
        lines.append(f"{feat:<22} " + "   ".join(cols))
    return "\n".join(lines)
