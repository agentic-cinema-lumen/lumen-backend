"""
🎬 SUBMISSION ORACLE WRAPPER & DETERMINISTIC COUNTERFACTUAL SWEEP 🎬

`QuantOracle.predict_craft` substitutes benchmark medians for any omitted
feature, so a partial call evaluates a *different film*. On Fight Club,
`predict_craft(dark_frame_ratio=0.45)` returns 8.12 while the correct value is
8.23 — an error that inverts the sign of the recommendation.

`SubmissionOracle` holds the submission's full feature vector and merges
overrides onto it. All oracle access goes through it; nothing else calls
`predict_craft` with partial features.
"""

from typing import Dict, Any, List, Optional, Tuple

from src.quant.oracle import QuantOracle, get_oracle


# Counterfactual grid: absolute target values per feature. The submission's own
# value is prepended, so each feature yields 5 points with the first at delta 0.
# ponytail: only the three features the PRD names have a defined range; the rest
# are left out rather than invented.
SWEEP_RANGES: Dict[str, List[float]] = {
    "dark_frame_ratio": [0.55, 0.40, 0.25, 0.10],
    "total_duration_min": [165.0, 150.0, 135.0, 120.0],
    "pacing_acceleration": [1.1, 1.3, 1.5, 1.75],
}


class SubmissionOracle:
    """Holds one submission's full feature vector; the only way to reach the oracle."""

    def __init__(self, features: Dict[str, Any], oracle: Optional[QuantOracle] = None):
        self.features = dict(features)
        self.oracle = oracle or get_oracle()

    @classmethod
    def from_movie_package(cls, movie_dir: str, oracle: Optional[QuantOracle] = None) -> "SubmissionOracle":
        oracle = oracle or get_oracle()
        features = oracle.features_for_movie_package(movie_dir)
        if features is None:
            raise ValueError(f"No usable feature vector in movie package: {movie_dir}")
        return cls(features, oracle=oracle)

    def predict(self, **overrides: Any) -> Dict[str, Any]:
        """Merge overrides onto the submission's full feature vector and predict."""
        return self.oracle.predict_craft({**self.features, **overrides})

    def baseline(self) -> Dict[str, Any]:
        return self.predict()


def run_sweep(submission: SubmissionOracle) -> List[Dict[str, Any]]:
    """Deterministic single-variable counterfactual sweep.

    Returns rows of {feature, value, delta, inside_noise_floor}, where
    inside_noise_floor is abs(delta) < the model's cross-validated MAE.
    """
    base = submission.baseline()
    base_rating = base["expected_rating"]
    noise_floor = float(base["model_metadata"].get("cv_mae") or 0.435)

    rows: List[Dict[str, Any]] = []
    for feature, grid in SWEEP_RANGES.items():
        own = submission.features.get(feature)
        values = ([float(own)] if own is not None else []) + [v for v in grid if own is None or float(v) != float(own)]
        for value in values[:5]:
            delta = round(submission.predict(**{feature: value})["expected_rating"] - base_rating, 3)
            rows.append({
                "feature": feature,
                "value": value,
                "delta": delta,
                "inside_noise_floor": abs(delta) < noise_floor,
            })
    return rows


# ---------------------------------------------------------------------------
# Compatibility surface carried over from main (PR #2).
#
# `SafeOracleWrapper`, `generate_probe_values`, `run_counterfactual_sweep` and
# `format_ascii_sweep_table` keep their original names, signatures and return
# keys, because `scripts/run_model.py --sweep` and `tests/test_quant_sweep.py`
# call them. They are adapters over `SubmissionOracle` above, so there is one
# wrapper holding the submission's full feature vector, not two.
# ---------------------------------------------------------------------------


class SafeOracleWrapper(SubmissionOracle):
    """`SubmissionOracle` under its main-branch name, with the baseline precomputed."""

    def __init__(self, base_features: Dict[str, Any], oracle: Optional[QuantOracle] = None):
        super().__init__(base_features or {}, oracle=oracle)
        self.base_features = self.features
        self.baseline_result = self.baseline()
        self.baseline_rating = float(self.baseline_result["expected_rating"])
        self.genre_baseline = float(self.baseline_result["genre_baseline_rating"])
        self.craft_residual = float(self.baseline_result["craft_residual_delta"])
        self.mae = float(self.baseline_result.get("model_metadata", {}).get("cv_mae") or 0.435)

    def predict_override(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a probe by merging overrides onto the submission's own features."""
        return self.predict(**(overrides or {}))

    def probe_delta(self, feature_name: str, new_value: float) -> Tuple[float, float]:
        """Return (new_prediction, delta_from_baseline) for one changed feature."""
        new_rating = float(self.predict_override({feature_name: new_value})["expected_rating"])
        return new_rating, round(new_rating - self.baseline_rating, 3)


def generate_probe_values(feature_name: str, baseline_val: float) -> List[float]:
    """Generate 5 target probe values for a craft feature, anchored by the baseline."""
    if feature_name == "dark_frame_ratio":
        candidates = [baseline_val] + SWEEP_RANGES["dark_frame_ratio"]
        unique: List[float] = []
        for c in candidates:
            c_round = round(c, 2)
            if c_round not in unique and 0.0 <= c_round <= 1.0:
                unique.append(c_round)
        while len(unique) < 5:
            extra = round(0.10 + len(unique) * 0.15, 2)
            if extra not in unique:
                unique.append(extra)
        return sorted(unique, reverse=True)[:5]

    if feature_name == "total_duration_min":
        candidates = [baseline_val] + SWEEP_RANGES["total_duration_min"]
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

    if feature_name == "pacing_acceleration":
        candidates = [baseline_val] + SWEEP_RANGES["pacing_acceleration"]
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

    # Generic +/-10%, +/-20% perturbations.
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
    """Deterministic 15-point sensitivity sweep across the three craft levers.

    Same return keys as the main-branch implementation: baseline_rating,
    genre_baseline, craft_residual_delta, noise_floor_mae, clears_noise_floor,
    max_delta, outcome, confidence_summary, sweep_table, formatted_table.
    """
    wrapper = SafeOracleWrapper(features, oracle=oracle)
    mae = wrapper.mae
    baseline_rating = wrapper.baseline_rating

    if features_to_sweep is None:
        features_to_sweep = list(SWEEP_RANGES.keys())

    sweep_rows: List[Dict[str, Any]] = []
    all_deltas: List[float] = []

    for feat in features_to_sweep:
        base_val = float(
            features.get(feat, wrapper.baseline_result["input_craft_features"].get(feat, 0.0))
        )
        row: Dict[str, Any] = {"feature": feat, "baseline_value": base_val, "probes": []}

        for p_val in generate_probe_values(feat, base_val):
            pred, delta = wrapper.probe_delta(feat, p_val)
            all_deltas.append(delta)
            row["probes"].append({
                "value": p_val,
                "predicted_rating": pred,
                "delta": delta,
                "clears_noise_floor": abs(delta) > mae,
            })

        sweep_rows.append(row)

    max_delta = max((abs(d) for d in all_deltas), default=0.0)
    clears_noise_floor = max_delta > mae

    if not clears_noise_floor:
        outcome = "inconclusive"
        confidence_summary = (
            f"All available craft modifications remain within the model's noise floor "
            f"(max delta {max_delta:+.2f} pts vs ±{mae:.2f} MAE error bar). "
            f"Directional tendencies (e.g. lighting correlation) hold, but specific "
            f"point gains cannot be guaranteed."
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
        "formatted_table": format_ascii_sweep_table(baseline_rating, sweep_rows),
    }


def format_ascii_sweep_table(baseline_rating: float, sweep_rows: List[Dict[str, Any]]) -> str:
    """Format the 15-point counterfactual table into clean text."""
    lines = [f"BASELINE = {baseline_rating:.3f}"]
    for row in sweep_rows:
        cols = []
        for p in row["probes"]:
            v, d = p["value"], p["delta"]
            d_str = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
            if isinstance(v, float) and v >= 10:
                cols.append(f"{v:5.1f} {d_str}")
            else:
                cols.append(f"{v:4.2f} {d_str}")
        lines.append(f"{row['feature']:<22} " + "   ".join(cols))
    return "\n".join(lines)
