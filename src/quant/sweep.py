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

from typing import Dict, Any, List, Optional

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
    noise_floor = float(base["model_metadata"].get("cv_mae") or 0.498)

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
