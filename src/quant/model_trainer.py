"""
🎬 QUANT MODEL TRAINER & RESIDUAL CALCULATOR 🎬

Trains classic ML regression models (Ridge, Random Forest, Gradient Boosting)
on cinematographic & structural features to predict expected IMDb ratings and
calculate residual craft variance.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Pure Pre-Production Features (Strictly observable from Script + Keyframes + Genre Baseline)
# Note: Post-release outcome variables (like log_votes or broadcast positioning) are excluded to prevent target leakage.
FEATURE_COLUMNS = [
    # Script Pacing & Structure
    "total_duration_min",
    "cuts_per_minute",
    "pacing_acceleration",
    "average_shot_length",
    # Script Dialogue Dynamics
    "words_per_minute",
    "lines_per_minute",
    "dialogue_shot_ratio",
    # Keyframe Visual Cinematography
    "mean_luminance",
    "luminance_std",
    "dark_frame_ratio",
    # Prior Genre Expectation (from screenplay pitch / genre)
    "show_historical_mean"
]

# Benchmark cinema median defaults (from 100 genuine feature film packages)
DEFAULT_FEATURE_VALUES: Dict[str, float] = {
    "total_duration_min": 120.0,
    "cuts_per_minute": 15.5,
    "pacing_acceleration": 1.05,
    "average_shot_length": 3.87,
    "words_per_minute": 55.0,
    "lines_per_minute": 4.5,
    "dialogue_shot_ratio": 0.38,
    "mean_luminance": 65.0,
    "luminance_std": 30.0,
    "dark_frame_ratio": 0.30,
    "show_historical_mean": 7.80
}


def train_champion(
    data_root: str = "data",
    model_type: str = "ridge",
    cv_splits: int = 5
) -> Dict[str, float]:
    """Retrain and persist data/models/champion_model.joblib. Returns CV metrics.

    ponytail: model_type is pinned to ridge rather than taken from the tournament
    winner. With the recentred genre prior random_forest scores a better cv_r2
    (0.245 vs -0.016), but explain_prediction() only produces real per-feature
    attributions for ridge — the tree branch returns a flat importance stand-in,
    which is what the product surfaces as "craft attributions". Switching
    architecture is a separate change from recentring the prior.
    """
    from src.quant.benchmark_dataset import get_full_training_dataset

    df = get_full_training_dataset(data_root)
    model = QuantResidualModel(model_type=model_type)
    metrics = model.train_and_evaluate(df, cv_splits=cv_splits)
    model.save(str(Path(data_root) / "models" / "champion_model.joblib"))
    return metrics


class QuantResidualModel:
    """End-to-end ML model for expected rating prediction and residual anomaly detection."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.feature_names = FEATURE_COLUMNS
        self.pipeline: Optional[Pipeline] = None
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.metrics: Dict[str, float] = {}
        self.feature_importances: List[Dict[str, Any]] = []

    def _build_estimator(self):
        if self.model_type == "ridge":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10.0, random_state=42))
            ])
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=3,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'")

    def train_and_evaluate(self, df: pd.DataFrame, cv_splits: int = 5) -> Dict[str, Any]:
        """Train the model on the DataFrame and evaluate using cross-validation."""
        X = df[self.feature_names].copy().fillna(0.0)
        y = df["imdb_rating"].values

        estimator = self._build_estimator()

        # Cross-validation
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        cv_res = cross_validate(
            estimator, X, y, cv=kf,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            return_train_score=False
        )

        rmse = float(-np.mean(cv_res["test_neg_root_mean_squared_error"]))
        mae = float(-np.mean(cv_res["test_neg_mean_absolute_error"]))
        r2 = float(np.mean(cv_res["test_r2"]))

        # Fit final model on full dataset
        estimator.fit(X, y)
        self.pipeline = estimator

        # Compute full-fit metrics
        y_pred = estimator.predict(X)
        full_rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        full_r2 = float(r2_score(y, y_pred))

        self.metrics = {
            "cv_rmse": round(rmse, 3),
            "cv_mae": round(mae, 3),
            "cv_r2": round(r2, 3),
            "fit_rmse": round(full_rmse, 3),
            "fit_r2": round(full_r2, 3),
            "samples_count": len(df)
        }

        # Extract Feature Importances
        self._compute_feature_importances(estimator, X)

        return self.metrics

    def _compute_feature_importances(self, estimator: Any, X: pd.DataFrame):
        """Extract sorted feature importances / coefficients."""
        importances = []
        if self.model_type == "ridge":
            coefs = estimator.named_steps["regressor"].coef_
            for name, coef in zip(self.feature_names, coefs):
                importances.append({
                    "feature": name,
                    "importance": round(float(abs(coef)), 4),
                    "direction": "positive" if coef > 0 else "negative",
                    "weight": round(float(coef), 4)
                })
        elif hasattr(estimator, "feature_importances_"):
            # Random Forest
            raw_imp = estimator.feature_importances_
            for name, imp in zip(self.feature_names, raw_imp):
                importances.append({
                    "feature": name,
                    "importance": round(float(imp), 4),
                    "direction": "neutral",
                    "weight": round(float(imp), 4)
                })
        else:
            # Fallback for HistGradientBoosting
            importances = [{"feature": f, "importance": 0.05, "direction": "neutral", "weight": 0.05} for f in self.feature_names]

        # Sort descending by importance
        importances.sort(key=lambda x: x["importance"], reverse=True)
        self.feature_importances = importances

    def predict_expected_rating(self, features: Dict[str, Any]) -> float:
        """Predict expected IMDb baseline rating from an episode feature dict."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call train_and_evaluate first.")

        row = []
        for f in self.feature_names:
            val = features.get(f)
            if val is None:
                val = DEFAULT_FEATURE_VALUES.get(f, 0.0)
            row.append(float(val))

        X_input = pd.DataFrame([row], columns=self.feature_names)
        pred = float(self.pipeline.predict(X_input)[0])
        return round(float(np.clip(pred, 1.0, 10.0)), 2)

    def explain_prediction(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Explain the prediction by calculating each feature's point contribution (delta)
        relative to the cinema average.
        """
        if self.pipeline is None:
            return []

        row = []
        for f in self.feature_names:
            val = features.get(f)
            if val is None:
                val = DEFAULT_FEATURE_VALUES.get(f, 0.0)
            row.append(float(val))

        explanations = []
        if self.model_type == "ridge" and hasattr(self.pipeline, "named_steps"):
            scaler = self.pipeline.named_steps.get("scaler")
            regressor = self.pipeline.named_steps.get("regressor")
            if scaler is not None and regressor is not None:
                means = scaler.mean_
                scales = scaler.scale_
                coefs = regressor.coef_

                for i, f in enumerate(self.feature_names):
                    val = row[i]
                    z_score = (val - means[i]) / (scales[i] + 1e-9)
                    impact = round(float(z_score * coefs[i]), 3)
                    explanations.append({
                        "feature": f,
                        "value": round(val, 2),
                        "benchmark_mean": round(float(means[i]), 2),
                        "point_impact": impact,
                        "direction": "positive" if impact > 0 else ("negative" if impact < 0 else "neutral")
                    })
        else:
            # Fallback for tree-based models using general importance weights
            pred = self.predict_expected_rating(features)
            for imp in self.feature_importances:
                explanations.append({
                    "feature": imp["feature"],
                    "value": round(float(features.get(imp["feature"], DEFAULT_FEATURE_VALUES.get(imp["feature"], 0.0))), 2),
                    "point_impact": round(imp["weight"] * 0.1, 3),
                    "direction": imp["direction"]
                })

        # Sort by absolute impact descending
        explanations.sort(key=lambda x: abs(x.get("point_impact", 0.0)), reverse=True)
        return explanations

    def calculate_residual(
        self,
        actual_rating: float,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute rating residual = Actual - Expected, and classify anomaly status.
        """
        expected_rating = self.predict_expected_rating(features)
        residual = round(actual_rating - expected_rating, 2)

        # Anomaly Classification
        if residual <= -1.5:
            anomaly_type = "SEVERE_CRAFT_DEFICIT"
            description = "Severe underperformance relative to historical baseline. High likelihood of audience backlash, review bombing, or major craft/visual failure."
        elif residual <= -0.5:
            anomaly_type = "MILD_UNDERPERFORMANCE"
            description = "Noticeable dip below expectation. Pacing lag or script disconnect."
        elif residual >= 1.2:
            anomaly_type = "HISTORIC_MASTERPIECE"
            description = "Historic overperformance. Transcends series baseline expectation."
        elif residual >= 0.5:
            anomaly_type = "MILD_OUTPERFORMANCE"
            description = "Solid overperformance above series average."
        else:
            anomaly_type = "EXPECTED_BASELINE"
            description = "Performance aligns directly with structural and craft expectation."

        return {
            "actual_rating": round(actual_rating, 2),
            "expected_rating": round(expected_rating, 2),
            "residual": round(residual, 2),
            "anomaly_type": anomaly_type,
            "anomaly_severity": round(abs(residual), 2),
            "description": description
        }

    def save(self, file_path: str) -> None:
        """Serialize trained model and metadata to disk."""
        import joblib
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances
        }, p)

    @classmethod
    def load(cls, file_path: str) -> "QuantResidualModel":
        """Load serialized model from disk."""
        import joblib
        p = Path(file_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        data = joblib.load(p)
        inst = cls(model_type=data["model_type"])
        inst.feature_names = data["feature_names"]
        inst.pipeline = data["pipeline"]
        inst.metrics = data["metrics"]
        inst.feature_importances = data["feature_importances"]
        return inst
