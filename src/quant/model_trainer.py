"""
🎬 QUANT MODEL TRAINER & RESIDUAL CALCULATOR 🎬

Trains classic ML regression models (Ridge, Random Forest, Gradient Boosting)
on cinematographic & structural features to predict expected IMDb ratings and
calculate residual craft variance.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


FEATURE_COLUMNS = [
    # Pacing
    "average_shot_length",
    "median_shot_length",
    "shot_length_std",
    "cuts_per_minute",
    "pacing_acceleration",
    # Dialogue & Script
    "words_per_minute",
    "lines_per_minute",
    "dialogue_shot_ratio",
    "max_silence_sec",
    # Visual
    "mean_luminance",
    "luminance_std",
    "dark_frame_ratio",
    # Structural Context
    "season_position",
    "is_premiere",
    "is_finale",
    "is_penultimate",
    "show_historical_mean",
    "log_votes"
]


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
            row.append(float(features.get(f, 0.0)))

        X_input = pd.DataFrame([row], columns=self.feature_names)
        pred = float(self.pipeline.predict(X_input)[0])
        return round(float(np.clip(pred, 1.0, 10.0)), 2)

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
            "expected_rating": expected_rating,
            "residual": residual,
            "anomaly_type": anomaly_type,
            "anomaly_severity": round(abs(residual), 2),
            "description": description
        }
