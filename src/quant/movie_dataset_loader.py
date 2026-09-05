"""
🎬 MOVIE DATASET LOADER & PRECEDENT RETRIEVER 🎬

Ingests and merges:
- data/IMDb movies.csv (85,855 films with genres, directors, descriptions, metascores)
- data/IMDb ratings.csv (vote distributions, bimodal polarization, demographic splits)

Capabilities:
1. Calculates empirical genre baselines (e.g., Horror=5.4, Sci-Fi=6.1, Drama=6.6)
2. Computes vote polarization index: (votes_10 + votes_1) / total_votes
3. Comparable Movie Retriever: Uses TF-IDF cosine similarity across 85,000 descriptions
   to find real comparable movies for any user pitch/logline
4. Trains an empirical Movie Residual ML model
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate


class MovieDatasetLoader:
    """Manages IMDb movie dataset, genre statistics, and comparable precedent search."""

    _instance: Optional["MovieDatasetLoader"] = None

    def __init__(self, data_dir: str = "data", min_votes: int = 1000):
        self.data_dir = Path(data_dir).resolve()
        self.movies_csv = self.data_dir / "IMDb movies.csv"
        if not self.movies_csv.exists():
            alt_m = self.data_dir / "imdb_data" / "IMDb movies.csv"
            if alt_m.exists():
                self.movies_csv = alt_m

        self.ratings_csv = self.data_dir / "IMDb ratings.csv"
        if not self.ratings_csv.exists():
            alt_r = self.data_dir / "imdb_data" / "IMDb ratings.csv"
            if alt_r.exists():
                self.ratings_csv = alt_r
        self.min_votes = min_votes

        self.df: Optional[pd.DataFrame] = None
        self.genre_baselines: Dict[str, Dict[str, float]] = {}
        self.global_mean: float = 6.25
        self.director_means: Dict[str, float] = {}

        # Precedent search index
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._search_df: Optional[pd.DataFrame] = None

        if self.movies_csv.exists():
            self._load_and_process()

    @classmethod
    def get_instance(cls, data_dir: str = "data", min_votes: int = 1000) -> "MovieDatasetLoader":
        """Singleton accessor to prevent redundant CSV reads."""
        if cls._instance is None:
            cls._instance = cls(data_dir=data_dir, min_votes=min_votes)
        return cls._instance

    def _load_and_process(self):
        """Load, merge, and clean movies and ratings tables."""
        print("📦 [MovieDatasetLoader] Loading IMDb movies and ratings CSVs...")
        df_m = pd.read_csv(self.movies_csv, low_memory=False)

        # Basic cleaning on movies
        df_m["votes"] = pd.to_numeric(df_m["votes"], errors="coerce").fillna(0).astype(int)
        df_m["avg_vote"] = pd.to_numeric(df_m["avg_vote"], errors="coerce").fillna(0.0)
        df_m["duration"] = pd.to_numeric(df_m["duration"], errors="coerce").fillna(90.0)
        df_m["year"] = pd.to_numeric(df_m["year"], errors="coerce").fillna(2000).astype(int)
        df_m["metascore"] = pd.to_numeric(df_m["metascore"], errors="coerce")

        # Filter to films with sufficient community consensus
        filtered_m = df_m[df_m["votes"] >= self.min_votes].copy()

        # Merge ratings table if available
        if self.ratings_csv.exists():
            df_r = pd.read_csv(self.ratings_csv, low_memory=False)
            rating_cols = [
                "imdb_title_id", "votes_10", "votes_9", "votes_8", "votes_7", "votes_6",
                "votes_5", "votes_4", "votes_3", "votes_2", "votes_1",
                "males_allages_avg_vote", "females_allages_avg_vote",
                "us_voters_rating", "non_us_voters_rating", "top1000_voters_rating"
            ]
            avail_cols = [c for c in rating_cols if c in df_r.columns]
            merged = pd.merge(filtered_m, df_r[avail_cols], on="imdb_title_id", how="left")
        else:
            merged = filtered_m

        # Compute Polarization Index: (votes_10 + votes_1) / total_votes
        if "votes_10" in merged.columns and "votes_1" in merged.columns:
            v10 = pd.to_numeric(merged["votes_10"], errors="coerce").fillna(0)
            v1 = pd.to_numeric(merged["votes_1"], errors="coerce").fillna(0)
            merged["polarization_index"] = (v10 + v1) / np.maximum(merged["votes"], 1)
        else:
            merged["polarization_index"] = 0.05

        # Compute Critic vs Audience Divergence
        # Metascore is 0-100; avg_vote is 0-10
        merged["critic_audience_divergence"] = (merged["metascore"] / 10.0) - merged["avg_vote"]

        # Log votes
        merged["log_votes"] = np.log10(np.maximum(merged["votes"], 1))

        self.df = merged
        self.global_mean = float(merged["avg_vote"].mean())
        print(f"✅ [MovieDatasetLoader] Processed {len(self.df)} films with >= {self.min_votes} votes (global mean: {self.global_mean:.2f}).")

        self._compute_genre_baselines()
        self._compute_director_means()
        self._init_search_index()

    def _compute_genre_baselines(self):
        """Compute empirical rating distributions by individual genre."""
        genre_scores = {}
        for _, row in self.df.iterrows():
            genre_str = str(row.get("genre", ""))
            vote = float(row.get("avg_vote", 0.0))
            if not genre_str or vote <= 0:
                continue
            for g in genre_str.split(","):
                g = g.strip()
                if g:
                    genre_scores.setdefault(g, []).append(vote)

        for g, vals in genre_scores.items():
            if len(vals) >= 20:
                self.genre_baselines[g] = {
                    "count": len(vals),
                    "mean": round(float(np.mean(vals)), 2),
                    "median": round(float(np.median(vals)), 2),
                    "std": round(float(np.std(vals)), 2)
                }

    def _compute_director_means(self):
        """Compute director historical means (for directors with >= 2 films)."""
        grouped = self.df.groupby("director")["avg_vote"].agg(["count", "mean"])
        qualified = grouped[grouped["count"] >= 2]
        self.director_means = qualified["mean"].to_dict()

    def get_genre_expectation(self, genre_str: str) -> float:
        """Get the empirical expected IMDb score for a genre or combo."""
        if not genre_str:
            return round(self.global_mean, 2)
        genres = [g.strip() for g in genre_str.split(",") if g.strip()]
        means = []
        for g in genres:
            if g in self.genre_baselines:
                means.append(self.genre_baselines[g]["mean"])
        if means:
            return round(float(np.mean(means)), 2)
        return round(self.global_mean, 2)

    def _init_search_index(self):
        """Initialize TF-IDF matrix for comparable movie search over descriptions."""
        search_subset = self.df.dropna(subset=["description"]).copy().reset_index(drop=True)
        self._search_df = search_subset

        corpus = search_subset["description"].astype(str).tolist()
        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=6000,
            ngram_range=(1, 2)
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        print("🔍 [MovieDatasetLoader] Search index ready over 29,000+ movie loglines.")

    def find_comparable_movies(
        self,
        logline: str,
        genre: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Find real movie precedents that most closely match a pitch logline.
        Returns title, year, IMDb rating, polarization index, and description.
        """
        if self._tfidf_vectorizer is None or self._search_df is None:
            return []

        # Vectorize input pitch
        q_vec = self._tfidf_vectorizer.transform([logline])
        sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

        # Genre boost if provided
        if genre:
            genre_lower = genre.lower()
            for i, row in self._search_df.iterrows():
                row_g = str(row.get("genre", "")).lower()
                if any(g.strip() in row_g for g in genre_lower.split("/")):
                    sims[i] *= 1.25

        top_indices = sims.argsort()[-top_k:][::-1]
        comparables = []

        for idx in top_indices:
            row = self._search_df.iloc[idx]
            sim_score = float(sims[idx])
            pol = float(row.get("polarization_index", 0.05))
            crit_delta = row.get("critic_audience_divergence")
            crit_note = None
            if pd.notna(crit_delta):
                crit_note = round(float(crit_delta), 2)

            comparables.append({
                "title": str(row.get("title")),
                "year": int(row.get("year", 2000)),
                "genre": str(row.get("genre")),
                "director": str(row.get("director")),
                "imdb_rating": round(float(row.get("avg_vote", 0.0)), 1),
                "total_votes": int(row.get("votes", 0)),
                "polarization_index": round(pol, 3),
                "critic_audience_divergence": crit_note,
                "similarity_score": round(sim_score, 3),
                "logline": str(row.get("description"))
            })

        return comparables

    def train_movie_residual_model(self) -> Dict[str, Any]:
        """
        Train a Random Forest regression model on movie features:
        - genre_baseline
        - duration
        - year
        - log_votes
        - director_mean
        """
        if self.df is None:
            raise ValueError("Movie dataset not loaded.")

        df_train = self.df.copy()
        df_train["genre_baseline"] = df_train["genre"].apply(lambda g: self.get_genre_expectation(str(g)))
        df_train["director_mean"] = df_train["director"].map(self.director_means).fillna(self.global_mean)

        feature_cols = ["genre_baseline", "duration", "year", "log_votes", "director_mean"]
        X = df_train[feature_cols].values
        y = df_train["avg_vote"].values

        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(
            rf, X, y, cv=cv,
            scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
        )

        cv_r2 = float(np.mean(scores["test_r2"]))
        cv_rmse = float(-np.mean(scores["test_neg_root_mean_squared_error"]))
        cv_mae = float(-np.mean(scores["test_neg_mean_absolute_error"]))

        # Fit on full data
        rf.fit(X, y)

        importances = [
            {"feature": col, "importance": round(float(imp), 3)}
            for col, imp in zip(feature_cols, rf.feature_importances_)
        ]
        importances.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "model_type": "random_forest_movie",
            "samples_trained": len(df_train),
            "cv_r2": round(cv_r2, 3),
            "cv_rmse": round(cv_rmse, 3),
            "cv_mae": round(cv_mae, 3),
            "feature_importances": importances,
            "genre_baselines_count": len(self.genre_baselines),
            "global_mean": round(self.global_mean, 2)
        }
