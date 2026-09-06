"""
🎬 BENCHMARK SHOWCASE PROVIDER 🎬

Loads genuine benchmark films from data/movies/movies_manifest.json and evaluates
them using the Quant Residual Model (QuantOracle) and 15-point counterfactual sweep.

Exposes instant pre-computed or live-evaluated pre-mortem evaluations on famous
films (Fight Club, Alien, The Matrix, Blade Runner, etc.) for the frontend demo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.quant.oracle import get_oracle
from src.quant.sweep import run_counterfactual_sweep

MOVIES_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "movies" / "movies_manifest.json"

# Featured iconic films highlighted for the frontend showcase
FEATURED_SLUGS = [
    "fight_club",
    "alien",
    "the_matrix",
    "blade_runner",
    "inception",
    "pulp_fiction",
    "the_social_network",
    "apocalypse_now",
    "whiplash",
    "fargo",
    "get_out",
    "interstellar",
]

_MANIFEST_CACHE: Optional[List[Dict[str, Any]]] = None
_BENCHMARK_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_manifest() -> List[Dict[str, Any]]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    if not MOVIES_MANIFEST_PATH.exists():
        return []
    with open(MOVIES_MANIFEST_PATH, "r", encoding="utf-8") as f:
        _MANIFEST_CACHE = json.load(f)
    return _MANIFEST_CACHE


def list_benchmark_movies(featured_only: bool = False) -> List[Dict[str, Any]]:
    """List available benchmark movies with essential metadata and high-level scores."""
    manifest = _load_manifest()
    results = []
    for item in manifest:
        slug = item.get("slug")
        if not slug:
            continue
        if featured_only and slug not in FEATURED_SLUGS:
            continue

        cv = item.get("cv_metrics") or {}
        sm = item.get("script_metrics") or {}

        results.append({
            "title": item.get("title"),
            "slug": slug,
            "year": item.get("year"),
            "director": item.get("director"),
            "genre": item.get("genre"),
            "primaryGenre": item.get("primary_bucket"),
            "imdbRating": item.get("imdb_rating"),
            "votes": item.get("votes"),
            "durationMin": item.get("duration"),
            "cutsPerMinute": sm.get("cuts_per_minute"),
            "climaxAcceleration": sm.get("climax_acceleration"),
            "darkFrameRatio": cv.get("dark_frame_ratio"),
            "isFeatured": slug in FEATURED_SLUGS,
        })

    # Sort featured first, then by title
    results.sort(key=lambda x: (not x["isFeatured"], x["title"] or ""))
    return results


def get_benchmark_movie(slug: str) -> Optional[Dict[str, Any]]:
    """Get full evaluation and 15-point counterfactual sweep for a specific movie slug."""
    if slug in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE[slug]

    manifest = _load_manifest()
    entry = next((item for item in manifest if item.get("slug") == slug), None)
    if not entry:
        return None

    cv = entry.get("cv_metrics") or {}
    sm = entry.get("script_metrics") or {}
    genre = entry.get("primary_bucket") or entry.get("genre", "Drama")
    duration = float(entry.get("duration") or sm.get("estimated_duration_min") or 120.0)

    # Assemble craft feature vector
    features = {
        "title": entry.get("title"),
        "genre": genre,
        "total_duration_min": duration,
        "cuts_per_minute": float(sm.get("cuts_per_minute") or 15.0),
        "pacing_acceleration": float(sm.get("climax_acceleration") or 1.0),
        "average_shot_length": float(round(60.0 / max(float(sm.get("cuts_per_minute") or 15.0), 1.0), 2)),
        "words_per_minute": float(sm.get("words_per_minute") or 80.0),
        "lines_per_minute": float(round(float(sm.get("words_per_minute") or 80.0) / 12.0, 2)),
        "dialogue_shot_ratio": float(sm.get("dialogue_ratio") or 0.40),
        "mean_luminance": float(cv.get("mean_luminance") or 60.0),
        "luminance_std": float(cv.get("luminance_std") or 30.0),
        "dark_frame_ratio": float(cv.get("dark_frame_ratio") or 0.30),
    }

    oracle = get_oracle()
    pred = oracle.predict_craft(features)
    sweep = run_counterfactual_sweep(features, oracle=oracle)

    # Score on 0-100 scale
    predicted_rating = float(pred["expected_rating"])
    score_100 = int(round(predicted_rating * 10.0))
    imdb_rating = float(entry.get("imdb_rating") or 0.0)

    # Construct complete showcase payload
    payload = {
        "title": entry.get("title"),
        "slug": slug,
        "year": entry.get("year"),
        "director": entry.get("director"),
        "genre": entry.get("genre"),
        "primaryGenre": genre,
        "imdbRating": imdb_rating,
        "predictedRating": round(predicted_rating, 2),
        "score100": score_100,
        "genreBaselineRating": round(float(pred["genre_baseline_rating"]), 2),
        "craftResidualDelta": round(float(pred["craft_residual_delta"]), 2),
        "craftVerdict": pred.get("craft_verdict"),
        "confidenceInterval": pred.get("confidence_interval"),
        "craftVulnerabilities": pred.get("craft_vulnerabilities", []),
        "topAttributions": pred.get("top_attributions", []),
        "scriptMetrics": {
            "totalScenes": sm.get("total_scenes"),
            "cutsPerMinute": sm.get("cuts_per_minute"),
            "climaxAcceleration": sm.get("climax_acceleration"),
            "wordsPerMinute": sm.get("words_per_minute"),
            "dialogueRatio": sm.get("dialogue_ratio"),
            "estimatedDurationMin": duration,
        },
        "visionMetrics": {
            "meanLuminance": cv.get("mean_luminance"),
            "luminanceStd": cv.get("luminance_std"),
            "darkFrameRatio": cv.get("dark_frame_ratio"),
            "severeDarknessPenalty": cv.get("severe_darkness_penalty"),
        },
        "counterfactualSweep": {
            "baselineRating": round(float(sweep["baseline_rating"]), 3),
            "maxDelta": round(float(sweep["max_delta"]), 3),
            "noiseFloorMae": round(float(sweep["noise_floor_mae"]), 3),
            "clearsNoiseFloor": bool(sweep["clears_noise_floor"]),
            "outcome": sweep["outcome"],
            "summary": sweep["confidence_summary"],
            "rows": sweep["sweep_table"],
        },
    }

    _BENCHMARK_CACHE[slug] = payload
    return payload
