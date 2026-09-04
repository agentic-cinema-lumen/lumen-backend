"""
🎬 QUANT FEATURE EXTRACTOR 🎬

Extracts statistical and cinematographic craft features from episode_features.json
and keyframe images:
- Pacing dynamics: Average Shot Length (ASL), Cuts Per Minute (CPM), variance, acceleration
- Dialogue density: Words Per Minute (WPM), lines per minute, silence ratio
- Visual metrics: Mean keyframe luminance (brightness), contrast, dark frame ratio
- Structural metadata: Position in season, is_premiere, is_finale
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image


def calculate_image_luminance(image_path: Path) -> Optional[float]:
    """Calculate mean perceived luminance (0-255) using ITU-R BT.601 standard."""
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            small_img = rgb_img.resize((100, 100))
            arr = np.array(small_img, dtype=np.float32)
            # Perceived brightness formula: 0.299*R + 0.587*G + 0.114*B
            luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            return float(np.mean(luminance))
    except Exception:
        return None


def extract_features_from_episode(
    episode_data: Dict[str, Any],
    base_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract a comprehensive tabular feature vector from episode_features.json data.
    """
    shots = episode_data.get("shots", [])
    total_duration_sec = float(episode_data.get("total_duration_sec", 0.0))
    total_shots = int(episode_data.get("total_shots", len(shots)))

    # Fallback duration if missing
    if total_duration_sec <= 0.0 and shots:
        total_duration_sec = sum(s.get("duration_sec", 2.0) for s in shots)

    duration_min = max(total_duration_sec / 60.0, 0.01)

    # 1. PACING FEATURES
    shot_durations = [float(s.get("duration_sec", 0.0)) for s in shots if s.get("duration_sec", 0.0) > 0]
    if not shot_durations:
        shot_durations = [2.5]  # safe fallback

    asl = float(np.mean(shot_durations))
    median_shot_length = float(np.median(shot_durations))
    shot_std = float(np.std(shot_durations))
    cuts_per_minute = total_shots / duration_min

    # Climax acceleration: ratio of ASL in the final 25% of episode vs first 75%
    n = len(shot_durations)
    if n >= 4:
        split_idx = int(n * 0.75)
        first_part_asl = float(np.mean(shot_durations[:split_idx]))
        climax_asl = float(np.mean(shot_durations[split_idx:]))
        pacing_acceleration = (first_part_asl / max(climax_asl, 0.1))
    else:
        pacing_acceleration = 1.0

    # 2. DIALOGUE & SCRIPT FEATURES
    dialogue_shots_count = 0
    total_word_count = 0
    dialogue_lines_count = 0
    silence_streaks = []
    current_silence = 0.0

    for s in shots:
        dlg = str(s.get("dialogue", "")).strip()
        dur = float(s.get("duration_sec", 2.0))
        lines = s.get("dialogue_lines", [])
        if lines:
            dialogue_lines_count += len(lines)
        elif dlg:
            dialogue_lines_count += 1

        if dlg:
            dialogue_shots_count += 1
            words = len(dlg.split())
            total_word_count += words
            if current_silence > 0:
                silence_streaks.append(current_silence)
                current_silence = 0.0
        else:
            current_silence += dur

    if current_silence > 0:
        silence_streaks.append(current_silence)

    words_per_minute = total_word_count / duration_min
    lines_per_minute = dialogue_lines_count / duration_min
    dialogue_shot_ratio = dialogue_shots_count / max(total_shots, 1)
    max_silence_duration = float(np.max(silence_streaks)) if silence_streaks else 0.0

    # 3. VISUAL & LUMINANCE FEATURES
    luminances = []
    if base_dir:
        for s in shots:
            keyframe_rel = s.get("keyframe_path")
            if keyframe_rel:
                kf_path = Path(base_dir) / keyframe_rel
                lum = calculate_image_luminance(kf_path)
                if lum is not None:
                    luminances.append(lum)

    if luminances:
        mean_luminance = float(np.mean(luminances))
        luminance_std = float(np.std(luminances))
        dark_frame_ratio = float(np.mean([1.0 if l < 40.0 else 0.0 for l in luminances]))
    else:
        # Default neutral visual baseline (mid daylight)
        mean_luminance = 110.0
        luminance_std = 25.0
        dark_frame_ratio = 0.05

    # 4. STRUCTURAL / METADATA FEATURES
    meta = metadata or {}
    season_num = int(meta.get("season", 1))
    episode_num = int(meta.get("episode", 1))
    total_season_eps = int(meta.get("total_season_episodes", 10))
    season_position = float(episode_num / max(total_season_eps, 1))

    is_premiere = 1.0 if episode_num == 1 else 0.0
    is_finale = 1.0 if episode_num >= total_season_eps else 0.0
    is_penultimate = 1.0 if episode_num == max(total_season_eps - 1, 1) else 0.0
    show_historical_mean = float(meta.get("show_historical_mean", 8.2))
    log_votes = float(math.log10(max(meta.get("vote_count", 5000), 10)))

    return {
        "episode_id": episode_data.get("episode_id", "unknown"),
        "show_name": episode_data.get("show_name", "Unknown"),
        "total_duration_min": round(duration_min, 2),
        "total_shots": total_shots,
        # Pacing
        "average_shot_length": round(asl, 3),
        "median_shot_length": round(median_shot_length, 3),
        "shot_length_std": round(shot_std, 3),
        "cuts_per_minute": round(cuts_per_minute, 2),
        "pacing_acceleration": round(pacing_acceleration, 3),
        # Dialogue
        "words_per_minute": round(words_per_minute, 2),
        "lines_per_minute": round(lines_per_minute, 2),
        "dialogue_shot_ratio": round(dialogue_shot_ratio, 3),
        "max_silence_sec": round(max_silence_duration, 2),
        # Visual
        "mean_luminance": round(mean_luminance, 2),
        "luminance_std": round(luminance_std, 2),
        "dark_frame_ratio": round(dark_frame_ratio, 3),
        # Context
        "season_position": round(season_position, 3),
        "is_premiere": is_premiere,
        "is_finale": is_finale,
        "is_penultimate": is_penultimate,
        "show_historical_mean": round(show_historical_mean, 2),
        "log_votes": round(log_votes, 2)
    }


def extract_from_file(features_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience helper to extract features directly from episode_features.json path."""
    p = Path(features_path).resolve()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return extract_features_from_episode(data, base_dir=p.parent, metadata=metadata)
