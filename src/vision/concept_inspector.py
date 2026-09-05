"""
🎬 CONCEPT & KEYFRAME VISION INSPECTOR 🎬

Analyzes visual keyframes and moodboard images:
1. Low-level computer vision metrics: Luminance, contrast, dark frame ratio, severe darkness penalty
2. Multimodal aesthetic appraisal: Color temperature, lighting style, and consumer screen legibility
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from src.utils.llm_client import LLMClient


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


def calculate_image_contrast(image_path: Path) -> Optional[float]:
    """Calculate RMS contrast of image."""
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("L")
            arr = np.array(rgb_img, dtype=np.float32)
            return float(np.std(arr))
    except Exception:
        return None


class ConceptInspector:
    """Inspects keyframes and concept art for craft metrics and aesthetic risks."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def inspect_images(self, image_paths_or_dir: Any) -> Dict[str, Any]:
        """
        Inspect a collection of images from a directory path or list of file paths.
        """
        paths: List[Path] = []
        if isinstance(image_paths_or_dir, (str, Path)):
            p = Path(image_paths_or_dir).resolve()
            if p.is_dir():
                paths = sorted([
                    f for f in p.iterdir()
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
                ])
            elif p.is_file():
                paths = [p]
        elif isinstance(image_paths_or_dir, list):
            paths = [Path(f).resolve() for f in image_paths_or_dir if Path(f).exists()]

        if not paths:
            # Safe default fallback
            return {
                "image_count": 0,
                "mean_luminance": 85.0,
                "luminance_std": 15.0,
                "dark_frame_ratio": 0.10,
                "severe_darkness_penalty": 0.10,
                "visual_style_summary": "No keyframe images provided; neutral baseline assumed.",
                "visual_risk_flags": []
            }

        luminances: List[float] = []
        contrasts: List[float] = []

        for img_path in paths:
            lum = calculate_image_luminance(img_path)
            if lum is not None:
                luminances.append(lum)
            cnt = calculate_image_contrast(img_path)
            if cnt is not None:
                contrasts.append(cnt)

        if not luminances:
            luminances = [85.0]

        mean_lum = float(np.mean(luminances))
        lum_std = float(np.std(luminances)) if len(luminances) > 1 else 10.0
        dark_frames = [l for l in luminances if l < 40.0]
        dark_frame_ratio = float(len(dark_frames) / max(len(luminances), 1))
        severe_darkness_penalty = float((dark_frame_ratio ** 2) * 10.0)

        risk_flags = []
        if dark_frame_ratio >= 0.40:
            risk_flags.append("HIGH_DARKNESS_RISK: Over 40% of frames are sub-40 luminance, prone to streaming compression artifacts.")
        if mean_lum < 35.0:
            risk_flags.append("SEVERE_UNDEREXPOSURE: Mean luminance is critically low, creating high risk of viewer visual fatigue.")
        if lum_std < 8.0:
            risk_flags.append("MONOTONE_LIGHTING: Very low lighting variance across shots; potential lack of dynamic visual contrast.")

        visual_summary = self._generate_visual_critique(paths, mean_lum, dark_frame_ratio, risk_flags)

        return {
            "image_count": len(paths),
            "mean_luminance": round(mean_lum, 2),
            "luminance_std": round(lum_std, 2),
            "dark_frame_ratio": round(dark_frame_ratio, 3),
            "severe_darkness_penalty": round(severe_darkness_penalty, 3),
            "visual_style_summary": visual_summary,
            "visual_risk_flags": risk_flags
        }

    def _generate_visual_critique(
        self,
        paths: List[Path],
        mean_lum: float,
        dark_ratio: float,
        risk_flags: List[str]
    ) -> str:
        """Synthesize visual critique via LLM or heuristic fallback."""
        if self.llm.provider == "mock":
            if dark_ratio > 0.40:
                return (
                    f"Visuals exhibit heavy low-key chiaroscuro with extreme shadows (mean luminance: {mean_lum:.1f}, "
                    f"dark frame ratio: {dark_ratio*100:.1f}%). Highly atmospheric in theater mastering, but carries severe "
                    f"risk of viewer backlash under standard consumer TV gamma and streaming bitrate compression."
                )
            else:
                return (
                    f"Balanced cinematography with healthy dynamic range (mean luminance: {mean_lum:.1f}, "
                    f"dark frame ratio: {dark_ratio*100:.1f}%). Clean visual clarity suitable for high-prestige drama."
                )

        # Real LLM call
        prompt = (
            f"Analyze cinematography metrics for television/film keyframes:\n"
            f"- Mean Luminance (0-255): {mean_lum:.1f}\n"
            f"- Dark Frame Ratio (<40 lux): {dark_ratio*100:.1f}%\n"
            f"- Detected Risk Flags: {risk_flags}\n"
            f"Provide a 2-sentence cinema craft verdict on lighting legibility, mood, and consumer viewing risks."
        )
        return self.llm.generate(
            system_prompt="You are a veteran Director of Photography and Cinema Visual Analyst.",
            user_prompt=prompt,
            temperature=0.3
        )
