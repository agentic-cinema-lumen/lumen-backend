"""
Synthetic & Demo Keyframe Generator.

Generates mock keyframe images and shot manifests for quick testing and development
when full episode video files are not available.
"""

import json
from pathlib import Path
from typing import Dict, Any

from src.utils.time_helpers import format_timestamp

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def generate_mock_episode_data(
    output_dir: str,
    show_name: str = "Game of Thrones",
    season_ep: str = "S08E03",
    num_shots: int = 12
) -> Dict[str, Any]:
    """
    Generate synthetic keyframe images and a valid shot_manifest.json.

    Args:
        output_dir: Target output directory.
        show_name: Name of the TV show.
        season_ep: Season and Episode string (e.g. S08E03).
        num_shots: Number of synthetic shots to generate.

    Returns:
        Dict representing shot_manifest.json contents.
    """
    out_dir_obj = Path(output_dir).resolve()
    keyframes_dir = out_dir_obj / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    shots = []
    current_time = 0.0

    print(f"🛠️ Generating {num_shots} mock keyframe records for {show_name} {season_ep} in {out_dir_obj}...")

    # Color palette for synthetic frames
    colors = [
        (25, 28, 36), (40, 45, 60), (15, 20, 30), (55, 30, 40),
        (30, 50, 45), (45, 40, 65), (20, 35, 50), (60, 55, 45)
    ]

    # Simple 1x1 GIF/BMP byte placeholder if PIL is missing
    MINIMAL_JPEG = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06'
        b'\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00'
        b'\x10\x00\x10\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00'
        b'\x00?\x00\xbf\x00\x07\xff\xd9'
    )

    for idx in range(1, num_shots + 1):
        shot_id = f"shot_{idx:04d}"
        duration = 2.5 + (idx % 3) * 1.2
        start_sec = current_time
        end_sec = start_sec + duration
        current_time = end_sec

        keyframe_filename = f"{shot_id}.jpg"
        keyframe_path = keyframes_dir / keyframe_filename

        if PIL_AVAILABLE:
            img = Image.new("RGB", (640, 360), color=colors[(idx - 1) % len(colors)])
            draw = ImageDraw.Draw(img)
            text = f"{show_name} {season_ep}\nShot #{idx:02d} ({shot_id})\nTime: {format_timestamp(start_sec)} - {format_timestamp(end_sec)}"
            draw.text((30, 140), text, fill=(230, 235, 245))
            img.save(keyframe_path, "JPEG", quality=85)
        else:
            with open(keyframe_path, "wb") as img_file:
                img_file.write(MINIMAL_JPEG)

        shots.append({
            "shot_id": shot_id,
            "shot_index": idx,
            "start_time": format_timestamp(start_sec),
            "end_time": format_timestamp(end_sec),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(duration, 3),
            "start_frame": int(start_sec * 24),
            "end_frame": int(end_sec * 24),
            "midpoint_frame": int(((start_sec + end_sec) / 2) * 24),
            "keyframe_path": f"keyframes/{keyframe_filename}"
        })

    manifest = {
        "video_filename": f"{show_name.lower().replace(' ', '_')}_{season_ep.lower()}.mp4",
        "fps": 24.0,
        "total_shots": num_shots,
        "duration_sec": round(current_time, 3),
        "is_mock": True,
        "shots": shots
    }

    manifest_path = out_dir_obj / "shot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Generated mock shot manifest at: {manifest_path}")
    return manifest
