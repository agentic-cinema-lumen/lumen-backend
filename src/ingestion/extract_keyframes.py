"""
Keyframe Extraction & Scene Boundary Detection Engine.

Uses PySceneDetect (with OpenCV backend) to identify shot transitions in an episode,
extract representative keyframes (midpoint of each shot), and output a structured
shot manifest JSON file.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from src.utils.time_helpers import format_timestamp

try:
    import cv2
    from scenedetect import ContentDetector, AdaptiveDetector, SceneManager, open_video
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False


def extract_shot_keyframes(
    video_path: str,
    output_dir: str,
    threshold: float = 50.0,
    min_scene_len_sec: float = 0.8,
    max_shots: Optional[int] = None,
    detector_type: str = "content"
) -> Dict[str, Any]:
    """
    Detect shot boundaries in video_path, save one keyframe per shot, and return manifest dict.

    Args:
        video_path: Path to input video file (e.g. mp4, mkv, avi).
        output_dir: Output directory where keyframes/ and shot_manifest.json will be saved.
        threshold: Sensitivity threshold (default 27.0 for ContentDetector).
        min_scene_len_sec: Minimum duration in seconds for a single shot.
        max_shots: Optional cap on total shots processed (useful for testing).
        detector_type: 'content' or 'adaptive'.

    Returns:
        Dict containing manifest metadata and list of shot entries.
    """
    if not SCENEDETECT_AVAILABLE:
        raise ImportError(
            "scenedetect or opencv-python is not installed. "
            "Please install dependencies via: pip install scenedetect[opencv] opencv-python Pillow tqdm"
        )

    video_path_obj = Path(video_path).resolve()
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    out_dir_obj = Path(output_dir).resolve()
    keyframes_dir = out_dir_obj / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Opening video: {video_path_obj.name}")
    video = open_video(str(video_path_obj))
    fps = video.frame_rate

    # Setup SceneManager
    scene_manager = SceneManager()
    min_scene_frames = int(min_scene_len_sec * fps)

    if detector_type.lower() == "adaptive":
        scene_manager.add_detector(AdaptiveDetector(min_scene_len=min_scene_frames))
    else:
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_frames))

    print(f"🔍 Running scene detection (threshold={threshold}, min_len={min_scene_len_sec}s)...")
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        print("⚠️ No scene transitions detected. Treating entire video as 1 shot.")
        total_frames = video.duration.get_frames() if video.duration else 100
        from scenedetect import FrameTimecode
        scene_list = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    if max_shots and len(scene_list) > max_shots:
        print(f"ℹ️ Truncating shot count to max_shots={max_shots} (detected {len(scene_list)} total)")
        scene_list = scene_list[:max_shots]

    total_shots = len(scene_list)
    print(f"📸 Extracted {total_shots} shots. Saving keyframes to: {keyframes_dir}")

    # Extract keyframes using OpenCV VideoCapture
    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    shots_manifest: List[Dict[str, Any]] = []

    for idx, (start_timecode, end_timecode) in enumerate(tqdm(scene_list, desc="Extracting Keyframes")):
        shot_id = f"shot_{idx + 1:04d}"

        start_sec = float(start_timecode.get_seconds())
        end_sec = float(end_timecode.get_seconds())
        duration_sec = float(end_sec - start_sec)

        start_frame = int(start_timecode.get_frames())
        end_frame = int(end_timecode.get_frames())
        midpoint_frame = int((start_frame + end_frame) // 2)

        # Extract midpoint frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint_frame)
        ret, frame = cap.read()

        keyframe_filename = f"{shot_id}.jpg"
        keyframe_path = keyframes_dir / keyframe_filename
        rel_keyframe_path = f"keyframes/{keyframe_filename}"

        if ret and frame is not None:
            # Save image using OpenCV (BGR to JPEG)
            cv2.imwrite(str(keyframe_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        else:
            print(f"⚠️ Failed to read frame {midpoint_frame} for {shot_id}")

        shots_manifest.append({
            "shot_id": shot_id,
            "shot_index": idx + 1,
            "start_time": format_timestamp(start_sec),
            "end_time": format_timestamp(end_sec),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(duration_sec, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "midpoint_frame": midpoint_frame,
            "keyframe_path": rel_keyframe_path
        })

    cap.release()

    manifest_data = {
        "video_filename": video_path_obj.name,
        "fps": float(fps),
        "total_shots": total_shots,
        "duration_sec": round(float(scene_list[-1][1].get_seconds()) if scene_list else 0.0, 3),
        "detector_settings": {
            "threshold": float(threshold),
            "min_scene_len_sec": float(min_scene_len_sec),
            "detector_type": str(detector_type)
        },
        "shots": shots_manifest
    }

    manifest_path = out_dir_obj / "shot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"✅ Keyframe extraction complete! Shot manifest written to: {manifest_path}")
    return manifest_data
