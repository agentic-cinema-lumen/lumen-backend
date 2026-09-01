#!/usr/bin/env python3
"""
🎬 SHOT PIPELINE ORCHESTRATOR 🎬

The master ingestion script for Agentic Cinema Detective.
Runs the entire end-to-end data ingestion pipeline in one command:
 1. Shot boundary detection & keyframe extraction (PySceneDetect + OpenCV)
 2. Subtitle parsing & timestamp alignment (.srt / Whisper AI)
 3. Shot manifest & episode feature packaging

Usage:
  python scripts/run_shot_pipeline.py --video data/space/space.mp4 --output-dir data/space --srt data/space/space.srt
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ingestion.extract_keyframes import extract_shot_keyframes
from src.ingestion.transcript_aligner import align_transcript_with_shots


PIPELINE_ASCII = r"""
     _______________________________________
    |                                       |
    |    🎬  SHOT PIPELINE ORCHESTRATOR  🎬  |
    |_______________________________________|
     Processing Shots & Dialogue Features...
"""


def run_shot_pipeline(
    video_path: str,
    output_dir: str,
    srt_path: str = None,
    threshold: float = 50.0,
    min_len: float = 0.8,
    use_whisper: bool = False,
    whisper_model: str = "base"
):
    """Run the complete end-to-end ingestion pipeline."""
    start_time = time.time()
    print(PIPELINE_ASCII)

    out_dir_obj = Path(output_dir).resolve()
    out_dir_obj.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Stage 1: Keyframe Extraction & Shot Detection
    # -------------------------------------------------------------
    print("--------------------------------------------------")
    print("🎬 STAGE 1: Shot Boundary Detection & Keyframe Extraction")
    print("--------------------------------------------------")

    manifest = extract_shot_keyframes(
        video_path=video_path,
        output_dir=output_dir,
        threshold=threshold,
        min_scene_len_sec=min_len
    )
    manifest_path = str(out_dir_obj / "shot_manifest.json")

    # -------------------------------------------------------------
    # Stage 2: Subtitle Parsing & Shot Alignment
    # -------------------------------------------------------------
    print("\n--------------------------------------------------")
    print("🎬 STAGE 2: Subtitle Parsing & Shot Alignment")
    print("--------------------------------------------------")

    # Auto-detect matching SRT file if not explicitly passed
    if not srt_path and not use_whisper:
        video_name = Path(video_path).stem
        auto_srt = out_dir_obj / f"{video_name}.srt"
        if auto_srt.exists():
            srt_path = str(auto_srt)
            print(f"ℹ️ Auto-detected matching subtitle file: {auto_srt}")

    if srt_path or use_whisper or video_path:
        try:
            manifest = align_transcript_with_shots(
                manifest_path=manifest_path,
                srt_path=srt_path,
                video_path=video_path,
                use_whisper=use_whisper,
                whisper_model=whisper_model
            )
        except Exception as err:
            print(f"⚠️ Subtitle alignment skipped or failed: {err}")
    else:
        print("⚠️ No subtitle file provided. Skipping transcript alignment.")

    # -------------------------------------------------------------
    # Stage 3: Package Final Episode Features
    # -------------------------------------------------------------
    print("\n--------------------------------------------------")
    print("🎬 STAGE 3: Packaging Final Episode Features")
    print("--------------------------------------------------")

    ep_features = {
        "show_name": out_dir_obj.name.split("_")[0].capitalize(),
        "episode_id": out_dir_obj.name,
        "video_file": Path(video_path).name,
        "total_shots": manifest.get("total_shots", 0),
        "total_duration_sec": manifest.get("duration_sec", 0.0),
        "total_dialogue_lines": manifest.get("total_dialogue_lines", 0),
        "shots": manifest.get("shots", [])
    }

    features_path = out_dir_obj / "episode_features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(ep_features, f, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print("\n==================================================")
    print(f"🎉 SHOT PIPELINE COMPLETE IN {elapsed}s!")
    print(f"📦 Final Episode Features: {features_path}")
    print(f"📸 Keyframes Directory:   {out_dir_obj / 'keyframes'}")
    print(f"📜 Manifest File:          {manifest_path}")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(
        description="🎬 Shot Pipeline: Master ingestion script for Agentic Cinema Detective."
    )
    parser.add_argument("--video", "-v", required=True, help="Path to input episode video file (.mp4, .mkv)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for keyframes and manifest")
    parser.add_argument("--srt", "-s", default=None, help="Path to subtitle .srt file")
    parser.add_argument("--threshold", "-t", type=float, default=50.0, help="Shot detection threshold (default: 50.0)")
    parser.add_argument("--min-len", type=float, default=0.8, help="Minimum shot length in seconds (default: 0.8s)")
    parser.add_argument("--whisper", action="store_true", help="Use Whisper AI speech recognition if no SRT is provided")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium"], help="Whisper model size")

    args = parser.parse_args()

    try:
        run_shot_pipeline(
            video_path=args.video,
            output_dir=args.output_dir,
            srt_path=args.srt,
            threshold=args.threshold,
            min_len=args.min_len,
            use_whisper=args.whisper,
            whisper_model=args.whisper_model
        )
    except Exception as err:
        print(f"\n❌ Shot Pipeline Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
