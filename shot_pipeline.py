#!/usr/bin/env python3
"""
🎬 SHOT PIPELINE MASTER CLI 🎬

Root execution script for Agentic Cinema Detective.
Automatically discovers all video files in data/ and processes them end-to-end.

Usage:
  python shot_pipeline.py
  python shot_pipeline.py --dir data/space
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.run_shot_pipeline import run_shot_pipeline, PIPELINE_ASCII


def discover_and_process(target_dir: str = "data", threshold: float = 50.0):
    """Discover all video files in target_dir and run Shot Pipeline."""
    target_path = Path(target_dir).resolve()
    if not target_path.exists():
        print(f"❌ Error: Directory '{target_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Search for video files recursively or in subdirectories
    video_extensions = [".mp4", ".mkv", ".avi", ".mov"]
    video_files = []

    for ext in video_extensions:
        video_files.extend(list(target_path.glob(f"**/*{ext}")))

    if not video_files:
        print(f"⚠️ No video files (.mp4, .mkv, .avi) found in '{target_dir}'.")
        print("💡 Place video files in data/<episode_folder>/<video_name>.mp4 and re-run python shot_pipeline.py")
        return

    print(PIPELINE_ASCII)
    print(f"🔍 Discovered {len(video_files)} video file(s) in '{target_dir}':")
    for idx, v in enumerate(video_files, 1):
        print(f"  {idx}. {v.relative_to(target_path.parent)}")

    print("\n==================================================")
    print("🎬 RUNNING SHOT PIPELINE FOR DISCOVERED MEDIA")
    print("==================================================\n")

    for v_file in video_files:
        output_dir = v_file.parent
        # Look for matching SRT file in same folder
        srt_file = output_dir / f"{v_file.stem}.srt"
        srt_path = str(srt_file) if srt_file.exists() else None

        print(f"\n🎬 Processing: {v_file.name}")
        print(f"📁 Output Directory: {output_dir}")
        if srt_path:
            print(f"📜 Subtitle File: {srt_file.name}")

        try:
            run_shot_pipeline(
                video_path=str(v_file),
                output_dir=str(output_dir),
                srt_path=srt_path,
                threshold=threshold
            )
        except Exception as err:
            print(f"❌ Error processing {v_file.name}: {err}", file=sys.stderr)

    print("\n🎬 ALL SHOT PIPELINE JOBS COMPLETED! 🎬\n")


def main():
    parser = argparse.ArgumentParser(
        description="🎬 Shot Pipeline: Master CLI for Agentic Cinema Detective."
    )
    parser.add_argument(
        "--dir", "-d",
        default="data",
        help="Target data directory to scan for video files (default: data)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=50.0,
        help="ContentDetector threshold sensitivity (default: 50.0)"
    )

    args = parser.parse_args()
    discover_and_process(target_dir=args.dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
