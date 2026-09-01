#!/usr/bin/env python3
"""
CLI script to run keyframe extraction and scene detection on an episode video file.

Usage:
  python scripts/run_keyframe_extraction.py --video path/to/episode.mp4 --output-dir data/episodes/got_s8e3
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ingestion.extract_keyframes import extract_shot_keyframes


def main():
    parser = argparse.ArgumentParser(
        description="Extract shot boundary keyframes and generate shot manifest JSON for an episode."
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to input video file (e.g. mp4, mkv, avi)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to save extracted keyframes and shot_manifest.json"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=50.0,
        help="ContentDetector threshold sensitivity (default: 50.0)"
    )
    parser.add_argument(
        "--min-len",
        type=float,
        default=0.8,
        help="Minimum scene duration in seconds (default: 0.8s)"
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=None,
        help="Optional max number of shots to process (for quick testing)"
    )
    parser.add_argument(
        "--detector",
        choices=["content", "adaptive"],
        default="content",
        help="Detector algorithm to use (default: content)"
    )

    args = parser.parse_args()

    try:
        extract_shot_keyframes(
            video_path=args.video,
            output_dir=args.output_dir,
            threshold=args.threshold,
            min_scene_len_sec=args.min_len,
            max_shots=args.max_shots,
            detector_type=args.detector
        )
    except ImportError as err:
        print(f"\n❌ Dependency Error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n❌ Error during keyframe extraction: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
