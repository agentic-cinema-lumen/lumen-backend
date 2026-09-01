#!/usr/bin/env python3
"""
CLI script to extract/align transcript dialogue with keyframe shots in shot_manifest.json.

Usage (with SRT file):
  python scripts/run_transcript_extraction.py --manifest data/space/shot_manifest.json --srt data/space/space.srt

Usage (with automatic Whisper transcription):
  python scripts/run_transcript_extraction.py --manifest data/space/shot_manifest.json --video data/space/space.mp4 --whisper
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ingestion.transcript_aligner import align_transcript_with_shots


def main():
    parser = argparse.ArgumentParser(
        description="Align SRT subtitles or Whisper transcriptions with shot keyframe manifests."
    )
    parser.add_argument(
        "--manifest", "-m",
        required=True,
        help="Path to shot_manifest.json file"
    )
    parser.add_argument(
        "--srt", "-s",
        default=None,
        help="Path to input .srt subtitle file"
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="Path to input video file (required if using Whisper)"
    )
    parser.add_argument(
        "--whisper",
        action="store_true",
        help="Use Whisper AI automatic speech recognition on the video file"
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium"],
        default="base",
        help="Whisper model size (default: base)"
    )

    args = parser.parse_args()

    try:
        align_transcript_with_shots(
            manifest_path=args.manifest,
            srt_path=args.srt,
            video_path=args.video,
            use_whisper=args.whisper,
            whisper_model=args.whisper_model
        )
    except Exception as err:
        print(f"\n❌ Error during transcript alignment: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
