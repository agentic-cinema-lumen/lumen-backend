"""
Transcript Extraction & Shot Timestamp Alignment Engine.

Parses subtitle files (.srt, .vtt) or runs automatic speech recognition (Whisper),
and aligns dialogue lines with detected shot keyframes from shot_manifest.json.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.time_helpers import format_timestamp, parse_timestamp


def parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse an SRT subtitle file into a list of timestamped entries.

    Returns:
        List of dicts: [{'index': 1, 'start_sec': 0.0, 'end_sec': 2.5, 'text': '...'}]
    """
    srt_path_obj = Path(srt_path).resolve()
    if not srt_path_obj.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    with open(srt_path_obj, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        # Line 0: index (optional/numeric)
        # Line 1: timestamp (00:00:01,200 --> 00:00:03,500)
        time_line_idx = 1 if lines[0].isdigit() else 0
        if time_line_idx >= len(lines):
            continue

        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[time_line_idx]
        )
        if not time_match:
            continue

        start_str, end_str = time_match.groups()
        start_sec = parse_timestamp(start_str.replace(",", "."))
        end_sec = parse_timestamp(end_str.replace(",", "."))

        text_lines = lines[time_line_idx + 1:]
        text = " ".join(text_lines)
        # Strip HTML tags (e.g. <i>...</i>)
        text = re.sub(r"<[^>]+>", "", text).strip()

        if text:
            entries.append({
                "index": len(entries) + 1,
                "start_time": format_timestamp(start_sec),
                "end_time": format_timestamp(end_sec),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "text": text
            })

    return entries


def transcribe_video_with_whisper(
    video_path: str,
    model_size: str = "base"
) -> List[Dict[str, Any]]:
    """
    Transcribe audio track from video file using faster-whisper or whisper.

    Returns:
        List of timestamped dialogue entries.
    """
    try:
        from faster_whisper import WhisperModel
        print(f"🎙️ Loading Whisper model ('{model_size}')...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(video_path, beam_size=5)

        entries = []
        for idx, seg in enumerate(segments):
            text = seg.text.strip()
            if text:
                entries.append({
                    "index": idx + 1,
                    "start_time": format_timestamp(seg.start),
                    "end_time": format_timestamp(seg.end),
                    "start_sec": round(float(seg.start), 3),
                    "end_sec": round(float(seg.end), 3),
                    "duration_sec": round(float(seg.end - seg.start), 3),
                    "text": text
                })
        return entries
    except ImportError:
        pass

    try:
        import whisper
        print(f"🎙️ Loading OpenAI Whisper model ('{model_size}')...")
        model = whisper.load_model(model_size)
        result = model.transcribe(video_path)

        entries = []
        for idx, seg in enumerate(result.get("segments", [])):
            text = seg["text"].strip()
            if text:
                entries.append({
                    "index": idx + 1,
                    "start_time": format_timestamp(seg["start"]),
                    "end_time": format_timestamp(seg["end"]),
                    "start_sec": round(float(seg["start"]), 3),
                    "end_sec": round(float(seg["end"]), 3),
                    "duration_sec": round(float(seg["end"] - seg["start"]), 3),
                    "text": text
                })
        return entries
    except ImportError:
        raise ImportError(
            "Whisper is not installed. Please install via: pip install faster-whisper "
            "or provide an SRT subtitle file using --srt path/to/file.srt"
        )


def align_transcript_with_shots(
    manifest_path: str,
    srt_path: Optional[str] = None,
    video_path: Optional[str] = None,
    use_whisper: bool = False,
    whisper_model: str = "base"
) -> Dict[str, Any]:
    """
    Align subtitle dialogue entries with shot keyframes in shot_manifest.json.

    Args:
        manifest_path: Path to shot_manifest.json.
        srt_path: Optional path to .srt subtitle file.
        video_path: Optional path to video file (required if using whisper).
        use_whisper: If True, run automatic speech recognition on video_path.
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium').

    Returns:
        Updated manifest dictionary with aligned dialogue.
    """
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Shot manifest not found: {manifest_path}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Fetch transcript entries
    entries: List[Dict[str, Any]] = []

    if srt_path and Path(srt_path).exists():
        print(f"📜 Parsing SRT subtitle file: {srt_path}")
        entries = parse_srt_file(srt_path)
    elif use_whisper or (video_path and not srt_path):
        if not video_path:
            # Fallback to video path specified in output directory or manifest
            parent_dir = manifest_file.parent
            v_file = manifest.get("video_filename")
            if v_file and (parent_dir / v_file).exists():
                video_path = str(parent_dir / v_file)

        if video_path and Path(video_path).exists():
            entries = transcribe_video_with_whisper(video_path, model_size=whisper_model)
        else:
            raise ValueError("No valid SRT file provided and video_path not found for Whisper transcription.")
    else:
        raise ValueError("Please provide either --srt <file.srt> or --video <file.mp4> for transcript extraction.")

    print(f"💬 Found {len(entries)} dialogue lines. Aligning with {manifest.get('total_shots', 0)} shots...")

    # 2. Align dialogue entries with shots
    shots = manifest.get("shots", [])

    for shot in shots:
        shot_start = shot["start_sec"]
        shot_end = shot["end_sec"]

        aligned_lines = []
        for entry in entries:
            # Calculate temporal overlap
            overlap_start = max(shot_start, entry["start_sec"])
            overlap_end = min(shot_end, entry["end_sec"])
            overlap = overlap_end - overlap_start

            if overlap > 0:
                aligned_lines.append(entry["text"])

        shot["dialogue_lines"] = aligned_lines
        shot["dialogue"] = " ".join(aligned_lines) if aligned_lines else None

    manifest["total_dialogue_lines"] = len(entries)
    manifest["transcript_entries"] = entries

    # Write updated manifest back to disk
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save standalone transcript.json in the same folder
    transcript_file = manifest_file.parent / "transcript.json"
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump({"total_lines": len(entries), "lines": entries}, f, indent=2)

    print(f"✅ Alignment complete! Updated manifest: {manifest_file}")
    print(f"✅ Standalone transcript saved: {transcript_file}")
    return manifest
