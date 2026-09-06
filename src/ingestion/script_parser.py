"""
🎬 SCRIPT & PREMISE INGESTION ENGINE 🎬

Parses screenplays (standard format, .fountain, or plain text) and pitch loglines into:
1. Scene breakdown and narrative tempo
2. Dialogue velocity (WPM), character line distribution, action-to-dialogue ratios
3. Climax acceleration (tempo progression across acts)
4. Pre-production craft feature representations compatible with the Quant ML residual engine
"""

import re
import math
from typing import Dict, Any, List, Optional
from pathlib import Path

# Common scene heading patterns (INT. / EXT. / SCENE / etc.)
# Separators are deliberately loose: scripts use "INT.", "EXT - ", "EXT -- ",
# "EXT — ", "INT/EXT" and occasionally no separator at all.
SCENE_HEADER_PATTERN = re.compile(
    r"^(?:[0-9]+[ \t.\-]*)?"
    r"(?:(?:INT|EXT|I)[./](?:EXT|INT|E)|INTERIOR|EXTERIOR|INT|EXT)(?=[\s.:,\-–—]|$)"
    r"|^(?:SCENE\s+[0-9]+|ACT\s+[0-9IVXLCDM]+)\b",
    re.IGNORECASE
)

# Camera / transition directions that look like character cues but are not people.
CAMERA_TRANSITION_PATTERN = re.compile(
    r"^(?:"
    r"(?:SMASH\s+|MATCH\s+|JUMP\s+|HARD\s+|QUICK\s+)?CUT|DISSOLVE|FADE|WIPE|"
    r"WE\s+(?:SEE|HEAR|FOLLOW|CUT|MOVE)|ANGLE|CLOSE|CLOSER|CLOSEUP|WIDE|WIDER|"
    r"TWO\s+SHOT|ONE\s+SHOT|REVERSE|INSERT|INTERCUT|MONTAGE|FLASHBACK|FLASH|"
    r"SERIES\s+OF|PAN|TILT|ZOOM|TRACKING|DOLLY|CRANE|POV|P\.O\.V|BACK\s+TO|"
    r"CONTINUED|CONTINUOUS|SUPER|TITLE|TITLES|CREDITS|THE\s+END|END\s+OF|"
    r"OMITTED|LATER|MEANWHILE|MOMENTS\s+LATER|SPLIT\s+SCREEN|STOCK\s+SHOT|"
    r"ESTABLISHING|MAIN\s+TITLE|OVER\s+BLACK|BLACK\s+SCREEN|FREEZE\s+FRAME"
    r")\b",
    re.IGNORECASE
)

# Character cue pattern (ALL CAPS name, optionally followed by (O.S.), (V.O.), (CONT'D))
CHARACTER_CUE_PATTERN = re.compile(
    r"^[ \t]*([A-Z0-9_\-\. ]{2,30})(?:\s*\([A-Za-z0-9_\-\. ']+\))?[ \t]*$"
)


def _is_character_cue(line: str) -> bool:
    """True when `line` looks like a speaker cue rather than a heading or camera direction."""
    if not CHARACTER_CUE_PATTERN.match(line) or len(line) >= 35:
        return False
    if any(p in line for p in [".", ",", "!", "?", "--", ":"]):
        return False
    if SCENE_HEADER_PATTERN.match(line):
        return False
    # ponytail: two cheap shape rules instead of a real classifier — a speaker cue is
    # short and contains a real word. Kills OCR fragments ("W7", "3 7") and stray
    # capitalised action ("DJANGO WHIPS HIM TO THE GROUND").
    if len(line.split()) > 4 or not re.search(r"[A-Za-z]{3,}", line):
        return False
    return not CAMERA_TRANSITION_PATTERN.match(line)


def validate_screenplay(metrics: Dict[str, Any]) -> List[str]:
    """Return reasons the parsed document is not a usable screenplay (empty list = valid)."""
    reasons: List[str] = []
    duration = metrics.get("estimated_duration_min", 0.0)
    if not 60.0 <= duration <= 240.0:
        reasons.append(f"implausible runtime: {duration} min (expected 60-240)")
    if metrics.get("total_scenes", 0) < 4:
        reasons.append(f"too few scenes: {metrics.get('total_scenes', 0)} (expected >= 4)")
    characters = metrics.get("characters_count", 0)
    # ponytail: ceiling 150, not 80 - large ensemble casts legitimately exceed 80.
    if not 3 <= characters <= 150:
        reasons.append(f"implausible character count: {characters} (expected 3-150)")
    ratio = metrics.get("dialogue_ratio", 0.0)
    if not 0.10 <= ratio <= 0.85:
        reasons.append(f"implausible dialogue ratio: {ratio} (expected 0.10-0.85)")
    return reasons


class ScriptParser:
    """Parses screenplays and computes structural craft & pacing metrics."""

    def __init__(self, words_per_minute_reading: float = 130.0, words_per_page: float = 186.0):
        self.words_per_minute_reading = words_per_minute_reading
        self.words_per_page = words_per_page

    def parse_script_text(self, text: str, title: str = "Untitled Script") -> Dict[str, Any]:
        """Parse raw screenplay text into structured scenes, dialogue blocks, and craft metrics."""
        lines = [line.rstrip() for line in text.splitlines()]
        
        scenes: List[Dict[str, Any]] = []
        current_scene: Dict[str, Any] = {
            "scene_number": 1,
            "heading": "PROLOGUE",
            "dialogue_blocks": [],
            "action_lines": [],
            "word_count": 0,
            "dialogue_word_count": 0,
            "characters": set()
        }
        
        i = 0
        n = len(lines)
        current_speaker: Optional[str] = None
        current_speech: List[str] = []

        def flush_dialogue():
            nonlocal current_speaker, current_speech
            if current_speaker and current_speech:
                speech_text = " ".join(current_speech).strip()
                words = len(speech_text.split())
                if speech_text:
                    current_scene["dialogue_blocks"].append({
                        "character": current_speaker,
                        "text": speech_text,
                        "word_count": words
                    })
                    current_scene["dialogue_word_count"] += words
                    current_scene["characters"].add(current_speaker)
            current_speaker = None
            current_speech = []

        while i < n:
            raw_line = lines[i]
            line = raw_line.strip()

            if not line:
                flush_dialogue()
                i += 1
                continue

            # Check if this line is a new Scene Heading
            if SCENE_HEADER_PATTERN.match(line):
                flush_dialogue()
                if current_scene["dialogue_blocks"] or current_scene["action_lines"]:
                    # Finish previous scene
                    current_scene["characters"] = sorted(list(current_scene["characters"]))
                    current_scene["word_count"] = current_scene["dialogue_word_count"] + sum(
                        len(a.split()) for a in current_scene["action_lines"]
                    )
                    scenes.append(current_scene)

                current_scene = {
                    "scene_number": len(scenes) + 1,
                    "heading": line,
                    "dialogue_blocks": [],
                    "action_lines": [],
                    "word_count": 0,
                    "dialogue_word_count": 0,
                    "characters": set()
                }
                i += 1
                continue

            # Check for Character Cue (precedes dialogue)
            match_char = CHARACTER_CUE_PATTERN.match(line)
            # Make sure it's not a general action line in all caps like "THE DOOR SLAMS SHUT",
            # a scene heading, or a camera/transition direction.
            if match_char and _is_character_cue(line):
                flush_dialogue()
                # Next line should be dialogue
                if i + 1 < n and lines[i + 1].strip():
                    current_speaker = match_char.group(1).strip()
                    i += 1
                    continue

            # If we have an active speaker, accumulate dialogue
            if current_speaker:
                current_speech.append(line)
            else:
                # Action line
                current_scene["action_lines"].append(line)

            i += 1

        flush_dialogue()
        if current_scene["dialogue_blocks"] or current_scene["action_lines"]:
            current_scene["characters"] = sorted(list(current_scene["characters"]))
            current_scene["word_count"] = current_scene["dialogue_word_count"] + sum(
                len(a.split()) for a in current_scene["action_lines"]
            )
            scenes.append(current_scene)

        # If no explicit scene headers were found, treat paragraph clusters as scenes
        if not scenes:
            scenes = self._fallback_chunking(lines)

        return self._compute_script_metrics(title, scenes)

    def parse_script_file(self, file_path: str) -> Dict[str, Any]:
        """Read and parse script from a file path."""
        p = Path(file_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Script file not found: {file_path}")
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return self.parse_script_text(text, title=p.stem.replace("_", " ").title())

    def _fallback_chunking(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Fallback chunking when no standard scene headers exist."""
        scenes = []
        chunk_lines: List[str] = []
        scene_idx = 1

        for line in lines:
            line_str = line.strip()
            if not line_str and len(chunk_lines) >= 15:
                # Create scene
                all_text = " ".join(chunk_lines)
                words = len(all_text.split())
                scenes.append({
                    "scene_number": scene_idx,
                    "heading": f"BEAT {scene_idx}",
                    "dialogue_blocks": [{"character": "SPEAKER", "text": all_text, "word_count": words}],
                    "action_lines": [],
                    "word_count": words,
                    "dialogue_word_count": words,
                    "characters": ["SPEAKER"]
                })
                chunk_lines = []
                scene_idx += 1
            elif line_str:
                chunk_lines.append(line_str)

        if chunk_lines:
            all_text = " ".join(chunk_lines)
            words = len(all_text.split())
            scenes.append({
                "scene_number": scene_idx,
                "heading": f"BEAT {scene_idx}",
                "dialogue_blocks": [{"character": "SPEAKER", "text": all_text, "word_count": words}],
                "action_lines": [],
                "word_count": words,
                "dialogue_word_count": words,
                "characters": ["SPEAKER"]
            })
        return scenes

    def _compute_script_metrics(self, title: str, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive screenplay tempo, dialogue, and act metrics."""
        total_scenes = len(scenes)
        total_words = sum(s["word_count"] for s in scenes)
        total_dialogue_words = sum(s["dialogue_word_count"] for s in scenes)
        total_action_words = total_words - total_dialogue_words

        # Estimate scene durations:
        # Standard rule: 1 standard screenplay page ≈ 1 minute (220 words ≈ 60 seconds)
        # Fast action beats ≈ 3.5 words/sec; spoken dialogue ≈ 2.2 words/sec
        scene_durations = []
        all_dialogue_lines = []
        all_characters = set()

        for s in scenes:
            # Duration model: the industry one-page-per-minute convention.
            # words_per_page is tuned against the 100 known runtimes in movies_manifest.json
            # (median absolute error 12.5%, vs 32.6% for the old words-per-second model).
            dur = max(4.0, s["word_count"] / self.words_per_page * 60.0)
            s["estimated_duration_sec"] = round(dur, 2)
            scene_durations.append(dur)
            for d in s["dialogue_blocks"]:
                all_dialogue_lines.append(d)
                all_characters.add(d["character"])

        total_duration_sec = sum(scene_durations)
        duration_min = max(total_duration_sec / 60.0, 0.1)

        # Pacing & Climax Acceleration:
        # Ratio of average scene duration in first 75% vs final 25%
        if total_scenes >= 4:
            split_idx = int(total_scenes * 0.75)
            early_asl = float(sum(scene_durations[:split_idx]) / split_idx)
            late_asl = float(sum(scene_durations[split_idx:]) / (total_scenes - split_idx))
            climax_acceleration = round(early_asl / max(late_asl, 1.0), 3)
        else:
            climax_acceleration = 1.0

        # Cuts & Shots estimation:
        # Professional screen drama averages ~3 to 5 cuts per scene or 1 cut per 3.5 seconds
        estimated_shots_per_scene = [
            max(2, int(s["estimated_duration_sec"] / 3.8))
            for s in scenes
        ]
        total_estimated_shots = sum(estimated_shots_per_scene)
        cuts_per_minute = round(total_estimated_shots / duration_min, 2)
        words_per_minute = round(total_dialogue_words / duration_min, 1)
        dialogue_ratio = round(total_dialogue_words / max(total_words, 1), 3)

        return {
            "title": title,
            "total_scenes": total_scenes,
            "total_words": total_words,
            "dialogue_words": total_dialogue_words,
            "action_words": total_action_words,
            "dialogue_ratio": dialogue_ratio,
            "estimated_duration_min": round(duration_min, 1),
            "estimated_duration_sec": round(total_duration_sec, 1),
            "estimated_total_shots": total_estimated_shots,
            "cuts_per_minute": cuts_per_minute,
            "words_per_minute": words_per_minute,
            "climax_acceleration": climax_acceleration,
            "characters_count": len(all_characters),
            "characters": sorted(list(all_characters)),
            "scenes": scenes
        }

    def to_episode_features(self, script_metrics: Dict[str, Any], keyframe_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert parsed script metrics into the standard episode_features.json schema
        used by src/quant/feature_extractor.py.
        """
        shots = []
        current_time = 0.0

        for s in script_metrics["scenes"]:
            dur = s["estimated_duration_sec"]
            dlg_text = " ".join(b["text"] for b in s["dialogue_blocks"])
            dlg_lines = [b["text"] for b in s["dialogue_blocks"]]

            # Break scene into simulated cuts
            num_cuts = max(1, int(dur / 4.0))
            cut_dur = dur / num_cuts

            for c in range(num_cuts):
                is_first = (c == 0)
                shots.append({
                    "shot_id": len(shots) + 1,
                    "start_time_sec": round(current_time, 2),
                    "end_time_sec": round(current_time + cut_dur, 2),
                    "duration_sec": round(cut_dur, 2),
                    "dialogue": dlg_text if is_first else "",
                    "dialogue_lines": dlg_lines if is_first else []
                })
                current_time += cut_dur

        data = {
            "title": script_metrics["title"],
            "total_duration_sec": script_metrics["estimated_duration_sec"],
            "total_shots": len(shots),
            "shots": shots,
            "script_metrics": script_metrics
        }

        if keyframe_features:
            data.update(keyframe_features)

        return data


def parse_premise_logline(logline: str, genre: Optional[str] = None) -> Dict[str, Any]:
    """
    Deconstructs a short pitch/logline (2-4 sentences) into archetype,
    stakes, conflict, and estimated pacing dynamics.
    """
    words = logline.strip().split()
    word_count = len(words)
    logline_lower = logline.lower()

    # Heuristic genre detection if not specified
    if not genre:
        if any(w in logline_lower for w in ["space", "lunar", "sci-fi", "alien", "cyborg", "galaxy", "future"]):
            genre = "Sci-Fi Thriller"
        elif any(w in logline_lower for w in ["murder", "courtroom", "detective", "investigation", "crime", "trial"]):
            genre = "Legal / Crime Drama"
        elif any(w in logline_lower for w in ["war", "battle", "kingdom", "throne", "sword", "army"]):
            genre = "Epic / Fantasy Action"
        elif any(w in logline_lower for w in ["corporate", "family", "boardroom", "billionaire", "dynasty"]):
            genre = "Prestige Corporate Drama"
        else:
            genre = "Dramatic Thriller"

    # Tone detection
    if any(w in logline_lower for w in ["dark", "claustrophobic", "grim", "bleak", "paranoia", "terror"]):
        tone = "Claustrophobic & Dark"
        expected_asl = 3.2
        expected_cpm = 18.0
        expected_wpm = 95.0
    elif any(w in logline_lower for w in ["fast", "chase", "explosive", "adrenaline", "heist"]):
        tone = "High-Octane Kinetic"
        expected_asl = 2.0
        expected_cpm = 28.0
        expected_wpm = 70.0
    elif any(w in logline_lower for w in ["courtroom", "debate", "dialogue", "conspiracy", "secrets"]):
        tone = "Intense Dialogue-Driven"
        expected_asl = 4.8
        expected_cpm = 12.5
        expected_wpm = 145.0
    else:
        tone = "Balanced Prestige"
        expected_asl = 3.5
        expected_cpm = 16.0
        expected_wpm = 115.0

    return {
        "logline": logline,
        "genre": genre,
        "tone": tone,
        "word_count": word_count,
        "expected_pacing": {
            "asl": expected_asl,
            "cuts_per_minute": expected_cpm,
            "words_per_minute": expected_wpm,
            "climax_acceleration": 1.45
        }
    }
