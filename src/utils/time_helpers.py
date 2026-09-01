"""
Time and formatting utility functions for video shot processing.
"""

def format_timestamp(seconds: float) -> str:
    """Format floating point seconds into HH:MM:SS.mmm string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_timestamp(timestamp_str: str) -> float:
    """Parse HH:MM:SS.mmm or MM:SS.mmm or SS.mmm into floating point seconds."""
    parts = timestamp_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    elif len(parts) == 1:
        return float(parts[0])
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")
