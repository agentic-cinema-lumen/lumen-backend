"""
Utility to load .env file into os.environ without requiring external packages.
"""

import os
from pathlib import Path


def load_env_file(env_path: str = ".env") -> None:
    """Read key=value pairs from .env and set them in os.environ if not already set."""
    p = Path(env_path).resolve()
    if not p.exists():
        # Try searching upwards to project root
        root_p = Path(__file__).resolve().parent.parent.parent / ".env"
        if root_p.exists():
            p = root_p
        else:
            return

    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


# Auto-load on import
load_env_file()
