#!/usr/bin/env python3
"""Re-parse every data/movies/<slug>/script.txt with the current ScriptParser and
refresh the cached script_metrics in movies_manifest.json and each metadata.json.

The training rows in src/quant/benchmark_dataset.py read the cached metrics, so the
cache has to be regenerated whenever the parser changes, or the model trains on
features the serving path no longer produces.

Films that fail validate_screenplay() keep their metrics but carry a
"validation_errors" list, which benchmark_dataset uses to exclude them from training.

Usage: python3 scripts/reextract_script_metrics.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.script_parser import ScriptParser, validate_screenplay

MOVIES_DIR = Path(__file__).resolve().parent.parent / "data" / "movies"


def reextract(dry_run: bool = False):
    parser = ScriptParser()
    manifest_path = MOVIES_DIR / "movies_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    excluded, missing = [], []
    for entry in manifest:
        slug = entry.get("slug")
        script_path = MOVIES_DIR / slug / "script.txt"
        if not script_path.exists() or script_path.stat().st_size <= 2000:
            entry["has_script"] = False
            entry["script_metrics"] = None
            missing.append(slug)
            continue

        metrics = parser.parse_script_text(
            script_path.read_text(encoding="utf-8", errors="ignore"),
            title=entry.get("title", slug),
        )
        metrics.pop("scenes", None)
        errors = validate_screenplay(metrics)
        if errors:
            metrics["validation_errors"] = errors
            excluded.append((slug, errors))
        entry["has_script"] = True
        entry["script_metrics"] = metrics

        meta_path = MOVIES_DIR / slug / "metadata.json"
        if meta_path.exists() and not dry_run:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["script_metrics"] = metrics
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"re-extracted {len(manifest) - len(missing)}/{len(manifest)} films")
    if missing:
        print(f"no usable script.txt: {', '.join(missing)}")
    print(f"excluded by validate_screenplay: {len(excluded)}")
    for slug, errors in excluded:
        print(f"  - {slug}: {'; '.join(errors)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    reextract(**vars(ap.parse_args()))
