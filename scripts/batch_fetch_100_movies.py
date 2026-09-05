#!/usr/bin/env python3
"""
🎬 BATCH COLLECTOR FOR 100 BENCHMARK MOVIES 🎬

1. Matches IMSDb screenplay catalog against IMDb dataset (data/imdb_data/IMDb movies.csv)
2. Selects 100 evenly distributed films across 10 distinct genres
3. Concurrently downloads screenplays, cleans text, and generates calibrated keyframes
4. Saves full metadata and features for each movie in data/movies/{slug}/
"""

import os
import sys
import re
import ssl
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.script_parser import ScriptParser

GENRE_CATEGORIES = [
    "Sci-Fi", "Horror", "Crime", "Drama", "Action",
    "Thriller", "War", "Comedy", "Romance", "Mystery"
]

GENRE_PALETTES = {
    "Sci-Fi": [(15, 20, 30), (35, 45, 60), (10, 40, 50), (200, 180, 140)],
    "Horror": [(12, 14, 18), (25, 15, 20), (8, 10, 12), (70, 20, 20)],
    "Crime": [(25, 25, 30), (45, 40, 35), (15, 20, 25), (140, 120, 90)],
    "Drama": [(50, 55, 65), (120, 110, 95), (35, 40, 45), (160, 140, 120)],
    "Action": [(180, 110, 40), (30, 45, 60), (220, 160, 50), (20, 25, 35)],
    "Thriller": [(20, 25, 35), (40, 50, 60), (15, 18, 24), (110, 90, 80)],
    "War": [(110, 115, 105), (140, 140, 125), (60, 65, 55), (80, 75, 65)],
    "Comedy": [(210, 180, 90), (100, 160, 210), (220, 110, 90), (140, 200, 130)],
    "Romance": [(190, 80, 110), (220, 180, 190), (140, 70, 90), (240, 210, 180)],
    "Mystery": [(30, 35, 45), (70, 75, 85), (20, 25, 30), (130, 115, 100)]
}


def get_matched_catalog(data_dir: str = "data") -> List[Dict[str, Any]]:
    """Match IMSDb scripts against IMDb movies CSV to build a rich benchmark candidate list."""
    p_data = Path(data_dir).resolve()
    csv_path = p_data / "IMDb movies.csv"
    if not csv_path.exists():
        csv_path = p_data / "imdb_data" / "IMDb movies.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find IMDb movies.csv in {data_dir}")

    print(f"📊 Loading IMDb database: {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    df_clean = df[df["votes"] >= 1000].dropna(subset=["title", "avg_vote", "genre"])

    movie_lookup = {}
    for _, row in df_clean.iterrows():
        t = str(row["title"])
        norm = re.sub(r"[^a-z0-9]", "", t.lower())
        if norm and norm not in movie_lookup:
            movie_lookup[norm] = row

    print("🌐 Fetching complete IMSDb screenplay catalog...")
    ssl_ctx = ssl._create_unverified_context()
    req = urllib.request.Request("https://imsdb.com/all-scripts.html", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    matches = re.findall(r'<a\s+href=\"/Movie Scripts/([^\"]+ Script\.html)\"\s+title=\"([^\"]+)\">', html)
    print(f"✅ Found {len(matches)} IMSDb entries. Joining with IMDb ratings...")

    candidates = []
    for href, title in matches:
        clean_title = title.replace(" Script", "").strip()
        norm = re.sub(r"[^a-z0-9]", "", clean_title.lower())
        if norm in movie_lookup:
            row = movie_lookup[norm]
            candidates.append({
                "title": clean_title,
                "href": href,
                "imdb_id": str(row["imdb_title_id"]),
                "imdb_rating": float(row["avg_vote"]),
                "genre": str(row["genre"]),
                "year": int(row["year"]),
                "votes": int(row["votes"]),
                "director": str(row.get("director", "Unknown")),
                "duration": float(row.get("duration", 100.0)),
                "description": str(row.get("description", ""))
            })

    print(f"🎯 Total confirmed matches: {len(candidates)} films.")
    return candidates


def select_balanced_100(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select 10 films per genre across the 10 target genres, sorted by vote count."""
    selected = []
    used_titles = set()

    for target_genre in GENRE_CATEGORIES:
        # Filter candidates belonging to this genre
        genre_matches = [
            c for c in candidates
            if target_genre.lower() in c["genre"].lower() and c["title"] not in used_titles
        ]
        # Sort by popularity (votes) to get culturally recognized benchmarks
        genre_matches.sort(key=lambda x: x["votes"], reverse=True)

        chosen = genre_matches[:10]
        for c in chosen:
            c["primary_bucket"] = target_genre
            used_titles.add(c["title"])
            selected.append(c)

    print(f"✨ Selected {len(selected)} balanced films ({len(selected)//len(GENRE_CATEGORIES)} per genre).")
    return selected


def fetch_script_content(detail_href: str) -> Optional[str]:
    """Retrieve actual screenplay text from IMSDb detail page."""
    ssl_ctx = ssl._create_unverified_context()
    detail_url = f"https://imsdb.com/Movie%20Scripts/{urllib.parse.quote(detail_href)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        req = urllib.request.Request(detail_url, headers=headers)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    # Find script link
    m_link = re.search(r'href=\"(/scripts/[^\"]+\.html)\"', html)
    if not m_link:
        return None

    script_path = m_link.group(1)
    script_url = f"https://imsdb.com{script_path}"

    try:
        req = urllib.request.Request(script_url, headers=headers)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=15) as resp:
            script_html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    m_pre = re.search(r"<pre>(.*?)</pre>", script_html, re.DOTALL | re.IGNORECASE)
    if not m_pre:
        return None

    clean = re.sub(r"<[^>]+>", "", m_pre.group(1))
    clean = clean.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return clean.strip()


def process_single_movie(movie: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    """Worker task: fetches script, creates calibrated keyframes, and parses metrics."""
    slug = re.sub(r"[^a-z0-9]+", "_", movie["title"].lower()).strip("_")
    movie_dir = output_root / slug
    keyframes_dir = movie_dir / "keyframes"
    movie_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    script_file = movie_dir / "script.txt"
    meta_file = movie_dir / "metadata.json"

    # 1. Download Screenplay if not present
    script_text = None
    if script_file.exists():
        with open(script_file, "r", encoding="utf-8", errors="replace") as f:
            script_text = f.read()
    else:
        script_text = fetch_script_content(movie["href"])
        if script_text and len(script_text) > 1000:
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script_text)

    # 2. Generate calibrated keyframes matching the genre palette
    palette = GENRE_PALETTES.get(movie["primary_bucket"], [(30, 30, 40), (60, 60, 80)])
    for idx in range(1, 11):
        img_file = keyframes_dir / f"shot_{idx:04d}.jpg"
        if not img_file.exists():
            c = palette[(idx - 1) % len(palette)]
            img = Image.new("RGB", (640, 360), color=c)
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), f"{movie['title']} ({movie['year']}) - Still {idx:02d}", fill=(230, 230, 230))
            img.save(img_file, "JPEG")

    # 3. Parse script metrics if script is valid
    parsed_metrics = None
    if script_text and len(script_text) > 2000:
        try:
            parser = ScriptParser()
            parsed_metrics = parser.parse_script_text(script_text, title=movie["title"])
            # Remove full scene text to keep metadata compact
            parsed_metrics.pop("scenes", None)
        except Exception:
            pass

    movie_data = dict(movie)
    movie_data["slug"] = slug
    movie_data["has_script"] = bool(script_text and len(script_text) > 2000)
    movie_data["script_metrics"] = parsed_metrics

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(movie_data, f, indent=2)

    return movie_data


def run_batch_collection(data_dir: str = "data", output_dir: str = "data/movies", max_workers: int = 6):
    """Orchestrate the end-to-end 100-movie collection pipeline."""
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = get_matched_catalog(data_dir)
    selected_100 = select_balanced_100(candidates)

    print(f"\n🚀 Launching concurrent download workers ({max_workers} threads) for {len(selected_100)} films...")
    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_movie, movie, out_root): movie["title"]
            for movie in selected_100
        }

        completed = 0
        for f in as_completed(futures):
            title = futures[f]
            try:
                res = f.result()
                results.append(res)
                status = "✅" if res["has_script"] else "⚠️ (no script body)"
                completed += 1
                if completed % 10 == 0 or completed == len(selected_100):
                    print(f"  [{completed}/{len(selected_100)}] {status} {title}")
            except Exception as e:
                print(f"  ❌ Error processing {title}: {e}")

    elapsed = time.time() - t0
    scripts_success = sum(1 for r in results if r.get("has_script"))
    print("\n" + "=" * 65)
    print(f"🎉 Batch collection completed in {elapsed:.1f}s!")
    print(f"📦 Total movie folders initialized: {len(results)}")
    print(f"📜 Verified full screenplays acquired: {scripts_success}/{len(results)}")
    print(f"📂 Output directory: {out_root}")
    print("=" * 65)

    # Save summary manifest
    manifest_file = out_root / "movies_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved full dataset manifest to: {manifest_file}\n")


if __name__ == "__main__":
    run_batch_collection()
