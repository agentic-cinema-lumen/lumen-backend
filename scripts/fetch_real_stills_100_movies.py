#!/usr/bin/env python3
"""
🎬 Fetch Real Photographic Film Stills for 100 Movies
Spaced evenly across the entire film timeline:
  - Shot 01: 0%  (Opening Tone & Visual Establishment)
  - Shot 02: 20% (Inciting Incident / Act I Turning Point)
  - Shot 03: 40% (Rising Action / Midpoint Approach)
  - Shot 04: 60% (Midpoint Reversal / Complications)
  - Shot 05: 80% (Climax Crisis / Climax Confrontation)
  - Shot 06: 100% (Resolution / Final Frame)
"""

import csv
import json
import os
import re
import shutil
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.ingestion.script_parser import ScriptParser


TARGET_BEATS = [
    (0.00, "Opening Tone & Visual Establishment"),
    (0.20, "Inciting Incident & Act I Break"),
    (0.40, "Rising Action & Midpoint Build"),
    (0.60, "Midpoint Reversal & Dark Night Crisis"),
    (0.80, "Climax Confrontation & Peak Action"),
    (1.00, "Resolution & Final Frame"),
]


def create_ssl_context():
    return ssl._create_unverified_context()


def fetch_filmgrab_index() -> Dict[str, Tuple[str, str]]:
    """Fetch all 4,100+ movie slugs and URLs from Film-Grab."""
    print("🌐 Fetching Film-Grab master movie directory...")
    ssl_ctx = create_ssl_context()
    req = urllib.request.Request(
        "https://film-grab.com/movies-a-z/",
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    urls = re.findall(r'href=[\"\'](https?://film-grab\.com/\d{4}/\d{2}/\d{2}/([^/\"#]+)/)[\"\']', html)
    print(f"✅ Found {len(urls)} movies in Film-Grab library.")
    filmgrab = {}
    for u, slug in urls:
        clean_slug = re.sub(r"[^a-z0-9]", "", slug.lower())
        filmgrab[clean_slug] = (slug, u)
    return filmgrab


def fetch_imsdb_catalog() -> List[Tuple[str, str]]:
    """Fetch complete list of screenplays from IMSDb."""
    print("🌐 Fetching IMSDb screenplay catalog...")
    ssl_ctx = create_ssl_context()
    req = urllib.request.Request(
        "https://imsdb.com/all-scripts.html",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    matches = re.findall(r'<a href=\"/Movie Scripts/([^\"]+)\"[^>]*>([^<]+)</a>', html)
    print(f"✅ Found {len(matches)} IMSDb screenplays.")
    return matches


def load_imdb_database(data_dir: Path) -> Dict[str, dict]:
    """Load IMDb movies database with >= 1000 votes."""
    csv_path = data_dir / "IMDb movies.csv"
    if not csv_path.exists():
        csv_path = data_dir / "imdb_data" / "IMDb movies.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find IMDb movies.csv in {data_dir}")

    print(f"📊 Loading IMDb database: {csv_path}...")
    imdb_titles = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            votes = int(row.get("votes") or 0)
            if votes < 1000:
                continue
            title = row.get("original_title") or row.get("title") or ""
            clean_title = re.sub(r"[^a-z0-9]", "", title.lower())
            if clean_title and clean_title not in imdb_titles:
                imdb_titles[clean_title] = row

    print(f"✅ Loaded {len(imdb_titles):,} validated IMDb films (>=1,000 votes).")
    return imdb_titles


def fetch_imsdb_script_content(href: str) -> Optional[str]:
    """Fetch full screenplay text from IMSDb."""
    base_url = "https://imsdb.com/scripts/"
    script_name = href.replace(" Script.html", ".html").replace(" ", "-")
    url = base_url + urllib.parse.quote(script_name)

    ssl_ctx = create_ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        try:
            url_alt = f"https://imsdb.com/Movie%20Scripts/{urllib.parse.quote(href)}"
            req_alt = urllib.request.Request(url_alt, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_alt, context=ssl_ctx, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None

    m = re.search(r"<pre>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
        text = re.sub(r"<[^<]+?>", "", text)
        return text.strip()
    return None


def fetch_filmgrab_stills(filmgrab_url: str, num_stills: int = 6) -> List[Tuple[int, float, str, str]]:
    """
    Fetch all stills from a Film-Grab page and sample `num_stills` evenly spaced across timeline.
    Returns: List of (frame_index, timeline_pct, narrative_beat, still_url)
    """
    ssl_ctx = create_ssl_context()
    req = urllib.request.Request(
        filmgrab_url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    links = re.findall(
        r'href=[\"\'](https?://film-grab\.com/wp-content/uploads/(?:photo-gallery/)?[^\"\'#]+\.jpe?g[^\"\'#]*)[\"\']',
        html
    )
    # Remove thumbnail images and site icons/logos
    full_links = [
        l for l in links
        if "/thumb/" not in l
        and not any(x in l.lower() for x in ["icon", "logo", "avatar", "header", "banner"])
    ]

    if not full_links:
        img_links = re.findall(
            r'(?:data-original|src)=[\"\'](https?://film-grab\.com/wp-content/uploads/(?:photo-gallery/)?[^\"\'#]+\.jpe?g[^\"\'#]*)[\"\']',
            html
        )
        full_links = [
            l for l in img_links
            if "/thumb/" not in l
            and not any(x in l.lower() for x in ["icon", "logo", "avatar", "header", "banner"])
        ]

    seen = set()
    ordered_links = []
    for l in full_links:
        base = l.split("?")[0]
        if base not in seen:
            seen.add(base)
            ordered_links.append(l)

    total = len(ordered_links)
    if total == 0:
        return []

    sampled = []
    for step_idx, (target_pct, beat_name) in enumerate(TARGET_BEATS[:num_stills]):
        frame_idx = int(round(target_pct * (total - 1)))
        frame_idx = max(0, min(total - 1, frame_idx))
        actual_pct = frame_idx / (total - 1) if total > 1 else 0.0
        sampled.append((frame_idx, actual_pct, beat_name, ordered_links[frame_idx]))

    return sampled


def download_image(url: str, dest_path: Path) -> bool:
    """Download single image safely with header and SSL handling."""
    ssl_ctx = create_ssl_context()
    parsed = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(parsed.path)
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, encoded_path, parsed.query, ""))

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    req = urllib.request.Request(safe_url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=12) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
        with Image.open(dest_path) as img:
            img.verify()
        return True
    except Exception:
        if dest_path.exists():
            dest_path.unlink()
        return False


def select_balanced_100_films(
    candidates: List[dict],
    target_count: int = 100
) -> List[dict]:
    """Select an evenly balanced collection of 100 films across 11 genres."""
    TARGET_GENRES = [
        "Action", "Drama", "Sci-Fi", "Comedy", "Crime",
        "Horror", "Thriller", "Adventure", "Romance", "Mystery", "War"
    ]
    per_genre_target = target_count // len(TARGET_GENRES)

    by_genre = defaultdict(list)
    for c in candidates:
        genres = [g.strip() for g in c["genre"].split(",")]
        assigned = False
        for tg in TARGET_GENRES:
            if tg in genres:
                by_genre[tg].append(c)
                assigned = True
                break
        if not assigned:
            by_genre["Drama"].append(c)

    for g in by_genre:
        by_genre[g].sort(key=lambda x: (x["votes"], x["imdb_rating"]), reverse=True)

    selected = []
    selected_ids = set()

    for g in TARGET_GENRES:
        count = 0
        for item in by_genre[g]:
            if count >= per_genre_target:
                break
            if item["imdb_id"] not in selected_ids:
                item["primary_bucket"] = g
                selected.append(item)
                selected_ids.add(item["imdb_id"])
                count += 1

    remaining_candidates = []
    for g in TARGET_GENRES:
        for item in by_genre[g]:
            if item["imdb_id"] not in selected_ids:
                remaining_candidates.append(item)

    remaining_candidates.sort(key=lambda x: x["votes"], reverse=True)
    for item in remaining_candidates:
        if len(selected) >= target_count:
            break
        selected.append(item)
        selected_ids.add(item["imdb_id"])

    return selected


def process_movie(movie: dict, movies_dir: Path) -> dict:
    """Download screenplay and 6 timeline-spaced real stills for a single movie."""
    slug = movie.get("slug") or re.sub(r"[^a-z0-9]+", "_", movie["title"].lower()).strip("_")
    movie_dir = movies_dir / slug
    keyframes_dir = movie_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    script_file = movie_dir / "script.txt"
    meta_file = movie_dir / "metadata.json"
    manifest_file = keyframes_dir / "shot_manifest.json"

    # 1. Script
    script_text = ""
    if script_file.exists() and script_file.stat().st_size > 2000:
        with open(script_file, "r", encoding="utf-8", errors="ignore") as f:
            script_text = f.read()
    else:
        script_text = fetch_imsdb_script_content(movie["href"])
        if script_text and len(script_text) > 2000:
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script_text)

    # 2. Film-Grab Real Stills
    stills_data = fetch_filmgrab_stills(movie["filmgrab_url"], num_stills=6)
    shots_manifest = []

    if stills_data:
        for idx, (frame_idx, timeline_pct, beat_name, img_url) in enumerate(stills_data, 1):
            shot_file = keyframes_dir / f"shot_{idx:04d}.jpg"
            # Always download or replace with authentic film still
            success = download_image(img_url, shot_file)
            if success:
                shots_manifest.append({
                    "shot_id": f"shot_{idx:04d}",
                    "shot_index": idx,
                    "keyframe_file": shot_file.name,
                    "timeline_pct": round(timeline_pct, 2),
                    "narrative_beat": beat_name,
                    "gallery_frame_index": frame_idx,
                    "source_url": img_url
                })

    # Save shot manifest
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "movie_title": movie["title"],
            "slug": slug,
            "filmgrab_url": movie["filmgrab_url"],
            "total_shots": len(shots_manifest),
            "shots": shots_manifest
        }, f, indent=2)

    # 3. Parse script metrics
    parsed_metrics = None
    if script_text and len(script_text) > 2000:
        try:
            parser = ScriptParser()
            parsed_metrics = parser.parse_script_text(script_text, title=movie["title"])
            parsed_metrics.pop("scenes", None)
        except Exception:
            pass

    movie_meta = dict(movie)
    movie_meta["slug"] = slug
    movie_meta["has_script"] = bool(script_text and len(script_text) > 2000)
    movie_meta["has_real_stills"] = bool(len(shots_manifest) >= 5)
    movie_meta["real_stills_count"] = len(shots_manifest)
    movie_meta["script_metrics"] = parsed_metrics

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(movie_meta, f, indent=2)

    return movie_meta


def main():
    root_dir = Path(__file__).parent.parent.resolve()
    data_dir = root_dir / "data"
    movies_dir = data_dir / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print("🎬  REAL FILM STILLS & SCRIPT HARVESTER (100+ MOVIES)  🎬")
    print("================================================================")

    imdb_titles = load_imdb_database(data_dir)
    filmgrab = fetch_filmgrab_index()
    imsdb_scripts = fetch_imsdb_catalog()

    print("\n🔍 Building 3-way intersection (IMSDb + Film-Grab + IMDb)...")
    candidates = []
    for href, title in imsdb_scripts:
        clean_title = re.sub(r"[^a-z0-9]", "", title.lower())
        if clean_title in filmgrab and clean_title in imdb_titles:
            row = imdb_titles[clean_title]
            candidates.append({
                "title": title,
                "href": href,
                "imdb_id": row["imdb_title_id"],
                "imdb_rating": float(row["avg_vote"]),
                "votes": int(row["votes"]),
                "genre": row["genre"],
                "year": int(row["year"]),
                "director": row.get("director", "Unknown"),
                "duration": float(row.get("duration") or 100.0),
                "description": row.get("description", ""),
                "filmgrab_url": filmgrab[clean_title][1],
                "filmgrab_slug": filmgrab[clean_title][0],
            })

    print(f"✅ Found {len(candidates)} high-pedigree films in 3-way intersection.")

    selected_100 = select_balanced_100_films(candidates, target_count=100)
    print(f"\n🎯 Selected {len(selected_100)} balanced films for complete real stills collection.")

    max_workers = 8
    print(f"\n🚀 Downloading real timeline-spaced stills ({max_workers} threads)...")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_movie, m, movies_dir): m["title"] for m in selected_100}
        completed = 0
        for f in as_completed(futures):
            title = futures[f]
            completed += 1
            try:
                res = f.result()
                results.append(res)
                stills_count = res.get("real_stills_count", 0)
                print(f"[{completed}/{len(selected_100)}] ✅ {title} -> {stills_count} real stills spaced across timeline")
            except Exception as e:
                print(f"[{completed}/{len(selected_100)}] ❌ {title} Error: {e}")

    dt = time.time() - t0
    print(f"\n✨ Processed {len(results)} movies in {dt:.1f}s.")

    valid_slugs = {r["slug"] for r in results if r.get("has_real_stills") and r.get("has_script")}
    print(f"\n🧹 Cleaning up any obsolete placeholder directories...")
    pruned = 0
    for p in movies_dir.iterdir():
        if p.is_dir() and p.name not in valid_slugs:
            shot_manifest = p / "keyframes" / "shot_manifest.json"
            if not shot_manifest.exists():
                shutil.rmtree(p)
                pruned += 1
    print(f"🗑️ Pruned {pruned} obsolete directories.")

    final_manifest_path = movies_dir / "movies_manifest.json"
    clean_results = [r for r in results if r.get("has_real_stills") and r.get("has_script")]
    with open(final_manifest_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\n🏆 Final Verified Movie Count: {len(clean_results)} movies.")
    print(f"📄 Updated manifest: {final_manifest_path}")


if __name__ == "__main__":
    main()
