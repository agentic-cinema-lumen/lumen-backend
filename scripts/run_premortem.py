#!/usr/bin/env python3
"""
🎬 CLI RUNNER: AGENTIC CINEMA PRE-MORTEM & GREENLIGHT COMPASS 🎬

Executes the autonomous Pre-Mortem evaluation on:
- Mode 1: Full Script / Draft + Keyframes
- Mode 2: Logline Idea + Keyframe Moodboard
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.premortem.premortem_agent import PreMortemAgent
from src.search.parallel_search_client import ParallelSearchClient
from src.utils.llm_client import LLMClient


def print_banner():
    print("=" * 70)
    print("🎬  AGENTIC CINEMA DETECTIVE: PRE-MORTEM & GREENLIGHT ENGINE  🎬")
    print("=" * 70)


def run_cli():
    parser = argparse.ArgumentParser(description="Autonomous Cinema Pre-Mortem & Craft Simulator")
    parser.add_argument("--script", type=str, help="Path to screenplay (.txt, .fountain)")
    parser.add_argument("--logline", type=str, help="Pitch logline (2-4 sentences)")
    parser.add_argument("--pitch-file", type=str, help="Path to pitch JSON file containing logline and metadata")
    parser.add_argument("--keyframes", type=str, default="data/movies/alien/keyframes", help="Directory or path with keyframe images")
    parser.add_argument("--title", type=str, default="Untitled Project", help="Title of script or concept")
    parser.add_argument("--show", type=str, default="Prestige Series", help="Parent show name (for scripts)")
    parser.add_argument("--export", type=str, help="Optional output JSON file path for report")
    parser.add_argument("--mock", action="store_true", help="Force offline mock mode for APIs")

    args = parser.parse_args()
    print_banner()

    llm = LLMClient(force_mock=args.mock)
    search_client = ParallelSearchClient(force_mock=args.mock)
    agent = PreMortemAgent(llm_client=llm)
    agent.trope_sleuth.client = search_client

    report = None

    # Determine execution mode
    if args.script:
        print(f"\n📑 [Mode 1: Script Pre-Mortem] Ingesting script: {args.script}")
        print(f"📸 Keyframes: {args.keyframes}")
        report = agent.run_script_premortem(
            script_path_or_text=args.script,
            keyframes_path_or_dir=args.keyframes,
            title=args.title,
            show_name=args.show
        )

        print("\n" + "-" * 70)
        print(f"📊 PROJECTED RATING: {report['projected_baseline_rating']} / 10.0")
        print(f"📈 Series Historical Mean: {report['historical_show_mean']} | Residual Delta: {report['projected_residual_delta']:+0.2f}")
        print("-" * 70)

        print("\n⚙️  CRAFT METRICS:")
        for k, v in report["craft_metrics"].items():
            print(f"  • {k:24s}: {v}")

        print("\n👁️  VISUAL CRAFT ASSESSMENT:")
        print(f"  {report['vision_summary']}")

        print("\n⚠️  DETECTED CRAFT FLAWS & RESIDUAL DRAG:")
        if report["detected_craft_flaws"]:
            for f in report["detected_craft_flaws"]:
                print(f"  ❌ [{f['category']} - {f['severity']}] {f['finding']}")
                print(f"     Impact: {f['predicted_penalty']}")
        else:
            print("  ✅ No severe craft drags identified in baseline script dynamics.")

        print("\n🌐 PARALLEL SEARCH AUDIENCE SLEUTH (Reddit & Critic Archives):")
        for c in report["audience_trope_intelligence"]["audience_consensus_claims"]:
            valence_icon = "🔻" if c["valence"] == "negative" else "🔹"
            print(f"  {valence_icon} [{c['category']}] {c['claim']}")
            if c.get("evidence"):
                print(f"     Snippet: \"{c['evidence'][:110]}...\"")

        print("\n💡 ACTIONABLE PRODUCTION RECOMMENDATIONS:")
        for r in report["recommendations"]:
            print(f"  • {r}")

    elif args.logline or args.pitch_file:
        logline = args.logline
        title = args.title
        keyframes = args.keyframes

        if args.pitch_file:
            with open(args.pitch_file, "r", encoding="utf-8") as f:
                pdata = json.load(f)
                logline = pdata.get("logline", logline)
                title = pdata.get("title", title)
                if "moodboard_dir" in pdata:
                    keyframes = pdata["moodboard_dir"]

        print(f"\n💡 [Mode 2: Greenlight Compass] Pitch: \"{title}\"")
        print(f"📝 Logline: {logline}")
        print(f"📸 Moodboard: {keyframes}")

        report = agent.run_premise_premortem(
            logline=logline,
            keyframes_path_or_dir=keyframes,
            title=title
        )

        print("\n" + "-" * 70)
        target = report['projected_rating_potential']
        print(f"🎯 PROJECTED POTENTIAL: {target['median_target']} / 10.0 (Floor: {target['floor']} | Ceiling: {target['ceiling']})")
        print(f"🎭 Genre: {report['genre']} | Tone: {report['detected_tone']}")
        print(f"🎨 Aesthetic Cohesion: {report['style_premise_cohesion']['rating']} ({report['style_premise_cohesion']['evaluation']})")
        print("-" * 70)

        comps = report.get("comparable_movie_precedents", [])
        if comps:
            print("\n🎞️  HISTORICAL MOVIE PRECEDENTS (IMDb Database Matches):")
            for c in comps:
                pol_str = f" | Pol: {c['polarization_index']:.1%}" if c.get("polarization_index") else ""
                print(f"  • {c['title']} ({c['year']}) — IMDb: {c['imdb_rating']}/10 ({c['total_votes']:,} votes{pol_str})")
                print(f"    Genre: {c['genre']} | Dir: {c['director']}")
                print(f"    Premise: \"{c['logline'][:100]}...\"")

        print("\n📌 MAKE-OR-BREAK CRAFT DEPENDENCIES:")
        for d in report["make_or_break_dependencies"]:
            print(f"  • {d}")

        print("\n🌐 PARALLEL SEARCH AUDIENCE FATIGUE RADAR:")
        for c in report["audience_fatigue_radar"]["audience_consensus_claims"]:
            valence_icon = "🔻" if c["valence"] == "negative" else "🔹"
            print(f"  {valence_icon} [{c['category']}] {c['claim']}")

        print("\n⚖️  FINAL GREENLIGHT VERDICT:")
        print(f"  {report['greenlight_verdict']}")

    else:
        parser.print_help()
        sys.exit(1)

    # Export if requested
    if args.export and report:
        out_p = Path(args.export).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Saved Pre-Mortem report to: {out_p}")


if __name__ == "__main__":
    run_cli()
