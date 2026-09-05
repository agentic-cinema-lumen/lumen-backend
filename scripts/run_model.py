#!/usr/bin/env python3
"""
🎬 CLI RUNNER: QUANT RESIDUAL ML ORACLE 🎬

Standalone CLI tool to query the trained Quant ML model:
- Mode 1: Evaluate a complete movie package (script + keyframes + metadata)
- Mode 2: Evaluate a custom screenplay file and keyframes directory
- Mode 3: Evaluate ad-hoc craft parameters (cutting tempo, visual darkness, pacing acceleration)
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quant.oracle import get_oracle


def print_banner():
    print("=" * 70)
    print("🎬  QUANT RESIDUAL ML MODEL ORACLE (PRE-PRODUCTION INFERENCE)  🎬")
    print("=" * 70)


def run_cli():
    parser = argparse.ArgumentParser(description="Query the trained Quant Residual ML Model")
    parser.add_argument("--movie", type=str, help="Path to movie package directory (e.g. data/movies/apocalypse_now_1979)")
    parser.add_argument("--script", type=str, help="Path to screenplay (.txt)")
    parser.add_argument("--keyframes", type=str, help="Path to keyframes directory")
    parser.add_argument("--genre", type=str, default="Drama", help="Project genre (e.g. Sci-Fi, Drama, Horror)")
    parser.add_argument("--title", type=str, default="Untitled Project", help="Project title")
    parser.add_argument("--runtime", type=float, default=None, help="Runtime in minutes")
    parser.add_argument("--cpm", type=float, default=None, help="Cuts per minute")
    parser.add_argument("--pacing-acc", type=float, default=None, help="Third-act climax acceleration")
    parser.add_argument("--dark-ratio", type=float, default=None, help="Dark keyframe ratio (0.0 to 1.0)")
    parser.add_argument("--luminance", type=float, default=None, help="Mean keyframe luminance (0 to 255)")
    parser.add_argument("--wpm", type=float, default=None, help="Dialogue velocity in words per minute")
    parser.add_argument("--export", type=str, help="Save evaluation JSON to destination file")
    parser.add_argument("--tool-spec", action="store_true", help="Print standard LLM / MCP tool schema")
    parser.add_argument("--decision-history", action="store_true", help="Print ML Engineering Agent decision history markdown")

    args = parser.parse_args()

    oracle = get_oracle()

    if args.decision_history:
        print(oracle.get_decision_history())
        return

    if args.tool_spec:
        print(json.dumps(oracle.get_tool_spec(), indent=2))
        return

    print_banner()

    result = None

    if args.movie:
        print(f"\n📦 [Mode 1: Movie Package] Evaluating directory: {args.movie}")
        result = oracle.predict_movie_package(args.movie)
    elif args.script:
        print(f"\n📑 [Mode 2: Script & Stills] Evaluating screenplay: {args.script}")
        if args.keyframes:
            print(f"📸 Keyframes: {args.keyframes}")
        result = oracle.predict_from_script_and_keyframes(
            script_path_or_text=args.script,
            keyframes_path_or_dir=args.keyframes,
            genre=args.genre,
            title=args.title
        )
    else:
        # Parameter mode
        params = {"genre": args.genre, "title": args.title}
        if args.runtime is not None:
            params["total_duration_min"] = args.runtime
        if args.cpm is not None:
            params["cuts_per_minute"] = args.cpm
        if args.pacing_acc is not None:
            params["pacing_acceleration"] = args.pacing_acc
        if args.dark_ratio is not None:
            params["dark_frame_ratio"] = args.dark_ratio
        if args.luminance is not None:
            params["mean_luminance"] = args.luminance
        if args.wpm is not None:
            params["words_per_minute"] = args.wpm

        print(f"\n⚙️  [Mode 3: Direct Craft Parameters] Title: \"{args.title}\" | Genre: {args.genre}")
        result = oracle.predict_craft(params)

    # Display results
    print("\n" + "-" * 70)
    print(f"📊 PROJECTED RATING: {result['expected_rating']:.2f} / 10.0")
    print(f"🎯 Genre Baseline ({result['genre']}): {result['genre_baseline_rating']:.2f} | Craft Residual Delta: {result['craft_residual_delta']:+0.2f}")
    ci = result['confidence_interval']
    print(f"🔒 Confidence Interval (±{ci['margin_of_error']}): [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    print(f"⚖️  Verdict: {result['craft_verdict']}")
    print(f"   {result['verdict_summary']}")
    print("-" * 70)

    if result.get("craft_vulnerabilities"):
        print("\n⚠️  ACTIONABLE CRAFT VULNERABILITIES:")
        for v in result["craft_vulnerabilities"]:
            print(f"  ❌ [{v['category']} - {v['severity']}] {v['issue']}")
            print(f"     Detail: {v['detail']}")
            print(f"     Remedy: {v['remedy']}")
    else:
        print("\n✅ Zero severe craft vulnerabilities detected in baseline parameters.")

    print("\n📈 TOP CRAFT FEATURE ATTRIBUTIONS:")
    for a in result["top_craft_attributions"]:
        direction_icon = "🟢" if a["direction"] == "positive" else ("🔴" if a["direction"] == "negative" else "⚪")
        print(f"  {direction_icon} {a['feature']:22s}: {a['point_impact']:+0.3f} pts (input: {a['input_value']})")
        print(f"     {a['interpretation']}")

    meta = result["model_metadata"]
    print(f"\n🤖 Champion Model: {meta['model_type'].upper()} (CV MAE: {meta['cv_mae']}, CV RMSE: {meta['cv_rmse']}, N={meta['training_samples']} real films)")

    if args.export:
        out_p = Path(args.export).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved prediction results to: {out_p}")


if __name__ == "__main__":
    run_cli()
