#!/usr/bin/env python3
"""
🎬 CLI RUNNER: ORCHESTRATED CINEMA PRE-MORTEM 🎬

Script mode (--script) or concept mode (--logline). The numbers come from the
deterministic orchestrator; the prose and the research claims come from the two
Gemini subagents. A degraded run says so instead of pretending.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import Orchestrator
from src.premortem.premortem_agent import PreMortemAgent
from src.search.parallel_search_client import ParallelSearchClient
from src.agents.research_agent import ResearchAgent


def run_cli():
    parser = argparse.ArgumentParser(description="Orchestrated cinema pre-mortem")
    parser.add_argument("--script", type=str, help="Path to screenplay (.txt, .fountain)")
    parser.add_argument("--logline", type=str, help="Pitch logline (2-4 sentences)")
    parser.add_argument("--keyframes", type=str, help="Directory of keyframe images")
    parser.add_argument("--title", type=str, default="Untitled Project")
    parser.add_argument("--genre", type=str, help="Genre; the agent infers one if omitted")
    parser.add_argument("--export", type=str, help="Write the full report JSON here")
    parser.add_argument("--mock", action="store_true", help="Force offline mock search")
    args = parser.parse_args()

    if not args.script and not args.logline:
        parser.print_help()
        sys.exit(1)

    script_text = None
    if args.script:
        script_text = Path(args.script).read_text(encoding="utf-8", errors="replace")

    orchestrator = Orchestrator(
        research_agent=ResearchAgent(
            search_client=ParallelSearchClient(force_mock=args.mock)
        )
    )
    story = args.logline or f"Screenplay submission: {args.title}."

    print("=" * 70)
    print("🎬  ORCHESTRATED CINEMA PRE-MORTEM  🎬")
    print("=" * 70)
    print(f"Mode: {'script' if script_text else 'concept'} | Title: {args.title}")

    report = PreMortemAgent(orchestrator=orchestrator).run_premortem(
        story=story,
        script_text=script_text,
        keyframes_dir=args.keyframes,
        genre=args.genre,
        title=args.title,
    )

    if report["degraded"]:
        print("\n⚠️  DEGRADED RUN:")
        for reason in report["degradation_reasons"]:
            print(f"  • {reason}")

    p = report["prediction"]
    print("\n" + "-" * 70)
    print(f"📊 PROJECTED {p['expected_rating']}/10 vs {p['genre_baseline_rating']}/10 "
          f"genre prior (residual {p['craft_residual_delta']:+.2f}, "
          f"cv_mae ±{p['cv_mae']}, cv_r2 {p['cv_r2']})")
    print(f"🎭 Genre: {report['genre']}"
          + ("" if report["genre"] == report["inferred_genre"] else
             f" (agent inferred: {report['inferred_genre']})"))
    print("-" * 70)

    if report["script_metrics"]:
        print("\n⚙️  CRAFT METRICS (measured):")
        for k, v in report["script_metrics"].items():
            print(f"  • {k:24s}: {v}")

    print("\n👁️  VISION MEASUREMENTS:")
    v = report["vision"]
    print(f"  • frames: {v['image_count']} | mean luminance: {v['mean_luminance']} "
          f"| dark frame ratio: {v['dark_frame_ratio']}")

    print("\n⚠️  DETERMINISTIC RISK FLAGS:")
    for f in report["risk_flags"] or []:
        print(f"  ❌ [{f['category']} — {f['severity']}] {f['issue']}: {f['detail']}")
    if not report["risk_flags"]:
        print("  ✅ none above threshold")

    print("\n🧮 COUNTERFACTUAL SWEEP (single-variable, deterministic):")
    for row in report["sweep"]:
        tag = " (inside the noise floor)" if row["inside_noise_floor"] else ""
        print(f"  • {row['feature']:22s} {row['value']:>8} {row['delta']:+.3f}{tag}")

    print(f"\n🌐 RESEARCH CLAIMS ({len(report['claims'])} survived grounding, "
          f"{len(report['dropped_claims'])} dropped):")
    for c in report["claims"]:
        print(f"  [{c['category']} / {c['valence']}] {c['claim']}")
        print(f"     evidence: \"{c['evidence'][:140]}\"")
        print(f"     source:   {c['source_url']}")

    synthesis = report["synthesis"]
    if synthesis:
        print("\n📝 SUMMARY:")
        print(f"  {synthesis['summary']}")
        print("\n🔎 PER-SLOT FINDINGS:")
        for f in synthesis["findings"]:
            print(f"  [{f['slot']}] {f['finding']}")
        print("\n💡 RECOMMENDATIONS:")
        sweep_flags = {r["feature"]: r["inside_noise_floor"] for r in report["sweep"]}
        for r in synthesis["recommendations"]:
            tag = ""
            if r["feature"] in sweep_flags:
                tag = (" [sweep row inside the noise floor]" if sweep_flags[r["feature"]]
                       else " [sweep row clears the noise floor]")
            print(f"  • {r['text']}{tag}")

    if report["tool_events"]:
        print(f"\n🔧 TOOL-CALL EVENTS: {len(report['tool_events'])}")
        for e in report["tool_events"]:
            print(f"  • {e['phase']}: {e['tool']} {e.get('args', {})}")

    if args.export:
        out = Path(args.export).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\n💾 Saved report to: {out}")


if __name__ == "__main__":
    run_cli()
