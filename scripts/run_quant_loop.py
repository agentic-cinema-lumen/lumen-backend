#!/usr/bin/env python3
"""
🎬 QUANT AGENT LOOP CLI 🎬

Executes the autonomous Quant ML training loop and generates residual evaluations.
Supports both classic tournament mode and full Agentic LLM feature engineering.

Usage:
  # Run full agentic training loop (Gemini/OpenAI or intelligent mock)
  python scripts/run_quant_loop.py --target got_s08e03_long_night

  # Force mock mode (offline / zero API keys)
  python scripts/run_quant_loop.py --target got_s08e03_long_night --mock

  # Run standard statistical tournament without LLM
  python scripts/run_quant_loop.py --target space --no-agentic
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.quant.quant_agent import QuantAgent
from src.quant.agentic_trainer import AgenticQuantTrainer


ASCII_ART = r"""
     _________________________________________
    |                                         |
    |   📊  QUANT RESIDUAL AGENT LOOP  📊     |
    |_________________________________________|
     Craft Pacing & Visual Residual Analytics
"""


def main():
    parser = argparse.ArgumentParser(description="🎬 Quant ML Residual Agent Loop")
    parser.add_argument(
        "--target", "-t",
        default="got_s08e03_long_night",
        help="Target episode ID (e.g. got_s08e03_long_night, got_s08e05_the_bells, succ_s04e03_connors_wedding, space, vikings)"
    )
    parser.add_argument(
        "--rating", "-r",
        type=float,
        default=None,
        help="Optional override for actual IMDb rating"
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        default=True,
        help="Run autonomous LLM Agent loop for feature engineering & error analysis (default: True)"
    )
    parser.add_argument(
        "--no-agentic",
        action="store_false",
        dest="agentic",
        help="Run classic statistical tournament instead of LLM agent"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force mock LLM agent mode (offline, zero API keys)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of agentic reasoning rounds (default: 2)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--save-json", "-s",
        default=None,
        help="Optional path to save JSON diagnosis"
    )

    args = parser.parse_args()

    print(ASCII_ART)

    if args.agentic:
        print("==================================================")
        print("🤖 RUNNING AGENTIC ML SCIENTIST LOOP")
        print("==================================================")
        trainer = AgenticQuantTrainer(
            data_root=args.data_dir,
            force_mock=args.mock,
            max_rounds=args.rounds
        )
        summary = trainer.run_agentic_loop(exclude_target=args.target)

        print("\n==================================================")
        print(f"🎬 EVALUATING TARGET EPISODE: '{args.target}'")
        print("==================================================")
        result = trainer.evaluate_episode(episode_identifier=args.target, actual_rating=args.rating)
        eval_data = result["quant_evaluation"]

        print(f"\n📺 Show:            {result['show_name']}")
        print(f"🎬 Title:           {result['title']}")
        print(f"⭐ Actual Rating:   {eval_data['actual_rating']}")
        print(f"🎯 Expected Rating: {eval_data['expected_rating']}")
        residual_str = f"{eval_data['residual']:+0.2f}"
        print(f"⚖️ Residual Variance: {residual_str}")
        print(f"🚨 Anomaly Status:  {eval_data['anomaly_type']}")
        print(f"📝 Diagnosis:       {eval_data['description']}")

        if args.save_json:
            out_path = Path(args.save_json)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Saved full JSON report to: {out_path}")

    else:
        # Standard tournament loop
        agent = QuantAgent(data_root=args.data_dir)
        summary = agent.run_training_loop(exclude_target=args.target)
        result = agent.evaluate_episode(episode_identifier=args.target, actual_rating=args.rating)
        eval_data = result["quant_evaluation"]

        print(f"\n📺 Show:            {result['show_name']}")
        print(f"🎬 Title:           {result['title']}")
        print(f"⭐ Actual Rating:   {eval_data['actual_rating']}")
        print(f"🎯 Expected Rating: {eval_data['expected_rating']}")
        residual_str = f"{eval_data['residual']:+0.2f}"
        print(f"⚖️ Residual Variance: {residual_str}")
        print(f"🚨 Anomaly Status:  {eval_data['anomaly_type']}")


if __name__ == "__main__":
    main()
