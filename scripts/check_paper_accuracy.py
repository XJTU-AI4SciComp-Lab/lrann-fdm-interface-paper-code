"""Run paper-parameter checks and compare distributions of error values."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrann_interface.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/accuracy_checks.json")
    parser.add_argument("--case", help="Run only the selected case name")
    parser.add_argument("--repeats", type=int, help="Override the configured run count")
    args = parser.parse_args()
    cases = json.loads(args.config.read_text())["cases"]
    if args.case:
        cases = [case for case in cases if case["name"] == args.case]
        if not cases:
            parser.error(f"Unknown case: {args.case}")
    summaries = []
    for case in cases:
        config = dict(case["config"])
        config["name"] = "accuracy_" + case["name"]
        if args.repeats is not None:
            config["repeats"] = args.repeats
        with threadpool_limits(limits=1):
            rows = run(config)
        values = [row[case["metric"]] for row in rows]
        summaries.append({
            "case": case["name"], "paper_location": case["paper_location"],
            "metric": case["metric"], "paper_value": case["paper_value"],
            "runs": len(values), "mean": float(np.mean(values)),
            "std": float(np.std(values)), "min": min(values), "max": max(values),
            "mean_to_paper": float(np.mean(values)) / case["paper_value"],
        })
        destination = ROOT / "results/paper_accuracy_latest.csv"
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"{case['name']}: mean {np.mean(values):.4e}; paper {case['paper_value']:.4e}", flush=True)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
