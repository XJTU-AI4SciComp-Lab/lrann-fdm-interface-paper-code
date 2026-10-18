"""Run the supplied table cases and collect their repeated-run means in CSV."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrann_interface.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    table = json.loads(args.config.read_text())
    rows = []
    for index, case in enumerate(table["cases"][:args.limit], 1):
        config = {**table["base"], **case}
        config["arguments"] = table["base"].get("arguments", []) + case.get("arguments", [])
        config["name"] = f"{table['name']}_case_{index:03d}"
        if args.repeats is not None:
            config["repeats"] = args.repeats
        # Match independent fresh-process table runs, including their initial dtype.
        results = run(config)
        keys = [key for key in results[0] if key != "run"]
        rows.append({"case": index, "label": case.get("label", ""), **{
            key: sum(item[key] for item in results) / len(results) for key in keys
        }})
        destination = ROOT / "results" / table["name"]
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "table.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {destination / 'table.csv'}")


if __name__ == "__main__":
    main()
