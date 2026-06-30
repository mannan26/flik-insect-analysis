"""
Combines all JSONL files in the sensors/ folder into a single CSV.

Each .jsonl file must have one JSON object per line:
  {"ts": "...", "value": 1.2}
  {"ts": "...", "value": 3.4}

A 'source_file' column is added so each row is traceable.

Usage:
  python merge_sensors.py                        # uses defaults below
  python merge_sensors.py <sensors_dir> <output_csv>
"""

import sys
import json
from pathlib import Path
import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] {path.name} line {i}: {e}")
    return records


def merge(sensors_dir: Path, output_csv: Path) -> None:
    files = sorted(sensors_dir.glob("*.jsonl"))
    if not files:
        print(f"No JSON files found in: {sensors_dir}")
        sys.exit(1)

    print(f"Found {len(files)} JSON file(s):")
    frames = []
    for f in files:
        records = load_jsonl(f)
        print(f"  {f.name}  ({len(records)} record(s))")
        df = pd.DataFrame(records)
        df.insert(0, "source_file", f.name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"\nMerged {len(merged):,} rows → {output_csv}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    sensors_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "sensor"
    output_csv  = Path(sys.argv[2]) if len(sys.argv) > 2 else script_dir / "merged_sensors.csv"

    merge(sensors_dir, output_csv)
