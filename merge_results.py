"""
Merges all *_results.csv files found in timestamped subfolders.

Expected structure:
  <root>/
    <timestamp>/          e.g. 20260623_155754/
      <timestamp>_results.csv

Produces a single merged CSV at <root>/merged_results.csv with an
extra 'source_folder' column so each row is traceable.
"""

import sys
from pathlib import Path
import pandas as pd


def find_results_csvs(root: Path) -> list[Path]:
    return sorted(root.glob("*/*_results.csv"))


def merge(root: Path, output_path: Path) -> None:
    files = find_results_csvs(root)

    if not files:
        print(f"No *_results.csv files found under: {root}")
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    frames = []
    for f in files:
        print(f"  {f.relative_to(root)}")
        df = pd.read_csv(f)
        ts = f.parent.name  # e.g. 20260623_155754
        try:
            dt = pd.to_datetime(ts, format="%Y%m%d_%H%M%S")
            date_val = dt.strftime("%Y-%m-%d")
            time_val = dt.strftime("%H:%M:%S")
        except ValueError:
            date_val = ""
            time_val = ""
        df.insert(0, "source_folder", ts)
        df.insert(1, "date", date_val)
        df.insert(2, "time", time_val)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"\nMerged {len(merged):,} rows → {output_path}")


if __name__ == "__main__":
    # Default: look in the 'output' subfolder next to this script
    script_dir = Path(__file__).parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "output"
    out  = Path(sys.argv[2]) if len(sys.argv) > 2 else script_dir / "merged_results.csv"

    merge(root, out)