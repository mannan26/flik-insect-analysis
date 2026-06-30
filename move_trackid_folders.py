"""
Moves all trackid folders from:
  <source_root>/<timestamp>/<timestamp>_crops/<trackid>/

to:
  <destination>/

Usage:
  python move_trackid_folders.py                          # uses defaults below
  python move_trackid_folders.py <source_root> <dest>
"""

import sys
import shutil
from pathlib import Path


def move_trackid_folders(source_root: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    crops_dirs = sorted(source_root.glob("*/*_crops"))
    if not crops_dirs:
        print(f"No *_crops folders found under: {source_root}")
        sys.exit(1)

    moved = 0
    skipped = 0

    for crops_dir in crops_dirs:
        timestamp = crops_dir.parent.name
        trackid_folders = [p for p in crops_dir.iterdir() if p.is_dir()]

        if not trackid_folders:
            print(f"  [empty] {crops_dir.relative_to(source_root)}")
            continue

        print(f"\n{timestamp}_crops  ({len(trackid_folders)} track folders)")
        for track in sorted(trackid_folders):
            target = dest / track.name
            if target.exists():
                print(f"  [skip] {track.name} already exists in destination")
                skipped += 1
            else:
                shutil.move(str(track), target)
                print(f"  [moved] {track.name}")
                moved += 1

    print(f"\nDone — moved: {moved}, skipped: {skipped}")
    print(f"Destination: {dest}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Volumes/T7/videobug_analysis/output")
    dest        = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Users/mannangupta/Documents/workflows/Flik insect analysis/output")

    move_trackid_folders(source_root, dest)