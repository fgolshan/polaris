#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Recursively remove all 'raw' subdirectories under a given base folder."
    )
    parser.add_argument(
        "base",
        help="Path to the base folder (or a folder of test folders) to clean up.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists() or not base.is_dir():
        print(f"[ERROR] Base folder {base!r} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    removed = 0
    # '**/raw' matches any directory named 'raw' under base (including base/raw)
    for raw_dir in base.glob("**/raw"):
        if raw_dir.is_dir():
            try:
                shutil.rmtree(raw_dir)
                print(f"[INFO] Removed: {raw_dir}")
                removed += 1
            except Exception as e:
                print(f"[WARN] Failed to remove {raw_dir}: {e}", file=sys.stderr)

    if removed == 0:
        print("[INFO] No 'raw' directories found.")
    else:
        print(f"[INFO] Total 'raw' directories removed: {removed}")

if __name__ == "__main__":
    main()
