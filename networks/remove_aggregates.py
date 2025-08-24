#!/usr/bin/env python3
"""
remove_aggregates.py

Recursively find and delete folders named "aggregated" under a given base path.
Everything else is left untouched.

Usage:
    python remove_aggregates.py /path/to/base
    python remove_aggregates.py /path/to/base --dry-run  # show what would be deleted
"""

import argparse
import os
import shutil
import sys


def remove_aggregated(base_path: str, dry_run: bool = False) -> int:
    """
    Delete directories named 'aggregated' under base_path (recursively).

    Returns the number of directories deleted (or that would be deleted in dry-run).
    """
    deleted = 0
    base_path = os.path.abspath(base_path)

    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}", file=sys.stderr)
        return 0
    if not os.path.isdir(base_path):
        print(f"Error: Base path is not a directory: {base_path}", file=sys.stderr)
        return 0

    # Walk without following symlinks to avoid surprises.
    for root, dirs, _files in os.walk(base_path, topdown=True, followlinks=False):
        # Work on a copy since we may modify dirs in-place.
        for dname in list(dirs):
            if dname == "aggregated":
                target = os.path.join(root, dname)

                # Prevent os.walk from descending into it after we remove it.
                try:
                    dirs.remove(dname)
                except ValueError:
                    pass

                if dry_run:
                    print(f"[DRY-RUN] Would delete: {target}")
                    deleted += 1
                    continue

                try:
                    shutil.rmtree(target)
                    print(f"Deleted: {target}")
                    deleted += 1
                except Exception as e:
                    print(f"Failed to delete {target}: {e}", file=sys.stderr)

    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Recursively delete directories named "aggregated" under BASE.'
    )
    parser.add_argument("base", help="Base folder to search under")
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without removing anything"
    )
    args = parser.parse_args()

    count = remove_aggregated(args.base, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[DRY-RUN] {count} 'aggregated' director{'y' if count == 1 else 'ies'} found.")
    else:
        print(f"{count} 'aggregated' director{'y' if count == 1 else 'ies'} deleted.")


if __name__ == "__main__":
    main()
