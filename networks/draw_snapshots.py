#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small convenience wrapper for draw_snapshots_param.py

Usage:
  # Single run (default)
  python draw_snapshots.py /path/to/base

  # Multiple: run for each subfolder inside /path/to/base
  python draw_snapshots.py /path/to/base --multiple
"""
import argparse
import sys
import subprocess
from pathlib import Path

def resolve_param_script() -> str:
    """
    Prefer draw_snapshots_param.py next to this script; fall back to name on PATH.
    """
    here = Path(__file__).resolve().parent
    candidate = here / "draw_snapshots_param.py"
    return str(candidate) if candidate.is_file() else "draw_snapshots_param.py"

def run_once(base: Path, param_script: str) -> int:
    outdir = base / "frames"
    cmd = [
        sys.executable, str(param_script), str(base),
        "--mode", "switches",
        "--outdir", str(outdir),
        "--label-curve-scale", "3",
    ]
    print(">>", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode

def main():
    ap = argparse.ArgumentParser(description="Run draw_snapshots_param.py conveniently.")
    ap.add_argument("base", type=Path, help="Base folder path")
    ap.add_argument("--multiple", action="store_true",
                    help="If set, run once for each immediate subfolder of the base path.")
    args = ap.parse_args()

    param_script = resolve_param_script()

    if not args.multiple:
        sys.exit(run_once(args.base, param_script))
    else:
        base = args.base
        if not base.is_dir():
            print(f"[!] --multiple expects a directory; got: {base}", file=sys.stderr)
            sys.exit(1)
        last_rc = 0
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            print(f"\n=== Processing: {sub} ===")
            rc = run_once(sub, param_script)
            if rc != 0:
                last_rc = rc
        sys.exit(last_rc)

if __name__ == "__main__":
    main()
