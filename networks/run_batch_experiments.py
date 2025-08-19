#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path

def find_spec(subdir: Path) -> Path | None:
    """Return the single *spec.json file in subdir (non-recursive), or None."""
    specs = sorted(p for p in subdir.iterdir()
                   if p.is_file() and p.name.endswith("spec.json"))
    if not specs:
        return None
    if len(specs) > 1:
        print(f"[WARN] Multiple spec files in {subdir}, using {specs[0].name}", file=sys.stderr)
    return specs[0]

def main():
    parser = argparse.ArgumentParser(
        description="Run run_single_experiment.py over each subfolder that contains a *spec.json."
    )
    parser.add_argument("base", help="Base directory containing per-experiment subfolders")
    parser.add_argument("--capacity", type=float, default=None,
                        help="Optional capacity in Mbps; passed through to run_single_experiment.py")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop immediately if any experiment fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands but do not execute")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    if not base.is_dir():
        print(f"[ERROR] {base} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Resolve the single-experiment runner (assumed to be alongside this script)
    script_dir = Path(__file__).resolve().parent
    single_runner = script_dir / "run_single_experiment.py"
    if not single_runner.exists():
        print(f"[ERROR] {single_runner} not found (expected next to this script).", file=sys.stderr)
        sys.exit(1)

    subdirs = sorted(p for p in base.iterdir() if p.is_dir())
    if not subdirs:
        print(f"[INFO] No subfolders in {base}", file=sys.stderr)
        sys.exit(0)

    total = 0
    ok = 0
    failed = 0

    for d in subdirs:
        spec = find_spec(d)
        if not spec:
            print(f"[WARN] No *spec.json in {d}; skipping", file=sys.stderr)
            continue

        cmd = [sys.executable, str(single_runner), str(spec), str(d)]
        if args.capacity is not None:
            cmd += ["--capacity", str(args.capacity)]

        total += 1
        print(f">>> Running: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            ok += 1
            continue

        try:
            subprocess.run(cmd, check=True)
            ok += 1
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"[ERROR] Experiment in {d.name} failed with exit {e.returncode}", file=sys.stderr)
            if args.stop_on_error:
                break

    print(f"\nDone. attempted={total}, succeeded={ok}, failed={failed}")

if __name__ == "__main__":
    main()
