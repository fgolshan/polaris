#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Filenames we treat as "reserved" optional specs and should not be used
# as the positional spec by accident during legacy scanning.
RESERVED_OPTIONAL_SPEC_BASENAMES = {"pcaspec.json", "tcspec.json"}

def find_positional_spec(subdir: Path, explicit_name: Optional[str]) -> Path | None:
    """
    Find the positional spec file in `subdir`.

    - If `explicit_name` is given, return subdir/explicit_name if it exists; else None.
    - Otherwise (legacy mode), return the single *spec.json file, excluding reserved
      optional spec names (pcaspec.json, tcspec.json). Warn if multiple remain.
    """
    if explicit_name:
        p = subdir / explicit_name
        if p.is_file():
            return p
        return None

    # Legacy behavior: search *spec.json (non-recursive), but exclude reserved optional spec names.
    specs = sorted(
        p for p in subdir.iterdir()
        if p.is_file()
        and p.name.endswith("spec.json")
        and p.name not in RESERVED_OPTIONAL_SPEC_BASENAMES
    )
    if not specs:
        return None
    if len(specs) > 1:
        print(f"[WARN] Multiple candidate positional spec files in {subdir}, using {specs[0].name}",
              file=sys.stderr)
    return specs[0]

def find_optional_spec(subdir: Path, filename: Optional[str]) -> Path | None:
    """
    Return subdir/filename if filename is provided and exists, else None.
    """
    if not filename:
        return None
    p = subdir / filename
    return p if p.is_file() else None

def main():
    parser = argparse.ArgumentParser(
        description="Run run_single_experiment.py over each subfolder that contains a positional *spec.json."
    )
    parser.add_argument("base", help="Base directory containing per-experiment subfolders")

    # Existing passthrough options
    parser.add_argument("--capacity", type=float, default=None,
                        help="Optional capacity in Mbps; passed through to run_single_experiment.py")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop immediately if any experiment fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands but do not execute")

    # New controls for spec resolution
    parser.add_argument(
        "--spec-name",
        default=None,
        help=("Filename to use for the mandatory positional spec in each subfolder "
              "(e.g., testspec.json). If omitted, falls back to legacy search for *spec.json "
              "excluding pcaspec.json and tcspec.json.")
    )
    parser.add_argument(
        "--pcaspec",
        default="pcaspec.json",
        help=("Filename for optional PCA spec in each subfolder. If the file exists, "
              "it will be passed as --pcaspec to run_single_experiment.py. "
              "Use a different name to override; set to '' to disable lookup.")
    )
    parser.add_argument(
        "--tcspec",
        default="tcspec.json",
        help=("Filename for optional TC spec in each subfolder. If the file exists, "
              "it will be passed as --tcspec to run_single_experiment.py. "
              "Use a different name to override; set to '' to disable lookup.")
    )

    args = parser.parse_args()

    # Normalize "disable" empty strings to None
    pcaspec_name = args.pcaspec if args.pcaspec.strip() else None
    tcspec_name = args.tcspec if args.tcspec.strip() else None

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
        spec = find_positional_spec(d, args.spec_name)
        if not spec:
            # Decide what to tell the user based on mode
            if args.spec_name:
                print(f"[WARN] Missing positional spec '{args.spec_name}' in {d}; skipping", file=sys.stderr)
            else:
                print(f"[WARN] No *spec.json in {d}; skipping", file=sys.stderr)
            continue

        # Optional specs: only pass if present
        pca_spec = find_optional_spec(d, pcaspec_name)
        tc_spec = find_optional_spec(d, tcspec_name)

        cmd = [sys.executable, str(single_runner), str(spec), str(d)]
        if args.capacity is not None:
            cmd += ["--capacity", str(args.capacity)]
        if pca_spec is not None:
            cmd += ["--pcaspec", str(pca_spec)]
        if tc_spec is not None:
            cmd += ["--tcspec", str(tc_spec)]

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