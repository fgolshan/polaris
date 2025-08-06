#!/usr/bin/env python3
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Wrapper: gather interface throughputs (step 1) and plot/utilization (step 2)"
    )
    # step 1 flags
    parser.add_argument(
        "--base", "-b", required=True,
        help="Base output dir (same as traffic generator output)"
    )
    parser.add_argument(
        "--direction", choices=["input", "output", "both"], default="both",
        help="Which direction(s) to process: input, output, or both"
    )
    parser.add_argument(
        "--isd-as", dest="isd_as",
        help="Optional ISD-AS to restrict to (e.g. 1-111)"
    )
    # step 2 flags
    parser.add_argument(
        "--capacity", type=float, default=None,
        help="(step 2) capacity in Mbps; if given, normalize y-axis to utilization (%%)"
    )
    parser.add_argument(
        "--types", default="external,local",
        help="(step 2) comma-separated types to plot: external,local"
    )
    parser.add_argument(
        "--out-dir", "-o", dest="out_dir", default=None,
        help="(step 2) where to write plots (default: metrics/interface_utilization_step2 under base)"
    )

    args = parser.parse_args()

    # --- STEP 1: collect CSVs ---
    dirs = []
    if args.direction in ("input", "both"):
        dirs.append("input")
    if args.direction in ("output", "both"):
        dirs.append("output")

    for d in dirs:
        cmd = [
            sys.executable, "plot_interface_data_step1.py",
            "--base", args.base,
            "--direction", d,
        ]
        if args.isd_as:
            cmd += ["--isd-as", args.isd_as]

        print(f"[RUNNING] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # --- STEP 2: plot ---
    cmd2 = [
        sys.executable, "plot_interface_data_step2.py",
        "--base", args.base,
        "--types", args.types,
    ]
    if args.isd_as:
        cmd2 += ["--isd-as", args.isd_as]
    if args.capacity is not None:
        cmd2 += ["--capacity", str(args.capacity)]
    if args.out_dir:
        cmd2 += ["--out-dir", args.out_dir]

    print(f"[RUNNING] {' '.join(cmd2)}")
    subprocess.run(cmd2, check=True)

if __name__ == "__main__":
    main()
