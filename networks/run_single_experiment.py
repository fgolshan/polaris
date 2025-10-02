#!/usr/bin/env python3
"""
Run a full Polaris experiment: generation, aggregation, and plotting.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, **kwargs):
    print(f"\n>>> Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error: command {' '.join(cmd)} failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single full Polaris experiment: generate traffic, aggregate metrics, and plot results."
    )
    parser.add_argument("spec_file", help="Path to the traffic generator spec file")
    parser.add_argument("out_dir", help="Directory where all outputs will be written")
    parser.add_argument("--capacity", type=float,
                        help="Optional link capacity in Mbps for normalization and summaries")
    parser.add_argument("--tcspec", dest="tc_spec", default=None,
                        help="Optional JSON file describing tc events to execute during the experiment")
    parser.add_argument("--pcaspec", dest="pca_spec", default=None,
                        help="Optional JSON file describing pca events to execute during the experiment")
    args = parser.parse_args()

    spec = Path(args.spec_file)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Traffic generation
    traffic_command = [sys.executable, "traffic_generator_v3.py", str(spec), str(out)]
    if args.tc_spec is not None:
        traffic_command += ["--tcspec", args.tc_spec]
    if args.pca_spec is not None:
        traffic_command += ["--pcaspec", args.pca_spec]
    run_cmd(traffic_command)

    # 2) Plot path switches
    run_cmd([sys.executable, "plot_path_switches_grouped.py", "--base", str(out)])

    # 3) Aggregation
    run_cmd([sys.executable, "aggregator.py", str(out)])

    # 4) Interface data plotting (step 1 & 2)
    #    direction both implies running step1 twice inside plot_interface_data.py if supported
    plot_int_cmd = [sys.executable, "plot_interface_data.py", "--base", str(out), "--direction", "both"]
    if args.capacity is not None:
        plot_int_cmd += ["--capacity", str(args.capacity)]
    run_cmd(plot_int_cmd)

    # 5) Interface summary plotting
    summary_cmd = [sys.executable, "plot_interface_summary.py", "--base", str(out)]
    if args.capacity is not None:
        summary_cmd += ["--capacity", str(args.capacity)]
    run_cmd(summary_cmd)

    # 6) Write capacity.txt if needed
    if args.capacity is not None:
        cap_file = out / "capacity.txt"
        with open(cap_file, "w") as f:
            f.write(f"{args.capacity} Mbps\n")
        print(f"Wrote capacity to {cap_file}")

    # 7) Loss Stats
    loss_stats_cmd = [sys.executable, "compute_loss_stats.py", "--base", str(out), "--print"]
    run_cmd(loss_stats_cmd)

    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()
