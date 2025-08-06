#!/usr/bin/env python3
import argparse
import csv
import os
import re
import statistics
from pathlib import Path

def parse_flow_map(flow_map_path: Path):
    """Read flow_map.txt and return dict mapping long_name → short_label."""
    mapping = {}
    with open(flow_map_path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                short, longn = parts
                mapping[longn] = short
    return mapping

def extract_losses(log_path: Path):
    """Parse one *_client.log file, return (s2c_loss, c2s_loss) as floats."""
    s2c = c2s = None
    lines = log_path.read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "S->C results":
            for l in lines[idx+1:idx+6]:
                m = re.match(r"Loss rate:\s*([\d\.]+)%", l)
                if m:
                    s2c = float(m.group(1))
                    break
        if line.strip() == "C->S results":
            for l in lines[idx+1:idx+6]:
                m = re.match(r"Loss rate:\s*([\d\.]+)%", l)
                if m:
                    c2s = float(m.group(1))
                    break
    return s2c, c2s

def flow_sort_key(flow_label: str):
    """Extract the first integer in the label for sorting, fallback to label."""
    m = re.search(r'(\d+)', flow_label)
    return int(m.group(1)) if m else float('inf')

def build_summary(records):
    """Return summary as a list of lines given list of (label, s2c, c2s)."""
    s2c_vals = [r[1] for r in records]
    c2s_vals = [r[2] for r in records]
    lines = []
    def summarize(name, vals):
        lines.append(f"{name} loss (%):")
        lines.append(f"  min:    {min(vals):.2f}")
        lines.append(f"  max:    {max(vals):.2f}")
        lines.append(f"  mean:   {statistics.mean(vals):.2f}")
        lines.append(f"  median: {statistics.median(vals):.2f}")
        lines.append("")  # blank line
    summarize("S->C", s2c_vals)
    summarize("C->S", c2s_vals)
    return lines

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-flow loss rates and summary from bandwidth-tester client logs."
    )
    parser.add_argument("--base", "-b", required=True,
                        help="Base output directory (where flow_map.txt and *_client.log live)")
    parser.add_argument("--print", action="store_true", dest="print_summary",
                        help="Also print summary stats to stdout")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.is_dir():
        parser.error(f"{base} is not a directory")

    flow_map_file = base / "flow_map.txt"
    if not flow_map_file.exists():
        parser.error(f"flow_map.txt not found in {base}")
    flow_map = parse_flow_map(flow_map_file)

    records = []  # will hold tuples (short_label, s2c_loss, c2s_loss)
    for root, _, files in os.walk(base):
        for fn in files:
            if not fn.endswith("_client.log"):
                continue
            longname = fn[:-len("_client.log")]
            short = flow_map.get(longname, longname)
            log_path = Path(root) / fn
            s2c, c2s = extract_losses(log_path)
            if s2c is None or c2s is None:
                print(f"[WARN] could not parse both losses in {log_path}", file=os.sys.stderr)
                continue
            records.append((short, s2c, c2s))

    if not records:
        print("No loss records found.", file=os.sys.stderr)
        return

    # 1) Sort by numeric part of flow label
    records.sort(key=lambda rec: flow_sort_key(rec[0]))

    # Write per‐flow CSV
    out_csv = base / "loss_stats.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flow", "S->C_loss_%", "C->S_loss_%"])
        for short, s2c, c2s in records:
            writer.writerow([short, f"{s2c:.2f}", f"{c2s:.2f}"])
    print(f"Wrote per‐flow loss stats to {out_csv}")

    # Build summary text and save to loss_summary.txt
    summary_lines = build_summary(records)
    out_txt = base / "loss_summary.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(summary_lines))
    if args.print_summary:
        print("\nSummary across all flows:")
        print("\n".join(summary_lines))

if __name__ == "__main__":
    main()
