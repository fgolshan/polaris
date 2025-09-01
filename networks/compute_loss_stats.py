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

def extract_flow_metrics(log_path: Path):
    """
    Parse one *_client.log file and extract bandwidth and loss metrics for both
    directions.  Returns a dictionary with keys for attempted/achieved
    bandwidth and loss metrics for S->C and C->S directions.  The values are
    floats (bps for bandwidths, percentage for loss rates).  If any metric
    cannot be found it will be returned as None.

    Keys returned:

        S2C_attempted  – attempted bandwidth in bps for server→client
        S2C_achieved   – achieved bandwidth in bps for server→client
        S2C_loss       – target miss rate (%) for server→client
        C2S_attempted  – attempted bandwidth in bps for client→server
        C2S_achieved   – achieved bandwidth in bps for client→server
        C2S_loss       – target miss rate (%) for client→server
        C2S_network    – network loss rate (%) for client→server

    Example log excerpt demonstrating expected patterns::

        S->C results
        Attempted bandwidth: 0 bps / 0.00 Mbps
        Achieved bandwidth: 0 bps / 0.00 Mbps
        Loss rate: 0.0%
        C->S results
        Attempted bandwidth: 1000000 bps / 1.00 Mbps
        Achieved bandwidth: 986400 bps / 0.99 Mbps
        Loss rate: 1.4%
        The network loss rate from client to server is 1.4%
    """
    # Initialize result dictionary with None defaults
    result = {
        "S2C_attempted": None,
        "S2C_achieved": None,
        "S2C_loss": None,
        "C2S_attempted": None,
        "C2S_achieved": None,
        "C2S_loss": None,
        "C2S_network": None,
    }
    try:
        lines = log_path.read_text().splitlines()
    except Exception:
        return result

    # First, attempt to find the network loss rate line (client → server)
    # Example: "The network loss rate from client to server is 1.4%"
    net_re = re.compile(r"network loss rate from client to server is\s*([\d\.]+)%", re.IGNORECASE)
    for line in lines:
        m = net_re.search(line)
        if m:
            try:
                result["C2S_network"] = float(m.group(1))
            except ValueError:
                pass
            break

    # Now look for the S->C and C->S sections and parse attempted/achieved/loss lines.
    # We'll iterate through the lines; upon encountering a section marker, we
    # parse subsequent lines until we hit another marker or until we've parsed
    # all three metrics.  This prevents accidental capture of metrics from the
    # wrong section when the sections are adjacent without a blank line.
    n = len(lines)
    i = 0
    while i < n:
        striped = lines[i].strip()
        # Server to client section
        if striped == "S->C results":
            # parse metrics for S->C until we hit another section or a blank line
            parsed = 0
            j = i + 1
            while j < n:
                l_strip = lines[j].strip()
                # break if next section header encountered
                if l_strip in ("C->S results", "S->C results"):  # new section
                    break
                # Attempted bandwidth
                m = re.match(r"Attempted bandwidth:\s*(\d+) bps", l_strip)
                if m and result["S2C_attempted"] is None:
                    try:
                        result["S2C_attempted"] = float(m.group(1))
                    except ValueError:
                        pass
                    parsed += 1
                # Achieved bandwidth
                m = re.match(r"Achieved bandwidth:\s*(\d+) bps", l_strip)
                if m and result["S2C_achieved"] is None:
                    try:
                        result["S2C_achieved"] = float(m.group(1))
                    except ValueError:
                        pass
                    parsed += 1
                # Loss rate
                m = re.match(r"Loss rate:\s*([\d\.]+)%", l_strip)
                if m and result["S2C_loss"] is None:
                    try:
                        result["S2C_loss"] = float(m.group(1))
                    except ValueError:
                        pass
                    parsed += 1
                # if we've parsed all three metrics, we can break early
                if parsed >= 3:
                    break
                j += 1
            i = j
            continue
        # Client to server section
        if striped == "C->S results":
            parsed = 0
            j = i + 1
            while j < n:
                l_strip = lines[j].strip()
                # break if next section header encountered
                if l_strip in ("S->C results", "C->S results"):
                    break
                m = re.match(r"Attempted bandwidth:\s*(\d+) bps", l_strip)
                if m and result["C2S_attempted"] is None:
                    try:
                        result["C2S_attempted"] = float(m.group(1))
                    except ValueError:
                        pass
                    parsed += 1
                m = re.match(r"Achieved bandwidth:\s*(\d+) bps", l_strip)
                if m and result["C2S_achieved"] is None:
                    try:
                        result["C2S_achieved"] = float(m.group(1))
                    except ValueError:
                        pass
                    parsed += 1
                m = re.match(r"Loss rate:\s*([\d\.]+)%", l_strip)
                if m and result["C2S_loss"] is None:
                    try:
                        result["C2S_loss"] = float(m.group(1))
                    except ValueError:
                        pass
                    parsed += 1
                if parsed >= 3:
                    break
                j += 1
            i = j
            continue
        i += 1
    return result

def flow_sort_key(flow_label: str):
    """Extract the first integer in the label for sorting, fallback to label."""
    m = re.search(r'(\d+)', flow_label)
    return int(m.group(1)) if m else float('inf')

def summarize_metric(name: str, values: list, lines: list):
    """Helper to append summary statistics for a single metric to lines.

    If the values list is empty, the function will skip summarization to avoid
    errors.  Otherwise, it computes min, max, mean and median.  This helper
    centralizes the summarization logic for both directions and all metrics.
    """
    if not values:
        return
    lines.append(f"{name}:")
    lines.append(f"  min:    {min(values):.2f}")
    lines.append(f"  max:    {max(values):.2f}")
    lines.append(f"  mean:   {statistics.mean(values):.2f}")
    lines.append(f"  median: {statistics.median(values):.2f}")
    lines.append("")

def build_summary(records):
    """
    Given a list of record dictionaries (each containing per-flow metrics), build
    a human-readable summary.  The summary includes, for both the server→client
    (S->C) and client→server (C->S) directions, the minimum, maximum, mean and
    median of attempted bandwidth, achieved bandwidth and target miss rate.
    Additionally, for the C->S direction the network loss rate statistics are
    computed.  Returns a list of lines ready to be written to a file.
    """
    # Collect values by metric and direction
    s2c_attempted = [r["S2C_attempted"] for r in records if r["S2C_attempted"] is not None]
    s2c_achieved  = [r["S2C_achieved"] for r in records if r["S2C_achieved"] is not None]
    s2c_loss      = [r["S2C_loss"] for r in records if r["S2C_loss"] is not None]
    c2s_attempted = [r["C2S_attempted"] for r in records if r["C2S_attempted"] is not None]
    c2s_achieved  = [r["C2S_achieved"] for r in records if r["C2S_achieved"] is not None]
    c2s_loss      = [r["C2S_loss"] for r in records if r["C2S_loss"] is not None]
    c2s_network   = [r["C2S_network"] for r in records if r["C2S_network"] is not None]

    lines = []
    # Summaries for S->C
    lines.append("S->C summary (server→client):")
    summarize_metric("  Attempted bandwidth (bps)", s2c_attempted, lines)
    summarize_metric("  Achieved bandwidth (bps)", s2c_achieved, lines)
    summarize_metric("  Target miss rate (%)", s2c_loss, lines)
    # Summaries for C->S
    lines.append("C->S summary (client→server):")
    summarize_metric("  Attempted bandwidth (bps)", c2s_attempted, lines)
    summarize_metric("  Achieved bandwidth (bps)", c2s_achieved, lines)
    summarize_metric("  Target miss rate (%)", c2s_loss, lines)
    summarize_metric("  Network loss rate (%)", c2s_network, lines)
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

    # Each record will be a dictionary containing the short flow label and
    # parsed metrics.  Metrics may be None if unavailable.  Example keys:
    #   'label', 'S2C_attempted', 'S2C_achieved', 'S2C_loss',
    #   'C2S_attempted', 'C2S_achieved', 'C2S_loss', 'C2S_network'
    records = []
    for root, _, files in os.walk(base):
        for fn in files:
            if not fn.endswith("_client.log"):
                continue
            longname = fn[:-len("_client.log")]
            short = flow_map.get(longname, longname)
            log_path = Path(root) / fn
            metrics = extract_flow_metrics(log_path)
            # We require at least some data to include this flow; specifically
            # attempted values in either direction.  If both directions are missing,
            # skip the record and warn the user.
            if all(metrics.get(key) is None for key in ["S2C_attempted", "C2S_attempted"]):
                print(f"[WARN] could not parse metrics in {log_path}", file=os.sys.stderr)
                continue
            metrics_with_label = {"label": short, **metrics}
            records.append(metrics_with_label)

    if not records:
        print("No loss records found.", file=os.sys.stderr)
        return

    # 1) Sort by numeric part of flow label
    records.sort(key=lambda rec: flow_sort_key(rec["label"]))

    # Write per-flow CSV with detailed metrics.  We'll include separate
    # columns for each direction and metric.  Blank values will be left empty.
    out_csv = base / "loss_stats.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "flow",
            "S2C_attempted_bps",
            "S2C_achieved_bps",
            "S2C_target_miss_rate_%",
            "C2S_attempted_bps",
            "C2S_achieved_bps",
            "C2S_target_miss_rate_%",
            "C2S_network_loss_rate_%",
        ])
        for rec in records:
            def fmt(val):
                return f"{val:.2f}" if isinstance(val, float) and not val.is_integer() else (f"{val:.0f}" if isinstance(val, float) else ("" if val is None else str(val)))
            writer.writerow([
                rec["label"],
                fmt(rec.get("S2C_attempted")),
                fmt(rec.get("S2C_achieved")),
                fmt(rec.get("S2C_loss")),
                fmt(rec.get("C2S_attempted")),
                fmt(rec.get("C2S_achieved")),
                fmt(rec.get("C2S_loss")),
                fmt(rec.get("C2S_network")),
            ])
    print(f"Wrote per-flow loss stats to {out_csv}")

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
