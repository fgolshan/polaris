#!/usr/bin/env python3
import argparse
import pathlib
import re
import datetime
import csv
from collections import defaultdict

METRICS_OF_INTEREST = {
    "router_input_bytes_total",
    "router_output_bytes_total",
    "router_input_pkts_total",
    "router_output_pkts_total",
    "router_dropped_pkts_total",
}


def parse_prom_line(line: str):
    m = re.match(r'^([^{\s]+)(\{[^}]*\})?\s+(.+)$', line)
    if not m:
        return None
    metric = m.group(1)
    labels_raw = m.group(2)
    val_raw = m.group(3).strip()
    try:
        value = float(val_raw)
    except ValueError:
        return None
    label_dict = {}
    if labels_raw:
        inner = labels_raw[1:-1]
        parts = re.findall(r'(\w+)="([^"\\]*(?:\\.[^"\\]*)*)"', inner)
        for k, v in parts:
            label_dict[k] = v
    return metric, label_dict, value


def aggregate_prom_file(prom_path: pathlib.Path):
    text = prom_path.read_text()
    lines = text.splitlines()

    help_lines = {}
    type_lines = {}
    for line in lines:
        if line.startswith("# HELP"):
            parts = line.split()
            if len(parts) >= 3:
                metric_name = parts[2]
                if metric_name in METRICS_OF_INTEREST:
                    help_lines[metric_name] = line
        elif line.startswith("# TYPE"):
            parts = line.split()
            if len(parts) >= 3:
                metric_name = parts[2]
                if metric_name in METRICS_OF_INTEREST:
                    type_lines[metric_name] = line

    groups = defaultdict(float)
    per_interface = defaultdict(float)

    for line in lines:
        if line.startswith("#"):
            continue
        parsed = parse_prom_line(line)
        if parsed is None:
            continue
        metric, labels, value = parsed
        if metric not in METRICS_OF_INTEREST:
            continue
        interface = labels.get("interface", "")
        isd_as = labels.get("isd_as", "")
        neighbor_isd_as = labels.get("neighbor_isd_as", "")
        key = (metric, interface, isd_as, neighbor_isd_as)
        groups[key] += value
        per_interface[key] += value

    # physical_local: internal + ->*
    physical_local = defaultdict(float)
    for (metric, interface, isd_as, neighbor_isd_as), value in per_interface.items():
        if interface == "internal" or interface.startswith("->"):
            key = (metric, "physical_local", isd_as, "local")
            physical_local[key] += value

    # physical_remote: numeric interfaces aggregated
    physical_remote = defaultdict(float)
    for (metric, interface, isd_as, neighbor_isd_as), value in per_interface.items():
        if re.fullmatch(r'\d+', interface):
            key = (metric, "physical_remote", isd_as, "remote")
            physical_remote[key] += value

    for k, v in physical_local.items():
        groups[k] += v
    for k, v in physical_remote.items():
        groups[k] += v

    # parse timestamp from filename
    ts = None
    name = prom_path.name
    try:
        iso = name.split(".prom.txt")[0]
        ts = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        ts = datetime.datetime.fromtimestamp(prom_path.stat().st_mtime, datetime.timezone.utc)
    return ts, groups, help_lines, type_lines


def write_aggregated(prom_path: pathlib.Path, aggregated: dict, help_lines: dict, type_lines: dict, out_path: pathlib.Path):
    with open(out_path, "w") as f:
        for metric in sorted(METRICS_OF_INTEREST):
            if metric in help_lines:
                f.write(f"{help_lines[metric]}\n")
            if metric in type_lines:
                f.write(f"{type_lines[metric]}\n")
            for (m, interface, isd_as, neighbor_isd_as), value in sorted(aggregated.items()):
                if m != metric:
                    continue
                labels = [
                    f'interface="{interface}"',
                    f'isd_as="{isd_as}"',
                    f'neighbor_isd_as="{neighbor_isd_as}"',
                ]
                label_str = "{" + ",".join(labels) + "}"
                if value.is_integer():
                    value_str = str(int(value))
                else:
                    value_str = repr(value)
                f.write(f"{metric}{label_str} {value_str}\n")


def compute_deltas(sorted_snapshots):
    deltas = []
    prev_ts = None
    prev_agg = None
    for ts, agg, _, _ in sorted_snapshots:
        if prev_agg is None:
            for (metric, interface, isd_as, neighbor_isd_as), value in sorted(agg.items()):
                deltas.append({
                    "timestamp": ts.isoformat(),
                    "metric": metric,
                    "interface": interface,
                    "isd_as": isd_as,
                    "neighbor_isd_as": neighbor_isd_as,
                    "raw_delta": 0.0,
                    "delta_seconds": 0.0,
                    "normalized_delta": 0.0,
                })
        else:
            delta_seconds = (ts - prev_ts).total_seconds()
            all_keys = set(prev_agg.keys()).union(set(agg.keys()))
            for key in sorted(all_keys):
                metric, interface, isd_as, neighbor_isd_as = key
                curr_val = agg.get(key, 0.0)
                prev_val = prev_agg.get(key, 0.0)
                raw_delta = curr_val - prev_val
                if raw_delta < 0:  # counter reset
                    raw_delta = curr_val
                normalized = raw_delta / delta_seconds if delta_seconds > 0 else 0.0
                deltas.append({
                    "timestamp": ts.isoformat(),
                    "metric": metric,
                    "interface": interface,
                    "isd_as": isd_as,
                    "neighbor_isd_as": neighbor_isd_as,
                    "raw_delta": raw_delta,
                    "delta_seconds": delta_seconds,
                    "normalized_delta": normalized,
                })
        prev_ts = ts
        prev_agg = agg
    return deltas


def main():
    parser = argparse.ArgumentParser(description="Aggregate SCION router Prometheus exports and compute deltas.")
    parser.add_argument("base", help="Base output folder with metrics/<router>/raw/*.prom.txt")
    parser.add_argument("--router", help="Optional: limit to a single router directory name")
    args = parser.parse_args()

    base = pathlib.Path(args.base)
    metrics_root = base / "metrics"
    if not metrics_root.is_dir():
        print(f"Error: metrics directory not found under {base}")
        return

    for router_dir in sorted(metrics_root.iterdir()):
        if args.router and router_dir.name != args.router:
            continue
        if not router_dir.is_dir():
            continue
        raw_dir = router_dir / "raw"
        if not raw_dir.is_dir():
            continue
        aggregated_dir = router_dir / "aggregated"
        aggregated_dir.mkdir(exist_ok=True)

        snapshots = []
        for prom_file in sorted(raw_dir.iterdir()):
            if not prom_file.name.endswith(".prom.txt"):
                continue
            ts, agg_groups, help_lines, type_lines = aggregate_prom_file(prom_file)
            out_agg_path = aggregated_dir / (prom_file.name.replace(".prom.txt", ".aggregated.prom.txt"))
            write_aggregated(prom_file, agg_groups, help_lines, type_lines, out_agg_path)
            snapshots.append((ts, agg_groups, help_lines, type_lines))

        if not snapshots:
            continue

        snapshots.sort(key=lambda x: x[0])
        delta_records = compute_deltas(snapshots)

        delta_csv_path = router_dir / "aggregated_deltas.csv"
        with open(delta_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "timestamp", "metric", "interface", "isd_as", "neighbor_isd_as",
                "raw_delta", "delta_seconds", "normalized_delta"])
            writer.writeheader()
            for rec in delta_records:
                writer.writerow(rec)

        print(f"Router {router_dir.name}: wrote {len(snapshots)} aggregated scrapes and deltas to {delta_csv_path}")

        snapshots.sort(key=lambda x: x[0])
        delta_records = compute_deltas(snapshots)

        # Write combined delta file (can be kept or removed if you prefer only per-metric)
        combined_csv_path = router_dir / "aggregated_deltas.csv"
        with open(combined_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "timestamp", "metric", "interface", "isd_as", "neighbor_isd_as",
                "raw_delta", "delta_seconds", "normalized_delta"])
            writer.writeheader()
            for rec in delta_records:
                writer.writerow(rec)

        # Write one delta file per metric for readability
        # Group by metric
        per_metric = defaultdict(list)
        for rec in delta_records:
            per_metric[rec["metric"]].append(rec)

        for metric, records in per_metric.items():
            metric_fname = f"aggregated_deltas_{metric}.csv"
            metric_csv_path = aggregated_dir / metric_fname
            with open(metric_csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "timestamp", "interface", "isd_as", "neighbor_isd_as",
                    "raw_delta", "delta_seconds", "normalized_delta"])
                writer.writeheader()
                for rec in records:
                    # drop the 'metric' field since it's fixed per file
                    filtered = {k: v for k, v in rec.items() if k != "metric"}
                    writer.writerow(filtered)



if __name__ == "__main__":
    main()
