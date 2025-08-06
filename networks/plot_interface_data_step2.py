#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

def log(msg):
    print(msg, file=sys.stderr)

def load_csv(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "timestamp" not in df.columns:
        return None
    # parse every timestamp as ISO-8601, coerce failures to NaT
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
        format="ISO8601",
    )
    # drop any rows pandas couldn’t parse
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return None
    # re‐index
    df = df.set_index("timestamp")
    return df


def shorten_label(col: str) -> str:
    # Try to extract the last number in the column name (interface ID)
    m = re.search(r'(\d+)(?!.*\d)', col)
    if m:
        return m.group(1)
    return col

def direction_to_human(direction: str) -> str:
    d = direction.lower()
    if d in ("input", "in", "recv", "received"):
        return "received"
    if d in ("output", "out", "sent"):
        return "sent"
    return direction

def plot_series(df: pd.DataFrame, as_name: str, iface_type: str, direction: str,
                capacity_mbps: float | None, out_path: Path):
    if df.empty:
        log(f"[WARN] empty dataframe for AS {as_name} {iface_type} {direction}, skipping plot")
        return

    # Compute relative time in seconds from first timestamp
    t0 = df.index.min()
    rel_seconds = (df.index - t0).total_seconds()

    # Determine whether to normalize
    if capacity_mbps is not None and capacity_mbps > 0:
        util = df / capacity_mbps * 100.0  # percent
        y = util
        ylabel = "Utilization (%)"
        cap_str = f"{capacity_mbps:g}"
        title_extra = f"normalized to {cap_str} Mbps"
        file_suffix = f"utilization_{cap_str}Mbps"
    else:
        y = df
        ylabel = "Throughput (Mbps)"
        title_extra = None
        file_suffix = "throughput"

    plt.figure(figsize=(10, 5))
    line_styles = ['-', '--', '-.', ':']
    for i, col in enumerate(sorted(y.columns)):
        label = shorten_label(col)
        linestyle = line_styles[i % len(line_styles)]
        plt.plot(rel_seconds, y[col], label=label, linestyle=linestyle)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)

    human_dir = direction_to_human(direction)
    # Improved title formatting: direction in parentheses, normalization appended clearly
    if title_extra:
        plt.title(f"AS {as_name} - {iface_type.capitalize()} Interfaces ({human_dir}), {title_extra}")
    else:
        plt.title(f"AS {as_name} - {iface_type.capitalize()} Interfaces ({human_dir})")

    # Deduplicate legend labels (in case multiple full columns map to same short label)
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = {}
    filtered = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen[l] = True
        filtered.append(h)
        filtered_labels.append(l)
    plt.legend(filtered, filtered_labels, title="Interface", fontsize="small", ncols=2)
    plt.grid(True)
    plt.tight_layout()

    out_file = out_path / f"{as_name}_{iface_type}_{direction}_{file_suffix}.png"
    plt.savefig(out_file)
    plt.close()
    log(f"[INFO] Wrote plot for AS {as_name} {iface_type} {direction} to {out_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: plot per-AS external/local interface throughput or utilization"
    )
    parser.add_argument("--base", "-b", required=True, help="Base output directory (same as traffic generator output)")
    parser.add_argument("--isd-as", dest="isd_as", help="Optional ISD-AS to restrict to (e.g. 1-111)")
    parser.add_argument("--capacity", type=float, default=None,
                        help="Optional capacity in Mbps; if provided, plots utilization (%%)")  # escaped % to avoid argparse formatting error
    parser.add_argument("--types", default="external,local", help="Comma-separated types to plot: external,local")
    parser.add_argument("--out-dir", "-o", dest="out_dir", default=None,
                        help="Where to write plots (default: metrics/interface_utilization_step2 under base)")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        log(f"[ERROR] base directory {base} does not exist")
        sys.exit(1)
    metrics_root = base / "metrics"
    if not metrics_root.exists():
        log(f"[ERROR] metrics/ subdirectory not found under {base}")
        sys.exit(1)

    step1_dir = metrics_root / "interface_utilization_step1"
    if not step1_dir.exists():
        log(f"[ERROR] expected step1 directory {step1_dir} does not exist")
        sys.exit(1)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = metrics_root / "interface_utilization_step2"
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_types = {t.strip() for t in args.types.split(",") if t.strip() in ("external", "local")}
    if not requested_types:
        log("[ERROR] no valid types specified (must include 'external' and/or 'local')")
        sys.exit(1)

    # Discover available AS / iface_type / direction combinations from filenames
    pattern = re.compile(r'^(?P<as_name>[^_]+)_(?P<iface_type>external|local)_(?P<direction>[^_]+)_raw_mbps\.csv$')
    available = {}  # as_name -> iface_type -> direction -> path
    for f in step1_dir.glob("*_*.csv"):
        m = pattern.match(f.name)
        if not m:
            continue
        as_name = m.group("as_name")
        iface_type = m.group("iface_type")
        direction = m.group("direction")
        available.setdefault(as_name, {}).setdefault(iface_type, {})[direction] = f

    if args.isd_as:
        if args.isd_as not in available:
            log(f"[WARN] requested ISD-AS {args.isd_as} has no step1 output, skipping")
            return
        as_list = [args.isd_as]
    else:
        as_list = sorted(available.keys())

    for as_name in as_list:
        if as_name not in available:
            continue
        for iface_type in ("external", "local"):
            if iface_type not in requested_types:
                continue
            dirs_for_type = available.get(as_name, {}).get(iface_type, {})
            if not dirs_for_type:
                log(f"[WARN] no data for AS {as_name} {iface_type}, skipping")
                continue
            for direction, csv_path in dirs_for_type.items():
                if not csv_path.exists():
                    log(f"[WARN] missing expected file {csv_path}, skipping {as_name} {iface_type} {direction}")
                    continue
                df = load_csv(csv_path)
                if df is None or df.empty:
                    log(f"[WARN] failed to load or empty csv for {as_name} {iface_type} {direction}")
                    continue
                plot_series(df, as_name, iface_type, direction, args.capacity, out_dir)

if __name__ == "__main__":
    main()
