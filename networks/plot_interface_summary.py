#!/usr/bin/env python3
import argparse
import sys
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------
def find_step1_files(step1_dir):
    pat = re.compile(
        r'^(?P<as>[^_]+)_'             
        r'(?P<iface>external|local)_'
        r'(?P<dir>input|output)_raw_mbps\.csv$'
    )
    m = {}
    for f in step1_dir.glob("*.csv"):
        match = pat.match(f.name)
        if not match:
            continue
        asn   = match.group("as")
        iface = match.group("iface")
        direction = match.group("dir")
        m.setdefault(asn, {}).setdefault(iface, {})[direction] = f
    return m

def extract_iface_id(colname: str) -> str:
    m = re.search(r'(\d+)$', colname)
    return m.group(1) if m else colname

def plot_combined_box(df_ext, df_loc, asn, direction, capacity, out_dir):
    # build unified interface list
    ext_ids = [extract_iface_id(c) for c in df_ext.columns]
    loc_ids = [extract_iface_id(c) for c in df_loc.columns]
    all_ids = sorted(set(ext_ids) | set(loc_ids), key=int)

    # reindex into aligned DataFrames
    ext = pd.DataFrame({iid: df_ext.get(c, pd.Series([], dtype=float))
                        for c, iid in zip(df_ext.columns, ext_ids)}).reindex(columns=all_ids)
    loc = pd.DataFrame({iid: df_loc.get(c, pd.Series([], dtype=float))
                        for c, iid in zip(df_loc.columns, loc_ids)}).reindex(columns=all_ids)

    # normalize if desired
    ylabel = "Throughput (Mbps)"
    if capacity:
        ext = ext.div(capacity).mul(100)
        loc = loc.div(capacity).mul(100)
        ylabel = "Utilization (%)"

    # prepare boxplot data & positions
    data, positions = [], []
    width = 0.35
    for i, iid in enumerate(all_ids):
        base = i * 2
        data.append(ext[iid].dropna().values)
        positions.append(base)
        data.append(loc[iid].dropna().values)
        positions.append(base + width)
    xticks = [i*2 + width/2 for i in range(len(all_ids))]

    # draw
    plt.figure(figsize=(len(all_ids)*0.5 + 2, 4))
    bps = plt.boxplot(
        data,
        positions=positions,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,          # no outliers
        manage_ticks=False
    )

    # style boxes & medians
    for idx, box in enumerate(bps['boxes']):
        box.set_facecolor(f"C{idx%2}")
        box.set_alpha(0.5)
        box.set_edgecolor("black")
    for med in bps['medians']:
        med.set_color("black")
        med.set_linewidth(1.5)

    plt.xticks(xticks, all_ids)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # legend outside
    legend_patches = [
        Patch(facecolor="C0", alpha=0.5, edgecolor="black", label="external"),
        Patch(facecolor="C1", alpha=0.5, edgecolor="black", label="local"),
    ]
    plt.legend(handles=legend_patches,
               title="Interface",
               loc="upper left",
               bbox_to_anchor=(1.02, 1))

    # title & labels
    dir_word = "Received" if direction == "input" else "Sent"
    title_main = f"AS {asn} — Bytes {dir_word}"
    if capacity:
        title_main += f" (normalized to {capacity:.3g} Mbps)"
    plt.title(title_main)
    plt.xlabel("SCION interface N")
    plt.ylabel(ylabel)

    # layout & save
    plt.tight_layout()
    fname = f"{asn}_external-local_{direction}"
    if capacity:
        fname += f"_util_{capacity:.3g}Mbps"
    fname += ".png"
    plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Boxplot of per-AS interface throughput"
    )
    parser.add_argument("--base", "-b", required=True,
                        help="Base output directory from traffic generator")
    parser.add_argument("--capacity", type=float, default=None,
                        help="Optional link capacity in Mbps (for utilization)")
    parser.add_argument("--out-dir", "-o", dest="out_dir", default=None,
                        help="Where to write summaries (default: <base>/metrics/interface_summary)")
    args = parser.parse_args()

    base = Path(args.base)
    step1_dir = base / "metrics" / "interface_utilization_step1"
    if not step1_dir.is_dir():
        print(f"[ERROR] Step 1 dir not found: {step1_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else base / "metrics" / "interface_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = find_step1_files(step1_dir)
    if not mapping:
        print(f"[WARN] No CSVs found in {step1_dir}", file=sys.stderr)
        return

    for asn, by_iface in sorted(mapping.items()):
        for direction in ("input", "output"):
            if ("external" in by_iface and direction in by_iface["external"] and
                "local"    in by_iface and direction in by_iface["local"]):
                df_ext = pd.read_csv(by_iface["external"][direction],
                                     index_col=0, parse_dates=True)
                df_loc = pd.read_csv(by_iface["local"][direction],
                                     index_col=0, parse_dates=True)
                # clean index name
                for df in (df_ext, df_loc):
                    if df.index.name and "timestamp" in df.index.name.lower():
                        df.index.name = None
                plot_combined_box(df_ext, df_loc,
                                  asn, direction,
                                  args.capacity, out_dir)

    print(f"[OK] summaries written to {out_dir}")

if __name__ == "__main__":
    main()
