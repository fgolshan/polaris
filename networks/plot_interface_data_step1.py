#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import pandas as pd
import sys
from collections import defaultdict

def log(msg):
    print(msg, file=sys.stderr)

def load_aggregated_df(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Identify time column
    time_cols = [c for c in df.columns if c.lower() in ("timestamp", "time", "ts")]
    if time_cols:
        tcol = time_cols[0]
    else:
        tcol = df.columns[0]
    try:
        df[tcol] = pd.to_datetime(df[tcol], utc=True)
    except Exception:
        pass
    df = df.set_index(tcol)
    return df

def extract_throughput_by_iface(df: pd.DataFrame):
    """
    Expect either long format with columns: interface, raw_delta, delta_seconds
    or wide format with per-interface columns like <iface>_raw_delta and <iface>_delta_seconds.
    Returns DataFrame indexed by timestamp with columns per interface (throughput in Mbps).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    if "interface" in cols_lower and "raw_delta" in cols_lower and "delta_seconds" in cols_lower:
        iface_col = cols_lower["interface"]
        raw_col = cols_lower["raw_delta"]
        delta_sec_col = cols_lower["delta_seconds"]
        tmp = df[[iface_col, raw_col, delta_sec_col]].copy()
        tmp["throughput_mbps"] = tmp.apply(
            lambda row: (row[raw_col] / row[delta_sec_col]) * 8 / 1e6
            if row[delta_sec_col] and row[delta_sec_col] > 0 else 0.0,
            axis=1,
        )
        pivot = tmp.pivot_table(index=df.index, columns=iface_col, values="throughput_mbps", aggfunc="first")
        pivot.columns = [str(c) for c in pivot.columns]
        pivot = pivot.fillna(0.0)
        return pivot

    # wide format heuristics
    pattern_raw = re.compile(r"^(?P<iface>.+?)_raw_delta$")
    pattern_dt = re.compile(r"^(?P<iface>.+?)_delta_seconds$")
    raw_ifaces = {}
    dt_ifaces = {}
    for c in df.columns:
        m = pattern_raw.match(c)
        if m:
            raw_ifaces[m.group("iface")] = c
        m2 = pattern_dt.match(c)
        if m2:
            dt_ifaces[m2.group("iface")] = c
    common_ifaces = set(raw_ifaces.keys()) & set(dt_ifaces.keys())
    if common_ifaces:
        out = {}
        for iface in common_ifaces:
            raw_c = raw_ifaces[iface]
            dt_c = dt_ifaces[iface]
            raw = df[raw_c]
            dt = df[dt_c]
            throughput = pd.Series(0.0, index=df.index)
            mask = dt > 0
            throughput[mask] = (raw[mask] / dt[mask]) * 8 / 1e6
            out[str(iface)] = throughput
        result = pd.DataFrame(out, index=df.index)
        result = result.fillna(0.0)
        return result

    # fallback: split on first underscore
    alt_iface_vals = defaultdict(dict)
    for c in df.columns:
        parts = c.split("_", 1)
        if len(parts) != 2:
            continue
        iface, rest = parts
        if rest.startswith("raw"):
            alt_iface_vals[iface]["raw"] = c
        elif rest.startswith("delta_seconds"):
            alt_iface_vals[iface]["dt"] = c
    common = [iface for iface, vals in alt_iface_vals.items() if "raw" in vals and "dt" in vals]
    if common:
        out = {}
        for iface in common:
            raw_c = alt_iface_vals[iface]["raw"]
            dt_c = alt_iface_vals[iface]["dt"]
            raw = df[raw_c]
            dt = df[dt_c]
            throughput = pd.Series(0.0, index=df.index)
            mask = dt > 0
            throughput[mask] = (raw[mask] / dt[mask]) * 8 / 1e6
            out[str(iface)] = throughput
        result = pd.DataFrame(out, index=df.index)
        result = result.fillna(0.0)
        return result

    return None

def find_aggregated_file(router_dir: Path, direction: str, metric: str):
    # search with priority: aggregated/<specific>, aggregated/aggregated_deltas.csv, root specific, root generic
    candidates = [
        router_dir / "aggregated" / f"aggregated_deltas_router_{direction}_{metric}.csv",
        router_dir / "aggregated" / "aggregated_deltas.csv",
        router_dir / f"aggregated_deltas_router_{direction}_{metric}.csv",
        router_dir / "aggregated_deltas.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: collect per-AS external and local physical interface throughput (Mbps)"
    )
    parser.add_argument("--base", "-b", required=True, help="Base output directory from traffic generator")
    parser.add_argument("--direction", choices=["input", "output"], default="input",
                        help="Whether to process input_bytes_total or output_bytes_total")
    parser.add_argument("--isd-as", dest="isd_as", help="Optional ISD-AS to restrict to (e.g. 1-111)")
    parser.add_argument("--out-dir", "-o", dest="out_dir", default=None,
                        help="Where to write per-AS CSVs (default: metrics/interface_utilization_step1 under base)")
    parser.add_argument("--metric", default="bytes_total",
                        help="Suffix of the metric file, e.g. bytes_total (used to build filenames like aggregated_deltas_router_input_bytes_total.csv)")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        log(f"[ERROR] Base directory {base} does not exist")
        sys.exit(1)
    metrics_root = base / "metrics"
    if not metrics_root.exists():
        log(f"[ERROR] metrics/ subdirectory not found under {base}")
        sys.exit(1)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = metrics_root / "interface_utilization_step1"
    out_dir.mkdir(parents=True, exist_ok=True)

    external_by_as = defaultdict(list)
    local_by_as = defaultdict(list)

    for router_dir in sorted(metrics_root.iterdir()):
        if not router_dir.is_dir():
            continue
        # skip the output directory if it resides inside metrics/
        if router_dir.resolve() == out_dir.resolve():
            continue

        topo_path = router_dir / "topology.json"
        if not topo_path.exists():
            log(f"[WARN] topology.json missing in {router_dir}, skipping")
            continue
        try:
            topo = json.loads(topo_path.read_text())
        except Exception as e:
            log(f"[WARN] failed to parse topology.json in {router_dir}: {e}, skipping")
            continue
        isd_as = topo.get("isd_as")
        if not isd_as:
            log(f"[WARN] isd_as missing in topology.json of {router_dir}, skipping")
            continue
        if args.isd_as and isd_as != args.isd_as:
            continue

        agg_path = find_aggregated_file(router_dir, args.direction, args.metric)
        if not agg_path:
            log(f"[WARN] no aggregated delta file for {router_dir.name} (tried router_{args.direction}_{args.metric} and fallback), skipping")
            continue

        if agg_path.name != f"aggregated_deltas_router_{args.direction}_{args.metric}.csv":
            log(f"[INFO] using fallback {agg_path.relative_to(router_dir)} for router {router_dir.name}")

        df = load_aggregated_df(agg_path)
        if df is None or df.empty:
            log(f"[WARN] failed to load or empty dataframe from {agg_path}, skipping")
            continue

        per_iface = extract_throughput_by_iface(df)
        if per_iface is None:
            log(f"[WARN] could not infer per-interface throughput for {router_dir.name}, skipping")
            continue

        # External interfaces: purely digit names
        external_iface_names = [c for c in per_iface.columns if re.fullmatch(r"\d+", c)]
        if external_iface_names:
            ext_named = per_iface[external_iface_names].copy()
            ext_named.columns = [f"{router_dir.name}:{iface}" for iface in ext_named.columns]
            external_by_as[isd_as].append(ext_named)
        else:
            log(f"[INFO] no external numeric interfaces found in {router_dir.name} for throughput")

        # Local physical interface: sum of 'internal' + all '->*'
        local_components = [c for c in per_iface.columns if c == "internal" or c.startswith("->")]
        if local_components:
            local_sum = per_iface[local_components].sum(axis=1)
        else:
            local_sum = pd.Series(0.0, index=per_iface.index)

        for ext_iface in external_iface_names:
            colname = f"{router_dir.name}:{ext_iface}"
            local_df = pd.DataFrame({colname: local_sum})
            local_by_as[isd_as].append(local_df)

    # write per-AS CSVs
    for as_name in sorted(set(list(external_by_as.keys()) + list(local_by_as.keys()))):
        if external_by_as.get(as_name):
            combined_ext = pd.concat(external_by_as[as_name], axis=1).sort_index()
            out_path = out_dir / f"{as_name}_external_{args.direction}_raw_mbps.csv"
            combined_ext.to_csv(out_path)
            log(f"[INFO] Wrote external data for AS {as_name} to {out_path}")
        else:
            log(f"[INFO] no external data for AS {as_name}, skipping external CSV")

        if local_by_as.get(as_name):
            combined_loc = pd.concat(local_by_as[as_name], axis=1).sort_index()
            out_path = out_dir / f"{as_name}_local_{args.direction}_raw_mbps.csv"
            combined_loc.to_csv(out_path)
            log(f"[INFO] Wrote local data for AS {as_name} to {out_path}")
        else:
            log(f"[INFO] no local data for AS {as_name}, skipping local CSV")

if __name__ == "__main__":
    main()
