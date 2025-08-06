import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import os
import json
import re
from collections import defaultdict

# Mapping of trigger to marker shape
TRIGGER_MARKERS = {
    "initialization": "o",
    "scheduled": "s",
    "P-CA": "D",
    "SCMP_path_down": "^",
    "other": "x",
}
# Legend labels we want to display (omit 'other')
LEGEND_LABELS = {
    "initialization": "init",
    "scheduled": "sched",
    "P-CA": "P-CA",
    "SCMP_path_down": "down",
}

def extract_json(line):
    if "POLARIS_LOG" not in line:
        return None
    start = line.find("{", line.find("POLARIS_LOG"))
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(line)):
        if line[i] == "{":
            depth += 1
        elif line[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = line[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # fallback cleanup heuristics
                    cleaned = candidate.replace("...", "")
                    cleaned = re.sub(r",\s*}", "}", cleaned)
                    cleaned = re.sub(r",\s*\]", "]", cleaned)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        return None
    return None

def parse_client_log(path):
    flow = os.path.basename(path).split("_client.log")[0]
    events = []
    with open(path) as f:
        for line in f:
            if "POLARIS_LOG" not in line or '"trigger":' not in line:
                continue
            js = extract_json(line)
            if not js:
                continue
            trigger = js.get("trigger", "other")
            ts = js.get("ts")
            if ts is None:
                continue
            old = js.get("old")
            new = None
            # infer new path from fields if not directly labeled
            for k, v in js.items():
                if k.startswith("current_") and k not in ("current_rate", "current_rate_raw"):
                    new = v
                    break
            if new is None and "new" in js:
                new = js.get("new")
            if old is None and "old" in js:
                old = js.get("old")
            events.append({
                "flow": flow,
                "ts": float(ts),
                "trigger": trigger,
                "old": old,
                "new": new,
            })
    return events

def collect_all_events(base_dir, router_filter=None):
    """
    Walk base_dir, parse every *_client.log (optionally filtering by router_filter),
    return (events, src_as, dst_as).
    src_as/dst_as come from the first client log's filename.
    """
    events = []
    src_as = dst_as = None
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.endswith("_client.log"):
                continue
            if router_filter and router_filter not in fn:
                continue

            # if we haven't yet extracted src/dst, do so now
            if src_as is None:
                # expects names like "111_cs_1_to_113_cs_1_..._client.log"
                m = re.match(r"(?P<src>\d+)_cs.*_to_(?P<dst>\d+)_cs", fn)
                if m:
                    src_as, dst_as = m.group("src"), m.group("dst")

            path = os.path.join(root, fn)
            ev = parse_client_log(path)
            events.extend(ev)

    return events, src_as, dst_as

def assign_short_labels(flows):
    sorted_flows = sorted(flows)
    mapping = {}
    for i, f in enumerate(sorted_flows, start=1):
        mapping[f] = f"F{i}"
    return mapping

def abbreviate_path(p):
    if not p:
        return ""
    if "->" in p:
        parts = p.split("->")
        if len(parts) > 2:
            return "→".join(parts[-2:])
    return p

def plot_path_switches(events, out_path, relative=True, src_as=None, dst_as=None):
    if not events:
        raise RuntimeError("No events to plot.")
    by_flow = defaultdict(list)
    for ev in events:
        by_flow[ev["flow"]].append(ev)
    flows = sorted(by_flow.keys())
    short_map = assign_short_labels(flows)

    # Write mapping file next to output
    plot_dir = os.path.dirname(out_path) or "."
    mapping_path = os.path.join(plot_dir, "flow_map.txt")
    with open(mapping_path, "w") as mf:
        for full, short in sorted(short_map.items(), key=lambda kv: kv[1]):
            mf.write(f"{short}\t{full}\n")
    print(f"Wrote flow mapping to {mapping_path}")

    all_ts = [ev["ts"] for ev in events]
    t0 = min(all_ts) if relative else 0
    times_relative = [(ev["ts"] - t0 if relative else ev["ts"]) for ev in events]
    min_time = min(times_relative) if times_relative else 0
    max_time = max(times_relative) if times_relative else 1
    span = max_time - min_time if max_time > min_time else 1

    # Threshold for considering two annotations 'close' in time (to avoid stacking unnecessarily)
    small_time = max(0.5, span * 0.02)

    fig_height = max(3, len(flows) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    for flow in flows:
        evs = sorted(by_flow[flow], key=lambda x: x["ts"])
        y = flows.index(flow)
        # Track previously annotated switch times per stack level to avoid overlap
        level_last_x = defaultdict(lambda: -1e9)  # level -> last event time

        # Initial event (show once)
        if evs:
            first_ev = evs[0]
            init_path = first_ev.get("new") or first_ev.get("old")
            if init_path:
                x_init = (first_ev["ts"] - t0) if relative else first_ev["ts"]
                txt = f"{abbreviate_path(init_path)}"
                # plot marker
                ax.scatter(x_init, y, marker=TRIGGER_MARKERS["initialization"], s=100,
                           edgecolors="black", facecolors="white", zorder=3)
                # annotation above
                ax.annotate(txt, (x_init, y),
                            xytext=(0, 8), textcoords="offset points",
                            fontsize=6, va="bottom", ha="center", alpha=0.85, clip_on=False)

        for ev in evs:
            x = ev["ts"] - t0 if relative else ev["ts"]
            trig = ev.get("trigger", "other")
            marker = TRIGGER_MARKERS.get(trig, TRIGGER_MARKERS["other"])
            # Skip drawing duplicate initialization marker if already drawn
            if not (trig == "initialization"):
                ax.scatter(x, y, marker=marker, s=100,
                           edgecolors="black", facecolors="white", zorder=3)
            # Annotate path switch (non-init) only if path changed
            if trig != "initialization" and ev.get("new") and ev.get("old") and ev["new"] != ev["old"]:
                # Determine stacking level: start at 0, and bump until no close-in-time conflict on that level
                level = 0
                while True:
                    last_x_at_level = level_last_x[level]
                    if abs(x - last_x_at_level) < small_time:
                        level += 1
                        continue
                    break
                level_last_x[level] = x
                vertical_offset = 8 + level * 12  # more spacing per level
                new_abbr = abbreviate_path(ev["new"])
                ax.annotate(new_abbr, (x, y),
                            xytext=(0, vertical_offset), textcoords="offset points",
                            fontsize=6, va="bottom", ha="center", alpha=0.75, clip_on=False)

    ax.set_yticks(list(range(len(flows))))
    ax.set_yticklabels([short_map[f] for f in flows])
    ax.set_ylabel("Flow")
    ax.set_xlabel("Time (s)" + ("" if relative else " (absolute)"))
    ax.set_title(f"Polaris Path Switch Timeline: AS {src_as} → AS {dst_as}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Axis padding to avoid clipping end events
    xmin, xmax = ax.get_xlim()
    pad = max(1.0, (xmax - xmin) * 0.05)
    ax.set_xlim(xmin - pad * 0.1, xmax + pad)

    # Y padding
    y_min, y_max = -0.5, len(flows) - 0.5
    y_span = y_max - y_min if y_max > y_min else 1
    extra = y_span * 0.05
    ax.set_ylim(y_min - extra, y_max + extra)

    # Legend with short labels
    handles = []
    for trig, lab in LEGEND_LABELS.items():
        marker = TRIGGER_MARKERS.get(trig, "o")
        handles.append(Line2D([0], [0], marker=marker, color="black",
                              label=lab, markerfacecolor="white",
                              markersize=8, linestyle=""))
    ax.legend(handles=handles, title="Event type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote path switch timeline to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Polaris path switch events timeline per flow.")
    parser.add_argument("--base", required=True, help="Traffic generator output base directory containing client logs.")
    parser.add_argument("--router", required=False, help="Optional substring to filter client log filenames.")
    parser.add_argument("--out", required=False, help="Output image path. Defaults to <base>/switches.png")
    parser.add_argument("--absolute", action="store_true", help="Use absolute timestamps instead of relative.")
    args = parser.parse_args()

    if args.out:
        out_path = args.out
    else:
        # default output in base directory
        out_path = os.path.join(args.base.rstrip("/"), "switches.png")

    events, src_as, dst_as = collect_all_events(args.base, router_filter=args.router)
    plot_path_switches(events, out_path, relative=not args.absolute, src_as=src_as, dst_as=dst_as)

if __name__ == "__main__":
    main()
