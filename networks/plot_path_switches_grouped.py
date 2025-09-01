"""
Extended Polaris path switch plotting script.

This script builds upon the original `plot_path_switches.py` functionality to
support experiments where multiple source–destination AS pairs are present.

Key features:
    • When only a single (src,dst) AS pair is observed, the plot output is
      identical to the original script: no per‑flow connecting lines are drawn,
      the title contains the AS pair and a single event‑type legend is shown.

    • When more than one distinct (src,dst) AS pair is detected, flows are
      grouped by their AS pair.  Flows in the same group are placed next to
      one another on the y–axis.  Each group is indicated using a distinct
      colour on the y–axis tick labels and in a secondary legend labelled
      "Source → Destination".  A subtle horizontal separator is drawn
      between groups to aid visual grouping.

    • For multi‑group plots, a faint horizontal line is drawn for each flow
      starting at its first event time and extending to its last observed
      event.  This line uses the group colour with reduced opacity so that
      it guides the eye without competing with the event markers.  (In
      single‑group plots these connecting lines are omitted.)

    • The event‑type legend is always rendered and will only display
      triggers that actually occur in the plotted data.

The command‑line interface mirrors the original script:

    python plot_path_switches_grouped.py --base /path/to/logs \
        [--router SUBSTR] [--out /path/switches.png] [--absolute]

"""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Mapping of trigger to marker shape
TRIGGER_MARKERS = {
    "initialization": "o",
    "scheduled": "s",
    "P-CA": "D",
    "SCMP_path_down": "^",
    "other": "x",
}

# Short labels for the legend.  We omit "other" since arbitrary triggers are
# annotated directly on the plot but not in the legend.
LEGEND_LABELS = {
    "initialization": "init",
    "scheduled": "sched",
    "P-CA": "P-CA",
    "SCMP_path_down": "down",
}


def extract_json(line: str):
    """Extract a JSON object embedded in a log line following the 'POLARIS_LOG' token.

    This helper mirrors the original script's logic: it finds the first JSON
    object in the line and attempts to parse it, applying simple cleanup
    heuristics when the log line contains trailing commas or ellipses.
    """
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


def parse_client_log(path: str):
    """Parse a single *_client.log file and return a list of event dicts.

    Each event dict has keys: flow (str), ts (float), trigger (str), old (str or None), new (str or None).
    The 'flow' key is derived from the filename (without the trailing '_client.log').
    """
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


def extract_src_dst(filename: str):
    """Extract (src,dst) AS numbers from a client log filename.

    The expected general form of filenames is:
        <src>_..._to_<dst>_..._client.log

    where <src> and <dst> are numeric AS identifiers.  The substrings
    between the numbers and the 'to' token can vary (e.g. 'cs', 'rs', 'host').
    If parsing fails, returns None.
    """
    # Split on '_to_' to find left and right sides.  This is more permissive
    # than matching a strict pattern and should work for a variety of token
    # separators.  Only the first occurrence is considered.
    if "_to_" not in filename:
        return None
    left, right = filename.split("_to_", 1)
    # Extract digits at the beginning of each side
    m_left = re.match(r"(\d+)", left)
    m_right = re.match(r"(\d+)", right)
    if m_left and m_right:
        return m_left.group(1), m_right.group(1)
    return None


def _convert_with_unit(value: str, unit: str) -> float:
    """Convert a numeric string with a unit suffix into seconds."""
    v = float(value)
    unit = unit.lower()
    if unit == "ms":
        return v / 1000.0
    if unit == "s":
        return v
    if unit == "m":
        return v * 60.0
    if unit == "h":
        return v * 3600.0
    # Fallback: treat as seconds
    return v


def parse_flow_offsets(filename: str) -> tuple[float | None, float | None]:
    """
    Parse the start and end offsets (in seconds) for a flow from its filename.

    Filenames may contain two consecutive time tokens before the '_client.log'
    suffix, for example '..._42s_258s_client.log'.  The first token is
    interpreted as the start offset from the experiment start, and the
    second token is interpreted as a duration.  The end offset is then
    computed as start_offset + duration.  Supported units are ms, s, m, h.

    Returns (start_offset, end_offset).  If parsing fails, both values are
    returned as None.
    """
    base = filename.rsplit("_client.log", 1)[0]
    # Look for two trailing time tokens of the form <number><unit>
    m = re.search(r"_(\d+(?:\.\d+)?)(ms|s|m|h)_(\d+(?:\.\d+)?)(ms|s|m|h)$", base)
    if m:
        start_val, start_unit, dur_val, dur_unit = m.groups()
        start_offset = _convert_with_unit(start_val, start_unit)
        duration = _convert_with_unit(dur_val, dur_unit)
        end_offset = start_offset + duration
        return start_offset, end_offset
    # If exactly one trailing time token is present, treat it as end offset
    m2 = re.search(r"_(\d+(?:\.\d+)?)(ms|s|m|h)$", base)
    if m2:
        val, unit = m2.groups()
        end_offset = _convert_with_unit(val, unit)
        return None, end_offset
    return None, None


def collect_events_and_groups(base_dir: str, log_include: str | None = None):
    """Walk base_dir, parse each *_client.log file, and group flows by (src,dst).

    Returns
    -------
    events : list of event dicts
        Each event dict has keys as produced by parse_client_log.
    flow_to_group : dict
        Maps flow name to a (src,dst) tuple (strings).  Unparseable files map to ('Unknown','Unknown').
    groups : list of (src,dst)
        Sorted list of unique groups present in the dataset.
    flow_to_start_offset : dict
        Maps flow name to its start offset (in seconds) if present in the filename, otherwise None.
    flow_to_end_offset : dict
        Maps flow name to its end offset (in seconds) if present, otherwise None.
    """
    events: list[dict] = []
    flow_to_group: dict[str, tuple[str, str]] = {}
    flow_to_start_offset: dict[str, float | None] = {}
    flow_to_end_offset: dict[str, float | None] = {}
    group_set: set[tuple[str, str]] = set()
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.endswith("_client.log"):
                continue
            if log_include and log_include not in fn:
                continue
            # Determine group (src,dst) from filename
            group = extract_src_dst(fn)
            if group is None:
                group = ("Unknown", "Unknown")
            flow_name = os.path.basename(fn).split("_client.log")[0]
            flow_to_group[flow_name] = group
            group_set.add(group)
            # Parse events
            ev = parse_client_log(os.path.join(root, fn))
            events.extend(ev)
            # Extract start and end offsets heuristically
            start_offset, end_offset = parse_flow_offsets(fn)
            flow_to_start_offset[flow_name] = start_offset
            flow_to_end_offset[flow_name] = end_offset
    # Sort groups for stable colour assignment (by src,dst lexicographically)
    groups_sorted = sorted(list(group_set))
    return events, flow_to_group, groups_sorted, flow_to_start_offset, flow_to_end_offset


def assign_short_labels(flows):
    """Assign short labels (F1, F2, …) to flows in sorted order."""
    sorted_flows = sorted(flows)
    mapping = {}
    for i, f in enumerate(sorted_flows, start=1):
        mapping[f] = f"F{i}"
    return mapping


def split_pairs_for_display(path: str) -> list[str]:
    """Split a whitespace‑separated path into pairs for diagonal display.

    Given a path string such as ``"1 1 6 6 5 2"``, which consists of an
    even number of tokens separated by whitespace, this helper returns a
    list of strings where each string represents a pair of tokens.  For
    example, ``"1 1 6 6 5 2"`` becomes ``["1 1", "6 6", "5 2"]``.  If the
    path has an odd number of tokens or no whitespace, the function
    returns a single‑element list containing the original path.  This
    ensures that full paths are still displayed in cases that do not
    conform to the expected pair structure.

    Parameters
    ----------
    path : str
        The raw path string from the event (``ev['new']`` or ``ev['old']``).

    Returns
    -------
    list of str
        A list of strings representing pairs for display.  If the path
        cannot be split into pairs, the original path is returned as
        the only element.
    """
    if not path:
        return []
    
    # Only split numeric paths separated by whitespace into pairs
    if " " in path:
        tokens = path.split()
        # If the number of tokens is even and greater than one, form pairs
        if len(tokens) % 2 == 0 and len(tokens) > 1:
            pairs = [" ".join(tokens[i:i + 2]) for i in range(0, len(tokens), 2)]
            return pairs
    # Fallback: return the whole path as a single element
    return [path]


def pair_tokens(path: str) -> list[tuple[str, str]]:
    """
    Given a whitespace‑separated path string (e.g. "1 1 6 6 5 2"),
    return a list of 2‑tuples representing each adjacent pair of tokens.

    If the path contains an odd number of tokens or no whitespace, the
    function returns an empty list.

    Parameters
    ----------
    path : str
        The raw path string.

    Returns
    -------
    list of (str, str)
        List of pairs of tokens.
    """
    if not path or " " not in path:
        return []
    tokens = path.split()
    # Only consider even number of tokens
    if len(tokens) % 2 != 0 or len(tokens) < 2:
        return []
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(tokens), 2):
        # Protect against odd lengths
        if i + 1 < len(tokens):
            pairs.append((tokens[i], tokens[i + 1]))
    return pairs


def plot_path_switches_grouped(
    events: list[dict],
    flow_to_group: dict[str, tuple[str, str]],
    groups: list[tuple[str, str]],
    flow_to_start_offset: dict[str, float | None],
    flow_to_end_offset: dict[str, float | None],
    out_path: str,
    relative: bool = True,
):
    """Plot the path switch timeline for all flows, grouping by (src,dst) AS pairs.

    Parameters
    ----------
    events : list of event dicts
        As returned by parse_client_log and collect_events_and_groups.
    flow_to_group : dict
        Mapping from flow names to (src,dst) tuples.
    groups : list of tuples
        Unique (src,dst) pairs present.  The order defines the colour mapping.
    out_path : str
        Where to write the resulting image.
    relative : bool, optional
        If True (default), times are shown relative to the earliest event; if False
        the absolute timestamps from the logs are used.
    """
    if not events:
        raise RuntimeError("No events to plot.")
    # Organise events by flow
    by_flow: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        by_flow[ev["flow"]].append(ev)
    flows = sorted(by_flow.keys())
    #
    # Compute the experiment start time and relative offsets.  We align flows
    # according to any start offsets specified in their filenames.  The
    # objective is to place the very beginning of the earliest flow at time
    # zero on the x‑axis and ensure all subsequent events and durations are
    # measured relative to that point.
    #
    # Determine the earliest possible experiment start time (t0).  For flows
    # with an explicit start offset in the filename, we assume the first
    # event of that flow occurs exactly start_offset seconds after the
    # experiment begins.  Thus, we calculate t0 candidates as
    # (first_event_ts - start_offset).  The smallest of these is taken as
    # t0.  If no start offsets are present, t0 falls back to the earliest
    # timestamp observed in the logs.
    event_ts_list = [ev["ts"] for ev in events]
    min_event_ts = min(event_ts_list)
    candidate_t0s = []
    # Track the first and last timestamp per flow for later use
    flow_first_ts: dict[str, float] = {}
    flow_last_ts: dict[str, float] = {}
    for flow_name, ev_list in by_flow.items():
        ts_first = min(ev["ts"] for ev in ev_list)
        ts_last = max(ev["ts"] for ev in ev_list)
        flow_first_ts[flow_name] = ts_first
        flow_last_ts[flow_name] = ts_last
        start_off = flow_to_start_offset.get(flow_name)
        if start_off is not None:
            candidate_t0s.append(ts_first - start_off)
    if candidate_t0s:
        t0 = min(candidate_t0s)
    else:
        t0 = min_event_ts
    # Compute relative times for events and per‑flow offsets.  For each
    # event timestamp ts, we set rel_ts = ts - t0.  If a flow has a start
    # offset given in its filename, its relative start time is that offset.
    # Otherwise it is the first event relative time.  End offsets are
    # similarly taken from the filename if present, or the last event time.
    relative_event_times = []
    for ev in events:
        relative_event_times.append(ev["ts"] - t0 if relative else ev["ts"])
    # Precompute relative start and end times per flow
    flow_rel_start: dict[str, float] = {}
    flow_rel_end: dict[str, float] = {}
    for flow_name in flows:
        start_off = flow_to_start_offset.get(flow_name)
        end_off = flow_to_end_offset.get(flow_name)
        if relative:
            if start_off is not None:
                flow_rel_start[flow_name] = start_off
            else:
                flow_rel_start[flow_name] = flow_first_ts[flow_name] - t0
            if end_off is not None:
                flow_rel_end[flow_name] = end_off
            else:
                flow_rel_end[flow_name] = flow_last_ts[flow_name] - t0
        else:
            # Absolute mode: no shifting
            flow_rel_start[flow_name] = flow_first_ts[flow_name]
            flow_rel_end[flow_name] = flow_last_ts[flow_name]
    # Determine the minimum relative time among all event times and start offsets.
    # We will shift all times so that this minimum is zero.  This avoids
    # showing large offsets like 1.756e9 on the axis in relative mode.
    all_relative_times = list(relative_event_times) + list(flow_rel_start.values()) + list(flow_rel_end.values())
    if relative and all_relative_times:
        min_rel = min(all_relative_times)
    else:
        min_rel = 0.0
    # Shift event times and offsets so that the minimum is zero
    if relative:
        for ev, rel_ts in zip(events, relative_event_times):
            ev["rel_ts"] = rel_ts - min_rel
        for flow_name in flows:
            flow_rel_start[flow_name] -= min_rel
            flow_rel_end[flow_name] -= min_rel
    else:
        # In absolute mode we preserve absolute timestamps; rel_ts is simply ts
        for ev in events:
            ev["rel_ts"] = ev["ts"]
    # At this point, each event dict has 'rel_ts' for plotting, and
    # flow_rel_start/flow_rel_end hold relative (or absolute) lifeline bounds.
    # Order flows by group before using them for axis limits and plotting.  Sort by
    # group index (based on the order in 'groups'), then by flow name.
    flows_ordered = sorted(
        flows,
        key=lambda f: (groups.index(flow_to_group.get(f, ("Unknown", "Unknown"))), f),
    )
    # Build list of times for determining axis limits
    times_relative: list[float] = []
    for ev in events:
        times_relative.append(ev.get("rel_ts", ev["ts"]))
    # Include lifeline start and end times to ensure the axis covers the full duration.
    for flow in flows_ordered:
        times_relative.append(flow_rel_start[flow])
        times_relative.append(flow_rel_end[flow])
    # Build mapping of flows to group indices and colours
    # Use a colormap with enough distinct hues.  tab10 provides 10 well separated colours.
    cmap = plt.get_cmap("tab10")
    group_to_colour = {g: cmap(i % 10) for i, g in enumerate(groups)}
    # Determine if this is a single group scenario
    single_group = len(groups) == 1
    # Assign short labels
    short_map = assign_short_labels(flows)
    # Prepare figure
    fig_height = max(3, len(flows) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    # Compute y positions; flows ordered first by group, then by flow
    # We produce a list of (flow, group) sorted accordingly
    # flows_ordered has already been computed above based on group ordering
    # Mapping flow -> y position
    flow_to_y = {flow: idx for idx, flow in enumerate(flows_ordered)}
    # Compute min and max times for axis limits.  Include any flow end times if
    # provided to ensure connecting lines extend fully across the timeline.
    min_time = min(times_relative) if times_relative else 0.0
    max_time = max(times_relative) if times_relative else 1.0
    span = max(max_time - min_time, 1e-9)
    # For annotation stacking, threshold for considering two annotations "close"
    small_time = max(0.5, span * 0.02)
    # Draw per‑flow lifelines.  In multi‑group plots each group uses a distinct colour;
    # in single‑group plots a neutral grey is used.
    for flow in flows_ordered:
        y = flow_to_y[flow]
        # Start and end times are precomputed in flow_rel_start/end dictionaries
        start_time = flow_rel_start[flow]
        end_time = flow_rel_end[flow]
        # Ensure start_time <= end_time
        if end_time < start_time:
            end_time = start_time
        # Choose colour: group colour or neutral grey
        if single_group:
            colour = (0.5, 0.5, 0.5, 1.0)
        else:
            group = flow_to_group.get(flow, ("Unknown", "Unknown"))
            colour = group_to_colour.get(group, (0.6, 0.6, 0.6, 1.0))
        # Draw the lifeline with reduced opacity.  We draw lines only in multi‑group
        # scenarios or when the user explicitly requested relative lines.  For
        # single‑group plots, these lifelines will still be drawn (grey).
        ax.hlines(y, start_time, end_time, colors=[colour], linewidth=1.0, alpha=0.45, zorder=1)
    # Plot events and annotations
    for flow in flows_ordered:
        evs = sorted(by_flow[flow], key=lambda x: x["ts"])
        y = flow_to_y[flow]
        # Track annotation stacking to avoid overlap
        level_last_x: dict[int, float] = defaultdict(lambda: -1e9)
        # We will remember whether we've drawn the initialization marker for this flow
        init_drawn = False
        for idx, ev in enumerate(evs):
            # Use precomputed relative timestamp for plotting
            x = ev.get("rel_ts", ev["ts"])
            trig = ev.get("trigger", "other")
            marker = TRIGGER_MARKERS.get(trig, TRIGGER_MARKERS["other"])
            # Draw initialization marker only once, even if multiple init triggers present
            if trig == "initialization":
                if init_drawn:
                    continue
                init_drawn = True
            # Draw marker
            ax.scatter(x, y, marker=marker, s=100, edgecolors="black", facecolors="white", zorder=3)
            # For non‑init events annotate the new path if it changed from old
            if trig != "initialization" and ev.get("new") and ev.get("old") and ev["new"] != ev["old"]:
                # Determine which vertical level to place this annotation on to avoid
                # collisions with previous annotations at nearby times.  Each level
                # corresponds to an increasing vertical offset.
                level = 0
                while True:
                    last_x = level_last_x[level]
                    if abs(x - last_x) < small_time:
                        level += 1
                        continue
                    break
                level_last_x[level] = x
                # Base vertical offset for annotation (in points) relative to the marker.
                # Use a smaller initial offset and tighter level spacing to reduce
                # vertical height and avoid overlapping neighbouring rows.
                # Base vertical offset for annotations.  Keep the
                # initial offset modest and use a relatively tight spacing
                # between successive levels to conserve vertical space.  The
                # offset values are tuned to balance the vertical and
                # horizontal footprint of the diagonal pairs without
                # overlapping the lifeline or the flow above.
                base_offset = 6 + level * 12
                new_path = ev["new"]
                # Decide whether to display numeric pairs or a simple label.  Numeric
                # paths (whitespace‑separated digits) are rendered as diagonal
                # pairs; other paths are rendered as a single label using the
                # abbreviation logic.
                pairs = pair_tokens(new_path)
                if pairs:
                    # Numeric paths: render each two‑digit pair horizontally, but draw
                    # the two digits of the pair along a shallow diagonal.  This
                    # preserves all digits while compressing horizontal space and
                    # minimising vertical intrusion into the row above.  Each
                    # subsequent pair is offset along the x‑axis to avoid overlap.
                    # Tune these offsets carefully to balance readability and
                    # compactness.  The horizontal spacing between pairs should
                    # exceed the combined width of a diagonal pair to avoid
                    # collisions when events occur near each other in time.  The
                    # vertical offset within a pair should be modest so that the
                    # upper digit does not overlap the flow line above.
                    # Tune offsets for numeric path annotation: pairs are spaced much
                    # closer together to make the entire path appear as a unit.  The
                    # two digits within a pair are also drawn closer together
                    # while retaining a diagonal relationship.  Reducing these
                    # offsets increases compactness and mitigates overlap in
                    # crowded plots.
                    # Adjusted offsets for numeric path annotations.  Pairs are
                    # drawn closer together while digits within a pair are
                    # separated enough to prevent overlap.  These values were
                    # tuned to keep the path compact and readable.
                    pair_spacing_x = 5   # horizontal separation between successive pairs (points)
                    digit_dx = 2         # horizontal shift between digits within a pair
                    digit_dy = 5         # vertical shift between digits within a pair
                    for j, (d1, d2) in enumerate(pairs):
                        dx_base = j * pair_spacing_x
                        # First digit of the pair at the base offset
                        ax.annotate(
                            d1,
                            (x, y),
                            xytext=(dx_base, base_offset),
                            textcoords="offset points",
                            fontsize=5,
                            va="bottom",
                            ha="left",
                            alpha=0.8,
                            clip_on=False,
                        )
                        # Second digit of the pair slightly up and to the right
                        ax.annotate(
                            d2,
                            (x, y),
                            xytext=(dx_base + digit_dx, base_offset + digit_dy),
                            textcoords="offset points",
                            fontsize=5,
                            va="bottom",
                            ha="left",
                            alpha=0.8,
                            clip_on=False,
                        )
                else:
                    # Non‑numeric path (e.g. arrow notation): abbreviate and display normally
                    label = new_path
                    ax.annotate(
                        label,
                        (x, y),
                        xytext=(0, base_offset),
                        textcoords="offset points",
                        fontsize=6,
                        va="bottom",
                        ha="center",
                        alpha=0.75,
                        clip_on=False,
                    )
        # Also annotate the initial path for each flow (first event), like original
        if evs:
            first_ev = evs[0]
            init_path = first_ev.get("new") or first_ev.get("old")
            if init_path:
                # Use relative timestamp for initial marker
                x_init = first_ev.get("rel_ts", first_ev["ts"])
                # Draw the marker if we haven't drawn one (but we have drawn above)
                if not init_drawn:
                    ax.scatter(x_init, y, marker=TRIGGER_MARKERS["initialization"], s=100,
                               edgecolors="black", facecolors="white", zorder=3)
                # Render initial path using the same logic as for other events
                init_pairs = pair_tokens(init_path)
                base_offset_init = 6
                if init_pairs:
                    # Render numeric initial paths using the same diagonal
                    # representation as for subsequent events.  Draw each pair
                    # horizontally, with digits in the pair drawn on a shallow
                    # diagonal.  This preserves the full path while keeping
                    # annotations compact.
                    pair_spacing_init = 5
                    digit_dx_init = 2
                    digit_dy_init = 5
                    for j, (d1, d2) in enumerate(init_pairs):
                        dx_base = j * pair_spacing_init
                        # First digit at base offset
                        ax.annotate(
                            d1,
                            (x_init, y),
                            xytext=(dx_base, base_offset_init),
                            textcoords="offset points",
                            fontsize=5,
                            va="bottom",
                            ha="left",
                            alpha=0.85,
                            clip_on=False,
                        )
                        # Second digit slightly up and right
                        ax.annotate(
                            d2,
                            (x_init, y),
                            xytext=(dx_base + digit_dx_init, base_offset_init + digit_dy_init),
                            textcoords="offset points",
                            fontsize=5,
                            va="bottom",
                            ha="left",
                            alpha=0.85,
                            clip_on=False,
                        )
                else:
                    # Non‑numeric initial path
                    label_init = init_path
                    ax.annotate(
                        label_init,
                        (x_init, y),
                        xytext=(0, base_offset_init),
                        textcoords="offset points",
                        fontsize=6,
                        va="bottom",
                        ha="center",
                        alpha=0.85,
                        clip_on=False,
                    )
    # Configure y–axis: ticks and labels
    ax.set_yticks(list(range(len(flows_ordered))))
    ytick_labels = []
    ytick_colours = []
    for flow in flows_ordered:
        short = short_map[flow]
        # Set colour for multi‑group, otherwise black
        if single_group:
            ytick_colours.append("black")
        else:
            group = flow_to_group.get(flow, ("Unknown", "Unknown"))
            colour = group_to_colour.get(group, (0.0, 0.0, 0.0, 1.0))
            ytick_colours.append(colour)
        ytick_labels.append(short)
    # Apply coloured labels by setting properties on tick objects
    ax.set_yticklabels(ytick_labels)
    for tick, colour in zip(ax.get_yticklabels(), ytick_colours):
        tick.set_color(colour)
    ax.set_ylabel("Flow")
    ax.set_xlabel("Time (s)" + ("" if relative else " (absolute)"))
    # Disable automatic offset on the x‑axis to prevent display of large numbers
    ax.ticklabel_format(style="plain", axis="x")
    # Title
    if single_group:
        src, dst = groups[0]
        ax.set_title(f"Polaris Path Switch Timeline: AS {src} → AS {dst}")
    else:
        ax.set_title("Polaris Path Switch Timeline")
    # x grid
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    # Y‑axis padding
    y_min = -0.5
    y_max = len(flows_ordered) - 0.5
    y_span = y_max - y_min if y_max > y_min else 1.0
    extra_y = y_span * 0.05
    ax.set_ylim(y_min - extra_y, y_max + extra_y)
    # x‑axis padding to avoid clipping
    xmin, xmax = ax.get_xlim()
    pad = max(1.0, (xmax - xmin) * 0.05)
    ax.set_xlim(xmin - pad * 0.1, xmax + pad * 0.1)
    # Draw group separators
    if not single_group and len(groups) > 1:
        # Determine boundaries between groups
        last_group = flow_to_group.get(flows_ordered[0], ("Unknown", "Unknown"))
        for i, flow in enumerate(flows_ordered[1:], start=1):
            current_group = flow_to_group.get(flow, ("Unknown", "Unknown"))
            if current_group != last_group:
                # draw horizontal line between i-1 and i
                y_sep = i - 0.5
                ax.axhline(y=y_sep, color="grey", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
            last_group = current_group
    # Build event legend (only triggers present in the data)
    present_triggers = {ev["trigger"] for ev in events if ev["trigger"] in LEGEND_LABELS}
    handles_event: list[Line2D] = []
    for trig, lab in LEGEND_LABELS.items():
        if trig not in present_triggers:
            continue
        marker = TRIGGER_MARKERS.get(trig, "o")
        handles_event.append(Line2D([0], [0], marker=marker, color="black",
                                     label=lab, markerfacecolor="white",
                                     markersize=8, linestyle=""))
    # Render legends
    if single_group:
        # Only one group: show event legend exactly like the original script.  We anchor
        # outside the axes and rely on tight bounding box to include it.
        # Draw event legend inside the axes for single‑group plots.  Placing
        # the legend within the axes avoids cropping issues and maintains the
        # compact layout of the original script.
        event_leg = ax.legend(
            handles=handles_event,
            title="Event type",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        # Save with tight bounding box so the external legend is included
        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        # Multi‑group case: draw both legends.  Preserve the event legend before
        # adding the group legend so it is not overridden.
        # Event legend for multi‑group case: place inside the axes at the top right
        event_legend = ax.legend(
            handles=handles_event,
            title="Event type",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        # Build group legend and place it below the event legend on the right.
        handles_group: list[Line2D] = []
        labels_group: list[str] = []
        for g in groups:
            colour = group_to_colour.get(g, (0.0, 0.0, 0.0, 1.0))
            handles_group.append(Line2D([0], [0], color=colour, linewidth=4))
            labels_group.append(f"{g[0]} → {g[1]}")
        # Add the event legend back to the axes before drawing the group legend
        ax.add_artist(event_legend)
        group_legend = ax.legend(
            handles=handles_group,
            labels=labels_group,
            title="Source → Destination",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.5),
        )
        # Save with tight bounding box so both legends are included outside
        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote path switch timeline to {out_path}")

    # Write mapping file next to output
    plot_dir = os.path.dirname(out_path) or "."
    mapping_path = os.path.join(plot_dir, "flow_map.txt")
    with open(mapping_path, "w") as mf:
        for full, short in sorted(short_map.items(), key=lambda kv: kv[1]):
            mf.write(f"{short}\t{full}\n")
    print(f"Wrote flow mapping to {mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Polaris path switch events timeline per flow, supporting multiple source–destination AS pairs.")
    parser.add_argument("--base", required=True, help="Traffic generator output base directory containing client logs.")
    parser.add_argument("--log-include", required=False, help="Optional substring to filter client log filenames.")
    parser.add_argument("--out", required=False, help="Output image path. Defaults to <base>/switches.png")
    parser.add_argument("--absolute", action="store_true", help="Use absolute timestamps instead of relative.")
    args = parser.parse_args()

    # Determine output path
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(args.base.rstrip("/"), "switches.png")
    # Collect events and group information
    events, flow_to_group, groups, flow_to_start_offset, flow_to_end_offset = collect_events_and_groups(args.base, log_include=args.log_include)
    # Plot
    plot_path_switches_grouped(events, flow_to_group, groups,
                               flow_to_start_offset, flow_to_end_offset,
                               out_path=out_path,
                               relative=not args.absolute)


if __name__ == "__main__":
    main()