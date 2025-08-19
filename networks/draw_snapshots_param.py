#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCION topology + flow visualization (Step 1 + Step 2)
(Parameterized curvature: edges vs labels)

New flags (defaults reproduce current plots):
  --edge-curve-scale  FLOAT   Scale factor for the curvature of drawn edges.
                              1.0 = current default (no change).
  --label-curve-scale FLOAT   Scale factor for the *sampling* curve used to
                              place flow labels. 1.0 = match edges (no change).
                              Change this to make labels more/less curved than edges.

Examples:
  # (Step 1) topology only (unchanged look)
  python3 draw_snapshots3_param.py BASE_DIR --out topology_t0.png

  # (Step 2) one snapshot per path switch (unchanged look)
  python3 draw_snapshots3_param.py BASE_DIR --mode switches --outdir frames

  # (Step 2) snapshots at specific times (unchanged look)
  python3 draw_snapshots3_param.py BASE_DIR --mode times --times 0,15,30 --outdir frames

  # Make edges bend more strongly while keeping labels as before
  python3 draw_snapshots3_param.py BASE_DIR --mode times --times 0,15,30 \
      --edge-curve-scale 1.4 --label-curve-scale 1.0

  # Keep edges as before, make label sampling slightly flatter
  python3 draw_snapshots3_param.py BASE_DIR --mode switches \
      --edge-curve-scale 1.0 --label-curve-scale 0.75
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict, namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import networkx as nx

# ======= global style (publication-friendly) =======
mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    # IMPORTANT: do not crop at save time; cropping reflows transforms
    "savefig.bbox": "standard",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "font.family": "DejaVu Sans",
})

def _rc_save_ctx():
    """
    Return a version-safe rcParams dict for saving that avoids unknown keys.
    """
    rc = {"savefig.bbox": "standard"}
    if "figure.autolayout" in mpl.rcParams:
        rc["figure.autolayout"] = False
    if "constrained_layout.use" in mpl.rcParams:
        rc["constrained_layout.use"] = False
    return rc

# ---------- helpers: IA/AS ----------
def parse_ia(ia_str: str) -> str:
    return str(ia_str).strip()

def as_num_for_label(ia_str: str) -> str:
    s = ia_str.split('-')[-1]
    s = s.split(':')[-1]
    return s

# ---------- discover topologies ----------
def discover_topologies(base_dir: Path) -> dict:
    metrics_dir = base_dir / "metrics"
    if not metrics_dir.is_dir():
        print(f"[!] 'metrics' directory not found under {base_dir}", file=sys.stderr)
        sys.exit(1)
    tops = {}
    for topo_path in metrics_dir.rglob("topology.json"):
        try:
            d = json.loads(topo_path.read_text())
        except Exception as e:
            print(f"[!] Skipping unreadable {topo_path}: {e}", file=sys.stderr)
            continue
        ia = parse_ia(d.get("isd_as", "").strip())
        if ia and ia not in tops:
            tops[ia] = d
    if not tops:
        print("[!] No topology.json found.", file=sys.stderr)
        sys.exit(1)
    return tops

# ---------- extract endpoints & match links ----------
Endpoint = namedtuple("Endpoint", "ia br ifid peer_ia ul_local ul_remote")

def extract_endpoints(topologies: dict) -> list:
    eps = []
    for ia, topo in topologies.items():
        for br_name, br in (topo.get("border_routers") or {}).items():
            for ifid_str, ifobj in (br.get("interfaces") or {}).items():
                try:
                    ifid = int(ifid_str)
                except ValueError:
                    continue
                peer_ia = parse_ia(ifobj.get("isd_as","").strip())
                ul = ifobj.get("underlay") or {}
                ul_local, ul_remote = ul.get("local"), ul.get("remote")
                if not (peer_ia and ul_local and ul_remote):
                    continue
                eps.append(Endpoint(ia=ia, br=br_name, ifid=ifid,
                                    peer_ia=peer_ia, ul_local=ul_local, ul_remote=ul_remote))
    return eps

def match_links(endpoints: list):
    """Pair endpoints to full links; return list of dicts with both IFIDs."""
    index = defaultdict(list)
    for ep in endpoints:
        index[(ep.ia, ep.peer_ia, ep.ul_local, ep.ul_remote)].append(ep)

    seen = set()
    links = []
    for ep in endpoints:
        if ep in seen:
            continue
        rev_key = (ep.peer_ia, ep.ia, ep.ul_remote, ep.ul_local)
        revs = index.get(rev_key, [])
        rev = next((r for r in revs if r is not ep and r not in seen), None)
        if not rev:
            continue
        seen.add(ep); seen.add(rev)
        a_ia, b_ia = sorted([ep.ia, rev.ia])
        if ep.ia == a_ia:
            links.append(dict(a_ia=ep.ia, a_ifid=ep.ifid, a_br=ep.br,
                              b_ia=rev.ia, b_ifid=rev.ifid, b_br=rev.br,
                              key=(a_ia, b_ia)))
        else:
            links.append(dict(a_ia=rev.ia, a_ifid=rev.ifid, a_br=rev.br,
                              b_ia=ep.ia, b_ifid=ep.ifid, b_br=ep.br,
                              key=(a_ia, b_ia)))
    return links

# ---------- layout & curvature ----------
def circular_positions(nodes):
    def sort_key(ia):
        tail = as_num_for_label(ia)
        try:
            return int(tail)
        except ValueError:
            return tail
    ordered = sorted(nodes, key=sort_key)
    n = len(ordered); R = 1.0
    return {ia: (R*math.cos(2*math.pi*i/n), R*math.sin(2*math.pi*i/n))
            for i, ia in enumerate(ordered)}

def spring_positions(G):
    return nx.spring_layout(G, seed=7, k=0.7)

def assign_arc_radii_and_slots(links):
    groups = defaultdict(list)
    for idx, L in enumerate(links):
        groups[L["key"]].append(idx)
    rad_by_idx, slot_by_idx = {}, {}
    for key, idxs in groups.items():
        m = len(idxs)
        if m == 1:
            rads = [0.0]
        else:
            step = 0.21
            if m % 2:
                half = (m-1)//2
                rads = [-(i+1)*step for i in range(half)][::-1] + [0.0] + [(i+1)*step for i in range(half)]
            else:
                half = m//2
                rads = [-(i+0.5)*step for i in range(half)][::-1] + [(i+0.5)*step for i in range(half)]
        for i, edge_idx in enumerate(idxs):
            rad_by_idx[edge_idx] = rads[i]
        order = [ei for ei,_ in sorted(zip(idxs, rads), key=lambda t: t[1])]
        if m % 2:
            slots = list(range(-(m//2), m//2+1))
        else:
            half = m//2
            slots = list(range(-half,0)) + list(range(1,half+1))
        for s, edge_idx in zip(slots, order):
            slot_by_idx[edge_idx] = s
    return rad_by_idx, slot_by_idx

# ---------- IFID pair -> directed edge mapping ----------
def build_ifid_pair_map(links):
    """
    Map (ia, local_ifid) -> (edge_idx, next_ia, remote_ifid, dir)
    dir = +1 for a_ia->b_ia, -1 for b_ia->a_ia
    """
    m = {}
    for idx, L in enumerate(links):
        m[(L["a_ia"], L["a_ifid"])] = (idx, L["b_ia"], L["b_ifid"], +1)
        m[(L["b_ia"], L["b_ifid"])] = (idx, L["a_ia"], L["a_ifid"], -1)
    return m

def map_ifid_pairs_to_edges(start_ia, seq_nums, pair_map):
    """seq like [3,3,4,1] -> [(edge_idx, dir), ...], following pairs from start_ia."""
    if len(seq_nums) % 2 != 0:
        raise ValueError(f"IFID sequence must be even-length pairs, got: {seq_nums}")
    cur = start_ia
    edges = []
    for i in range(0, len(seq_nums), 2):
        local, remote = seq_nums[i], seq_nums[i+1]
        key = (cur, local)
        if key not in pair_map:
            raise KeyError(f"Unknown hop: IA={cur} local IFID={local}")
        idx, nxt, expected_remote, d = pair_map[key]
        # Optional remote check
        if expected_remote != remote:
            pass
        edges.append((idx, d))
        cur = nxt
    return edges

# ---------- log parsing ----------
LOG_LINE_RE = re.compile(r'POLARIS_LOG\s+(\{.*\})')
FNAME_RE    = re.compile(r'(?P<start>\d+)_cs_1_to_(?P<end>\d+)_cs_1_port(?P<port>\d+)_(?P<s>\d+)s_(?P<e>\d+)s_client\.log$')


def read_flow_map(path: Path):
    m = {}
    if not path.is_file():
        print(f"[!] flow_map.txt not found at {path}", file=sys.stderr)
        return m
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        label, key = line.split("\t", 1)
        m[key] = label
    return m

def find_client_logs(base: Path):
    return list(base.rglob("*_client.log"))

def parse_client_log(path: Path):
    """Return list of events dicts with ts, trigger, new(list[int]), old(list[int] or [] )."""
    out = []
    for line in path.read_text().splitlines():
        m = LOG_LINE_RE.search(line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        if obj.get("event") != "path_switch":
            continue
        try:
            ts = float(obj["ts"])
            new = [int(x) for x in str(obj["new"]).split()]
            old = [int(x) for x in str(obj.get("old","")).split()] if "old" in obj else []
            trig = str(obj.get("trigger", ""))
        except Exception:
            continue
        out.append({"ts": ts, "trigger": trig, "new": new, "old": old})
    return out

def derive_flow_meta_from_fname(fname: str):
    m = FNAME_RE.search(fname)
    if not m:
        return None
    s_as = f"1-{m.group('start')}"
    e_as = f"1-{m.group('end')}"
    start_off = int(m.group('s'))  # may be useful if you ever want to check
    return (s_as, e_as, start_off)

# ======= Curvature math helpers (for label-only custom curve) =======
# Note: Matplotlib's "arc3,rad=r" connectionstyle generates a quadratic Bezier-like
# curve in DISPLAY coordinates. When label_curve_scale != edge_curve_scale, we
# synthesize a similar quadratic curve purely for label placement, with the same
# endpoints as the visible edge but a different curvature. This lets you tune the
# bend of labels independently without changing edge geometry or spacing logic.
#
# The control point offset uses: offset = ARC3_CTRL_FACTOR * rad * chord_length.
# ARC3_CTRL_FACTOR ~= 0.5 matches Matplotlib's visual look reasonably well.
ARC3_CTRL_FACTOR = 0.5

def _display_polyline_from_patch(patch, steps=256):
    """
    Dense polyline of the *actually drawn* edge in display (pixel) coords.
    Transform the patch path with the patch transform, then densify.
    Keep only the longest sub-path and collapse duplicate vertices.
    """
    tpath = patch.get_path().transformed(patch.get_transform())
    try:
        tpath = tpath.interpolated(steps)
    except Exception:
        pass
    V = tpath.vertices
    codes = tpath.codes

    if codes is not None:
        MOVETO = mpath.Path.MOVETO
        starts = [i for i, c in enumerate(codes) if c == MOVETO] + [len(V)]
        best_seg = None
        best_len = -1.0
        for i in range(len(starts) - 1):
            seg = V[starts[i]:starts[i+1]]
            if len(seg) < 2:
                continue
            L = 0.0
            for j in range(1, len(seg)):
                dx = seg[j][0] - seg[j-1][0]
                dy = seg[j][1] - seg[j-1][1]
                L += math.hypot(dx, dy)
            if L > best_len:
                best_len = L
                best_seg = seg
        P = best_seg if best_seg is not None else V
    else:
        P = V

    # collapse exact duplicates
    Q = []
    for x, y in P:
        if not Q or (Q[-1][0] != x or Q[-1][1] != y):
            Q.append((float(x), float(y)))
    return Q

def _polyline_even_samples_to_pre(poly_disp, n, margin_frac, inv_transform):
    """
    Given a display-space polyline, sample n evenly spaced points between
    margins and map them back to PRE-transform coordinates via inv_transform.
    """
    if len(poly_disp) < 2:
        if poly_disp:
            return [tuple(inv_transform.transform(poly_disp[0]))] * max(1, n)
        return [(0.0, 0.0)] * max(1, n)

    # cumulative display arc-length
    cum = [0.0]
    for i in range(1, len(poly_disp)):
        dx = poly_disp[i][0] - poly_disp[i-1][0]
        dy = poly_disp[i][1] - poly_disp[i-1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1] or 1.0

    a = margin_frac * total
    b = (1.0 - margin_frac) * total
    if b <= a:  # very short curve fallback
        a, b = 0.25 * total, 0.75 * total

    if n <= 0:
        targets = []
    elif n == 1:
        targets = [0.5 * (a + b)]
    else:
        gap = (b - a) / (n + 1)
        targets = [a + gap * (i + 1) for i in range(n)]

    pts_pre = []
    j = 1
    for s in targets:
        while j < len(cum) and cum[j] < s:
            j += 1
        j = max(1, min(j, len(cum) - 1))
        seg_len = cum[j] - cum[j-1]
        t = 0.0 if seg_len == 0 else (s - cum[j-1]) / seg_len
        xd = (1.0 - t) * poly_disp[j-1][0] + t * poly_disp[j][0]
        yd = (1.0 - t) * poly_disp[j-1][1] + t * poly_disp[j][1]
        pts_pre.append(tuple(inv_transform.transform((xd, yd))))
    return pts_pre

def _arc3_polyline_display(P0, P1, rad, steps=256):
    """
    Build a display-space polyline approximating an 'arc3' quadratic curve
    between P0 and P1 with curvature 'rad'. The sign of 'rad' gives the bend side.
    """
    x0, y0 = P0; x1, y1 = P1
    vx, vy = (x1 - x0), (y1 - y0)
    L = math.hypot(vx, vy)
    if L == 0.0:
        return [P0, P1]
    # Perpendicular (unit) normal
    nx, ny = (-vy / L, vx / L)
    # Control point offset from midpoint
    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    off = ARC3_CTRL_FACTOR * rad * L
    cx, cy = (mx + off * nx, my + off * ny)

    poly = []
    for i in range(max(2, steps)):
        t = i / (steps - 1)
        # Quadratic Bezier: B(t) = (1-t)^2 P0 + 2(1-t)t C + t^2 P1
        omt = 1.0 - t
        bx = (omt*omt)*x0 + 2*omt*t*cx + (t*t)*x1
        by = (omt*omt)*y0 + 2*omt*t*cy + (t*t)*y1
        poly.append((bx, by))
    return poly

def _sample_points_along_patch_display_dense(patch, n, margin_frac=0.18, steps=256):
    """
    Evenly spaced points *on the visible edge*, computed in display coords,
    then mapped back to the patch's pre-transform coordinates so we can
    annotate with xycoords=patch.get_transform().
    """
    P = _display_polyline_from_patch(patch, steps=steps)
    inv = patch.get_transform().inverted()

    return _polyline_even_samples_to_pre(P, n, margin_frac, inv)

def _sample_points_along_scaled_arc_display(patch, rad_base, edge_scale, label_scale,
                                            n, margin_frac=0.18, steps=256):
    """
    When label_scale != edge_scale, build a synthetic arc3-like curve with the
    SAME endpoints as the visible edge but curvature 'rad_base * label_scale',
    then sample evenly along it (display coords -> pre-transform).
    """
    # If scales match, just use the visible edge exactly
    if math.isclose(label_scale, edge_scale, rel_tol=1e-12, abs_tol=1e-12):
        return _sample_points_along_patch_display_dense(patch, n, margin_frac, steps)

    # Extract the drawn edge's endpoints in DISPLAY coords
    P = _display_polyline_from_patch(patch, steps=steps)
    if len(P) < 2:
        return _sample_points_along_patch_display_dense(patch, n, margin_frac, steps)
    P0, P1 = P[0], P[-1]

    # Build synthetic polyline with desired curvature
    rad_label = rad_base * label_scale
    Q = _arc3_polyline_display(P0, P1, rad_label, steps=steps)

    inv = patch.get_transform().inverted()
    return _polyline_even_samples_to_pre(Q, n, margin_frac, inv)

# ---------- drawing primitives ----------
def base_graph(G, pos, links, rad_by_idx, edge_style, ax, edge_curve_scale=1.0):
    """
    Draw edges using 'arc3' with per-pair base curvature (rad_by_idx) scaled
    by 'edge_curve_scale'. Return geom metadata with both base and drawn rad.
    """
    geom = {}
    for idx, L in enumerate(links):
        u, v = L["a_ia"], L["b_ia"]
        rad_base = rad_by_idx[idx]
        rad_drawn = rad_base * edge_curve_scale
        try:
            artists = nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad_drawn}",
                ax=ax, **edge_style
            )
        except TypeError:
            # Older Matplotlib doesn't know min_*_margin
            _kw = dict(edge_style)
            _kw.pop("min_source_margin", None)
            _kw.pop("min_target_margin", None)
            artists = nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad_drawn}",
                ax=ax, **_kw
            )
        patch = artists[0] if isinstance(artists, list) else artists
        try:
            patch.set_zorder(2)
        except Exception:
            pass
        geom[idx] = {"patch": patch, "rad_base": rad_base, "rad_drawn": rad_drawn}
    return geom

# ---------- IFID labels (placed on visible edge near nodes) ----------
def draw_ifid_labels(ax, pos, links, geom, node_radius_px, edge_clear_px, IFID_BOX):
    to_px   = ax.transData.transform
    to_data = ax.transData.inverted().transform
    anchor_r_px = node_radius_px + edge_clear_px

    for idx, L in enumerate(links):
        u, v = L["a_ia"], L["b_ia"]
        x1, y1 = pos[u]; x2, y2 = pos[v]
        patch = geom[idx]["patch"]
        V = patch.get_path().transformed(patch.get_transform()).vertices
        c_u_px = to_px((x1, y1))
        c_v_px = to_px((x2, y2))

        def intersect_circle_along_poly(Vseq, C, r):
            for i in range(1, len(Vseq)):
                Ax, Ay = Vseq[i-1][0]-C[0], Vseq[i-1][1]-C[1]
                Bx, By = Vseq[i][0]-C[0],   Vseq[i][1]-C[1]
                d0, d1 = math.hypot(Ax,Ay), math.hypot(Bx,By)
                if d0 <= r <= d1:
                    Dx, Dy = Bx-Ax, By-Ay
                    a = Dx*Dx + Dy*Dy
                    b = 2*(Ax*Dx + Ay*Dy)
                    c = Ax*Ax + Ay*Ay - r*r
                    disc = b*b - 4*a*c
                    if a != 0 and disc >= 0:
                        rt = math.sqrt(disc)
                        for t in ((-b+rt)/(2*a), (-b-rt)/(2*a)):
                            if 0.0 <= t <= 1.0:
                                Px = Vseq[i-1][0] + t*(Vseq[i][0]-Vseq[i-1][0])
                                Py = Vseq[i-1][1] + t*(Vseq[i][1]-Vseq[i-1][1])
                                return (Px, Py)
            return tuple(Vseq[0])

        P_u_px = intersect_circle_along_poly(V,        c_u_px, anchor_r_px)
        P_v_px = intersect_circle_along_poly(V[::-1],  c_v_px, anchor_r_px)

        Q_u = to_data(P_u_px); Q_v = to_data(P_v_px)
        ax.annotate(str(L["a_ifid"]), xy=Q_u, xycoords='data',
                    ha='center', va='center', bbox=IFID_BOX,
                    annotation_clip=False)
        ax.annotate(str(L["b_ifid"]), xy=Q_v, xycoords='data',
                    ha='center', va='center', bbox=IFID_BOX,
                    annotation_clip=False)

# ---------- Step 1 renderer ----------
def draw_topology(links, topologies, out_path, layout="circular",
                  dpi=200, size=(7,5), title_time_s=0, flipx=True,
                  edge_curve_scale=1.0):
    nodes = set(topologies.keys())
    G = nx.Graph(); G.add_nodes_from(nodes)
    for L in links:
        G.add_edge(L["a_ia"], L["b_ia"])
    pos = spring_positions(G) if layout == "spring" else circular_positions(nodes)
    if flipx:
        pos = {k:(-x, y) for k,(x, y) in pos.items()}

    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    ax.axis("off")
    ax.set_title(f"Flows at t={title_time_s}s", fontsize=11, pad=6)

    NODE_SIZE = 3000
    nodes_artist = nx.draw_networkx_nodes(
        G, pos, node_size=NODE_SIZE, node_color="#ffffff",
        edgecolors="#222222", linewidths=1.4, ax=ax
    )
    try:
        nodes_artist.set_zorder(3)
    except Exception:
        pass
    node_labels = {ia: f"AS {as_num_for_label(ia)}" for ia in topologies.keys()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold", ax=ax)

    # Final limits BEFORE edges/labels
    xs, ys = zip(*pos.values()); pad = 0.28
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    # Edges
    EDGE_STYLE = dict(width=1.3, edge_color="#444444",
                      arrows=True, arrowstyle='-',
                      min_source_margin=12, min_target_margin=12)
    rad_by_idx, _ = assign_arc_radii_and_slots(links)
    geom = base_graph(G, pos, links, rad_by_idx, EDGE_STYLE, ax, edge_curve_scale=edge_curve_scale)

    # Patches must finalize their paths before IFID anchors
    fig.canvas.draw()

    # IFID labels
    pt2px = fig.dpi / 72.0
    node_radius_px = math.sqrt(NODE_SIZE / math.pi) * pt2px
    EDGE_CLEAR_PX = 8
    IFID_BOX = dict(boxstyle="round,pad=0.12", facecolor="white",
                    alpha=0.95, linewidth=0.4, edgecolor="#666")
    draw_ifid_labels(ax, pos, links, geom, node_radius_px, EDGE_CLEAR_PX, IFID_BOX)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(_rc_save_ctx()):
        fig.savefig(out_path)
    plt.close(fig)
    print(f"[✓] Wrote {out_path.resolve()}")

# ---------- Step 2 snapshot renderer ----------
def render_snapshot(links, topologies, pos, geom, rad_by_idx,
                    flows_on_edges, highlight, out_path, title,
                    edge_curve_scale=1.0, label_curve_scale=1.0):
    """
    Draw one snapshot with labels. By default (label_curve_scale == edge_curve_scale)
    labels are placed *exactly* on the visible edge by sampling the drawn patch
    in display space. When label_curve_scale != edge_curve_scale, we synthesize a
    similar 'arc3' curve with the same endpoints but different curvature and
    sample along that for label placement.
    """
    ax = plt.gca()
    # Set title first, then force a draw so all transforms are final
    ax.set_title(title, fontsize=11, pad=6)
    ax.figure.canvas.draw()

    # Styles
    BOX_NEUTRAL = dict(boxstyle="round,pad=0.12", facecolor="#ffffff",
                       alpha=0.95, linewidth=0.6, edgecolor="#444")
    BOX_GREEN   = dict(boxstyle="round,pad=0.12", facecolor="#dff5e6",
                       alpha=0.95, linewidth=0.8, edgecolor="#2e7d32")
    BOX_RED     = dict(boxstyle="round,pad=0.12", facecolor="#f9dede",
                       alpha=0.95, linewidth=0.8, edgecolor="#c62828")

    FLOW_MARGIN_FRAC = 0.18  # keep labels away from ends

    switching_flow = highlight.get("flow") if highlight else None
    add_edges = set(highlight.get("add_edges", set())) if highlight else set()
    rem_edges = set(highlight.get("rem_edges", set())) if highlight else set()
    exclude_edges = set(highlight.get("exclude_edges", set())) if highlight else set()
    fl_label = highlight.get("flow", "") if highlight else ""

    # Build items per edge (neutral + overlays together)
    items_per_edge = defaultdict(list)

    # Neutral items (suppress switching-flow neutral labels on highlighted edges)
    for idx, flows in flows_on_edges.items():
        for fl in sorted(flows):
            if switching_flow and idx in exclude_edges and fl == switching_flow:
                continue
            items_per_edge[idx].append({"text": fl, "box": BOX_NEUTRAL})

    # Overlays integrated (red = removed, green = added)
    if highlight:
        for idx in sorted(rem_edges):
            items_per_edge[idx].append({"text": fl_label, "box": BOX_RED})
        for idx in sorted(add_edges):
            items_per_edge[idx].append({"text": fl_label, "box": BOX_GREEN})

    # Place along each actual drawn or synthetic curve
    for idx, items in items_per_edge.items():
        if not items:
            continue
        patch = geom[idx]["patch"]
        rad_base = geom[idx]["rad_base"]  # base (pre-scale) curvature used for this pair

        pts_path = _sample_points_along_scaled_arc_display(
            patch, rad_base, edge_curve_scale, label_curve_scale,
            n=len(items), margin_frac=FLOW_MARGIN_FRAC, steps=256
        )
        tr = patch.get_transform()
        for it, pt in zip(items, pts_path):
            ax.annotate(
                it["text"],
                xy=pt, xycoords=tr,      # pre-transform coords + patch transform
                ha='center', va='center',
                bbox=it["box"],
                annotation_clip=False
            )

# ---------- Step 2 coordinator ----------
def run_step2(base: Path, layout: str, dpi: int, size: tuple, mode: str, times: list,
              outdir: Path, flipx=True, edge_curve_scale=1.0, label_curve_scale=1.0):
    topologies = discover_topologies(base)
    endpoints  = extract_endpoints(topologies)
    links      = match_links(endpoints)
    if not links:
        print("[!] No inter-AS links matched.", file=sys.stderr); sys.exit(1)

    # Layout + base canvas
    nodes = set(topologies.keys())
    G = nx.Graph(); G.add_nodes_from(nodes)
    for L in links:
        G.add_edge(L["a_ia"], L["b_ia"])
    pos = spring_positions(G) if layout == "spring" else circular_positions(nodes)
    if flipx:
        pos = {k:(-x, y) for k,(x, y) in pos.items()}

    # Styles
    EDGE_STYLE = dict(width=1.3, edge_color="#444444",
                      arrows=True, arrowstyle='-',
                      min_source_margin=12, min_target_margin=12)

    rad_by_idx, _slots = assign_arc_radii_and_slots(links)

    # IFID pair mapper
    pair_map = build_ifid_pair_map(links)

    # Flow map and logs
    fmap = read_flow_map(base / "flow_map.txt")
    logs = find_client_logs(base)

    events = []  # list of dicts: flow, ts, trigger, start_ia, new_edges(list idx), old_edges(list idx)
    init_ts_list = []

    for lf in logs:
        meta = derive_flow_meta_from_fname(lf.name)
        if not meta:
            continue
        start_ia, end_ia, _start_off = meta

        # flow label from flow_map
        key_prefix = lf.name.replace("_client.log", "")
        flow_label = fmap.get(key_prefix)
        if not flow_label:
            for k, v in fmap.items():
                if key_prefix.startswith(k):
                    flow_label = v; break
        if not flow_label:
            continue  # silently skip unknown flows

        evs = parse_client_log(lf)
        for ev in evs:
            try:
                new_edges = [idx for (idx, _d) in map_ifid_pairs_to_edges(start_ia, ev["new"], pair_map)]
            except Exception as e:
                print(f"[!] {lf.name}: cannot map NEW {ev['new']} -> {e}", file=sys.stderr)
                continue
            old_edges = []
            if ev["old"]:
                try:
                    old_edges = [idx for (idx, _d) in map_ifid_pairs_to_edges(start_ia, ev["old"], pair_map)]
                except Exception:
                    pass
            events.append({
                "flow": flow_label,
                "ts": ev["ts"],
                "trigger": ev["trigger"],
                "start_ia": start_ia,
                "new_edges": new_edges,
                "old_edges": old_edges,
            })
            if ev["trigger"] == "initialization":
                init_ts_list.append(ev["ts"])

    if not events:
        print("[!] No path_switch events parsed.", file=sys.stderr); sys.exit(1)

    # Relative time origin: earliest initialization event (fallback to earliest event)
    t0 = min(init_ts_list) if init_ts_list else min(e["ts"] for e in events)

    outdir.mkdir(parents=True, exist_ok=True)

    if mode == "times":
        # Build state up to each time, then render (no overlays)
        def state_at(ts_abs):
            per_flow = {}
            for ev in sorted(events, key=lambda x: x["ts"]):
                if ev["ts"] <= ts_abs:
                    per_flow[ev["flow"]] = ev["new_edges"]
            flows_on_edges = defaultdict(list)
            for fl, eds in per_flow.items():
                for idx in eds:
                    flows_on_edges[idx].append(fl)
            return flows_on_edges

        for t_rel in times:
            fig, ax = plt.subplots(figsize=size, dpi=dpi); ax.axis("off")

            # Nodes first
            NODE_SIZE = 3000
            nodes_artist = nx.draw_networkx_nodes(
                G, pos, node_size=NODE_SIZE, node_color="#ffffff",
                edgecolors="#222222", linewidths=1.4, ax=ax
            )
            try:
                nodes_artist.set_zorder(3)
            except Exception:
                pass
            node_labels = {ia: f"AS {as_num_for_label(ia)}" for ia in topologies.keys()}
            nx.draw_networkx_labels(G, pos, labels=node_labels,
                                    font_size=10, font_weight="bold", ax=ax)

            # Final limits BEFORE edges/IFIDs/labels
            xs, ys = zip(*pos.values()); pad = 0.28
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)

            # Edges
            geom = base_graph(G, pos, links, rad_by_idx, EDGE_STYLE, ax, edge_curve_scale=edge_curve_scale)

            # Patches must be finalized before IFID placement
            fig.canvas.draw()

            # IFID labels
            pt2px = fig.dpi / 72.0
            node_radius_px = math.sqrt(NODE_SIZE / math.pi) * pt2px
            EDGE_CLEAR_PX = 8
            IFID_BOX = dict(boxstyle="round,pad=0.12", facecolor="white",
                            alpha=0.95, linewidth=0.4, edgecolor="#666")
            draw_ifid_labels(ax, pos, links, geom, node_radius_px, EDGE_CLEAR_PX, IFID_BOX)

            # State & flow labels (render_snapshot sets title & forces draw)
            flows_on_edges = state_at(t0 + t_rel)
            title = f"Snapshot @ t={t_rel:.3f}s"
            render_snapshot(links, topologies, pos, geom, rad_by_idx, flows_on_edges,
                            highlight=None, out_path=None, title=title,
                            edge_curve_scale=edge_curve_scale, label_curve_scale=label_curve_scale)

            out = outdir / f"time_{t_rel:08.3f}s.png"
            with mpl.rc_context(_rc_save_ctx()):
                fig.savefig(out)
            plt.close(fig)
            print(f"[✓] Wrote {out.resolve()}")

    else:
        # mode == switches
        cur_per_flow = {}
        for i, ev in enumerate(sorted(events, key=lambda x: x["ts"])):
            old_set = set(cur_per_flow.get(ev["flow"], ev["old_edges"] or []))
            new_set = set(ev["new_edges"])
            cur_per_flow[ev["flow"]] = ev["new_edges"]

            # Neutral state after commit
            flows_on_edges = defaultdict(list)
            for fl, eds in cur_per_flow.items():
                for idx in eds:
                    flows_on_edges[idx].append(fl)

            exclude_edges = new_set | old_set

            fig, ax = plt.subplots(figsize=size, dpi=dpi); ax.axis("off")

            NODE_SIZE = 3000
            nodes_artist = nx.draw_networkx_nodes(
                G, pos, node_size=NODE_SIZE, node_color="#ffffff",
                edgecolors="#222222", linewidths=1.4, ax=ax
            )
            try:
                nodes_artist.set_zorder(3)
            except Exception:
                pass
            node_labels = {ia: f"AS {as_num_for_label(ia)}" for ia in topologies.keys()}
            nx.draw_networkx_labels(G, pos, labels=node_labels,
                                    font_size=10, font_weight="bold", ax=ax)

            # Final limits BEFORE edges/IFIDs/labels
            xs, ys = zip(*pos.values()); pad = 0.28
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)

            # Edges
            geom = base_graph(G, pos, links, rad_by_idx, EDGE_STYLE, ax, edge_curve_scale=edge_curve_scale)

            # Patches must be finalized before IFID anchors
            fig.canvas.draw()

            # IFID labels
            pt2px = fig.dpi / 72.0
            node_radius_px = math.sqrt(NODE_SIZE / math.pi) * pt2px
            EDGE_CLEAR_PX = 8
            IFID_BOX = dict(boxstyle="round,pad=0.12", facecolor="white",
                            alpha=0.95, linewidth=0.4, edgecolor="#666")
            draw_ifid_labels(ax, pos, links, geom, node_radius_px, EDGE_CLEAR_PX, IFID_BOX)

            # Highlight overlays combined with neutral placement
            highlight = dict(flow=ev["flow"], add_edges=new_set, rem_edges=old_set, exclude_edges=exclude_edges)
            t_rel = ev["ts"] - t0
            title = f"Snapshot @ t={t_rel:.3f}s"
            render_snapshot(links, topologies, pos, geom, rad_by_idx, flows_on_edges,
                            highlight=highlight, out_path=None, title=title,
                            edge_curve_scale=edge_curve_scale, label_curve_scale=label_curve_scale)

            out = outdir / f"frame_{i:04d}_t{t_rel:08.3f}s.png"
            with mpl.rc_context(_rc_save_ctx()):
                fig.savefig(out)
            plt.close(fig)
            print(f"[✓] Wrote {out.resolve()}")

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Draw SCION AS topology and flow snapshots (parameterized curvature).")
    p.add_argument("base", type=Path, help="Base experiment folder")
    p.add_argument("--out", type=Path, default=Path("topology_t0.png"), help="(Step 1) output image path")
    p.add_argument("--layout", choices=["circular","spring"], default="circular")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--size", nargs=2, type=float, default=(7.0,5.0), metavar=("W","H"))
    p.add_argument("--no-flipx", dest="flipx", action="store_false",
                   help="Disable left↔right mirroring (default: mirrored)")
    p.set_defaults(flipx=True)
    p.add_argument("--flipx", action="store_true", help="Mirror left↔right")  # optional override

    # NEW: curvature scales
    p.add_argument("--edge-curve-scale", type=float, default=1.0,
                   help="Multiply the base rad of each edge by this factor (visual bend of edges). Default: 1.0")
    p.add_argument("--label-curve-scale", type=float, default=1.0,
                   help="Multiply the base rad used *only* for label placement. Default: 1.0 (match edges)")

    # Step 2
    p.add_argument("--mode", choices=["switches","times"], help="Enable Step 2")
    p.add_argument("--times", type=str, help="Comma-separated relative times (for --mode times)")
    p.add_argument("--outdir", type=Path, default=Path("frames"), help="Output directory for Step 2 frames")
    args = p.parse_args()

    tops = discover_topologies(args.base)
    endpoints = extract_endpoints(tops)
    if not endpoints:
        print("[!] No interfaces found.", file=sys.stderr); sys.exit(1)
    links = match_links(endpoints)
    if not links:
        print("[!] No inter-AS links matched.", file=sys.stderr); sys.exit(1)

    if not args.mode:
        draw_topology(links, tops, out_path=args.out, layout=args.layout,
                      dpi=args.dpi, size=tuple(args.size), title_time_s=0, flipx=args.flipx,
                      edge_curve_scale=args.edge_curve_scale)
    else:
        times = []
        if args.mode == "times":
            if not args.times:
                print("[!] --times required for --mode times", file=sys.stderr); sys.exit(1)
            try:
                times = [float(x) for x in args.times.split(",") if x.strip() != ""]
            except Exception:
                print("[!] Could not parse --times", file=sys.stderr); sys.exit(1)
        run_step2(args.base, args.layout, args.dpi, tuple(args.size), args.mode, times, args.outdir,
                  flipx=args.flipx, edge_curve_scale=args.edge_curve_scale, label_curve_scale=args.label_curve_scale)

if __name__ == "__main__":
    main()
