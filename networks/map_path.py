#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Tuple, List, Optional

class Interface:
    def __init__(self, iface_id: str, data: dict):
        self.id = iface_id
        self.data = data
        self.underlay = data.get("underlay", {})

    def local(self) -> Optional[str]:
        return self.underlay.get("local")

    def remote(self) -> Optional[str]:
        return self.underlay.get("remote")

class BorderRouter:
    def __init__(self, isd_as: str, name: str, topo: dict):
        self.isd_as = isd_as
        self.name = name
        self.interfaces: Dict[str, Interface] = {}
        br = topo.get("border_routers", {}).get(name, {})
        for iface_id, iface_data in br.get("interfaces", {}).items():
            self.interfaces[iface_id] = Interface(iface_id, iface_data)

def load_all_routers(topo_root: str) -> Dict[Tuple[str, str], BorderRouter]:
    routers: Dict[Tuple[str, str], BorderRouter] = {}
    for root, dirs, files in os.walk(topo_root):
        if "topology.json" in files:
            path = os.path.join(root, "topology.json")
            try:
                with open(path) as f:
                    topo = json.load(f)
            except Exception as e:
                print(f"Warning: failed to parse {path}: {e}")
                continue
            isd_as = topo.get("isd_as")
            if not isd_as:
                continue
            for br_name in topo.get("border_routers", {}):
                br = BorderRouter(isd_as, br_name, topo)
                routers[(isd_as, br_name)] = br
    return routers

def build_underlay_map(routers: Dict[Tuple[str, str], BorderRouter]):
    # Map (local, remote) -> ((isd_as, router_name), iface_id)
    underlay_map: Dict[Tuple[str, str], Tuple[Tuple[str, str], str]] = {}
    for key, br in routers.items():
        for iface_id, iface in br.interfaces.items():
            local = iface.local()
            remote = iface.remote()
            if local and remote:
                underlay_map[(local, remote)] = (key, iface_id)
    return underlay_map

def find_router_with_iface(routers: Dict[Tuple[str, str], BorderRouter],
                           isd_as: str,
                           iface_id: str,
                           specified_router: Optional[str]=None):
    candidates = []
    for (a, br_name), br in routers.items():
        if a != isd_as:
            continue
        if iface_id in br.interfaces:
            if specified_router and br_name != specified_router:
                continue
            candidates.append(((a, br_name), br))
    if not candidates:
        raise ValueError(f"No border router in {isd_as} has interface {iface_id}.")
    if len(candidates) > 1 and not specified_router:
        names = [br_name for (_, br_name), _ in candidates]
        raise ValueError(f"Ambiguous router in {isd_as} for interface {iface_id}: {names}. Use --start-router to disambiguate.")
    return candidates[0]  # (key, BorderRouter)

def find_any_router_in_as_with_iface(routers: Dict[Tuple[str, str], BorderRouter],
                                     isd_as: str,
                                     iface_id: str,
                                     exclude_router: Optional[str]=None):
    for (a, br_name), br in routers.items():
        if a != isd_as:
            continue
        if iface_id in br.interfaces and br_name != exclude_router:
            return ((a, br_name), br)
    return None

def map_path_pairwise(path_seq: List[str],
                      start_isd_as: str,
                      routers: Dict[Tuple[str, str], BorderRouter],
                      underlay_map: Dict[Tuple[str, str], Tuple[Tuple[str, str], str]],
                      start_router: Optional[str]=None,
                      verbose: bool=False):
    if len(path_seq) < 2 or len(path_seq) % 2 != 0:
        raise ValueError("Path sequence must have even length >=2 (exit, entry, exit, entry, ...).")

    hops = []

    # Initial exit: locate router in start AS with the first exit interface.
    (cur_key, cur_br) = find_router_with_iface(routers, start_isd_as, path_seq[0], specified_router=start_router)
    cur_as, cur_router_name = cur_key
    cur_iface_id = path_seq[0]
    if verbose:
        print(f"[INIT] Starting in AS {cur_as}, router {cur_router_name}, exit via interface {cur_iface_id}")

    for pair_idx in range(0, len(path_seq), 2):
        exit_iface_id = path_seq[pair_idx]
        entry_iface_expected = path_seq[pair_idx + 1]

        from_as = cur_as  # AS before crossing

        # Ensure we are on a router that has the exit interface; if not, try intra-AS router switch.
        if exit_iface_id not in cur_br.interfaces:
            fallback = find_any_router_in_as_with_iface(routers, cur_as, exit_iface_id, exclude_router=cur_router_name)
            if fallback:
                old_router = cur_router_name
                (cur_key, cur_br) = fallback
                cur_as, cur_router_name = cur_key
                if verbose:
                    print(f"[PAIR {pair_idx//2}] Intra-AS router switch before exit: {old_router} -> {cur_router_name} to use exit iface {exit_iface_id}")
            else:
                available = ", ".join(sorted(cur_br.interfaces.keys()))
                raise ValueError(f"At pair starting index {pair_idx}: expected exit interface {exit_iface_id} on router {cur_router_name} in {cur_as}, but it is missing. Available on current router: {available}")
        cur_iface_id = exit_iface_id
        if verbose:
            print(f"[PAIR {pair_idx//2}] Exiting AS {cur_as} on router {cur_router_name} via iface {cur_iface_id}")

        # Cross the exit interface to peer
        cur_iface = cur_br.interfaces[cur_iface_id]
        local = cur_iface.local()
        remote = cur_iface.remote()
        if not local or not remote:
            raise ValueError(f"Interface {cur_iface_id} on {cur_router_name} missing underlay info for crossing at pair {pair_idx//2}")
        peer = underlay_map.get((remote, local))
        if not peer:
            raise ValueError(f"Cannot cross interface {cur_iface_id} on {cur_router_name}; peer not found for underlay ({local}->{remote}) at pair {pair_idx//2}")
        (peer_key, peer_incoming_iface) = peer
        peer_as, peer_router_name = peer_key
        peer_br = routers.get(peer_key)
        if not peer_br:
            raise ValueError(f"Internal: missing border router object for {peer_key}")
        if verbose:
            print(f"[PAIR {pair_idx//2}] Crossed into AS {peer_as}, router {peer_router_name}, incoming iface {peer_incoming_iface}")

        # Now resolve entry interface in the new AS; allow intra-AS router switch if needed.
        post_cross_as = peer_as
        post_cross_router = peer_router_name
        post_cross_br = peer_br
        entry_iface_final = None

        if entry_iface_expected == peer_incoming_iface:
            entry_iface_final = peer_incoming_iface
        elif entry_iface_expected in post_cross_br.interfaces:
            entry_iface_final = entry_iface_expected
        else:
            # try intra-AS router switch to satisfy expected entry interface
            fallback = find_any_router_in_as_with_iface(routers, post_cross_as, entry_iface_expected, exclude_router=post_cross_router)
            if fallback:
                (cur_key, cur_br) = fallback
                cur_as, cur_router_name = cur_key
                cur_iface_id = entry_iface_expected
                entry_iface_final = entry_iface_expected
                if verbose:
                    print(f"[PAIR {pair_idx//2}] Intra-AS router switch after crossing: {post_cross_router} -> {cur_router_name} to satisfy entry iface {entry_iface_expected}")
                # Build hop entry reflecting this pair (from->to is still from_as -> post_cross_as)
                hops.append({
                    "pair": pair_idx // 2,
                    "from": from_as,
                    "to": post_cross_as,
                    "router": cur_router_name,
                    "exit_iface": exit_iface_id,
                    "entry_iface": entry_iface_expected,
                })
                continue  # proceed to next pair with updated current router/state
            else:
                available = []
                if post_cross_br:
                    available = sorted(post_cross_br.interfaces.keys())
                raise ValueError(f"At pair {pair_idx//2}: after crossing expected entry interface {entry_iface_expected} on {post_cross_router} in {post_cross_as}, but it does not exist. Available: {available}")

        # Successful landing on entry interface (possibly on same router)
        # Update current position to the entry side
        cur_as = post_cross_as
        cur_router_name = post_cross_router
        cur_br = post_cross_br
        cur_iface_id = entry_iface_final

        hops.append({
            "pair": pair_idx // 2,
            "from": from_as,
            "to": post_cross_as,
            "router": cur_router_name,
            "exit_iface": exit_iface_id,
            "entry_iface": entry_iface_final,
        })
        if verbose:
            print(f"[PAIR {pair_idx//2}] Now at AS {cur_as}, router {cur_router_name}, entry iface {cur_iface_id}")

    return hops

def main():
    parser = argparse.ArgumentParser(description="Map a SCION path (exit,entry,exit,entry...) to hops")
    parser.add_argument("--topo-root", required=True, help="Root dir containing topology.json files (e.g., metrics/)")
    parser.add_argument("--start-isd-as", required=True, help="Starting ISD-AS (e.g., 1-111)")
    parser.add_argument("--path", required=True, help="Space-separated sequence of interface IDs, e.g. '2 2 5 2'")
    parser.add_argument("--start-router", help="Optional: disambiguate starting border router name")
    parser.add_argument("--verbose", action="store_true", help="Verbose trace of mapping")
    parser.add_argument("--json", action="store_true", help="Emit hops as JSON")
    args = parser.parse_args()

    routers = load_all_routers(args.topo_root)
    underlay_map = build_underlay_map(routers)
    path_seq = args.path.strip().split()

    try:
        hops = map_path_pairwise(path_seq,
                                 args.start_isd_as,
                                 routers,
                                 underlay_map,
                                 start_router=args.start_router,
                                 verbose=args.verbose)
    except Exception as e:
        print("Error during mapping:", e)
        return

    if args.json:
        print(json.dumps(hops, indent=2))
    else:
        for h in hops:
            print(json.dumps(h))

if __name__ == "__main__":
    main()
