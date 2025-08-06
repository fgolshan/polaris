#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import threading
import time
import re
from pathlib import Path
from os import path
import shutil

# default starting port for auto-assignment
DEFAULT_PORT_START = 30100
# slack time (seconds) after test end before cleanup
SLACK = 6.0

class PortPool:
    def __init__(self, start=DEFAULT_PORT_START):
        self._next = start
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            p = self._next
            self._next += 2
            return p

def parse_duration(s):
    """Parse a duration like '10s', '2m', '150' (seconds) into float seconds."""
    m = re.match(r'^(\d+(?:\.\d+)?)([sm]?)$', s.strip())
    if not m:
        raise ValueError(f"invalid duration: {s}")
    val, unit = m.groups()
    sec = float(val)
    if unit == 'm':
        sec *= 60
    return sec

def find_container(host_pattern):
    as_id, node = host_pattern.split('/', 1)
    out = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}"]
    ).decode().splitlines()
    for name in out:
        if as_id in name and node in name:
            return name
    raise RuntimeError(f"no container matching '{host_pattern}'")


def get_scion_address(container):
    """
    Inside the container, run 'scion address' and return the first line,
    e.g. "1-113,10.113.0.71".
    """
    out = subprocess.check_output(
        ["docker", "exec", container, "scion", "address"]
    ).decode().splitlines()
    if not out:
        raise RuntimeError(f"'scion address' yielded no output in {container}")
    return out[0].strip()

def run_session(spec, out_dir, port_pool):
    server_proc = None
    client_proc = None
    try:
        start = parse_duration(spec["start"])
        end   = parse_duration(spec["end"])
        if end <= start:
            raise ValueError(f"end {spec['end']} ≤ start {spec['start']}")
        duration = end - start

        client_host = spec["client"]
        server_spec = spec["server"]
        if ":" in server_spec:
            server_host, port_str = server_spec.split(":", 1)
            port = int(port_str)
        else:
            server_host = server_spec
            port = port_pool.get()

        transport = spec.get("transport", "udp")
        selector  = spec.get("selector", "default")

        # optional numeric fields; None -> "?"
        pkt_size = spec.get("pkt_size")
        num_pkts = spec.get("num_pkts")
        rate     = spec.get("rate")

        # wait until start
        time.sleep(start)

        # find containers
        client_ctr = find_container(client_host)
        server_ctr = find_container(server_host)

        # get SCION address inside container
        server_addr = get_scion_address(server_ctr)
        client_addr = get_scion_address(client_ctr)

        # build server command
        server_cmd = [
            "scion-bwtestserver",
            f"--listen=:{port}",
            "--transport=" + transport,
        ]
        if selector != "default":
            # server_cmd.append("--selector=" + selector)
            pass  # Polaris selector is currently not supported by server

        # build client command parameters
        cs = []
        # cs: seconds, pktSize, numPkts, rate
        cs.append(str(int(duration)))
        cs.append(str(pkt_size) if pkt_size else "?")
        cs.append(str(num_pkts)  if num_pkts  else "?")
        cs.append(rate if rate    else "?")
        cs_arg = ",".join(cs)

        client_cmd = [
            "scion-bwtestclient",
            "-s", f"{server_addr}:{port}",
            "-cs", cs_arg,
            "--transport=" + transport,
        ]
        if selector != "default":
            client_cmd.append("--selector=" + selector)

        # prepare output files
        sid = (
            f"{client_host.replace('/', '_')}"
            f"_to_{server_host.replace('/', '_')}"
            f"_port{port}"
            f"_{int(start)}s"
            f"_{int(duration)}s"
        )
        server_log = open(Path(out_dir) / f"{sid}_server.log", "w")
        client_log = open(Path(out_dir) / f"{sid}_client.log", "w")

        # launch server
        print(f"[{start:6.1f}s] starting server in {server_ctr} on port {port}")
        server_proc = subprocess.Popen(
            ["docker", "exec", server_ctr] + server_cmd,
            stdout=server_log, stderr=server_log,
        )
        # give server a moment
        time.sleep(1.0)

        # launch client
        print(f"[{start:6.1f}s] starting client in {client_ctr}: {' '.join(client_cmd)}")
        client_proc = subprocess.Popen(
            ["docker", "exec", client_ctr] + client_cmd,
            stdout=client_log, stderr=client_log,
        )

        # wait until end + slack
        time.sleep(duration + SLACK)
    
    except Exception as e:
         print(f"[ERROR] session {spec['client']}→{spec['server']} failed: {e}")

    finally:
        # cleanup—always runs, even on crash
        print(f"[{start+duration:6.1f}s] stopping session {sid}")
        for p in (client_proc, server_proc):
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except Exception:
                    pass
        try:
            subprocess.run(
                ["docker", "exec", server_ctr,
                 "pkill", "-SIGTERM", "-f", "scion-bwtestserver"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                 )
        except subprocess.CalledProcessError:
            pass

        # close logs
        if 'client_log' in locals():
            client_log.close()
        if 'server_log' in locals():
            server_log.close()

def main():
    parser = argparse.ArgumentParser(
        description="Automate scion-bwtest sessions in seed-labs Docker emulation"
    )
    parser.add_argument("spec_file",
        help="JSON file with an array of session specs")
    parser.add_argument("out_dir",
        help="directory to store logs (created if missing)")
    args = parser.parse_args()

    with open(args.spec_file) as f:
        specs = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    port_pool = PortPool()
    # sort by start time (seconds)
    specs.sort(key=lambda s: parse_duration(s["start"]))

    threads = []
    for spec in specs:
        t = threading.Thread(target=run_session,
                             args=(spec, args.out_dir, port_pool),
                             daemon=True)
        t.start()
        threads.append(t)

    # wait for all sessions to finish
    for t in threads:
        t.join()
    
    # copy the spec file into the results folder for record
    spec_basename = path.basename(args.spec_file)
    dst = path.join(args.out_dir, spec_basename)
    shutil.copy(args.spec_file, dst)
    print(f"Copied spec file to {dst}")

if __name__ == "__main__":
    main()
