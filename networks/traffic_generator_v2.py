#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import threading
import time
import re
from pathlib import Path
import shutil
from datetime import datetime

DEFAULT_PORT_START = 30100
SLACK = 6.0
DEFAULT_SCRAPE_INTERVAL = 1.0
METRICS_PORT = 9000

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

def list_border_routers():
    out = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}"]
    ).decode().splitlines()
    return [n for n in out if "br_" in n]

def get_scion_address(container):
    out = subprocess.check_output(
        ["docker", "exec", container, "scion", "address"]
    ).decode().splitlines()
    if not out:
        raise RuntimeError(f"'scion address' yielded no output in {container}")
    return out[0].strip()

def copy_topology(router_name: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "topology.json"
    # First try docker cp
    cp_proc = subprocess.run(
        ["docker", "cp", f"{router_name}:/etc/scion/topology.json", str(dest_path)],
        capture_output=True, text=True
    )
    if cp_proc.returncode == 0:
        return
    # Fallback: docker exec cat
    exec_proc = subprocess.run(
        ["docker", "exec", router_name, "cat", "/etc/scion/topology.json"],
        capture_output=True, text=True
    )
    if exec_proc.returncode == 0 and exec_proc.stdout:
        try:
            with open(dest_path, "w") as f:
                f.write(exec_proc.stdout)
        except Exception as e:
            print(f"[metrics] failed writing topology for {router_name}: {e}")
    else:
        print(f"[metrics] warning: could not retrieve topology.json for {router_name}")

class MetricsCollector:
    def __init__(self, base_out_dir: Path, interval=DEFAULT_SCRAPE_INTERVAL):
        self.interval = interval
        self.base = base_out_dir / "metrics"
        self._stop_event = threading.Event()
        self._thread = None
        self.routers = []

    def start(self):
        self.routers = list_border_routers()
        if not self.routers:
            print("[metrics] warning: no border routers found")
        for r in self.routers:
            (self.base / r / "raw").mkdir(parents=True, exist_ok=True)
            # copy topology.json once at start
            copy_topology(r, self.base / r)
        cfg = {
            "interval": self.interval,
            "routers": self.routers,
            "metrics_port": METRICS_PORT,
            "start_time": datetime.utcnow().isoformat() + "Z",
        }
        (self.base / "config.json").parent.mkdir(parents=True, exist_ok=True)
        with open(self.base / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[metrics] started collector (interval={self.interval}s) for routers: {self.routers}")

    def _loop(self):
        while not self._stop_event.is_set():
            ts = time.time()
            iso_ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            for r in self.routers:
                try:
                    cmd = ["docker", "exec", r, "curl", "-s", f"http://localhost:{METRICS_PORT}/metrics"]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
                    if proc.returncode != 0:
                        print(f"[metrics] scrape failed for {r}: exit {proc.returncode} stderr={proc.stderr.strip()}")
                        continue
                    body = proc.stdout
                    raw_path = self.base / r / "raw" / f"{iso_ts}.prom.txt"
                    with open(raw_path, "w") as f:
                        f.write(body)
                except Exception as e:
                    print(f"[metrics] exception scraping {r} at {iso_ts}: {e}")
            elapsed = time.time() - ts
            to_sleep = self.interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
        try:
            with open(self.base / "config.json", "r+") as f:
                cfg = json.load(f)
                cfg["stop_time"] = datetime.utcnow().isoformat() + "Z"
                f.seek(0)
                json.dump(cfg, f, indent=2)
                f.truncate()
        except Exception:
            pass
        print("[metrics] stopped collector")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

def run_session(spec, out_dir, port_pool):
    server_proc = None
    client_proc = None
    client_ctr = None
    server_ctr = None
    server_log = None
    client_log = None
    sid = None
    try:
        start = parse_duration(spec["start"])
        end = parse_duration(spec["end"])
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
        selector = spec.get("selector", "default")

        pkt_size = spec.get("pkt_size")
        num_pkts = spec.get("num_pkts")
        rate = spec.get("rate")

        time.sleep(start)

        client_ctr = find_container(client_host)
        server_ctr = find_container(server_host)
        server_addr = get_scion_address(server_ctr)
        client_addr = get_scion_address(client_ctr)

        server_cmd = [
            "scion-bwtestserver",
            f"--listen=:{port}",
            "--transport=" + transport,
        ]
        if selector != "default":
            pass  # Polaris selector support not on server

        cs = [
            str(int(duration)),
            str(pkt_size) if pkt_size else "?",
            str(num_pkts) if num_pkts else "?",
            rate if rate else "?",
        ]
        cs_arg = ",".join(cs)

        # For S->C always use 0 Mbps rate
        sc = [
            str(int(duration)),
            str(pkt_size) if pkt_size else "?",
            "?",
            "0Mbps",
        ]
        sc_arg = ",".join(sc)

        client_cmd = [
            "scion-bwtestclient",
            "-s", f"{server_addr}:{port}",
            "-cs", cs_arg,
            "-sc", sc_arg,
            "--transport=" + transport,
        ]
        if selector != "default":
            client_cmd.append("--selector=" + selector)

        sid = (
            f"{client_host.replace('/', '_')}"
            f"_to_{server_host.replace('/', '_')}"
            f"_port{port}"
            f"_{int(start)}s"
            f"_{int(duration)}s"
        )
        server_log = open(Path(out_dir) / f"{sid}_server.log", "w")
        client_log = open(Path(out_dir) / f"{sid}_client.log", "w")

        print(f"[{start:6.1f}s] starting server in {server_ctr} on port {port}")
        server_proc = subprocess.Popen(
            ["docker", "exec", server_ctr] + server_cmd,
            stdout=server_log, stderr=server_log,
        )
        time.sleep(1.0)
        print(f"[{start:6.1f}s] starting client in {client_ctr}: {' '.join(client_cmd)}")
        client_proc = subprocess.Popen(
            ["docker", "exec", client_ctr] + client_cmd,
            stdout=client_log, stderr=client_log,
        )
        time.sleep(duration + SLACK)

    except Exception as e:
        print(f"[ERROR] session {spec.get('client')}→{spec.get('server')} failed: {e}")

    finally:
        if sid is None:
            sid = f"{spec.get('client','').replace('/', '_')}_to_{spec.get('server','').replace('/', '_')}"
        print(f"[session] stopping session {sid}")
        for p in (client_proc, server_proc):
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except Exception:
                    pass
        if server_ctr:
            try:
                subprocess.run(
                    ["docker", "exec", server_ctr,
                     "pkill", "-SIGTERM", "-f", "scion-bwtestserver"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                )
            except Exception:
                pass
        if client_log:
            client_log.close()
        if server_log:
            server_log.close()

def main():
    parser = argparse.ArgumentParser(
        description="Automate scion-bwtest sessions in seed-labs Docker emulation"
    )
    parser.add_argument("spec_file", help="JSON file with an array of session specs")
    parser.add_argument("out_dir", help="directory to store logs (created if missing)")
    args = parser.parse_args()

    with open(args.spec_file) as f:
        specs = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    port_pool = PortPool()
    specs.sort(key=lambda s: parse_duration(s["start"]))

    metrics_collector = MetricsCollector(Path(args.out_dir))
    metrics_collector.start()

    threads = []
    for spec in specs:
        t = threading.Thread(target=run_session, args=(spec, args.out_dir, port_pool), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    metrics_collector.stop()

    # spec_basename = os.path.basename(args.spec_file)
    # dst = os.path.join(args.out_dir, spec_basename)
    dst = Path(args.out_dir) / Path(args.spec_file).name
    try:
    # Only copy if source and destination are not the same file
        if not (dst.exists() and os.path.samefile(args.spec_file, dst)):
            shutil.copy2(args.spec_file, dst)
            print(f"Copied spec file to {dst}")
        else:
            print(f"[spec] {dst} already in output dir; skipping copy")
    except FileNotFoundError:
        # If args.spec_file vanished or path issues—just skip copying
        print(f"[spec] could not copy spec (missing?) {args.spec_file}, skipping")

if __name__ == "__main__":
    main()
    