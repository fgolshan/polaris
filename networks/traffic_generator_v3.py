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
SLACK = 8.0
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
    as_id = "as" + as_id
    out = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}"]
    ).decode().splitlines()
    for name in out:
        if as_id in name and node in name:
            return name
    raise RuntimeError(f"no container matching '{host_pattern}'")


class TCExecutor:
    """
    Execute traffic control (tc) events at specified times during an experiment.

    A tc_spec JSON file may define a list of events, each containing at minimum
    the following fields:

    - ``time``: A relative time offset (e.g. "115s" or "2m") at which the event
      should be executed. Durations without a suffix are interpreted as seconds.
    - ``host``: A SCION host identifier in the form ``AS/br_name`` (same format
      used in the session specs) used to locate the corresponding Docker
      container via :func:`find_container`.
    - ``event``: The type of event to execute. Currently only ``change_bw`` is
      supported.

    For ``change_bw`` events the following additional fields are required:

    - ``interface``: The name of the network interface inside the container on
      which to apply the bandwidth change.
    - ``bw``: The new bandwidth rate (e.g. "1Mbit"). Only the rate is
      modified; existing burst and limit values are preserved.

    Example tc_spec::

        {
          "events": [
            {
              "time": "30s",
              "host": "111/br_111_112_2",
              "event": "change_bw",
              "interface": "EKHVVsQBEr",
              "bw": "1Mbit"
            }
          ]
        }
    """

    def __init__(self, tc_spec):
        """Initialize the executor with a tc specification dictionary."""
        self._stop_event = threading.Event()
        self._thread = None
        # Record original qdisc parameters (rate, burst, limit) for interfaces that get modified
        # Keys are tuples of (container, interface), values are (rate, burst, limit)
        self._original_qdisc = {}
        # Flatten events list and ensure correct structure
        events = tc_spec.get("events", []) if tc_spec else []
        # Normalize and sort events by time
        def _parse_time(e):
            try:
                return parse_duration(str(e.get("time", "0")))
            except Exception:
                return 0.0
        self.events = sorted(events, key=_parse_time)

    def start(self):
        """Start executing tc events in a background thread."""
        if not self.events:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self):
        """Wait for the executor thread to finish executing all events."""
        if self._thread:
            self._thread.join()
        # After the thread finishes, restore original network conditions if any were saved
        self._restore_original_qdisc()

    def stop(self):
        """Signal the executor to stop; remaining events will not be executed."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        last_time = 0.0
        for ev in self.events:
            if self._stop_event.is_set():
                break
            try:
                t = parse_duration(str(ev.get("time", "0")))
            except Exception:
                t = 0.0
            delay = t - last_time
            if delay > 0:
                # Wait for the next event or until stopped
                if self._stop_event.wait(delay):
                    break
            # Dispatch the event
            evt_type = ev.get("event")
            if evt_type == "change_bw":
                self._handle_change_bw(ev)
            else:
                print(f"[tc] unknown event type '{evt_type}' in {ev}")
            last_time = t
        # All events processed or stop requested

    def _handle_change_bw(self, ev):
        host = ev.get("host")
        interface = ev.get("interface")
        bw = ev.get("bw")
        if not host or not interface or not bw:
            print(f"[tc] missing fields in change_bw event: {ev}")
            return
        try:
            container = find_container(host)
        except Exception as e:
            print(f"[tc] could not find container for host {host}: {e}")
            return
        # Retrieve existing tbf qdisc parameters to preserve burst and limit
        try:
            show_cmd = ["docker", "exec", container, "tc", "qdisc", "show", "dev", interface]
            out = subprocess.check_output(show_cmd, stderr=subprocess.STDOUT).decode()
        except subprocess.CalledProcessError as e:
            print(f"[tc] failed to inspect qdisc on {container}/{interface}: {e.output.decode().strip()}")
            return
        # Parse burst and limit from tbf qdisc line
        rate = None
        burst = None
        limit = None
        # Look for a tbf root line containing rate, burst and limit
        for line in out.splitlines():
            if 'tbf' in line and 'root' in line and 'rate' in line and 'burst' in line and 'limit' in line:
                m = re.search(r'rate\s+(\S+)\s+burst\s+(\S+)\s+limit\s+(\S+)', line)
                if m:
                    rate, burst, limit = m.group(1), m.group(2), m.group(3)
                    break
        if burst is None or limit is None or rate is None:
            print(f"[tc] could not parse rate/burst/limit on {container}/{interface}; output was: {out.strip()}")
            return
        # Save original qdisc parameters if not already saved
        key = (container, interface)
        if key not in self._original_qdisc:
            self._original_qdisc[key] = (rate, burst, limit)
        # Issue tc qdisc change to update rate while preserving burst and limit
        cmd = [
            "docker", "exec", container, "tc", "qdisc", "change",
            "dev", interface, "root", "tbf",
            "rate", bw, "burst", burst, "limit", limit
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[tc] changed bandwidth on {container}/{interface} to {bw}")
        except subprocess.CalledProcessError as e:
            print(f"[tc] failed to change bandwidth on {container}/{interface}: {e}")

    def _restore_original_qdisc(self):
        """Restore any qdisc parameters that were modified during the experiment."""
        for (container, interface), (rate, burst, limit) in self._original_qdisc.items():
            # Compose command to restore original rate using saved burst and limit
            cmd = [
                "docker", "exec", container, "tc", "qdisc", "change",
                "dev", interface, "root", "tbf",
                "rate", rate, "burst", burst, "limit", limit
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[tc] restored bandwidth on {container}/{interface} to {rate}")
            except subprocess.CalledProcessError as e:
                print(f"[tc] failed to restore bandwidth on {container}/{interface}: {e}")

class PCAExecutor:
    """
    Execute policy-push events to the router admin endpoint (/polaris/te/rules)
    at scheduled times during the experiment.

    pcaspec JSON:
    {
      "events": [
        {
          "time": "150s",
          "host": "112/br_112_113_2",
          "event": "polaris_rule",
          "mode": "merge",             # or "replace"
          "rules": "rules": [
            {
            "id": "any-unique-string",
            "scope": { "ingress_ifid": null, "egress_ifid": null }, 
            "match": { "code": null },             
            "action": {
                "emit_pca": true,
                "pca_code": 1,                   
                "switch_to_ifid": 2,
                "budget": 5,            # or 30%
                "expires_after": "300s"
            }
          ]
        }
      ]
    }
    """

    def __init__(self, pca_spec):
        self._stop_event = threading.Event()
        self._thread = None
        events = pca_spec.get("events", []) if pca_spec else []
        def _parse_time(e):
            try:
                return parse_duration(str(e.get("time", "0")))
            except Exception:
                return 0.0
        # Sort by absolute time like TCExecutor
        self.events = sorted(events, key=_parse_time)

    def start(self):
        if not self.events:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self):
        if self._thread:
            self._thread.join()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        last_time = 0.0
        for ev in self.events:
            if self._stop_event.is_set():
                break
            try:
                t = parse_duration(str(ev.get("time", "0")))
            except Exception:
                t = 0.0
            delay = t - last_time
            if delay > 0:
                if self._stop_event.wait(delay):
                    break
            self._handle_event(ev)
            last_time = t

    def _handle_event(self, ev):
        evt_type = ev.get("event")
        if evt_type != "polaris_rule":
            print(f"[pca] unknown event type '{evt_type}' in {ev}")
            return

        host = ev.get("host")
        mode = ev.get("mode", "merge")
        rules = ev.get("rules", [])

        if not host:
            print(f"[pca] missing host in event: {ev}")
            return

        try:
            container = find_container(host)  # reuse existing logic
        except Exception as e:
            print(f"[pca] could not find container for host {host}: {e}")
            return

        payload = {"mode": mode, "rules": rules}
        url = f"http://localhost:{METRICS_PORT}/polaris/te/rules"
        cmd = [
            "docker", "exec", container, "curl", "-sS", "-X", "POST",
            "-H", "Content-Type: application/json",
            "--data", json.dumps(payload),
            url,
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            print(f"[pca] pushed rules to {container}: {out.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[pca] ERROR pushing rules to {container}: {e.output.strip()}")
        except Exception as e:
            print(f"[pca] ERROR pushing rules to {container}: {e}")

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
    # Optional tc_spec argument for traffic control events
    parser.add_argument("--tcspec", dest="tc_spec", default=None,
                        help="JSON file describing tc events to execute during the experiment")
    parser.add_argument("--pcaspec", dest="pca_spec", default=None,
                    help="JSON file describing P-CA (TE rule) events to push to routers")

    args = parser.parse_args()

    with open(args.spec_file) as f:
        specs = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    port_pool = PortPool()
    specs.sort(key=lambda s: parse_duration(s["start"]))

    metrics_collector = MetricsCollector(Path(args.out_dir))
    metrics_collector.start()

    # Load tc spec if provided and start executor
    tc_executor = None
    if args.tc_spec:
        try:
            with open(args.tc_spec) as f:
                tc_spec_data = json.load(f)
            tc_executor = TCExecutor(tc_spec_data)
            tc_executor.start()
            print(f"[tc] loaded tc spec from {args.tc_spec} with {len(tc_executor.events)} events")
        except Exception as e:
            print(f"[tc] failed to load tc spec {args.tc_spec}: {e}")

    # Load pca spec if provided and start executor
    pca_executor = None
    if args.pca_spec:
        try:
            with open(args.pca_spec) as f:
                pca_spec_data = json.load(f)
            pca_executor = PCAExecutor(pca_spec_data)
            pca_executor.start()
            print(f"[pca] loaded pca spec from {args.pca_spec} with {len(pca_executor.events)} events")
        except Exception as e:
            print(f"[pca] failed to load pca spec {args.pca_spec}: {e}")


    threads = []
    for spec in specs:
        t = threading.Thread(target=run_session, args=(spec, args.out_dir, port_pool), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Wait for tc events to finish executing before shutting down metrics
    if 'tc_executor' in locals() and tc_executor is not None:
        # join without timeout to allow all events to complete
        tc_executor.join()

    if pca_executor:
        pca_executor.join()


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

    # Also copy tc_spec file if provided and copy succeeds
    if args.tc_spec:
        try:
            tc_dst = Path(args.out_dir) / Path(args.tc_spec).name
            if not (tc_dst.exists() and os.path.samefile(args.tc_spec, tc_dst)):
                shutil.copy2(args.tc_spec, tc_dst)
                print(f"Copied tc spec file to {tc_dst}")
            else:
                print(f"[spec] {tc_dst} already in output dir; skipping copy")
        except FileNotFoundError:
            print(f"[spec] could not copy tc spec (missing?) {args.tc_spec}, skipping")

    if args.pca_spec:
        try:
            pca_dst = Path(args.out_dir) / Path(args.pca_spec).name
            if not (pca_dst.exists() and os.path.samefile(args.pca_spec, pca_dst)):
                shutil.copy2(args.pca_spec, pca_dst)
                print(f"Copied pca spec file to {pca_dst}")
            else:
                print(f"[spec] {pca_dst} already in output dir; skipping copy")
        except FileNotFoundError:
            print(f"[spec] could not copy pca spec (missing?) {args.pca_spec}, skipping")

if __name__ == "__main__":
    main()
    