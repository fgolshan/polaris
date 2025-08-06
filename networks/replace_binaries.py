#!/usr/bin/env python3
import argparse
import subprocess
import sys
import tempfile
import shutil
import os
from pathlib import Path

def run(cmd, capture=False, check=False):
    try:
        if capture:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            res = subprocess.run(cmd)
        if check and res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, cmd, output=getattr(res, "stdout", ""), stderr=getattr(res, "stderr", ""))
        return res
    except Exception as e:
        print(f"ERROR running {' '.join(cmd)}: {e}", file=sys.stderr)
        return None

def list_containers():
    res = run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture=True)
    if not res:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]

def filter_cs(containers):
    return [c for c in containers if "cs_" in c]

def filter_br(containers):
    return [c for c in containers if "br_" in c]

def copy_binaries(cs_containers, br_containers, bin_dir, bin_apps_dir):
    for ctr in cs_containers:
        print(f"\n→ {ctr} (control-service)")
        run(["docker", "cp", str(Path(bin_dir) / "scion"), f"{ctr}:/bin/scion/scion"])
        run(["docker", "cp", str(Path(bin_apps_dir) / "scion-bwtestclient"), f"{ctr}:/bin/scion/scion-bwtestclient"])
        run(["docker", "cp", str(Path(bin_apps_dir) / "scion-bwtestserver"), f"{ctr}:/bin/scion/scion-bwtestserver"])
    for ctr in br_containers:
        print(f"\n→ {ctr} (border-router)")
        run(["docker", "cp", str(Path(bin_dir) / "scion"), f"{ctr}:/bin/scion/scion"])
        run(["docker", "cp", str(Path(bin_dir) / "router"), f"{ctr}:/bin/scion/router"])

def edit_toml_in_container(ctr, toml_path, log_level, prometheus_addr):
    print(f"Editing config in {ctr} ({toml_path})")
    # Copy out
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "config.toml"
        cp_out = run(["docker", "cp", f"{ctr}:{toml_path}", str(local)], capture=True)
        if not cp_out or not local.exists():
            print(f"warn: failed to fetch {toml_path} from {ctr}", file=sys.stderr)
            return False
        # Read and edit
        lines = local.read_text().splitlines(keepends=True)
        modified = False

        # Log level
        if log_level is not None:
            for i, l in enumerate(lines):
                if l.strip().startswith("level") and "=" in l:
                    lines[i] = f'level = "{log_level}"\n'
                    modified = True
                    break
            if not modified:
                # no existing line? append under [log.console]
                for i, l in enumerate(lines):
                    if l.strip() == "[log.console]":
                        # insert next line
                        lines.insert(i+1, f'level = "{log_level}"\n')
                        modified = True
                        break

        # Prometheus
        if prometheus_addr is not None:
            has_metrics_section = False
            prometheus_line_idx = None
            metrics_section_idx = None
            # locate
            for i, l in enumerate(lines):
                if l.strip() == "[metrics]":
                    has_metrics_section = True
                    metrics_section_idx = i
                if "prometheus" in l:
                    prometheus_line_idx = i
            if not has_metrics_section:
                # append both lines
                lines.append("\n[metrics]\n")
                lines.append(f'prometheus = "{prometheus_addr}"\n')
                modified = True
            else:
                if prometheus_line_idx is not None:
                    # replace existing
                    lines[prometheus_line_idx] = f'prometheus = "{prometheus_addr}"\n'
                    modified = True
                else:
                    # insert after [metrics]
                    insert_at = metrics_section_idx + 1
                    lines.insert(insert_at, f'prometheus = "{prometheus_addr}"\n')
                    modified = True

        if modified:
            local.write_text("".join(lines))
            # copy back
            cp_in = run(["docker", "cp", str(local), f"{ctr}:{toml_path}"], capture=True)
            if not cp_in:
                print(f"warn: failed to copy updated toml back to {ctr}", file=sys.stderr)
                return False
        else:
            print("No modifications needed for TOML.")
    return True

def restart_router(ctr, toml_path):
    print(f"Restarting router in {ctr}")
    # find existing pid
    pid = None
    res = run(["docker", "exec", ctr, "pidof", "router"], capture=True)
    if res and res.stdout.strip():
        pid = res.stdout.strip()
    else:
        # fallback to ps+grep
        res = run(["docker", "exec", ctr, "sh", "-c", "ps -eo pid,cmd | grep '[r]outer --config' | awk '{print $1}'"], capture=True)
        if res and res.stdout.strip():
            pid = res.stdout.strip()
    if pid:
        print(f"Killing old router PID {pid} in {ctr}")
        run(["docker", "exec", ctr, "kill", "-9", pid])
    else:
        print("No existing router found (starting fresh).")
    # start new router
    start_cmd = f"/bin/scion/router --config {toml_path} >> /var/log/scion-border-router.log 2>&1 &"
    run(["docker", "exec", ctr, "sh", "-c", start_cmd])
    print(f"Router start command issued in {ctr}")

def main():
    parser = argparse.ArgumentParser(description="Replace SCION binaries and optionally enable Prometheus.")
    parser.add_argument("--debug", action="store_true", help="Set log level to debug")
    parser.add_argument("--error", action="store_true", help="Set log level to error")
    parser.add_argument("--prometheus", type=str, default="", help="Enable prometheus on this addr (e.g. :9000)")
    parser.add_argument("--bin-dir", type=Path, default=Path("../scionproto/bin"), help="Directory with scion/router binaries")
    parser.add_argument("--apps-dir", type=Path, default=Path("../scion-apps/bin"), help="Directory with scion-bwtest binaries")
    args = parser.parse_args()

    log_level = None
    if args.debug:
        log_level = "debug"
    elif args.error:
        log_level = "error"

    # verify binaries exist
    required_bins = ["scion", "router"]
    for b in required_bins:
        if not (args.bin_dir / b).exists():
            print(f"Missing binary {b} in {args.bin_dir}", file=sys.stderr)
            sys.exit(1)
    for b in ["scion-bwtestclient", "scion-bwtestserver"]:
        if not (args.apps_dir / b).exists():
            print(f"Missing binary {b} in {args.apps_dir}", file=sys.stderr)
            sys.exit(1)

    containers = list_containers()
    cs_containers = filter_cs(containers)
    br_containers = filter_br(containers)

    copy_binaries(cs_containers, br_containers, args.bin_dir, args.apps_dir)

    for ctr in br_containers:
        # derive config name similar to original: cut after first '-'
        # e.g., as113brd-br_113_114_2-10.113.0.251 -> br_113_114_2
        parts = ctr.split('-')
        if len(parts) < 2:
            print(f"unexpected border-router name format: {ctr}", file=sys.stderr)
            continue
        config = parts[1]
        toml = f"/etc/scion/_{config}_.toml"

        success = edit_toml_in_container(ctr, toml, log_level, args.prometheus if args.prometheus else None)
        if not success:
            print(f"warning: editing toml failed for {ctr}", file=sys.stderr)
        restart_router(ctr, toml)

    print("\n✅ Done updating binaries and restarting routers.")

if __name__ == "__main__":
    main()
