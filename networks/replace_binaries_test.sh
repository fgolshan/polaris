#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="$(pwd)/../scionproto/bin"

# Find your containers by name
BR_CONTAINERS=( $(docker ps -a --format "{{.Names}}" | grep 'br_') )
CS_CONTAINERS=( $(docker ps -a --format "{{.Names}}" | grep 'cs_') )

# Ensure we have the binaries locally
for bin in scion control dispatcher daemon router; do
  if [ ! -f "$BIN_DIR/$bin" ]; then
    echo "Error: Missing binary $BIN_DIR/$bin"
    exit 1
  fi
done

# Update CONTROL‐SERVICE containers
for ctr in "${CS_CONTAINERS[@]}"; do
  echo; echo "→ $ctr (control-service)"
  # docker cp "$BIN_DIR/scion"      "$ctr:usr/bin/scion"
  # docker cp "$BIN_DIR/control"    "$ctr:/usr/bin/scion-control-service"
  # docker cp "$BIN_DIR/dispatcher" "$ctr:/usr/bin/scion-dispatcher"
  # docker cp "$BIN_DIR/daemon"     "$ctr:/usr/bin/scion-daemon"
   docker cp "$BIN_DIR/scion"      "$ctr:/bin/scion/scion"
  # docker cp "$BIN_DIR/control"    "$ctr:/bin/scion/control"
  # docker cp "$BIN_DIR/dispatcher" "$ctr:/bin/scion/dispatcher"
  # docker cp "$BIN_DIR/daemon"     "$ctr:/bin/scion/daemon"
  # docker restart "$ctr"
done

# Update BORDER-ROUTER containers
for ctr in "${BR_CONTAINERS[@]}"; do
  echo; echo "→ $ctr (border-router)"
  # docker cp "$BIN_DIR/scion"      "$ctr:/usr/bin/scion"
  docker cp "$BIN_DIR/scion"      "$ctr:/bin/scion/scion"
  # copy router binary to both possible entrypoints
  # docker cp "$BIN_DIR/router"     "$ctr:/usr/bin/scion-border-router"
  docker cp "$BIN_DIR/router"     "$ctr:/bin/scion/router"
  # docker restart "$ctr"
done

echo; echo "✅  Binaries copied. Please restart each container to load the new code."
