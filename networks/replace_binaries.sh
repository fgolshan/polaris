#!/usr/bin/env bash
set -euo pipefail

# Usage: $0 [--debug|--error]

# Parse optional log-level flag
LOG_LEVEL=""
for arg in "$@"; do
  case "$arg" in
    --debug) LOG_LEVEL="debug" ;;  
    --error) LOG_LEVEL="error" ;;  
    *) echo "Unknown option: $arg"; exit 1 ;;  
  esac
done

BIN_DIR="$(pwd)/../scionproto/bin"
BIN_DIR_APPS="$(pwd)/../scion-apps/bin"

# Find containers by role
BR_CONTAINERS=( $(docker ps -a --format "{{.Names}}" | grep 'br_') )
CS_CONTAINERS=( $(docker ps -a --format "{{.Names}}" | grep 'cs_') )

# Ensure we have the binaries locally
for bin in scion control dispatcher daemon router; do
  if [ ! -x "${BIN_DIR}/${bin}" ]; then
    echo "Error: Missing binary ${BIN_DIR}/${bin}" >&2
    exit 1
  fi
done
for bin in scion-bwtestclient scion-bwtestserver; do
  if [ ! -x "${BIN_DIR_APPS}/${bin}" ]; then
    echo "Error: Missing binary ${BIN_DIR_APPS}/${bin}" >&2
    exit 1
  fi
done

# Update CONTROL-SERVICE containers
for ctr in "${CS_CONTAINERS[@]}"; do
  echo; echo "→ ${ctr} (control-service)"
  docker cp "${BIN_DIR}/scion" "${ctr}:/bin/scion/scion"
  docker cp "${BIN_DIR_APPS}/scion-bwtestclient" "${ctr}:/bin/scion/scion-bwtestclient"
  docker cp "${BIN_DIR_APPS}/scion-bwtestserver" "${ctr}:/bin/scion/scion-bwtestserver"
done

# Update BORDER-ROUTER containers
for ctr in "${BR_CONTAINERS[@]}"; do
  echo; echo "→ ${ctr} (border-router)"
  docker cp "${BIN_DIR}/scion"  "${ctr}:/bin/scion/scion"
  docker cp "${BIN_DIR}/router" "${ctr}:/bin/scion/router"

  config=$(echo "${ctr}" | cut -d'-' -f2)
  toml="/etc/scion/_${config}_.toml"

  # Override log level if requested
  if [ -n "${LOG_LEVEL}" ]; then
    echo "Updating log level to ${LOG_LEVEL} in ${toml}"
    docker exec "${ctr}" sed -i 's/^\s*level\s*=.*/level = "'"${LOG_LEVEL}"'"/' "${toml}"
  fi

  # Determine router PID using pidof or ps
  echo "Finding router PID in ${ctr}"
  pid=$(docker exec "${ctr}" pidof router || echo "")
  if [ -z "${pid}" ]; then
    # fallback to ps
    pid=$(docker exec "${ctr}" ps -eo pid,cmd | grep '[r]outer --config' | awk '{print $1}')
  fi

  if [ -n "${pid}" ]; then
    echo "Killing router process (${pid}) in ${ctr}"
    if docker exec "${ctr}" kill -9 ${pid}; then
      echo "Successfully killed router"
    else
      echo "Error: failed to kill router (${pid}) in ${ctr}" >&2
      exit 1
    fi
  else
    echo "Warning: no router process found in ${ctr}" >&2
  fi

  # Give socket time to free
  # sleep 1

  # Start new router
  echo "Starting router in ${ctr}"
  if ! docker exec "${ctr}" sh -c "/bin/scion/router --config ${toml} >> /var/log/scion-border-router.log 2>&1 &"; then
    echo "Error: failed to start router in ${ctr}" >&2
    exit 1
  fi
done


echo; echo "✅  Binaries updated and routers restarted."