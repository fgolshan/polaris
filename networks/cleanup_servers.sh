# Kill any scion-bwtestclient or scion-bwtestserver in all control‐service containers
docker ps --format '{{.Names}}' | grep -E 'cs_' | while read ctr; do
  echo "Stopping bwtest in container $ctr…"
  docker exec "$ctr" pkill -f scion-bwtestclient || true
  docker exec "$ctr" pkill -f scion-bwtestserver || true
done
