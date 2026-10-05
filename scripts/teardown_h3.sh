#!/bin/bash
# Tear down the H3 lane from inside its own container. A host-side kill of the lane's processes is
# silently permission-denied, because the lane runs as root:
#
#     docker exec <container> bash /opt/h3/scripts/teardown_h3.sh
#
# The pattern keys on this lane's own processes: the argv of `vllm serve` carries the served model
# directory, which always begins with `MiniMax-H3` (whichever partition), plus the launcher and
# worker names. The brackets keep the pattern from matching this script's own command line.
PAT='[M]iniMax-H3-|[v]LLM-Omni::Diff|[s]erve_arwire|[h]3_boot_arm'

list() { ps -eo pid,stat,etime,args | awk '$2 !~ /Z/' | grep -E "$PAT" | grep -v ' grep ' | cut -c1-130; }

echo "=== teardown (inside the lane container) $(date +%H:%M:%S) ==="
echo "--- will match:"; list
for sig in TERM KILL; do
  pids=$(ps -eo pid,stat,args | awk '$2 !~ /Z/' | grep -E "$PAT" | grep -v ' grep ' | awk '{print $1}')
  if [ -n "$pids" ]; then
    echo "--- kill -$sig: $pids"
    kill -$sig $pids 2>&1 | sed 's/^/    /'
  else
    echo "--- kill -$sig: nothing left"
  fi
  sleep 4
done
echo "--- surviving:"; list
echo "--- zombies (informational, filter :: Z): $(ps -eo stat,args | awk '$1 ~ /Z/' | grep -cE '[v]LLM-Omni::Diff')"
if ss -ltn 2>/dev/null | grep -q ':8000 '; then
  echo "--- WARNING: something still LISTENs on 8000 inside this namespace"
  ss -ltnp 2>/dev/null | grep ':8000 '
else
  echo "--- port 8000 free"
fi
