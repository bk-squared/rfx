#!/usr/bin/env bash
# Poll one VESSL run until it leaves running/pending, tailing its log.
#   bash validation/vessl/wait.sh <run-id> [max-minutes] [poll-seconds]
# Terminates the run when max-minutes is exceeded: cluster etiquette is that
# no job of mine is left running when I stop watching.
set -eu
ID=${1:?usage: wait.sh <run-id> [max-minutes] [poll-seconds]}
MAXM=${2:-130}
POLL=${3:-60}
export PATH="$HOME/.local/bin:$PATH"
END=$(( $(date +%s) + MAXM * 60 ))
while :; do
  STATE=$(vessl run list 2>/dev/null | awk -v id="$ID" '$1==id{print $4}')
  echo "[$(date -u +%H:%M:%SZ)] $ID state=${STATE:-unknown}"
  case "${STATE:-}" in
    running|pending|idle|initializing|queued|"") : ;;
    *) echo "final state: $STATE"; vessl run logs "$ID" --tail 40 2>/dev/null | tail -40; exit 0 ;;
  esac
  if [ "$(date +%s)" -ge "$END" ]; then
    echo "TIMEOUT after ${MAXM} min: terminating $ID"
    vessl run terminate "$ID" || true
    exit 2
  fi
  vessl run logs "$ID" --tail 6 2>/dev/null | tail -6 || true
  sleep "$POLL"
done
