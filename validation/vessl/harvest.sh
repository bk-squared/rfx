#!/usr/bin/env bash
# Copy the lane artifacts off the read-only NFS mount into the repo.
#   bash validation/vessl/harvest.sh
# The NFS can hang, so every read is wrapped in a 30 s alarm.
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NFS=/Users/bk-squared/nfs-remilab/personal-workspaces/claude-workspace/rfx/runs
DEST="$REPO/validation/vessl/runs"
mkdir -p "$DEST"
t() { perl -e 'alarm 30; exec @ARGV' "$@"; }
for d in $(t ls "$NFS" 2>/dev/null | grep '^fdfd-gpu-' || true); do
  mkdir -p "$DEST/$d"
  for f in $(t ls "$NFS/$d" 2>/dev/null || true); do
    t cp -p "$NFS/$d/$f" "$DEST/$d/$f" 2>/dev/null || echo "skip $d/$f"
  done
  echo "harvested $d: $(ls "$DEST/$d" | wc -l | tr -d ' ') file(s)"
done
