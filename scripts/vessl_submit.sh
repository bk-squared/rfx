#!/bin/sh
# Submit a VESSL job and record its run id where the artifacts land.
#
# VESSL does not export VESSL_RUN_ID into the pod on this cluster (noticed
# 2026-09-04 on run 369367258205, unchanged as of 2026-09-08), so a job cannot
# write its own id and every artifact that tried carries a placeholder. The
# submitter is the only party that knows the id, so the submitter records it.
#
#   scripts/vessl_submit.sh <yaml> <runs-dir-glob-prefix> [artifact-root]
# Concurrent submissions must use disjoint artifact prefix patterns. The
# fixture campaign gives base, long, refinement and confirmation separate lanes.
#
# Example:
#   scripts/vessl_submit.sh /tmp/repin.yaml issue931-chain-repin
#
# Writes <newest matching run dir>/run_id.txt once the job has created it, and
# prints the id. Exits non-zero if the id cannot be parsed - a submission whose
# id was not captured is not a provenance-complete run.
set -eu
YAML="$1"; PREFIX="$2"
RUNS=${3:-/root/workspace/claude-workspace/rfx/runs}

# Record submission start BEFORE create: a fast job can make its output
# directory before the CLI returns. Own and remove only this marker directory.
MARKER_DIR=$(mktemp -d)
trap 'rm -f "$MARKER_DIR/marker" "$MARKER_DIR/run.yaml"; rmdir "$MARKER_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
STAMP=$MARKER_DIR/marker
: > "$STAMP"

# The CLI inspects cwd's .git at import time and cannot handle a worktree's
# .git file. Submit a copied specification from our plain temporary directory.
cp "$YAML" "$MARKER_DIR/run.yaml"
OUT=$(cd "$MARKER_DIR" && vessl run create -f run.yaml 2>&1)
echo "$OUT"
ID=$(echo "$OUT" | grep -o 'runs/byungkwan/[0-9]*' | tail -1 | grep -o '[0-9]*$')
[ -n "$ID" ] || { echo "FATAL: could not parse a run id from the create output"; exit 3; }
echo "run id: $ID"

# Image pulls and queue admission can precede the output directory by minutes.
# Keep the submitter alive through that delay so the run identity is not lost.
# Only a directory created AFTER this submission counts. Matching "newest
# without a run_id.txt" is not enough: a terminated earlier run under the same
# label leaves exactly that, and the first use of this script wrote two ids into
# stale directories from runs that had already been killed.
i=0
while [ "$i" -lt 720 ]; do
  D=$(find "$RUNS" -maxdepth 1 -type d -name "$PREFIX*" -newer "$STAMP" 2>/dev/null | sort | tail -1 || true)
  if [ -n "${D:-}" ] && [ -d "$D" ] && [ ! -f "$D/run_id.txt" ]; then
    echo "$ID" > "$D/run_id.txt"
    echo "recorded $ID in $D/run_id.txt"
    exit 0
  fi
  i=$((i + 1)); sleep 5
done
echo "WARNING: no new $PREFIX* directory appeared in 60 minutes; id $ID not recorded"
exit 4
