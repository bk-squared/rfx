#!/bin/sh
# One `git bisect run` step for #801: does the isolated-patch arm still grow at THIS commit?
#
# THE PREDICATE IS INVERTED ON PURPOSE.  git bisect looks for the first commit where a
# property APPEARS, walking from an ancestor that lacks it.  Here the property that appears
# is "the arm settles": fa3a99bd grows (0.00 dB) and fc7f7202 settles (-43.37 dB), so the
# job runs `git bisect start fc7f7202 fa3a99bd` -- bad = settles = fc7f7202, good = grows =
# fa3a99bd -- and git's "first bad commit" is the first commit where the growth STOPPED.
# That wording is what the report must use; "bad" here does not mean broken.
#
#   exit 0   the arm GROWS  (settling_db > -40)   -> git reads "good"  (old behaviour)
#   exit 1   the arm SETTLES (settling_db <= -40) -> git reads "bad"   (new behaviour)
#   exit 125 the arm could not be measured at this commit -> git skips it
#
# 125 is reserved by git for skip and must never be produced by a real measurement, so the
# driver's own exit code is never forwarded: it is translated here.
#
# Env: OUT (artifact dir), DRIVER (the driver to copy in), ARM_ARGS (extra driver args).
set -eu

OUT=${OUT:?OUT must be set}
DRIVER=${DRIVER:?DRIVER must be set}
ARM_ARGS=${ARM_ARGS:-}

SHA=$(git rev-parse HEAD)
SHORT=$(echo "$SHA" | cut -c1-12)
TAG="bisect_$SHORT"
echo "===== bisect step at $SHA ====="
git --no-pager log -1 --format='%h %ad %s' --date=short "$SHA" || true

# The driver is not in these trees (it is new on the diagnosis branch), so it is copied in
# as an untracked file.  git checkout between bisect steps leaves untracked files alone, so
# this is idempotent; it is repeated anyway because a `git bisect skip` can move the tree.
mkdir -p scripts/diagnostics
cp "$DRIVER" scripts/diagnostics/patch_pad_cpml_ringdown.py

set +e
PYTHONPATH=$(pwd) timeout 3600 python -u scripts/diagnostics/patch_pad_cpml_ringdown.py \
  --n 4 --pad 10 --periods 150 --cpml 8 --no-raster \
  --tag "$TAG" --out-dir "$OUT" --rfx-tree-sha "$SHA" --expect-rfx-root "$(pwd)" \
  $ARM_ARGS > "$OUT/$TAG.log" 2>&1
rc=$?
set -e
echo "$rc" > "$OUT/$TAG.driver_rc"

if [ "$rc" -ne 0 ]; then
  echo "  driver exited $rc at $SHORT -- SKIP (log tail below)"
  tail -n 15 "$OUT/$TAG.log" || true
  echo "skip" > "$OUT/$TAG.verdict"
  exit 125
fi

# Read the verdict from the record the driver persisted, not from stdout.
VERDICT=$(python -c "import json,sys; r=json.load(open(sys.argv[1])); s=r['settling_db']; print('grows' if s > -40.0 else 'settles')" "$OUT/$TAG.json")
SETTLE=$(python -c "import json,sys; print('%.2f' % json.load(open(sys.argv[1]))['settling_db'])" "$OUT/$TAG.json")
echo "$VERDICT" > "$OUT/$TAG.verdict"
echo "  $SHORT  settling_db=$SETTLE dB  -> $VERDICT"
printf '%s\t%s\t%s\n' "$SHA" "$SETTLE" "$VERDICT" >> "$OUT/bisect_trace.tsv"

if [ "$VERDICT" = "grows" ]; then
  exit 0
fi
exit 1
