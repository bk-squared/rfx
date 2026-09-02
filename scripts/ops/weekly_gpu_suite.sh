#!/bin/bash
# Weekly gpu-marked test suite on remilab-c0, driven from the lab Mac.
#
#   1. shallow-clone main into a scratch dir and rsync it to the NFS checkout
#      claude-workspace/rfx/checkouts/gpu-suite-<sha>/
#   2. render one VESSL YAML per shard (scripts/ops/gpu_suite_shards.json) and
#      submit them all -- each shard gets its own RTX 4090
#   3. poll until every run is terminal; harvest each full log to
#      ~/Documents/vessl-run-logs/ and delete the run (run hygiene)
#   4. fold the shard JUnit files into one markdown report, keep it under
#      ~/Documents/rfx-ops/reports/, and post it as a comment on the standing
#      GitHub issue "Weekly GPU suite (remilab-c0)" -- creating the issue once
#
# Installed by ~/Library/LaunchAgents/com.rfx.weekly-gpu-suite.plist (Mon 03:00 KST).
# Run by hand:  bash scripts/ops/weekly_gpu_suite.sh [--shards N] [--no-post]
set -euo pipefail

REPO_URL="https://github.com/bk-squared/rfx.git"
PROJECT="byungkwan"
NFS="$HOME/mnt/remilab-fs/personal-workspaces/claude-workspace/rfx"
OPS="$HOME/Documents/rfx-ops"
LOGS="$HOME/Documents/vessl-run-logs"
HARVEST="$LOGS/harvest.sh"
ISSUE_TITLE="Weekly GPU suite (remilab-c0)"
POST=1
for a in "$@"; do case "$a" in --no-post) POST=0;; esac; done

mkdir -p "$OPS/reports" "$OPS/work"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WORK="$OPS/work/$STAMP"
mkdir -p "$WORK"
exec > >(tee -a "$WORK/orchestrator.log") 2>&1
echo "=== weekly gpu suite $STAMP ==="

# --- 1. checkout ---------------------------------------------------------
mount | grep -q "remilab-fs" || { echo "NFS not mounted (sudo mount ... ~/mnt/remilab-fs); abort"; exit 2; }
git clone -q --depth 1 --branch main "$REPO_URL" "$WORK/rfx"
SHA=$(git -C "$WORK/rfx" rev-parse --short HEAD)
SLUG="gpu-suite-$SHA"
echo "main @ $SHA -> $SLUG"
mkdir -p "$NFS/checkouts/$SLUG"
rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ "$WORK/rfx/" "$NFS/checkouts/$SLUG/"

# --- 2. render + submit --------------------------------------------------
PY=$(command -v python3)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this script's own copy of the ops tools
"$PY" "$HERE/render_gpu_suite_shards.py" --slug "$SLUG" --stamp "$STAMP" --sha "$SHA" --out "$WORK/yamls" > /dev/null
RUN_IDS=()
cd "$WORK"   # vessl CLI must not run inside a git repo
for y in "$WORK"/yamls/gpu_suite_shard*.yaml; do
  id=$(vessl run create -f "$y" --project "$PROJECT" 2>&1 | grep -oE "runs/$PROJECT/[0-9]+" | head -1 | grep -oE "[0-9]+$" || true)
  [ -n "$id" ] || { echo "submit failed for $y"; continue; }
  echo "submitted $(basename "$y") -> $id"
  RUN_IDS+=("$id")
done
[ "${#RUN_IDS[@]}" -gt 0 ] || { echo "no runs submitted; abort"; exit 3; }

# --- 3. poll, harvest, delete --------------------------------------------
declare -A DONE
for i in $(seq 1 120); do            # up to 6 h at 3-minute polls
  all=1
  for id in "${RUN_IDS[@]}"; do
    [ -n "${DONE[$id]:-}" ] && continue
    s=$(vessl run read "$id" --project "$PROJECT" 2>/dev/null | grep -E "^ *Status " | head -1 | awk '{print $2}' || true)
    if echo "$s" | grep -qiE "completed|failed|terminated|stopped|cancel"; then
      echo "$(date -u +%H:%M:%SZ) run $id $s"
      DONE[$id]="$s"
    else
      all=0
    fi
  done
  [ "$all" -eq 1 ] && break
  sleep 180
done
for id in "${RUN_IDS[@]}"; do
  [ -n "${DONE[$id]:-}" ] || { echo "run $id still not terminal after 6 h; left in place"; continue; }
  bash "$HARVEST" "$id" "gpu-suite-$STAMP" >/dev/null 2>&1 || echo "harvest failed for $id (run left in place)"
done

# --- 4. report + signal --------------------------------------------------
RUN_DIR="$NFS/runs/gpu-suite/$STAMP"
REPORT="$OPS/reports/gpu-suite-$STAMP-$SHA.md"
set +e
"$PY" "$HERE/summarize_junit.py" "$RUN_DIR" --sha "$SHA" --runs "${RUN_IDS[*]}" > "$REPORT"
SUITE_RC=$?
set -e
cat "$REPORT"
echo "report: $REPORT  (suite rc=$SUITE_RC)"

if [ "$POST" -eq 1 ]; then
  num=$(gh issue list --repo bk-squared/rfx --state open --search "\"$ISSUE_TITLE\" in:title" --json number,title --jq ".[] | select(.title==\"$ISSUE_TITLE\") | .number" | head -1)
  if [ -z "$num" ]; then
    num=$(gh issue create --repo bk-squared/rfx --title "$ISSUE_TITLE" \
      --body "Standing report thread for the weekly gpu-marked pytest suite, run on remilab-c0 in line-balanced shards (scripts/ops/gpu_suite_shards.json) from the lab Mac by scripts/ops/weekly_gpu_suite.sh. These files ran in no CI lane before 2026-09-02 (issue #741 / #717). One comment per run; a red shard is a real finding, not a flake, until shown otherwise." \
      | grep -oE "[0-9]+$")
    echo "created issue #$num"
  fi
  gh issue comment "$num" --repo bk-squared/rfx --body-file "$REPORT" >/dev/null && echo "posted to #$num"
fi

# --- cleanup ---------------------------------------------------------------
rm -rf "$NFS/checkouts/$SLUG" "$WORK/rfx"
echo "=== done (suite rc=$SUITE_RC) ==="
exit "$SUITE_RC"
