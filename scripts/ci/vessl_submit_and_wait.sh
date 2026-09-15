#!/usr/bin/env bash
# Submit one VESSL lane at a pinned rfx ref, wait for it, surface its verdict, and
# release the GPU/CPU seat whatever happens.
#
# WHY THIS IS SHARED AND NOT INLINE. Two jobs in .github/workflows/validation.yml
# submit a VESSL lane on the weekly schedule (weekly-a6000-lane,
# crossval-solvers-lane) and the lab runs a two-seat GPU policy. The EXIT trap
# below is the part that must not drift between copies: a watcher that dies
# without terminating its run leaves a seat held until somebody notices by hand.
# tests/contracts/test_vessl_lane_submission_is_shared.py pins that no job
# re-derives this.
#
# Usage:
#   vessl_submit_and_wait.sh <lane-yaml> <ref-env-var> <label> [poll-minutes] [verdict-grep]
#
#   ref-env-var   the env: key inside the lane file holding the rfx ref to test.
#                 Its committed value must be "origin/main"; this script rewrites
#                 it to $GITHUB_SHA so the lane tests the triggering commit rather
#                 than whatever main drifted to while the job queued.
#   verdict-grep  optional ERE. Lines matching it are pulled out of the run log and
#                 written to the job summary. Needed because a lane's artifacts land
#                 on the NFS mount, which a GitHub runner cannot read, and only the
#                 log tail crosses.
#
# Requires: VESSL_ACCESS_TOKEN, VESSL_ORGANIZATION, VESSL_PROJECT, GITHUB_SHA.

set -uo pipefail

LANE_YAML="${1:?lane yaml path required}"
REF_VAR="${2:?ref env var name required}"
LABEL="${3:?label required}"
POLL_MINUTES="${4:-330}"
VERDICT_GREP="${5:-}"

: "${VESSL_ACCESS_TOKEN:?}"; : "${VESSL_ORGANIZATION:?}"; : "${VESSL_PROJECT:?}"; : "${GITHUB_SHA:?}"
[ -f "$LANE_YAML" ] || { echo "no such lane file: $LANE_YAML"; exit 1; }

vessl configure -t "$VESSL_ACCESS_TOKEN" -o "$VESSL_ORGANIZATION" -p "$VESSL_PROJECT" >/dev/null \
  || { echo "vessl configure failed"; exit 1; }

# Pin the triggering SHA. The grep-back is not decoration: if the lane file's env
# key is renamed the sed silently no-ops and the lane would run origin/main while
# the job reports it tested $GITHUB_SHA.
PINNED="/tmp/${LABEL}-lane.yaml"
sed "s|${REF_VAR}: \"origin/main\"|${REF_VAR}: \"${GITHUB_SHA}\"|" "$LANE_YAML" > "$PINNED"
grep -q "${REF_VAR}: \"${GITHUB_SHA}\"" "$PINNED" || {
  echo "could not pin the SHA into $LANE_YAML via ${REF_VAR} -- has the env key been renamed?"
  exit 1
}

OUT=$(vessl run create -f "$PINNED" 2>&1 || true)
printf '%s\n' "$OUT" | grep -vi token || true
RID=$(printf '%s\n' "$OUT" | grep -oE 'runs/[^/]+/[0-9]+' | grep -oE '[0-9]+$' | head -1 || true)
[ -n "$RID" ] || {
  echo "could not parse a run id from the create output (a YAML verification failure prints nothing at exit 0)"
  exit 1
}
echo "VESSL run $RID  ($LABEL, ${REF_VAR}=${GITHUB_SHA})"

trap 'if [ "${DONE:-0}" != 1 ]; then echo "terminating VESSL run $RID"; vessl run terminate "$RID" </dev/null >/dev/null 2>&1 || true; fi' EXIT

ST=""; MISSES=0
for _ in $(seq 1 "$POLL_MINUTES"); do
  ST=$(vessl run read "$RID" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^\s*Status\s' | head -1 | awk '{print $2}' || true)
  if [ -z "$ST" ]; then
    MISSES=$((MISSES+1)); echo "$(date -u +%H:%M) status unreadable ($MISSES in a row)"
    [ "$MISSES" -ge 15 ] && { echo "status unreadable 15 times in a row; giving up"; exit 1; }
  else
    MISSES=0; echo "$(date -u +%H:%M) status=$ST"
    case "$ST" in completed|failed|terminated|stopped) break;; esac
  fi
  sleep 60
done

LOG=$(vessl run logs "$RID" --tail 400 2>/dev/null | grep -vi token || true)
printf '%s\n' "$LOG" | tail -60

if [ -n "$VERDICT_GREP" ] && [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### ${LABEL} — VESSL run ${RID}"
    echo ""
    MATCHED=$(printf '%s\n' "$LOG" | grep -E "$VERDICT_GREP" || true)
    if [ -n "$MATCHED" ]; then
      echo '```'
      printf '%s\n' "$MATCHED"
      echo '```'
    else
      # Say so rather than print an empty block: a missing verdict means the lane
      # died before it got there, and that reads identically to "no problems".
      echo "**No line matching \`${VERDICT_GREP}\` in the last 400 log lines.**"
      echo "The lane did not reach its verdict; read the full run log."
    fi
  } >> "$GITHUB_STEP_SUMMARY"
fi

case "$ST" in
  completed|failed|terminated|stopped) DONE=1 ;;
  *) echo "$LABEL still $ST after the poll budget; terminating"; exit 1 ;;
esac
[ "$ST" = completed ] || { echo "$LABEL ended with status $ST"; exit 1; }
echo "$LABEL completed (VESSL run $RID)"
