#!/usr/bin/env bash
# Compute the changed-path list for this event and write `code_changed` to
# $GITHUB_OUTPUT. The decision itself lives in scripts/ci/changed_paths.py,
# which is unit-tested; this file is only the diff and the fail-open policy.
#
# Every input arrives through the environment. A `${{ }}` expression inside a
# `run:` block is a script injection when the value is attacker-controlled, and
# tests/contracts/test_ci_workflows_contract.py forbids the pattern outright
# rather than case by case.
#
# Inputs (all optional; a missing one falls open to "run everything"):
#   EVENT_NAME   github.event_name
#   BASE_SHA     github.event.pull_request.base.sha
#   HEAD_SHA     github.event.pull_request.head.sha
#   PUSH_BEFORE  github.event.before
#   PUSH_AFTER   github.sha
#   GITHUB_OUTPUT  where to append `code_changed=`; stdout only when unset.
#
# Run it locally against any two refs:
#   EVENT_NAME=pull_request BASE_SHA=origin/main HEAD_SHA=HEAD \
#     bash scripts/ci/classify_changes.sh

set -uo pipefail

EVENT_NAME="${EVENT_NAME:-}"
BASE_SHA="${BASE_SHA:-}"
HEAD_SHA="${HEAD_SHA:-}"
PUSH_BEFORE="${PUSH_BEFORE:-}"
PUSH_AFTER="${PUSH_AFTER:-}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
classifier="$here/changed_paths.py"

# The all-zero sha is what GitHub sends for the first push to a new branch.
ZERO="0000000000000000000000000000000000000000"

emit() {
  # $1 = true|false, $2 = the reason, printed into the job log either way.
  echo "code_changed=$1 — $2"
  if [ -n "$GITHUB_OUTPUT" ]; then
    echo "code_changed=$1" >> "$GITHUB_OUTPUT"
  fi
  exit 0
}

if [ "$EVENT_NAME" = "push" ]; then
  # A push to main is never skipped. Every merge is exercised in full: main is
  # what the release tags, the GPU lane and the weekly suite are cut from, and a
  # squash merge can carry content no pull-request diff showed.
  if [ -n "$PUSH_BEFORE" ] && [ "$PUSH_BEFORE" != "$ZERO" ] && [ -n "$PUSH_AFTER" ]; then
    echo "push $PUSH_BEFORE..$PUSH_AFTER changed:"
    git -c core.quotePath=false diff --name-only "$PUSH_BEFORE" "$PUSH_AFTER" || true
  fi
  emit true "push to main always runs the full lane"
fi

if [ -z "$BASE_SHA" ] || [ -z "$HEAD_SHA" ]; then
  emit true "no base/head sha for a '$EVENT_NAME' event — running the full lane"
fi

changed_file="$(mktemp)"
trap 'rm -f "$changed_file"' EXIT

if ! git -c core.quotePath=false diff --name-only "$BASE_SHA...$HEAD_SHA" > "$changed_file" 2>&1; then
  echo "git diff $BASE_SHA...$HEAD_SHA failed:"
  cat "$changed_file"
  emit true "could not compute the diff — running the full lane"
fi

echo "changed paths ($BASE_SHA...$HEAD_SHA):"
sed 's/^/  /' "$changed_file"

verdict="$(python3 "$classifier" --stdin --explain < "$changed_file")"
status=$?
if [ "$status" -ne 0 ] || { [ "$verdict" != "true" ] && [ "$verdict" != "false" ]; }; then
  emit true "the classifier did not answer cleanly (exit $status, said '$verdict')"
fi

if [ "$verdict" = "true" ]; then
  emit true "the diff touches code the test lanes can observe"
fi
emit false "the diff touches no file under rfx/, tests/, validation/, examples/ or the build/lane configuration"
