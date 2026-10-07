#!/usr/bin/env bash
# Every short gate a pull request must pass. Run it before every push.
# Measured 2026-09-18: tests/contracts is 5 min 16 s of it, the rest under 10 s.
#
# Two CI failures on one PR on 2026-09-18 could not be reproduced locally,
# because the commands that produced them lived inline in workflow yaml. The
# gates now live in scripts; this file is the one command that runs them in the
# order CI does, and stops at the first failure.
#
# What it does NOT run in full: the fast suite and the guard/preflight suite.
# Those are 35 minutes, and `.github/workflows/pr-tests.yml` only starts them
# when the diff touches code (`scripts/ci/changed_paths.py` decides).
#
#   scripts/ci/local.sh                 # the nine steps; pr-body reports skipped
#   scripts/ci/local.sh /tmp/body.md    # also check that PR body
#   PYTHON=.venv/bin/python scripts/ci/local.sh
#   CHANGELOG_BASE=origin/main CHANGELOG_HEAD=HEAD scripts/ci/local.sh
#
# $PYTHON is exported, so every step and every script this one calls uses it.
#
# The first eight step names are listed in docs/agent/agent-runbook.mdx.
# The final selected-tests session extends that baseline (issue #1528).

set -uo pipefail

STEP_NAMES=(ruff docs-hygiene changelog-fragment data-budget file-size-ratchet pr-body workflow-yaml contract-tests selected-tests)

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Exported, so the scripts this file calls use the same interpreter. Without it
# scripts/ci/lint.sh fell back to a bare `ruff` that is usually only in the venv.
export PYTHON="${PYTHON:-python3}"
BODY_PATH="${1:-}"

# What the changelog, data-budget and file-size gates diff against. CI uses the PR's base
# and head shas; locally the merge base with origin/main is the same question.
CHANGELOG_BASE="${CHANGELOG_BASE:-origin/main}"
CHANGELOG_HEAD="${CHANGELOG_HEAD:-HEAD}"
step_index=0

fail() {
  echo
  echo "FAILED: ${STEP_NAMES[$step_index]}"
  echo "(steps after it were not run)"
  exit 1
}

begin() {
  step_index="$1"
  echo
  echo "=== [$(($1 + 1))/${#STEP_NAMES[@]}] ${STEP_NAMES[$1]} ==="
}

begin 0
bash scripts/ci/lint.sh || fail

begin 1
bash scripts/ci/docs_hygiene.sh || fail

begin 2
# The SAME script the changelog-fragment workflow runs. `assemble.py --check`
# validates fragment names and headings, which is a different question: it never
# notices a missing fragment on an rfx/ change, nor a CHANGELOG.md edit without
# the release label, and those are the two rules CI actually enforces.
# PR_LABELS_JSON is empty locally, which is the strict reading (no release
# label), and that is the right default for a pre-push check.
# It is silent on success, so say what was asked: a green step that printed
# nothing is indistinguishable from a step that did not run.
echo "fragment required? $CHANGELOG_BASE...$CHANGELOG_HEAD"
"$PYTHON" scripts/ci/check_changelog_fragment.py \
  --base "$CHANGELOG_BASE" --head "$CHANGELOG_HEAD" || fail
echo "  ok — a fragment exists for every rfx/ change, and CHANGELOG.md is untouched"
# Fragment names and headings, which the workflow leaves to the release run.
"$PYTHON" scripts/changelog/assemble.py --check || fail

begin 3
# The SAME script the data-budget step of guards-and-preflight runs. Locally
# PR_LABELS_JSON is empty, so the exception label never applies here: a red
# step means CI will be red until the records move or the label is applied.
"$PYTHON" scripts/ci/check_data_budget.py \
  --base "$CHANGELOG_BASE" --head "$CHANGELOG_HEAD" || fail

begin 4
# The working tree, so uncommitted growth counts too.
BASE_SHA="$CHANGELOG_BASE" \
  "$PYTHON" scripts/ci/check_file_size_ratchet.py || fail

begin 5
if [ -n "$BODY_PATH" ]; then
  "$PYTHON" scripts/ci/check_pr_body.py --file "$BODY_PATH" || fail
  echo "PR body OK: $BODY_PATH"
else
  echo "skipped — pass a path to check a PR body: scripts/ci/local.sh /tmp/body.md"
fi

begin 6
"$PYTHON" - <<'PYEOF' || fail
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    sys.exit("PyYAML is not installed; `pip install pyyaml` or set PYTHON= to a venv")

workflows = sorted(Path(".github/workflows").glob("*.yml"))
if not workflows:
    sys.exit(".github/workflows/ holds no *.yml -- did the directory move?")
bad = 0
for path in workflows:
    try:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        bad += 1
        print(f"  {path}: {error}")
    else:
        print(f"  {path}: ok")
if bad:
    sys.exit(f"{bad} workflow file(s) do not parse")
PYEOF

begin 7
# The same flags the required gate in pr-tests.yml uses. Without `-o addopts=""`
# the local run silently collects two fewer tests than CI does, which is the
# local-is-weaker-than-CI gap these scripts exist to close. Costs about 27 s.
"$PYTHON" -m pytest tests/contracts -q -x \
  -o addopts="" -m "not gpu and not docs_consistency" --strict-markers || fail

begin 8
selection_dir=$(mktemp -d) || fail
trap 'rm -rf "$selection_dir"' EXIT
central_path_args=()
if [ "${RFX_GATE_CENTRAL_PATHS:-}" = "1" ]; then
  central_path_args+=(--central-paths)
fi
"$PYTHON" scripts/ci/select_gate_tests.py ${central_path_args[@]+"${central_path_args[@]}"} \
  --base "$CHANGELOG_BASE" --head "$CHANGELOG_HEAD" \
  --durations .test_durations --summary "$selection_dir/summary" \
  > "$selection_dir/files" || fail
selected_tests=()
while IFS= read -r test_file; do
  selected_tests+=("$test_file")
done < "$selection_dir/files"
selected_status=0
if [ "${#selected_tests[@]}" -eq 0 ]; then
  echo "nothing selected"
else
  "$PYTHON" -m pytest "${selected_tests[@]}" -q \
    -o addopts="" -m "not gpu and not slow and not slow_physics and not docs_consistency" --strict-markers || selected_status=$?
fi
cat "$selection_dir/summary"
if [ "$selected_status" -eq 5 ]; then
  echo "nothing to run (all selected tests deselected or no tests collected)"
elif [ "$selected_status" -ne 0 ]; then
  fail
fi

echo
echo "all ${#STEP_NAMES[@]} steps passed"
