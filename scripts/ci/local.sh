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
#   scripts/ci/local.sh                 # the ten steps; pr-body reports skipped
#   scripts/ci/local.sh /tmp/body.md    # also check that PR body
#   PYTHON=.venv/bin/python scripts/ci/local.sh
#   CHANGELOG_BASE=origin/main CHANGELOG_HEAD=HEAD scripts/ci/local.sh
#
#   RFX_GATE_STAGE2_WORKERS=8 scripts/ci/local.sh  # positive integer; empty inherits pytest
# Worker-crash-only failures retry crashed and unreported files once with the same marks.
#
# $PYTHON is exported, so every step and every script this one calls uses it.
#
# The first nine step names are listed in docs/agent/agent-runbook.mdx.
# The final selected-tests session extends that baseline (issue #1528).

set -uo pipefail

STEP_NAMES=(ruff docs-hygiene changelog-fragment data-budget file-size-ratchet pr-body workflow-yaml api-reference contract-tests selected-tests)

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
# The inventory half of the api-reference job of public-docs-source: a changed
# public signature without a regenerated docs/guides/api_symbol_inventory.json.
# That job is path-filtered and not a required check, so nothing else stops it
# before the merge (PR 1570 reached main red on it, 2026-10-09). The rendered
# HTML half needs pdoc and stays in CI.
"$PYTHON" scripts/check_api_reference.py || fail

begin 8
# The same flags the required gate in pr-tests.yml uses. Without `-o addopts=""`
# the local run silently collects two fewer tests than CI does, which is the
# local-is-weaker-than-CI gap these scripts exist to close. Costs about 27 s.
PYTHONPATH="$PWD/scripts/ci${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m pytest tests/contracts -q -x -p rfx_gate_memory \
  -o addopts="" -m "not gpu and not docs_consistency" --strict-markers || fail

begin 9
selection_dir=$(mktemp -d) || fail
# Quote the concrete path now so EXIT always removes this exact scratch directory.
printf -v selection_cleanup 'rm -rf -- %q' "$selection_dir"
trap "$selection_cleanup" EXIT
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
# Bash 3.2 needs guarded expansion for empty arrays under set -u.
stage2_worker_args=()
if [ -n "${RFX_GATE_STAGE2_WORKERS:-}" ]; then
  if [[ ! "$RFX_GATE_STAGE2_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RFX_GATE_STAGE2_WORKERS must be a positive integer" >&2
    fail
  fi
  stage2_worker_args=(-n "$RFX_GATE_STAGE2_WORKERS" --dist loadfile)
fi
stage2_pytest() {
  RFX_GATE_COLLECTION_DIR="$selection_dir/$1-collection" \
    PYTHONPATH="$PWD/scripts/ci${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m pytest "${@:2}" -q \
    -p rfx_gate_collection -p rfx_gate_memory -p no:cacheprovider \
    -o addopts="" -m "not gpu and not slow and not slow_physics and not docs_consistency" --strict-markers \
    --junitxml="$selection_dir/$1.xml"
}
cat "$selection_dir/summary"
if [ "${#selected_tests[@]}" -eq 0 ]; then
  echo "nothing selected"
else
  selected_status=0
  stage2_pytest selected "${selected_tests[@]}" \
    ${stage2_worker_args[@]+"${stage2_worker_args[@]}"} || selected_status=$?
  outcome_status=0
  "$PYTHON" scripts/ci/gate_stage2_outcome.py "$selected_status" \
    "$selection_dir/selected.xml" "$selection_dir/selected-collection" \
    "$selection_dir/first-outcome" || outcome_status=$?
  echo "Thread timeout worker deaths use the same retry path; a repeated timeout fails."
  if [ "$outcome_status" -eq 5 ]; then
    echo "nothing to run (all selected tests deselected or no tests collected)"
  elif [ "$outcome_status" -eq 10 ]; then
    retry_tests=()
    while IFS= read -r test_file; do
      retry_tests+=("$test_file")
    done < "$selection_dir/first-outcome.files"
    retry_worker_args=(-n 0)
    if [ "$(cat "$selection_dir/first-outcome.mode")" = parallel ]; then
      retry_worker_args=(${stage2_worker_args[@]+"${stage2_worker_args[@]}"})
    fi
    retry_status=0
    stage2_pytest retry "${retry_tests[@]}" \
      ${retry_worker_args[@]+"${retry_worker_args[@]}"} || retry_status=$?
    retry_outcome=0
    "$PYTHON" scripts/ci/gate_stage2_outcome.py "$retry_status" \
      "$selection_dir/retry.xml" "$selection_dir/retry-collection" \
      "$selection_dir/retry-outcome" --must-run "$selection_dir/first-outcome.json" || retry_outcome=$?
    echo "selected-tests first execution:"
    cat "$selection_dir/first-outcome.json"
    echo
    echo "retried files: ${#retry_tests[@]}; retry exit status: $retry_status"
    echo "selected-tests retry result:"
    cat "$selection_dir/retry-outcome.json"
    echo
    if [ "$retry_outcome" -ne 0 ]; then
      fail
    fi
  elif [ "$outcome_status" -ne 0 ]; then
    echo "retried files: 0; retry result: not run"
    fail
  else
    echo "retried files: 0; retry result: not needed"
  fi
fi

echo
echo "all ${#STEP_NAMES[@]} steps passed"
