#!/usr/bin/env bash
# Every short gate a pull request must pass. Run it before every push.
# Measured 2026-09-18: tests/contracts is 5 min 16 s of it, the rest under 10 s.
#
# Two CI failures on one PR on 2026-09-18 could not be reproduced locally,
# because the commands that produced them lived inline in workflow yaml. The
# gates now live in scripts; this file is the one command that runs them in the
# order CI does, and stops at the first failure.
#
# What it does NOT run: the six-shard fast suite and the guard/preflight suite.
# Those are 35 minutes, and `.github/workflows/pr-tests.yml` only starts them
# when the diff touches code (`scripts/ci/changed_paths.py` decides).
#
#   scripts/ci/local.sh                 # the six steps; pr-body reports skipped
#   scripts/ci/local.sh /tmp/body.md    # also check that PR body
#   PYTHON=.venv/bin/python scripts/ci/local.sh
#   CHANGELOG_BASE=origin/main CHANGELOG_HEAD=HEAD scripts/ci/local.sh
#
# $PYTHON is exported, so every step and every script this one calls uses it.
#
# The step names below are listed in the same order in docs/agent/agent-runbook.mdx
# and pinned against it by tests/contracts/test_ci_workflows_contract.py.

set -uo pipefail

STEP_NAMES=(ruff docs-hygiene changelog-fragment pr-body workflow-yaml contract-tests)

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Exported, so the scripts this file calls use the same interpreter. Without it
# scripts/ci/lint.sh fell back to a bare `ruff` that is usually only in the venv.
export PYTHON="${PYTHON:-python3}"
BODY_PATH="${1:-}"

# What the changelog gate diffs against. CI uses the PR's base and head shas;
# locally the merge base with origin/main is the same question.
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
if [ -n "$BODY_PATH" ]; then
  "$PYTHON" scripts/ci/check_pr_body.py --file "$BODY_PATH" || fail
  echo "PR body OK: $BODY_PATH"
else
  echo "skipped — pass a path to check a PR body: scripts/ci/local.sh /tmp/body.md"
fi

begin 4
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

begin 5
"$PYTHON" -m pytest tests/contracts -q -x || fail

echo
echo "all ${#STEP_NAMES[@]} steps passed"
