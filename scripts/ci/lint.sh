#!/usr/bin/env bash
# The ruff gate, in one place. `.github/workflows/lint.yml` calls this file and
# nothing else, so the line CI runs is the line you can run.
#
# Scope: the packages under test plus the CI/dev helpers. The rest of the tree
# has 197 findings under this selector and is its own cleanup, but an unlinted
# gate is a gate nobody notices breaking, so every script that gates a merge is
# in. tests/contracts/test_pr_body_contract.py pins the path list.
#
# `$PYTHON -m ruff`, not a bare `ruff`: ruff usually lives in the project venv
# and not on PATH, so the bare name made `scripts/ci/local.sh` fail at its first
# step for anyone who had not activated the venv -- the exact "you cannot run
# what CI runs" problem this file exists to remove. $PYTHON is honoured by every
# step of local.sh, so one setting covers the whole run.
#
#   scripts/ci/lint.sh                          # lint the pinned scope
#   scripts/ci/lint.sh --fix                    # extra arguments go to ruff
#   PYTHON=.venv/bin/python scripts/ci/lint.sh  # pick the interpreter

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m ruff check rfx/ tests/ validation/ scripts/ci/ scripts/dev/ scripts/changelog/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402 "$@"
