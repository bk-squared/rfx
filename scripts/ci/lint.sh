#!/usr/bin/env bash
# The ruff gate, in one place. `.github/workflows/lint.yml` calls this file and
# nothing else, so the line CI runs is the line you can run.
#
# Scope: the packages under test plus the CI/dev helpers. The rest of the tree
# has 197 findings under this selector and is its own cleanup, but an unlinted
# gate is a gate nobody notices breaking, so every script that gates a merge is
# in. tests/contracts/test_pr_body_contract.py pins the path list.
#
#   scripts/ci/lint.sh            # lint the pinned scope
#   scripts/ci/lint.sh --fix      # any extra argument is passed to ruff

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ruff check rfx/ tests/ validation/ scripts/ci/ scripts/dev/ scripts/changelog/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402 "$@"
