#!/usr/bin/env bash
# Forbid references to gitignored internal paths in what a public clone ships.
# `.github/workflows/lint.yml` calls this file and nothing else, so a failure
# here reproduces with one command instead of being readable only in yaml.
#
# docs/agent/ pages are sanitized public exports of internal agent guidance, and
# shipped code (docstrings, comments, runtime messages) is read by the same
# public-clone reader. Neither has docs/agent-memory/, docs/research_notes/,
# CLAUDE.md or .claude/ to point at.
#
#   scripts/ci/docs_hygiene.sh

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

status=0

# `grep` exits 1 on no match, which is the PASSING case here, hence `|| true`.
hits=$(grep -rn 'research_notes\|agent-memory\|CLAUDE\.md\|\.claude/' docs/agent/ || true)
if [ -n "$hits" ]; then
  echo "docs/agent/ must not reference gitignored internal paths:"
  echo "$hits"
  status=1
else
  echo "docs/agent/ clean"
fi

# research_notes is deliberately absent from this second scan: two examples have
# functional research_notes paths (the multilayer_ar_coating optional Meep
# reference loader, and the research subgrid material example's ARTIFACT_PATH).
# The docs/agent/ scan above still forbids it in exported agent pages.
hits=$(grep -rn 'agent-memory\|CLAUDE\.md\|\.claude/' rfx/ examples/ || true)
if [ -n "$hits" ]; then
  echo "rfx/ and examples/ must not reference gitignored internal paths:"
  echo "$hits"
  status=1
else
  echo "rfx/ and examples/ clean"
fi

exit "$status"
