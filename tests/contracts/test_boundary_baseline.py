"""B1 refusal prefixes and reproducible record writes."""

from copy import deepcopy
import json

import pytest

from tests.contracts.boundary_baseline import ROOT, refusal_prefix, write_baseline


BASELINE = json.loads((ROOT / "scripts/diagnostics/boundary_model/B1/MATRIX.json").read_text())


def test_baseline_rewrite_keeps_both_leader_conclusions(tmp_path):
    baseline = deepcopy(BASELINE)
    baseline["conclusion"] = "Leader's JSON text.\nKeep punctuation and spacing."
    baseline["kernel_base_commit"] = "new-base"
    markdown = (ROOT / "scripts/diagnostics/boundary_model/B1/MATRIX.md").read_text()
    markdown = markdown.rsplit("\n\n", 1)[0] + "\n\nLeader's Markdown text.  Keep two spaces.\n"
    write_baseline(baseline, markdown, tmp_path)
    assert json.loads((tmp_path / "MATRIX.json").read_text())["conclusion"] == baseline["conclusion"]
    updated = (tmp_path / "MATRIX.md").read_text()
    assert updated.split("## Stopped comparisons", 1)[1] == markdown.split("## Stopped comparisons", 1)[1]
    assert "Kernel base: `new-base`" in updated


def test_refusal_prefix_ignores_count_and_explanation():
    assert refusal_prefix("[run] preflight found 1 blocking error(s): old explanation") == "[run] preflight found"
    assert refusal_prefix("[run] preflight found 2 blocking error(s): new explanation") == "[run] preflight found"
    with pytest.raises(AssertionError, match="Unregistered refusal"):
        refusal_prefix("unrelated numerical error")
