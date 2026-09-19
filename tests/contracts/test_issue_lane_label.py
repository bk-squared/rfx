"""Reading the Lane out of an issue body, and deciding which labels to move.

The lane label is the only thing a session in another pod can query to learn
what a lane owns (`gh issue list -l lane:msl-port`). Everything downstream of
that query trusts one parser over a body an author can edit, so the cases that
must NOT produce a label matter as much as the one that must: a hand-written
issue with no form section, a dropdown left unanswered, a lane spelled
something the dropdown cannot emit.

The move half is the other risk. An edit that leaves the old lane behind puts
one issue in two lanes and the query starts lying. So is a SECOND `### Lane`
heading pasted into an earlier answer: taking the first match would let the
paste outrank the dropdown, silently and invisibly.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "issue_lane_label.py"

_SPEC = importlib.util.spec_from_file_location("issue_lane_label", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

LANES = (
    "lane:waveguide-port",
    "lane:msl-port",
    "lane:absorber",
    "lane:ci-infra",
    "lane:plan",
)

# The three sample bodies the workflow actually meets.
FORM_BODY = """### What is wrong

`|S21| = 1.043` on a passive thru at 10 GHz; realized aperture 14 cells, declared 16.

### Reproduction

pytest tests/unit/sparams/test_waveguide_twoport_contract_v1.py -q

### What happens next

someone will FIX this — say who, or which lane, in Evidence

### Lane

lane:waveguide-port

### Evidence

rfx/sparams/waveguide.py, VESSL run 369367261081
"""

FREEFORM_BODY = """Noticed while reading the extractor: the msl phase reference
looks off by half a cell. No lane, no form, filed by hand.
"""

UNANSWERED_BODY = """### The question

Do we promote the inverse to f64?

### Lane

_No response_
"""


# --------------------------------------------------------------------------
# lane_from_body
# --------------------------------------------------------------------------


def test_a_form_body_yields_the_dropdown_value() -> None:
    assert mod.lane_from_body(FORM_BODY) == "lane:waveguide-port"


def test_a_form_body_has_exactly_one_lane_section() -> None:
    sections = mod.lane_sections(FORM_BODY)
    assert [section.value for section in sections] == ["lane:waveguide-port"]


def test_a_hand_written_body_yields_nothing() -> None:
    """Not an error: the weekly audit reports it, this job does nothing.

    Failing here would put a red check on every issue transferred from another
    repository, every time anyone edited it.
    """
    assert mod.lane_from_body(FREEFORM_BODY) is None


def test_an_unanswered_dropdown_yields_nothing() -> None:
    assert mod.lane_from_body(UNANSWERED_BODY) is None


def test_the_heading_must_be_level_three() -> None:
    """Issue forms render a field label as `###`, always.

    Accepting `##` would let a prose heading someone wrote in a hand-filed
    issue be read as an answer.
    """
    assert mod.lane_from_body("## Lane\n\nlane:absorber\n") is None
    assert mod.lane_from_body("### Lane\n\nlane:absorber\n") == "lane:absorber"


def test_a_lane_mentioned_in_prose_is_not_an_answer() -> None:
    body = "This belongs to lane:msl-port I think.\n\nNo heading anywhere.\n"
    assert mod.lane_from_body(body) is None


def test_an_empty_section_falls_through_to_nothing() -> None:
    """A heading with the next heading straight after it answered nothing."""
    assert mod.lane_from_body("### Lane\n\n### Evidence\n\nfoo\n") is None


def test_crlf_bodies_are_handled() -> None:
    """GitHub returns CRLF for a body edited in some clients."""
    assert mod.lane_from_body("### Lane\r\n\r\nlane:ci-infra\r\n") == "lane:ci-infra"


def test_only_the_first_line_of_the_answer_is_taken() -> None:
    """A note under a single-select answer is a note, not part of the label."""
    body = "### Lane\n\nlane:absorber\n\nbecause the CPML pad is what moved\n"
    assert mod.lane_from_body(body) == "lane:absorber"


def test_an_empty_body_is_not_an_error() -> None:
    assert mod.lane_from_body("") is None


# --------------------------------------------------------------------------
# Two `### Lane` sections: the form cannot produce one, so nothing is applied
# --------------------------------------------------------------------------

#: The attack the first draft had: the heading pasted into an earlier free-text
#: answer, where it comes BEFORE the dropdown's own section.
PASTED_BODY = """### What is wrong

Broken. Also:

### Lane

lane:ci-infra

### Lane

lane:waveguide-port
"""


def test_a_pasted_lane_heading_does_not_outrank_the_dropdown() -> None:
    lane, conflicts = mod.resolve_lane(PASTED_BODY)
    assert lane is None
    assert len(conflicts) == 1
    assert "lane:ci-infra" in conflicts[0] and "lane:waveguide-port" in conflicts[0]


def test_the_conflict_message_names_the_lines() -> None:
    """Somebody has to find the paste, so the report says where it is."""
    _, conflicts = mod.resolve_lane(PASTED_BODY)
    numbers = [section.line for section in mod.lane_sections(PASTED_BODY)]
    assert numbers == [5, 9]
    for number in numbers:
        assert f"line {number}" in conflicts[0]


def test_two_sections_naming_the_SAME_lane_still_conflict() -> None:
    """The form renders one. Two means the body was edited, whatever they say."""
    body = "### Lane\n\nlane:absorber\n\n### Lane\n\nlane:absorber\n"
    lane, conflicts = mod.resolve_lane(body)
    assert lane is None and len(conflicts) == 1


def test_a_second_UNANSWERED_section_does_not_conflict() -> None:
    """Only answers compete. An empty heading decides nothing."""
    body = "### Lane\n\nlane:absorber\n\n### Lane\n\n_No response_\n"
    lane, conflicts = mod.resolve_lane(body)
    assert lane == "lane:absorber" and conflicts == []


def test_a_conflicting_body_changes_no_label_and_stays_green() -> None:
    """A body must never be able to turn the job red.

    It runs on every edit of every issue, so a red one would follow the issue
    around until somebody with write access noticed.
    """
    result = mod.plan(PASTED_BODY, ["lane:msl-port"], LANES)
    assert result.add is None
    assert result.remove == []
    assert result.problems == []
    assert len(result.notes) == 1


def test_the_cli_is_green_on_a_conflicting_body(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.write_text("", encoding="utf-8")
    result = _run({
        "ISSUE_BODY": PASTED_BODY,
        "CURRENT_LABELS_JSON": json.dumps(["lane:msl-port"]),
        "LANE_LABELS": "\n".join(LANES),
        "GITHUB_OUTPUT": str(output),
    })
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sections answer differently" in result.stderr
    assert output.read_text(encoding="utf-8").strip().splitlines() == [
        "lane=", "remove=",
    ]


# --------------------------------------------------------------------------
# has_lane_section: what the weekly audit asks
# --------------------------------------------------------------------------


def test_a_form_body_has_a_lane_section() -> None:
    assert mod.has_lane_section(FORM_BODY) is True


def test_a_hand_written_body_has_none() -> None:
    assert mod.has_lane_section(FREEFORM_BODY) is False


def test_an_unanswered_dropdown_still_counts_as_a_form() -> None:
    """The audit asks whether the FORM was used, not whether it was answered."""
    assert mod.has_lane_section(UNANSWERED_BODY) is True
    assert mod.lane_from_body(UNANSWERED_BODY) is None


# --------------------------------------------------------------------------
# plan: add one, remove the rest
# --------------------------------------------------------------------------


def test_an_unlabelled_issue_gets_its_lane() -> None:
    assert mod.plan(FORM_BODY, [], LANES) == (
        "lane:waveguide-port", [], [], [],
    )


def test_editing_the_dropdown_moves_the_label() -> None:
    result = mod.plan(FORM_BODY, ["lane:msl-port", "bug"], LANES)
    assert result.add == "lane:waveguide-port"
    assert result.remove == ["lane:msl-port"]
    assert result.problems == [] and result.notes == []


def test_a_non_lane_label_is_left_alone() -> None:
    """The job owns `lane:*` and nothing else."""
    result = mod.plan(FORM_BODY, ["bug", "release", "lane:plan"], LANES)
    assert result.remove == ["lane:plan"]


def test_an_edit_that_did_not_change_the_lane_changes_nothing() -> None:
    """Re-adding a label re-notifies every watcher for no reason."""
    assert mod.plan(FORM_BODY, ["lane:waveguide-port"], LANES) == (
        None, [], [], [],
    )


def test_more_than_one_stale_lane_is_removed() -> None:
    result = mod.plan(FORM_BODY, ["lane:msl-port", "lane:absorber"], LANES)
    assert result.remove == ["lane:absorber", "lane:msl-port"]


def test_a_hand_written_body_leaves_existing_labels_alone() -> None:
    """No answer is not an instruction to unlabel.

    An issue triaged by hand months ago must not lose its lane the next time
    somebody fixes a typo in it.
    """
    assert mod.plan(FREEFORM_BODY, ["lane:msl-port"], LANES) == (
        None, [], [], [],
    )


def test_a_lane_the_dropdown_cannot_produce_is_reported() -> None:
    result = mod.plan("### Lane\n\nlane:nope\n", [], LANES)
    assert result.add is None and result.remove == []
    assert len(result.problems) == 1 and "lane:nope" in result.problems[0]


# --------------------------------------------------------------------------
# parse_labels
# --------------------------------------------------------------------------


def test_labels_come_from_json_so_a_comma_survives() -> None:
    assert mod.parse_labels('["lane:plan","needs,triage"]') == [
        "lane:plan", "needs,triage",
    ]


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_no_labels_is_an_empty_list(raw) -> None:
    assert mod.parse_labels(raw) == []


def test_a_non_array_payload_is_rejected() -> None:
    with pytest.raises(ValueError):
        mod.parse_labels('{"name": "lane:plan"}')


# --------------------------------------------------------------------------
# The CLI, which is what the workflow runs
# --------------------------------------------------------------------------


def _run(env: dict, extra: list | None = None) -> subprocess.CompletedProcess:
    child = dict(os.environ)
    child.pop("ISSUE_BODY", None)
    child.pop("CURRENT_LABELS_JSON", None)
    child.pop("GITHUB_OUTPUT", None)
    child.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(extra or [])],
        env=child, capture_output=True, text=True, timeout=60,
    )


def test_the_cli_publishes_the_plan_as_step_outputs(tmp_path: Path) -> None:
    """The workflow's `if:` and its `gh` call read these two outputs."""
    output = tmp_path / "out"
    output.write_text("", encoding="utf-8")
    result = _run({
        "ISSUE_BODY": FORM_BODY,
        "CURRENT_LABELS_JSON": json.dumps(["lane:msl-port"]),
        "LANE_LABELS": "\n".join(LANES),
        "GITHUB_OUTPUT": str(output),
    })
    assert result.returncode == 0, result.stderr
    written = output.read_text(encoding="utf-8")
    assert "lane=lane:waveguide-port" in written
    assert "remove=lane:msl-port" in written


def test_the_cli_is_quiet_and_green_on_a_hand_written_issue(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.write_text("", encoding="utf-8")
    result = _run({
        "ISSUE_BODY": FREEFORM_BODY,
        "LANE_LABELS": "\n".join(LANES),
        "GITHUB_OUTPUT": str(output),
    })
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip().splitlines() == [
        "lane=", "remove=",
    ]


def test_the_cli_fails_on_a_lane_the_dropdown_cannot_produce() -> None:
    result = _run({
        "ISSUE_BODY": "### Lane\n\nlane:invented\n",
        "LANE_LABELS": "\n".join(LANES),
    })
    assert result.returncode == 1
    assert "lane:invented" in result.stderr


def test_the_cli_with_no_body_fails_rather_than_passing_vacuously() -> None:
    result = _run({"LANE_LABELS": "\n".join(LANES)})
    assert result.returncode == 1
    assert "ISSUE_BODY" in result.stderr


def test_the_cli_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "body.md"
    path.write_text(FORM_BODY, encoding="utf-8")
    result = _run({"LANE_LABELS": "\n".join(LANES)}, ["--file", str(path)])
    assert result.returncode == 0, result.stderr
    assert "add lane:waveguide-port" in result.stdout


def test_the_fallback_lane_list_is_the_pr_gates_own() -> None:
    """Two hardcoded lane lists drift, and then the two gates disagree.

    The issue lane and the PR lane have to be the same set of names, so this
    script imports the PR gate's list rather than keeping a second copy.
    """
    spec = importlib.util.spec_from_file_location(
        "check_pr_body_for_lane_test", REPO / "scripts" / "ci" / "check_pr_body.py"
    )
    assert spec is not None and spec.loader is not None
    cpb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpb)
    assert mod.allowed_lanes({}) == tuple(cpb.FALLBACK_LANE_LABELS)


def test_the_live_lane_list_wins_over_the_fallback() -> None:
    assert mod.allowed_lanes({"LANE_LABELS": "lane:only\n"}) == ("lane:only",)
