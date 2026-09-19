"""The intake surface: the path->lane map, the issue forms, and their workflows.

Three pieces of configuration decide who owns a change and whether a finding
becomes an issue at all. None of them runs in a test suite by itself, and each
fails SILENTLY when it drifts: a glob that stops matching owns nothing and says
nothing; a renamed form field leaves the lane parser reading a heading that is
no longer there; a `pull_request_target` job that grows a checkout step becomes
a way to run pull-request code with a write token.

So the shapes are pinned here rather than discovered the week somebody notices
the labels stopped appearing.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
LABELER = REPO / ".github" / "labeler.yml"
WORKFLOWS = REPO / ".github" / "workflows"
FORM_DIR = REPO / ".github" / "ISSUE_TEMPLATE"

_SPEC = importlib.util.spec_from_file_location(
    "check_pr_body_for_intake", REPO / "scripts" / "ci" / "check_pr_body.py"
)
assert _SPEC is not None and _SPEC.loader is not None
cpb = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cpb
_SPEC.loader.exec_module(cpb)

_LANE_SPEC = importlib.util.spec_from_file_location(
    "issue_lane_label_for_intake", REPO / "scripts" / "ci" / "issue_lane_label.py"
)
assert _LANE_SPEC is not None and _LANE_SPEC.loader is not None
lane_script = importlib.util.module_from_spec(_LANE_SPEC)
sys.modules[_LANE_SPEC.name] = lane_script
_LANE_SPEC.loader.exec_module(lane_script)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def triggers(workflow: dict) -> dict:
    """The `on:` block. YAML 1.1 reads a bare `on` as the boolean True."""
    return workflow.get("on", workflow.get(True))


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return [path for path in out.split("\0") if path]


def glob_to_regex(glob: str) -> re.Pattern:
    """`actions/labeler` glob semantics, enough of them to check a match.

    minimatch, which labeler v5 uses: `**` spans path separators and may match
    zero segments, `*` and `?` stay inside one segment. Written out rather than
    reached for through `fnmatch`, whose `*` crosses `/` and would call a dead
    glob alive.
    """
    out: list[str] = []
    index, end = 0, len(glob)
    while index < end:
        if glob.startswith("**/", index):
            out.append("(?:[^/]*/)*")
            index += 3
        elif glob.startswith("**", index) and index + 2 == end:
            out.append(".*")
            index += 2
        elif glob[index] == "*":
            out.append("[^/]*")
            index += 1
        elif glob[index] == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(glob[index]))
            index += 1
    return re.compile("^" + "".join(out) + "$")


def labeler_globs() -> list[tuple[str, str]]:
    """`[(label, glob), ...]` for every entry in `.github/labeler.yml`."""
    pairs: list[tuple[str, str]] = []
    for label, rules in load(LABELER).items():
        for rule in rules:
            for group in rule["changed-files"]:
                for glob in group["any-glob-to-any-file"]:
                    pairs.append((label, glob))
    return pairs


# --------------------------------------------------------------------------
# .github/labeler.yml
# --------------------------------------------------------------------------


def test_the_labeler_config_exists_and_is_not_empty() -> None:
    """A glob list that silently became empty makes every test below vacuous."""
    assert LABELER.is_file(), f"missing {LABELER}"
    assert len(labeler_globs()) >= 20, labeler_globs()


@pytest.mark.parametrize(
    "label,glob", labeler_globs(), ids=lambda value: str(value)
)
def test_every_glob_matches_something_tracked(label: str, glob: str) -> None:
    """A glob matching nothing is configuration that owns nothing.

    It is also the shape a rename leaves behind: `rfx/ports/waveguide*` was in
    the first draft of this file and matched zero files, because there is no
    `rfx/ports/`. Nothing would have said so.
    """
    pattern = glob_to_regex(glob)
    matches = [path for path in tracked_files() if pattern.match(path)]
    assert matches, (
        f"{label}: the glob {glob!r} matches no tracked file. Either the paths "
        f"moved and the glob needs updating, or the glob was never right."
    )


def test_every_labeled_lane_is_a_lane_label() -> None:
    """The keys are label names, and the PR gate has to accept them."""
    keys = set(load(LABELER))
    assert keys, "no lanes in the labeler config"
    unknown = keys - set(cpb.FALLBACK_LANE_LABELS)
    assert not unknown, (
        f"{sorted(unknown)} are not lane labels that check_pr_body.py knows. "
        f"A PR labelled with one would fail the `Lane:` agreement check."
    )


def test_changelog_fragments_are_not_owned_by_a_lane() -> None:
    """Every PR adds a fragment; owning `changelog.d/` labels every PR.

    A second lane label on every pull request would make the `Lane:` line
    mandatory on all of them, which is the opposite of what deriving the lane
    from paths is for.
    """
    fragment = "changelog.d/1234.fixed.md"
    owners = [
        label for label, glob in labeler_globs()
        if glob_to_regex(glob).match(fragment)
    ]
    assert not owners, f"changelog.d/ is owned by {owners}"


#: Files a lane glob must NOT own, and why. A glob wide enough to be convenient
#: is usually wide enough to be wrong, and the wrongness is invisible: the file
#: just quietly belongs to a lane whose owner has never heard of it.
MUST_NOT_MATCH = (
    (
        "lane:coax-mixed-port",
        "tests/unit/grid/test_mixed_precision.py",
        "f32/f64 arithmetic, not a mixed-mode port",
    ),
)


@pytest.mark.parametrize(
    "label,path,why", MUST_NOT_MATCH, ids=lambda value: str(value)
)
def test_a_lane_does_not_own_a_file_that_merely_shares_a_word(
    label: str, path: str, why: str
) -> None:
    assert path in tracked_files(), f"{path} moved; update MUST_NOT_MATCH"
    owning = [
        glob for owner, glob in labeler_globs()
        if owner == label and glob_to_regex(glob).match(path)
    ]
    assert not owning, f"{label} owns {path} via {owning}: {why}"


def test_the_lane_plan_label_has_no_paths() -> None:
    """`lane:plan` marks trackers and deferred decisions, which have no code."""
    assert "lane:plan" not in load(LABELER)


# --------------------------------------------------------------------------
# .github/workflows/labeler.yml
# --------------------------------------------------------------------------


def test_the_labeler_workflow_runs_on_pull_request_target_with_write() -> None:
    data = load(WORKFLOWS / "labeler.yml")
    assert set(triggers(data)["pull_request_target"]["types"]) == {
        "opened", "synchronize", "reopened",
    }
    assert data["permissions"]["pull-requests"] == "write"
    assert data["permissions"]["contents"] == "read"


def test_the_labeler_workflow_checks_out_nothing() -> None:
    """`pull_request_target` runs with a write token in the BASE repo's context.

    Checking out the pull request's head and running anything from it -- a
    setup script, a dependency install, a test -- is remote code execution with
    that token. The job is safe only because it reads the diff through the API
    and never materializes the branch.
    """
    steps = load(WORKFLOWS / "labeler.yml")["jobs"]["lane-label"]["steps"]
    uses = [str(step.get("uses", "")) for step in steps]
    assert not any(u.startswith("actions/checkout") for u in uses), uses
    assert not any("run" in step for step in steps), steps


def _uses_entries(workflow: dict) -> list[str]:
    """Every `uses:` value anywhere in a workflow document."""
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "uses" and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(workflow)
    return found


def pull_request_target_workflows() -> list[Path]:
    return [
        path for path in sorted(WORKFLOWS.glob("*.yml"))
        if "pull_request_target" in (triggers(load(path)) or {})
    ]


def test_at_least_one_workflow_uses_pull_request_target() -> None:
    """Otherwise the SHA-pinning test below silently checks nothing."""
    assert pull_request_target_workflows()


@pytest.mark.parametrize(
    "path", pull_request_target_workflows(), ids=lambda p: p.name
)
def test_every_action_in_a_write_token_workflow_is_pinned_by_sha(path: Path) -> None:
    """A `pull_request_target` job runs in the base repo's context with a token.

    `@v5` is a mutable tag: whoever can move it runs their own code here with
    whatever permissions the job holds, and for the labeler that is
    `pull-requests: write`. A 40-hex commit cannot be moved. The repository
    already pins this way in validation.yml.
    """
    unpinned = [
        entry for entry in _uses_entries(load(path))
        if not re.fullmatch(r"[^@]+@[0-9a-f]{40}", entry)
    ]
    assert not unpinned, (
        f"{path.name} runs under pull_request_target; pin these by commit SHA "
        f"with the version in a trailing comment -- {unpinned}"
    )


def test_the_pinned_labeler_still_says_which_version_it_is() -> None:
    """A bare 40-hex is unreadable; the trailing comment is how it gets bumped."""
    text = (WORKFLOWS / "labeler.yml").read_text(encoding="utf-8")
    assert re.search(r"uses: actions/labeler@[0-9a-f]{40} # v\d+\.\d+\.\d+", text), text


def test_the_labeler_workflow_syncs_labels() -> None:
    """Without `sync-labels` a lane picked up by one commit never leaves."""
    steps = load(WORKFLOWS / "labeler.yml")["jobs"]["lane-label"]["steps"]
    labeler = [s for s in steps if str(s.get("uses", "")).startswith("actions/labeler@")]
    assert len(labeler) == 1, steps
    assert labeler[0]["with"]["sync-labels"] is True
    assert labeler[0]["with"]["configuration-path"] == ".github/labeler.yml"


# --------------------------------------------------------------------------
# .github/ISSUE_TEMPLATE/
# --------------------------------------------------------------------------


def test_blank_issues_are_disabled() -> None:
    """A blank issue is how "will someone DO or DECIDE something" stopped being asked."""
    assert load(FORM_DIR / "config.yml")["blank_issues_enabled"] is False


FORMS = ("finding.yml", "decision.yml")


@pytest.mark.parametrize("name", FORMS)
def test_the_form_parses_and_names_itself(name: str) -> None:
    data = load(FORM_DIR / name)
    assert data["name"] and data["description"]
    assert isinstance(data["body"], list) and data["body"]


def fields(name: str) -> dict[str, dict]:
    return {
        item["id"]: item
        for item in load(FORM_DIR / name)["body"]
        if "id" in item
    }


@pytest.mark.parametrize(
    "name,expected",
    [
        ("finding.yml", {"what_is_wrong", "reproduction", "action", "lane", "evidence"}),
        ("decision.yml", {"question", "options", "what_blocks_on_it", "lane"}),
    ],
)
def test_the_form_asks_for_every_required_field(name: str, expected: set) -> None:
    assert set(fields(name)) == expected


@pytest.mark.parametrize("name", FORMS)
def test_every_field_is_required(name: str) -> None:
    """An optional field renders `_No response_`, which records nothing.

    The forms exist to make the rule's questions unskippable; a field somebody
    can leave blank is back to prose.
    """
    for field_id, field in fields(name).items():
        assert field.get("validations", {}).get("required") is True, field_id


@pytest.mark.parametrize("name", FORMS)
def test_the_lane_dropdown_offers_exactly_the_lane_labels(name: str) -> None:
    options = fields(name)["lane"]["attributes"]["options"]
    assert sorted(options) == sorted(cpb.FALLBACK_LANE_LABELS), options


@pytest.mark.parametrize("name", FORMS)
def test_the_lane_field_label_is_the_one_the_parser_looks_for(name: str) -> None:
    """The parser reads `### Lane` out of the RENDERED body.

    Rename the field label and the heading changes with it, the parser finds
    nothing, and every new issue silently arrives without a lane.
    """
    label = fields(name)["lane"]["attributes"]["label"]
    assert label == lane_script.LANE_HEADING, (
        f"{name}: the Lane field is labelled {label!r}, but "
        f"scripts/ci/issue_lane_label.py looks for a "
        f"`### {lane_script.LANE_HEADING}` heading."
    )


def test_the_finding_form_offers_the_do_not_file_answer() -> None:
    """Most findings belong in the ledger, and the form has to say so.

    Without that option the form only ever produces issues, which is the
    behaviour the three-way rule exists to stop.
    """
    options = fields("finding.yml")["action"]["attributes"]["options"]
    assert any("ledger" in option and "do not file" in option for option in options), options


@pytest.mark.parametrize("name", FORMS)
def test_the_form_titles_the_problem_rather_than_the_fix(name: str) -> None:
    """The seeded title is a placeholder, so it cannot be submitted as written."""
    title = load(FORM_DIR / name)["title"]
    assert title.startswith("<") and title.endswith(">"), title


# --------------------------------------------------------------------------
# .github/workflows/issue-lane-label.yml and governance-audit.yml
# --------------------------------------------------------------------------


def test_the_issue_lane_workflow_runs_on_open_and_edit_with_issue_write() -> None:
    data = load(WORKFLOWS / "issue-lane-label.yml")
    assert set(triggers(data)["issues"]["types"]) == {"opened", "edited"}
    assert data["permissions"]["issues"] == "write"


def test_the_issue_lane_workflow_calls_the_tested_script() -> None:
    data = load(WORKFLOWS / "issue-lane-label.yml")
    blocks = [
        step["run"]
        for step in data["jobs"]["lane-label"]["steps"]
        if isinstance(step.get("run"), str)
    ]
    assert any("scripts/ci/issue_lane_label.py" in block for block in blocks), blocks


def test_the_issue_body_reaches_the_script_through_env() -> None:
    """An issue body is author-controlled text; `${{ }}` in a shell is injection."""
    steps = load(WORKFLOWS / "issue-lane-label.yml")["jobs"]["lane-label"]["steps"]
    values = [
        value
        for step in steps
        for value in (step.get("env") or {}).values()
        if isinstance(value, str)
    ]
    assert "${{ github.event.issue.body }}" in values


def test_the_audit_workflow_is_scheduled_and_dispatchable() -> None:
    data = load(WORKFLOWS / "governance-audit.yml")
    on = triggers(data)
    assert "workflow_dispatch" in on
    crons = [entry["cron"] for entry in on["schedule"]]
    assert crons == ["0 0 * * 1"], crons


def test_the_audit_workflow_calls_the_tested_script() -> None:
    data = load(WORKFLOWS / "governance-audit.yml")
    blocks = [
        step["run"]
        for step in data["jobs"]["audit"]["steps"]
        if isinstance(step.get("run"), str)
    ]
    assert any("scripts/ci/governance_audit.py" in block for block in blocks), blocks


def test_the_audit_says_it_is_not_a_gate() -> None:
    """It exits 1 on any row. Made required, it would block every merge."""
    text = (WORKFLOWS / "governance-audit.yml").read_text(encoding="utf-8")
    assert "NOT a required check" in text


@pytest.mark.parametrize(
    "script", ["issue_lane_label.py", "governance_audit.py"]
)
def test_the_new_ci_scripts_are_executable(script: str) -> None:
    import os

    path = REPO / "scripts" / "ci" / script
    assert path.is_file(), f"missing {path}"
    assert os.access(path, os.X_OK), f"{path} is not executable (chmod +x)"
