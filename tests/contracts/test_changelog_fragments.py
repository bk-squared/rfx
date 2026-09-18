"""The changelog-fragment assembler and its CI gate do what they claim (#934).

``CHANGELOG.md`` was the repo's main merge-conflict source. The fix is
mechanical -- one fragment file per PR, assembled at release -- and so is
everything that can go wrong with it: a fragment silently dropped, entries
written in the wrong order, ``--dry-run`` writing anyway, a release cutting
into the wrong heading, a gate that passes a PR with no entry. All of those are
cheap to pin, so they are pinned here rather than trusted.

The tests never touch the repo's own ``CHANGELOG.md`` or ``changelog.d/``: the
assembler runs against a synthetic copy in ``tmp_path`` and the checker against
a git repository this file builds. No network, no package import.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from tests._git_tracked import git_available

REPO = Path(__file__).resolve().parents[2]
ASSEMBLER_PATH = REPO / "scripts" / "changelog" / "assemble.py"
CHECKER_PATH = REPO / "scripts" / "ci" / "check_changelog_fragment.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assemble = _load("_test_changelog_assemble", ASSEMBLER_PATH)
checker = _load("_test_changelog_checker", CHECKER_PATH)

SYNTHETIC = """# Changelog

Preamble that must survive byte-identical.

## [Unreleased — 2.0.0]

### Fixed — an entry that was already here (#700)

- Its bullet.

## [1.8.0] - 2026-09-06

### Added — a released entry (#1)

- Released bullet.
"""


def _fragment(number: int, kind: str, heading_word: str) -> str:
    return (
        f"### {heading_word} — headline number {number} (#{number})\n"
        f"\n"
        f"- bullet for {number}.\n"
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SYNTHETIC, encoding="utf-8")
    (tmp_path / "changelog.d").mkdir()
    return tmp_path


def _write_fragments(root: Path, specs) -> None:
    for number, kind, word in specs:
        (root / "changelog.d" / f"{number}.{kind}.md").write_text(
            _fragment(number, kind, word), encoding="utf-8"
        )


# --------------------------------------------------------------------------
# assembler
# --------------------------------------------------------------------------


def test_unreleased_inserts_newest_first_and_deletes_fragments(workspace: Path):
    _write_fragments(workspace, [(5, "added", "Added"),
                                 (100, "fixed", "Fixed"),
                                 (42, "breaking", "BREAKING")])
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d")])
    assert rc == 0

    text = (workspace / "CHANGELOG.md").read_text(encoding="utf-8")
    lines = text.split("\n")
    heading = lines.index("## [Unreleased — 2.0.0]")
    # Directly under the heading, in descending number order.
    assert lines[heading + 1] == ""
    assert lines[heading + 2] == "### Fixed — headline number 100 (#100)"
    order = [lines.index(f"### {word} — headline number {n} (#{n})")
             for n, word in ((100, "Fixed"), (42, "BREAKING"), (5, "Added"))]
    assert order == sorted(order)
    # The pre-existing entry and the released section are untouched and still
    # below every inserted one.
    assert lines.index("### Fixed — an entry that was already here (#700)") > order[-1]
    assert "## [1.8.0] - 2026-09-06" in text
    assert text.startswith("# Changelog\n\nPreamble that must survive byte-identical.\n")
    # Fragments are gone; README.md (absent here) is never a fragment.
    assert list((workspace / "changelog.d").iterdir()) == []


def test_unreleased_leaves_every_other_byte_untouched(workspace: Path):
    _write_fragments(workspace, [(7, "changed", "Changed")])
    assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                   "--fragments", str(workspace / "changelog.d")])
    after = (workspace / "CHANGELOG.md").read_text(encoding="utf-8")
    inserted = "\n### Changed — headline number 7 (#7)\n\n- bullet for 7.\n"
    assert after.replace(inserted, "", 1) == SYNTHETIC


def test_release_renames_the_heading_and_opens_a_fresh_one(workspace: Path):
    _write_fragments(workspace, [(9, "fixed", "Fixed")])
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"),
                        "--release", "2.0.0", "--date", "2026-09-18"])
    assert rc == 0
    lines = (workspace / "CHANGELOG.md").read_text(encoding="utf-8").split("\n")
    fresh = lines.index("## [Unreleased]")
    cut = lines.index("## [2.0.0] - 2026-09-18")
    assert cut == fresh + 2 and lines[fresh + 1] == ""
    # The fragment landed in the cut version, not in the fresh Unreleased.
    entry = lines.index("### Fixed — headline number 9 (#9)")
    assert entry > cut
    assert "## [Unreleased — 2.0.0]" not in "\n".join(lines)
    assert list((workspace / "changelog.d").iterdir()) == []


def test_release_rejects_a_bad_version_or_date(workspace: Path):
    _write_fragments(workspace, [(9, "fixed", "Fixed")])
    with pytest.raises(assemble.FragmentError):
        assemble.cut_release(SYNTHETIC, "2.0", "2026-09-18")
    with pytest.raises(assemble.FragmentError):
        assemble.cut_release(SYNTHETIC, "2.0.0", "18-09-2026")


def test_dry_run_prints_the_diff_and_touches_nothing(workspace: Path, capsys):
    _write_fragments(workspace, [(11, "removed", "Removed")])
    before = (workspace / "CHANGELOG.md").read_text(encoding="utf-8")
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"),
                        "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "+### Removed — headline number 11 (#11)" in out
    assert (workspace / "CHANGELOG.md").read_text(encoding="utf-8") == before
    assert (workspace / "changelog.d" / "11.removed.md").exists()


def test_check_accepts_valid_fragments(workspace: Path):
    _write_fragments(workspace, [(3, "deprecated", "Deprecated")])
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"), "--check"])
    assert rc == 0


@pytest.mark.parametrize("name", [
    "abc.fixed.md",          # no number
    "12.wrong.md",           # unknown type
    "12.fixed.txt",          # not markdown
    "12.md",                 # no type
    "0012.fixed.md",         # leading zero
])
def test_check_rejects_malformed_names(workspace: Path, name: str):
    (workspace / "changelog.d" / name).write_text(
        "### Fixed — headline (#12)\n", encoding="utf-8")
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"), "--check"])
    assert rc == 1


@pytest.mark.parametrize("first", [
    "### Fixed - headline (#12)",        # ASCII hyphen, not an em dash
    "### Added — headline (#12)",   # heading word contradicts the filename
    "### Fixed — headline (#13)",   # references someone else's number
    "### Fixed — ",                 # no headline
    "Fixed — headline (#12)",       # not a heading
])
def test_check_rejects_malformed_first_lines(workspace: Path, first: str):
    (workspace / "changelog.d" / "12.fixed.md").write_text(
        first + "\n\n- bullet.\n", encoding="utf-8")
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"), "--check"])
    assert rc == 1


def test_a_heading_may_reference_more_than_one_number(workspace: Path):
    (workspace / "changelog.d" / "12.fixed.md").write_text(
        "### Fixed — headline (#12, #34)\n\n- bullet.\n", encoding="utf-8")
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"), "--check"])
    assert rc == 0


def test_readme_is_not_a_fragment(workspace: Path):
    (workspace / "changelog.d" / "README.md").write_text("# docs\n", encoding="utf-8")
    assert assemble.iter_fragments(workspace / "changelog.d") == []
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"), "--check"])
    assert rc == 0


def test_refuses_a_changelog_without_an_unreleased_heading(workspace: Path):
    (workspace / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.8.0] - 2026-09-06\n", encoding="utf-8")
    _write_fragments(workspace, [(8, "fixed", "Fixed")])
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d")])
    assert rc == 1
    assert (workspace / "changelog.d" / "8.fixed.md").exists()


def test_the_repo_changelog_and_fragments_are_currently_valid():
    """The live directory passes its own gate."""
    assert assemble.validate_all(REPO / "changelog.d") == []
    lines = (REPO / "CHANGELOG.md").read_text(encoding="utf-8").split("\n")
    assert assemble.find_unreleased_index(lines) >= 0


def test_a_fragment_body_may_not_forge_a_section_heading(workspace: Path):
    """A `## ` line in a body would split the Unreleased block once assembled."""
    (workspace / "changelog.d" / "12.fixed.md").write_text(
        "### Fixed \u2014 headline (#12)\n\n## [1.8.0] - 2026-09-06\n\n- bullet.\n",
        encoding="utf-8")
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"), "--check"])
    assert rc == 1
    with pytest.raises(assemble.FragmentError, match="line 3"):
        assemble.validate_fragment_text(
            "12.fixed.md", "### Fixed \u2014 h (#12)\n\n## [1.8.0] - 2026-09-06\n")


def test_a_fragment_body_may_use_deeper_headings(workspace: Path):
    (workspace / "changelog.d" / "12.fixed.md").write_text(
        "### Fixed \u2014 headline (#12)\n\n#### Detail\n\n- bullet.\n",
        encoding="utf-8")
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"), "--check"])
    assert rc == 0


def test_same_number_fragments_have_a_deterministic_order(workspace: Path):
    _write_fragments(workspace, [(12, "fixed", "Fixed"), (12, "added", "Added")])
    order = [path.name for path in assemble.iter_fragments(workspace / "changelog.d")]
    assert order == ["12.added.md", "12.fixed.md"]


def test_release_refuses_an_empty_unreleased_section(workspace: Path, capsys):
    (workspace / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.8.0] - 2026-09-06\n\n- old.\n",
        encoding="utf-8")
    rc = assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                        "--fragments", str(workspace / "changelog.d"),
                        "--release", "2.0.0", "--date", "2026-09-18"])
    assert rc == 1
    assert "Unreleased section is empty" in capsys.readouterr().err
    assert "## [2.0.0]" not in (workspace / "CHANGELOG.md").read_text(encoding="utf-8")


def test_release_and_unreleased_are_mutually_exclusive(workspace: Path):
    with pytest.raises(SystemExit):
        assemble.main(["--changelog", str(workspace / "CHANGELOG.md"),
                       "--fragments", str(workspace / "changelog.d"),
                       "--unreleased", "--release", "2.0.0"])


def test_a_crlf_changelog_is_not_rewritten_wholesale(workspace: Path):
    changelog = workspace / "CHANGELOG.md"
    changelog.write_bytes(SYNTHETIC.replace("\n", "\r\n").encode("utf-8"))
    _write_fragments(workspace, [(7, "changed", "Changed")])
    assemble.main(["--changelog", str(changelog),
                   "--fragments", str(workspace / "changelog.d")])
    raw = changelog.read_bytes().decode("utf-8")
    assert "\r\n" in raw
    # Every line ending is still CRLF -- no lone LF survived the round trip.
    assert raw.replace("\r\n", "") .count("\n") == 0
    inserted = "\r\n### Changed \u2014 headline number 7 (#7)\r\n\r\n- bullet for 7.\r\n"
    assert raw.replace(inserted, "", 1) == SYNTHETIC.replace("\n", "\r\n")


def test_parse_labels_reads_the_json_array():
    assert checker.parse_labels('["release", "lane:ci-infra"]') == ["release", "lane:ci-infra"]
    assert checker.parse_labels("") == []
    assert checker.parse_labels("[]") == []
    with pytest.raises(ValueError):
        checker.parse_labels("release")
    with pytest.raises(ValueError):
        checker.parse_labels('{"name": "release"}')


def test_a_label_containing_a_comma_is_not_the_release_label(git_repo: Path):
    """`pre,release` split on ',' used to yield a 'release' element."""
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "CHANGELOG.md").write_text(SYNTHETIC + "\nmore\n", encoding="utf-8")
    head = _commit(git_repo, "edit changelog")
    labels = checker.parse_labels('["pre,release"]')
    assert labels == ["pre,release"]
    failures = _run_check(git_repo, base, head, labels=labels)
    assert len(failures) == 1 and "without the 'release' label" in failures[0]

# --------------------------------------------------------------------------
# CI checker
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    if not git_available():
        pytest.skip("git unavailable")
    repo = tmp_path / "repo"
    (repo / "rfx").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "changelog.d").mkdir()
    (repo / "CHANGELOG.md").write_text(SYNTHETIC, encoding="utf-8")
    (repo / "rfx" / "simulation.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "docs" / "guide.md").write_text("doc\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _run_check(repo: Path, base: str, head: str, labels=(), pr_number="77"):
    return checker.check(base, head, list(labels), repo, pr_number)


def test_package_change_without_a_fragment_fails(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "rfx" / "simulation.py").write_text("x = 2\n", encoding="utf-8")
    head = _commit(git_repo, "touch rfx")
    failures = _run_check(git_repo, base, head)
    assert len(failures) == 1
    assert "adds no changelog fragment" in failures[0]
    assert "changelog.d/77.fixed.md" in failures[0]


def test_package_change_with_a_fragment_passes(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "rfx" / "simulation.py").write_text("x = 2\n", encoding="utf-8")
    (git_repo / "changelog.d" / "77.fixed.md").write_text(
        _fragment(77, "fixed", "Fixed"), encoding="utf-8")
    head = _commit(git_repo, "touch rfx with fragment")
    assert _run_check(git_repo, base, head) == []


def test_a_modified_fragment_does_not_satisfy_the_rule(git_repo: Path):
    """Only an ADDED fragment counts -- editing someone else's is not an entry."""
    (git_repo / "changelog.d" / "50.fixed.md").write_text(
        _fragment(50, "fixed", "Fixed"), encoding="utf-8")
    base = _commit(git_repo, "pre-existing fragment")
    (git_repo / "rfx" / "simulation.py").write_text("x = 3\n", encoding="utf-8")
    (git_repo / "changelog.d" / "50.fixed.md").write_text(
        _fragment(50, "fixed", "Fixed") + "- edited.\n", encoding="utf-8")
    head = _commit(git_repo, "edit an existing fragment")
    failures = _run_check(git_repo, base, head)
    assert len(failures) == 1 and "adds no changelog fragment" in failures[0]


def test_changelog_edit_without_the_release_label_fails(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "CHANGELOG.md").write_text(SYNTHETIC + "\nmore\n", encoding="utf-8")
    head = _commit(git_repo, "edit changelog")
    failures = _run_check(git_repo, base, head)
    assert len(failures) == 1
    assert "without the 'release' label" in failures[0]


def test_changelog_edit_with_the_release_label_passes(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "CHANGELOG.md").write_text(SYNTHETIC + "\nmore\n", encoding="utf-8")
    head = _commit(git_repo, "release assembly")
    assert _run_check(git_repo, base, head, labels=["release"]) == []


def test_docs_only_change_without_a_fragment_passes(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "docs" / "guide.md").write_text("doc edited\n", encoding="utf-8")
    head = _commit(git_repo, "docs only")
    assert _run_check(git_repo, base, head) == []


def test_a_malformed_added_fragment_fails(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "rfx" / "simulation.py").write_text("x = 4\n", encoding="utf-8")
    (git_repo / "changelog.d" / "77.fixed.md").write_text(
        "### Fixed - ASCII hyphen (#77)\n\n- bullet.\n", encoding="utf-8")
    head = _commit(git_repo, "malformed fragment")
    failures = _run_check(git_repo, base, head)
    assert len(failures) == 1 and "em dash" in failures[0]


def test_a_deleted_fragment_is_not_validated(git_repo: Path):
    """A release PR removes fragments; the gate must not read them at head."""
    (git_repo / "changelog.d" / "50.fixed.md").write_text(
        _fragment(50, "fixed", "Fixed"), encoding="utf-8")
    base = _commit(git_repo, "pre-existing fragment")
    (git_repo / "changelog.d" / "50.fixed.md").unlink()
    (git_repo / "CHANGELOG.md").write_text(SYNTHETIC + "\nassembled\n", encoding="utf-8")
    head = _commit(git_repo, "assemble")
    assert _run_check(git_repo, base, head, labels=["release"]) == []


def test_the_checker_cli_exits_nonzero_on_failure(git_repo: Path, monkeypatch):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "rfx" / "simulation.py").write_text("x = 5\n", encoding="utf-8")
    head = _commit(git_repo, "touch rfx")
    monkeypatch.setenv("PR_LABELS_JSON", "[]")
    monkeypatch.setenv("PR_NUMBER", "77")
    rc = checker.main(["--base", base, "--head", head, "--repo", str(git_repo)])
    assert rc == 1


def test_the_workflow_invokes_the_checker_with_both_shas():
    workflow = (REPO / ".github" / "workflows" / "changelog-fragment.yml").read_text(
        encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    assert "timeout-minutes: 5" in workflow
    assert "scripts/ci/check_changelog_fragment.py" in workflow
    assert "github.event.pull_request.base.sha" in workflow
    assert "github.event.pull_request.head.sha" in workflow
    assert "PR_LABELS" in workflow and "PR_NUMBER" in workflow


def test_renaming_a_file_out_of_the_package_still_owes_a_fragment(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "mv", "rfx/simulation.py", "docs/simulation.py")
    head = _commit(git_repo, "move a module out of rfx/")
    statuses = _git(git_repo, "diff", "--name-status", f"{base}...{head}")
    assert statuses.startswith("R")
    failures = _run_check(git_repo, base, head)
    assert len(failures) == 1 and "adds no changelog fragment" in failures[0]


def test_renaming_a_file_into_the_package_owes_a_fragment(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "mv", "docs/guide.md", "rfx/guide.md")
    head = _commit(git_repo, "move a file into rfx/")
    failures = _run_check(git_repo, base, head)
    assert len(failures) == 1 and "adds no changelog fragment" in failures[0]


def test_a_rename_source_is_not_read_at_head(git_repo: Path):
    """The old path is gone at head; reading it would crash the gate."""
    (git_repo / "changelog.d" / "50.fixed.md").write_text(
        _fragment(50, "fixed", "Fixed"), encoding="utf-8")
    base = _commit(git_repo, "pre-existing fragment")
    _git(git_repo, "mv", "changelog.d/50.fixed.md", "changelog.d/51.fixed.md")
    (git_repo / "changelog.d" / "51.fixed.md").write_text(
        _fragment(51, "fixed", "Fixed"), encoding="utf-8")
    head = _commit(git_repo, "renumber a fragment")
    assert _run_check(git_repo, base, head) == []


def test_an_unreachable_base_is_a_message_not_a_traceback(git_repo: Path):
    head = _git(git_repo, "rev-parse", "HEAD")
    failures = _run_check(git_repo, "0" * 40, head)
    assert len(failures) == 1
    assert "fetch-depth: 0" in failures[0]


def test_a_forged_section_heading_in_an_added_fragment_fails(git_repo: Path):
    base = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "rfx" / "simulation.py").write_text("x = 6\n", encoding="utf-8")
    (git_repo / "changelog.d" / "77.fixed.md").write_text(
        "### Fixed \u2014 headline (#77)\n\n## [1.8.0] - 2026-09-06\n", encoding="utf-8")
    head = _commit(git_repo, "forged heading")
    failures = _run_check(git_repo, base, head)
    assert len(failures) == 1 and "may only use '###'" in failures[0]


def test_the_workflow_names_the_job_and_passes_labels_as_json():
    workflow = (REPO / ".github" / "workflows" / "changelog-fragment.yml").read_text(
        encoding="utf-8")
    assert "name: changelog-fragment" in workflow
    assert "PR_LABELS_JSON: ${{ toJSON(github.event.pull_request.labels.*.name) }}" in workflow
    # The comma form is gone, not merely unused (the comment still names it).
    assert "PR_LABELS:" not in workflow
    assert "${{ join(" not in workflow
    assert "REQUIRED check in branch protection" in workflow
