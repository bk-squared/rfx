"""Main and release trees must contain only numbered changelog fragments."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests._git_tracked import git_available


REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_main_changelog_assemble", REPO / "scripts/changelog/assemble.py",
)
assert _spec is not None and _spec.loader is not None
assemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assemble)


def topic_fragment_problems(fragment_dir: Path) -> list[str]:
    return [
        f"{path.name}: topic fragment remains after landing; the train must number it"
        for path in sorted(fragment_dir.iterdir())
        if path.is_file() and assemble.TOPIC_NAME_RE.fullmatch(path.name)
    ]


def test_no_topic_fragments_on_main():
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        pytest.skip("topic fragments are allowed before the merge train lands a pull request")
    assert topic_fragment_problems(REPO / "changelog.d") == []


def test_post_landing_tree_without_rename_is_reported(tmp_path):
    """Mutation: commit the member as main with its topic rename step omitted."""
    if not git_available():
        pytest.skip("git unavailable")
    directory = tmp_path / "changelog.d"
    directory.mkdir()
    (directory / "41.fixed.md").write_text("### Fixed — earlier change (#41)\n", encoding="utf-8")
    (directory / "correct-value.fixed.md").write_text(
        "### Fixed — correct the value (#PR)\n", encoding="utf-8")
    for args in (("init", "-q", "-b", "main"), ("add", "."),
                 ("-c", "user.name=Contract test", "-c", "user.email=contract@example.test",
                  "commit", "-qm", "Land member without topic rename")):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)
    assert topic_fragment_problems(directory) == [
        "correct-value.fixed.md: topic fragment remains after landing; the train must number it",
    ]


@pytest.mark.parametrize("event", [None, "push", "merge_group", "workflow_dispatch", "pull_request"])
def test_main_contract_skips_only_pull_requests(tmp_path, monkeypatch, event):
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "changelog.d/unrenamed.fixed.md").write_text("### Fixed — value (#PR)\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    if event is None:
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    else:
        monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    if event == "pull_request":
        with pytest.raises(pytest.skip.Exception, match="before the merge train lands"):
            test_no_topic_fragments_on_main()
    else:
        with pytest.raises(AssertionError):
            test_no_topic_fragments_on_main()
