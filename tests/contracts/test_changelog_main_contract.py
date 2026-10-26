"""Main and release trees must contain only numbered changelog fragments."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from tests._git_tracked import git_available
from tests._changelog_context import allow_topic_fragments


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
    if allow_topic_fragments():
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


@pytest.mark.parametrize("event,base,context,lenient", [
    (None, "main", None, False), ("", "main", None, False),
    (None, "main", "pre-landing", True), ("", "main", "pre-landing", True),
    (None, "main", "other", False),
    ("push", "main", "pre-landing", False),
    ("merge_group", "main", "pre-landing", False),
    ("workflow_dispatch", "main", "pre-landing", False),
    ("pull_request", "main", None, True),
    ("pull_request", "feature/base", None, True),
    ("pull_request", "release", "pre-landing", False),
    ("pull_request", "release/2.0", "pre-landing", False),
    ("pull_request", "release-next", "pre-landing", False),
])
def test_main_contract_pre_landing_policy(tmp_path, monkeypatch, event, base, context, lenient):
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "changelog.d/unrenamed.fixed.md").write_text("### Fixed — value (#PR)\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    for name, value in (("GITHUB_EVENT_NAME", event), ("GITHUB_BASE_REF", base),
                        ("RFX_CHANGELOG_CONTEXT", context)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    if lenient:
        with pytest.raises(pytest.skip.Exception, match="before the merge train lands"):
            test_no_topic_fragments_on_main()
    else:
        with pytest.raises((AssertionError, pytest.skip.Exception)) as failure:
            test_no_topic_fragments_on_main()
        assert isinstance(failure.value, AssertionError), "strict contexts must not skip the contract"
