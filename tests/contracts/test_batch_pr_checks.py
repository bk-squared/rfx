"""Exercise batch checks through the CLI, real temp git history, and fake gh."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/check_batch.py"
GOOD_BODY = "Lane: lane:ci-infra\nReview: reader (separate instance) - ACCEPT\n"


class Batch:
    def __init__(self, root: Path, member_changes=None, batch_changes=None, order=(11, 22)):
        self.origin = root / "origin"
        self.repo = root / "checkout"
        self.origin.mkdir()
        self.env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Batch test",
            "GIT_AUTHOR_EMAIL": "batch@example.test",
            "GIT_COMMITTER_NAME": "Batch test",
            "GIT_COMMITTER_EMAIL": "batch@example.test",
            "GIT_TERMINAL_PROMPT": "0",
            "GITHUB_REPOSITORY": "bk-squared/rfx",
            "LANE_LABELS": "lane:ci-infra\nlane:research\nlane:new-test",
            # These belong to the batch and must never substitute for a member's.
            "PR_LABELS_JSON": '["lane:research", "release"]',
            "ISSUE_LANES_JSON": '["lane:research"]',
        }
        self.git(self.origin, "init", "-b", "main")
        self.write(self.origin, {"rfx/first.py": "value = 0\n", "rfx/second.py": "value = 0\n"})
        self.base = self.commit(self.origin, "Base")
        changes = {
            11: {"rfx/first.py": "value = 11\n", "changelog.d/11.fixed.md": self.fragment(11)},
            22: {"rfx/second.py": "value = 22\n", "changelog.d/22.fixed.md": self.fragment(22)},
        }
        if member_changes:
            changes.update(member_changes)
        self.shas = {}
        self.prs = {}
        self.issues = {}
        for number in (11, 22):
            self.git(self.origin, "checkout", "-B", f"member-{number}", self.base)
            self.write(self.origin, changes[number])
            sha = self.commit(self.origin, f"Member {number}")
            self.git(self.origin, "update-ref", f"refs/pull/{number}/head", sha)
            self.shas[number] = sha
            self.prs[str(number)] = {
                "state": "OPEN", "baseRefName": "main", "headRefOid": sha,
                "labels": [{"name": "merge-train"}, {"name": "lane:ci-infra"}],
                "body": GOOD_BODY, "closingIssuesReferences": [],
            }
        self.git(self.origin, "checkout", "main")
        self.git(root, "clone", "--no-local", str(self.origin), str(self.repo))
        self.git(self.repo, "checkout", "-b", "merge-train/batch-test")
        self.built = []
        for number in order:
            self.git(self.repo, "cherry-pick", "--no-commit", self.shas[number])
            self.write(self.repo, (batch_changes or {}).get(number, {}))
            message = (
                f"Batch member {number}\n\nMerge-train-member: #{number}\n"
                f"Merge-train-head: {self.shas[number]}\n"
            )
            self.built.append(self.commit(self.repo, message))
        self.head = self.built[-1]
        self.body = "".join(f"Merge-train-member: #{n} {self.shas[n]}\n" for n in (11, 22))
        self.data = root / "responses.json"
        self.calls = root / "gh-calls.jsonl"
        bindir = root / "bin"
        bindir.mkdir()
        fake = bindir / "gh"
        fake.write_text("#!" + sys.executable + "\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_GH_CALLS"], "a") as log:
    log.write(json.dumps(args) + "\\n")
data = json.loads(Path(os.environ["FAKE_GH_DATA"]).read_text())
assert len(args) == 5 and args[1] == "view" and args[3] == "--json", args
assert args[0] in ("pr", "issue"), args
payload = data[args[0]][args[2]]
fields = args[4].split(",")
print(json.dumps({key: payload[key] for key in fields}))
''')
        fake.chmod(0o755)
        self.env.update({
            "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
            "FAKE_GH_DATA": str(self.data), "FAKE_GH_CALLS": str(self.calls),
        })

    @staticmethod
    def fragment(number):
        return f"### Fixed — member change (#{number})\n\n- Correct the value.\n"

    @staticmethod
    def write(repo, changes):
        for name, text in changes.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def git(self, repo, *args):
        return subprocess.run(
            ["git", *args], cwd=repo, env=self.env, capture_output=True, text=True, check=True,
        ).stdout.strip()

    def commit(self, repo, message):
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-m", message)
        return self.git(repo, "rev-parse", "HEAD")

    def run(self, command="body"):
        self.data.write_text(json.dumps({"pr": self.prs, "issue": self.issues}))
        return subprocess.run(
            [sys.executable, str(SCRIPT), command, "--repo", str(self.repo)],
            env={**self.env, "PR_BODY": self.body, "BASE_SHA": self.base, "HEAD_SHA": self.head},
            cwd=self.repo, capture_output=True, text=True,
        )


@pytest.fixture
def batch(tmp_path):
    return Batch(tmp_path)


def assert_rejected(result, message):
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_two_members_and_faithful_batch_pass(batch):
    # The batch itself has neither Lane nor Review lines.
    for command in ("body", "fragments"):
        result = batch.run(command)
        assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_review_text_after_verdict_is_rejected(batch):
    batch.prs["11"]["body"] = GOOD_BODY.replace("ACCEPT", "ACCEPT after corrections")
    assert_rejected(batch.run(), "#11: no well-formed `Review:` line")


def test_moved_head_is_rejected(batch):
    batch.prs["11"]["headRefOid"] = "f" * 40
    assert_rejected(batch.run(), "#11: member head moved since the batch was built")


def test_missing_merge_train_is_rejected(batch):
    batch.prs["11"]["labels"] = [{"name": "lane:ci-infra"}]
    assert_rejected(batch.run(), "#11: member is missing merge-train")


def test_blocked_member_is_rejected(batch):
    batch.prs["11"]["labels"].append({"name": "merge-train-blocked"})
    assert_rejected(batch.run(), "#11: member carries merge-train-blocked")


@pytest.mark.parametrize("field,value,message", [
    ("state", "CLOSED", "#11: member must be OPEN"),
    ("baseRefName", "develop", "#11: member base must be main"),
])
def test_member_state_and_base(batch, field, value, message):
    batch.prs["11"][field] = value
    assert_rejected(batch.run(), message)


def test_extra_unlisted_commit_is_rejected(batch):
    batch.write(batch.repo, {"extra.txt": "unlisted\n"})
    batch.head = batch.commit(batch.repo, "Extra commit")
    assert_rejected(batch.run(), "batch must have exactly one commit per listed member")


def test_commit_listed_twice_is_rejected(batch):
    batch.body = batch.body.splitlines()[0] + "\n" + batch.body.splitlines()[0] + "\n"
    assert_rejected(batch.run(), "#11: duplicate Merge-train-member number")


def test_commit_order_differing_from_body_is_rejected(tmp_path):
    batch = Batch(tmp_path, order=(22, 11))
    result = batch.run()
    assert_rejected(result, "#11: batch commit trailers do not match the listed member and head")
    assert "#22: batch commit trailers do not match the listed member and head" in result.stderr


def test_one_line_change_is_rejected(tmp_path):
    batch = Batch(tmp_path, batch_changes={11: {"rfx/first.py": "value = 12\n"}})
    assert_rejected(batch.run(), "#11: batch commit differs from the PR's own change")


def test_added_file_is_rejected(tmp_path):
    batch = Batch(tmp_path, batch_changes={11: {"extra.txt": "unreviewed\n"}})
    assert_rejected(batch.run(), "#11: batch commit differs from the PR's own change")


def test_member_touching_rfx_without_fragment_is_rejected(tmp_path):
    batch = Batch(tmp_path, member_changes={11: {"rfx/first.py": "value = 11\n"}})
    assert_rejected(batch.run("fragments"), "#11: this PR changes rfx/ but adds no changelog fragment")


def test_thirteen_line_fragment_is_rejected(tmp_path):
    batch = Batch(tmp_path, member_changes={11: {
        "changelog.d/11.fixed.md": "### Fixed — member change (#11)\n" + "- Detail.\n" * 12,
    }})
    assert_rejected(batch.run("fragments"), "#11: changelog.d/11.fixed.md is 13 lines; a fragment is at most 12")


@pytest.mark.parametrize("body,message", [
    ("No members\n", "batch body has no Merge-train-member lines"),
    (" Merge-train-member: #11 " + "a" * 40, "malformed Merge-train-member line"),
    ("Merge-train-member: #11 abc", "malformed Merge-train-member line"),
])
def test_member_line_shape(batch, body, message):
    batch.body = body
    assert_rejected(batch.run(), message)


def test_member_lane_labels_override_batch_labels(batch):
    batch.prs["11"]["body"] = GOOD_BODY.replace("lane:ci-infra", "lane:research")
    assert_rejected(batch.run(), "#11: `Lane: lane:research` is not among the lane labels")


@pytest.mark.parametrize("owner,repo,expected", [
    ("bk-squared", "rfx", 0), ("elsewhere", "rfx", 1), ("bk-squared", "other", 1),
])
def test_only_members_same_repository_closing_issues_supply_lanes(batch, owner, repo, expected):
    batch.prs["11"]["body"] = GOOD_BODY.replace("lane:ci-infra", "lane:new-test") + "Closes #7\n"
    batch.prs["11"]["closingIssuesReferences"] = [{
        "number": 7, "repository": {"owner": {"login": owner}, "name": repo},
    }]
    batch.issues["7"] = {"labels": [{"name": "lane:new-test"}]}
    result = batch.run()
    assert result.returncode == expected, result.stderr
    calls = [json.loads(line) for line in batch.calls.read_text().splitlines()]
    issue_calls = [call for call in calls if call[0] == "issue"]
    assert issue_calls == ([["issue", "view", "7", "--json", "labels"]] if expected == 0 else [])


@pytest.mark.parametrize("release,expected", [(True, 0), (False, 1)])
def test_fragments_use_member_release_label(tmp_path, release, expected):
    batch = Batch(tmp_path, member_changes={11: {"CHANGELOG.md": "Release notes\n"}})
    if release:
        batch.prs["11"]["labels"].append({"name": "release"})
    result = batch.run("fragments")
    assert result.returncode == expected, result.stderr
    if not release:
        assert "#11: CHANGELOG.md was modified without the 'release' label" in result.stderr


def test_workflow_shell_dispatches_by_head_ref(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!" + sys.executable + "\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    for filename, job, step_name, command, ordinary in (
        ("pr-body.yml", "pr-body-contract", "Check the PR body", "body", "check_pr_body.py"),
        ("changelog-fragment.yml", "fragment", "Require a changelog fragment", "fragments", "check_changelog_fragment.py"),
    ):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
        step = next(step for step in workflow["jobs"][job]["steps"] if step["name"] == step_name)
        assert step["env"]["PR_HEAD_REF"] == "${{ github.event.pull_request.head.ref }}"
        assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
        for ref, is_batch in (("merge-train/batch-x", True), ("feature/x", False),
                              ("xmerge-train/batch-x", False), ("merge-train/batch", False)):
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
                env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                     "PR_HEAD_REF": ref, "BASE_SHA": "base value", "HEAD_SHA": "head value"},
                capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stderr
            expected = (["scripts/ci/check_batch.py", command] if is_batch else ["scripts/ci/" + ordinary])
            if command == "fragments":
                expected += ["--base", "base value", "--head", "head value"]
            assert json.loads(result.stdout) == expected
