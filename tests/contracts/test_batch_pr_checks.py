"""Exercise batch checks through the CLI, real temp git history, and fake gh."""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/check_batch.py"
GOOD_BODY = "Physics: none — CI process\nLane: lane:ci-infra\nReview: reader (separate instance) - ACCEPT\n"
# Homebrew git on the development Mac is 2.39; Apple git supports merge-tree's
# explicit merge base. Child checkers resolve the same binary through PATH.
GIT = "/usr/bin/git" if sys.platform == "darwin" else shutil.which("git")


def _git_version(git):
    out = subprocess.run([git, "version"], capture_output=True, text=True).stdout
    match = re.search(r"(\d+)\.(\d+)", out)
    return (int(match[1]), int(match[2])) if match else (0, 0)


# merge-tree --merge-base needs git 2.40. The VESSL gate image has 2.34, so the
# module skips there; on GitHub Actions (ubuntu-latest ships a newer git) an old
# git is a failure, never a skip, so the required lane always runs these tests.
if GIT is None or _git_version(GIT) < (2, 40):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise RuntimeError(f"batch checks need git >= 2.40 on CI; found {GIT and _git_version(GIT)}")
    pytest.skip("git >= 2.40 not available (merge-tree --merge-base)", allow_module_level=True)


class Batch:
    def __init__(self, root: Path, member_changes=None, batch_changes=None, order=(11, 22),
                 initial_files=None, main_changes=None):
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
        self.write(self.origin, initial_files or {})
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
                "headRepository": {"nameWithOwner": "bk-squared/rfx"}, "isCrossRepository": False,
                "labels": [{"name": "merge-train"}, {"name": "lane:ci-infra"}],
                "body": GOOD_BODY, "closingIssuesReferences": [],
                "createdAt": "2026-10-27T00:00:00Z",
            }
        self.git(self.origin, "checkout", "main")
        if main_changes:
            self.write(self.origin, main_changes)
            self.base = self.commit(self.origin, "Main advanced after the members branched")
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
if os.environ.get("FAKE_GH_ERROR"):
    print(os.environ["FAKE_GH_ERROR"], file=sys.stderr)
    sys.exit(7)
assert len(args) == 5 and args[1] == "view" and args[3] == "--json", args
assert args[0] in ("pr", "issue"), args
payload = data[args[0]][args[2]]
fields = args[4].split(",")
print(json.dumps({key: payload[key] for key in fields}))
''')
        fake.chmod(0o755)
        (bindir / "git").symlink_to(GIT)
        self.bindir = bindir
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
            if text is None:
                path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))

    def git(self, repo, *args):
        return subprocess.run(
            [GIT, *args], cwd=repo, env=self.env, capture_output=True, text=True, check=True,
        ).stdout.strip()

    def fake_git(self, source):
        fake = self.bindir / "git"
        fake.unlink()
        fake.write_text(
            "#!" + sys.executable + "\nimport os, sys\n" + source
            + f"\nos.execv({GIT!r}, [{GIT!r}, *sys.argv[1:]])\n"
        )
        fake.chmod(0o755)

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


def test_member_before_physics_cutover_in_later_batch_passes(batch):
    batch.env["PR_CREATED_AT"] = "2026-10-28T00:00:00Z"
    batch.prs["11"]["createdAt"] = "2026-10-26T14:59:59Z"
    batch.prs["11"]["body"] = "Lane: lane:ci-infra\nReview: reader (separate instance) - ACCEPT\n"
    result = batch.run()
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_member_after_physics_cutover_in_later_batch_requires_physics(batch):
    batch.env["PR_CREATED_AT"] = "2026-10-28T00:00:00Z"
    batch.prs["11"]["createdAt"] = "2026-10-27T00:00:00Z"
    batch.prs["11"]["body"] = "Lane: lane:ci-infra\nReview: reader (separate instance) - ACCEPT\n"
    assert_rejected(batch.run(), "#11: no well-formed `Physics:` first line")


@pytest.mark.parametrize("creation_fields", [{}, {"createdAt": ""}, {"createdAt": None}],
                         ids=["missing", "empty", "null"])
def test_member_missing_creation_time_does_not_inherit_batch_exemption(monkeypatch, creation_fields):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import check_batch

    pr = {"labels": [], "closingIssuesReferences": [], **creation_fields}
    env = check_batch.member_env(ROOT, pr, {"PR_CREATED_AT": "2026-10-26T14:59:59Z"})
    assert env["PR_CREATED_AT"] == ""
    text = "Lane: lane:ci-infra\nReview: reader (separate instance) - ACCEPT\n"
    problems = check_batch.check_pr_body.check(text, env)
    assert len(problems) == 1
    assert "no well-formed `Physics:` first line" in problems[0]


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


TOPIC_PATH = "changelog.d/correct-value.fixed.md"
TOPIC_CONTENT = "### Fixed — member change (#PR)\r\n\r\n- Preserve (#PR) in the body.\r\n".encode("utf-8")
LANDED_CONTENT = "### Fixed — member change (#11)\r\n\r\n- Preserve (#PR) in the body.\r\n".encode("utf-8")


def topic_batch(tmp_path, landed=LANDED_CONTENT, *, initial_files=None, main_changes=None):
    return Batch(
        tmp_path,
        member_changes={11: {"rfx/first.py": "value = 11\n", TOPIC_PATH: TOPIC_CONTENT}},
        batch_changes={11: {TOPIC_PATH: None, "changelog.d/11.fixed.md": landed}},
        initial_files=initial_files, main_changes=main_changes,
    )


def test_renamed_topic_member_is_faithful(tmp_path):
    batch = topic_batch(tmp_path)
    index = Path(batch.git(batch.repo, "rev-parse", "--absolute-git-dir")) / "index"
    index_before = index.read_bytes()
    # An unrelated checkout edit must survive use of the temporary index.
    (batch.repo / "rfx/first.py").write_text("uncommitted = True\n")
    for command in ("body", "fragments"):
        result = batch.run(command)
        assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert index.read_bytes() == index_before
    assert (batch.repo / "rfx/first.py").read_text() == "uncommitted = True\n"
    assert (batch.repo / "changelog.d/11.fixed.md").read_bytes() == LANDED_CONTENT


@pytest.mark.parametrize("landed", [
    "### Fixed — member change (#11)\r\n\r\n- Changed the body.\r\n".encode("utf-8"),
    "### Fixed — member change (#11)\r\n\r\n- Preserve (#11) in the body.\r\n".encode("utf-8"),
    "### Fixed — member change (#11)\n\n- Preserve (#PR) in the body.\n".encode("utf-8"),
    TOPIC_CONTENT,
])
def test_topic_landing_with_other_fragment_changes_is_unfaithful(tmp_path, landed):
    batch = topic_batch(tmp_path, landed)
    assert_rejected(batch.run(), "#11: batch commit differs from the PR's own change in: changelog.d/11.fixed.md")


def test_unrenamed_topic_member_fails_fragments(tmp_path):
    batch = Batch(tmp_path, member_changes={11: {"rfx/first.py": "value = 11\n", TOPIC_PATH: TOPIC_CONTENT}})
    assert_rejected(batch.run("fragments"), "only the train may land it")
    assert_rejected(batch.run(), "#11: batch commit differs from the PR's own change")


def test_topic_member_behind_main_is_faithful(tmp_path):
    batch = topic_batch(tmp_path, main_changes={"unrelated.txt": "main advanced\n"})
    result = batch.run()
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_topic_destination_collision_is_rejected(tmp_path):
    batch = topic_batch(tmp_path, initial_files={"changelog.d/11.fixed.md": Batch.fragment(11)})
    assert_rejected(batch.run(), "changelog target already exists: changelog.d/11.fixed.md")


def test_topic_missing_from_merge_tree_is_a_validation_error(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import check_batch

    batch = topic_batch(tmp_path)
    # The base tree predates the member's added topic fragment.
    with pytest.raises(ValueError, match=f"topic fragment missing from merge tree: {TOPIC_PATH}"):
        check_batch.transform_topic_tree(
            batch.repo, batch.base, batch.base, batch.shas[11], 11,
        )


def test_two_topics_of_one_type_are_rejected(tmp_path):
    batch = Batch(tmp_path, member_changes={11: {
        TOPIC_PATH: TOPIC_CONTENT, "changelog.d/another-fix.fixed.md": TOPIC_CONTENT,
    }})
    assert_rejected(batch.run(), "two topic fragments of one type target changelog.d/11.fixed.md")


def test_topic_without_heading_placeholder_is_rejected(tmp_path):
    batch = Batch(tmp_path, member_changes={11: {
        TOPIC_PATH: "### Fixed — member change (#11)\n\n- Body (#PR).\n",
    }})
    assert_rejected(batch.run(), "topic fragment heading is missing (#PR)")


def test_added_file_is_rejected(tmp_path):
    batch = Batch(tmp_path, batch_changes={11: {"extra.txt": "unreviewed\n"}})
    assert_rejected(batch.run(), "#11: batch commit differs from the PR's own change")


def test_dedented_added_line_is_rejected(tmp_path):
    batch = Batch(tmp_path,
                  initial_files={"rfx/first.py": "if True:\n    value = 0\n"},
                  member_changes={11: {"rfx/first.py": "if True:\n    value = 0\n    added = 11\n"}},
                  batch_changes={11: {"rfx/first.py": "if True:\n    value = 0\nadded = 11\n"}})
    assert_rejected(batch.run(), "#11: batch commit differs from the PR's own change in: rfx/first.py")


def test_hunk_relocated_into_identical_context_is_rejected(tmp_path):
    block = "a\nb\nc\nvalue = 0\nd\ne\nf\n"
    changed = block.replace("value = 0", "value = 11")
    batch = Batch(tmp_path,
                  initial_files={"rfx/first.py": block + "separator\n" + block},
                  member_changes={11: {"rfx/first.py": changed + "separator\n" + block}},
                  batch_changes={11: {"rfx/first.py": block + "separator\n" + changed}})
    assert_rejected(batch.run(), "#11: batch commit differs from the PR's own change in: rfx/first.py")


def test_two_members_editing_lines_five_and_seven_pass(tmp_path):
    original = "".join(f"value_{line} = 0\n" for line in range(1, 12))
    batch = Batch(tmp_path,
                  initial_files={"rfx/first.py": original},
                  member_changes={
                      11: {"rfx/first.py": original.replace("value_5 = 0", "value_5 = 11")},
                      22: {"rfx/first.py": original.replace("value_7 = 0", "value_7 = 22")},
                  })
    result = batch.run()
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_member_behind_main_with_nearby_edit_passes(tmp_path):
    original = "".join(f"value_{line} = 0\n" for line in range(1, 12))
    batch = Batch(tmp_path,
                  initial_files={"rfx/first.py": original},
                  main_changes={"rfx/first.py": original.replace("value_5 = 0", "value_5 = 99")},
                  member_changes={11: {"rfx/first.py": original.replace("value_7 = 0", "value_7 = 11")}})
    result = batch.run()
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_fetching_new_main_does_not_change_pinned_base(batch):
    batch.git(batch.origin, "cherry-pick", batch.shas[11])
    result = batch.run()
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_conflicting_member_is_rejected(tmp_path):
    batch = Batch(tmp_path, order=(11,), member_changes={22: {"rfx/first.py": "value = 22\n"}})
    batch.write(batch.repo, {"rfx/first.py": "value = 22\n"})
    batch.head = batch.commit(batch.repo,
                              f"Resolve conflict\n\nMerge-train-member: #22\nMerge-train-head: {batch.shas[22]}\n")
    assert_rejected(batch.run(), "#22: does not apply cleanly onto the batch")


@pytest.mark.parametrize("command", ["body", "fragments"])
def test_merge_commit_is_rejected(batch, command):
    batch.git(batch.repo, "checkout", "-b", "side", batch.base)
    batch.write(batch.repo, {"side.txt": "side branch\n"})
    side = batch.commit(batch.repo, "Side commit")
    batch.git(batch.repo, "checkout", "merge-train/batch-test")
    batch.git(batch.repo, "merge", "--no-ff", "-m", "Merge side", side)
    batch.head = batch.git(batch.repo, "rev-parse", "HEAD")
    assert_rejected(batch.run(command), "batch commits must be linear, with no merge commits")


@pytest.mark.parametrize("repository,cross_repository", [
    ({"nameWithOwner": "fork-owner/rfx"}, True),
    ({"nameWithOwner": "fork-owner/rfx"}, False),
    ({"nameWithOwner": "bk-squared/rfx"}, True),
    (None, False),
])
def test_fork_member_is_rejected(batch, repository, cross_repository):
    batch.prs["11"].update(headRepository=repository, isCrossRepository=cross_repository)
    assert_rejected(batch.run(), "#11: member is not from this repository")


@pytest.mark.parametrize("command", ["body", "fragments"])
def test_old_git_is_rejected_at_start(batch, command):
    batch.fake_git('if sys.argv[1:] == ["version"]:\n    print("git version 2.39.2")\n    sys.exit(0)\n')
    assert_rejected(batch.run(command), "git 2.40 or later is required for batch checks; found git version 2.39.2")
    assert not batch.calls.exists()


@pytest.mark.parametrize("command", ["body", "fragments"])
def test_gh_failure_includes_last_twenty_stderr_lines(batch, command):
    batch.env["FAKE_GH_ERROR"] = "\n".join(f"gh diagnostic {line}" for line in range(25))
    result = batch.run(command)
    assert_rejected(result, "#11: Command")
    assert "stderr (last 20 lines):\n" + "\n".join(f"gh diagnostic {line}" for line in range(5, 25)) in result.stderr
    assert "gh diagnostic 4\n" not in result.stderr


@pytest.mark.parametrize("git_command", ["version", "rev-list", "fetch", "cat-file", "merge-tree"])
def test_git_failure_includes_stderr(batch, git_command):
    batch.fake_git(f'''if sys.argv[1] == {git_command!r}:
    print("git failure detail", file=sys.stderr)
    sys.exit(128)
''')
    assert_rejected(batch.run(), "stderr (last 20 lines):\ngit failure detail")


@pytest.mark.parametrize("git_args", [["diff-tree"], ["cat-file", "blob"]])
def test_fragment_git_failure_includes_stderr(batch, git_args):
    batch.fake_git(f'''if sys.argv[1:{len(git_args) + 1}] == {git_args!r}:
    print("fragment git failure detail", file=sys.stderr)
    sys.exit(128)
''')
    assert_rejected(batch.run("fragments"), "stderr (last 20 lines):\nfragment git failure detail")


@pytest.mark.parametrize("command", ["body", "fragments"])
def test_checker_uses_plumbing_for_git_reads(batch, command):
    batch.fake_git('''if sys.argv[1] in ("show", "diff", "patch-id"):
    print("porcelain or patch-id used", file=sys.stderr)
    sys.exit(128)
''')
    result = batch.run(command)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


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


@pytest.mark.parametrize("filename,job,step_name,command,ordinary", [
    ("pr-body.yml", "pr-body-contract", "Check the PR body", "body", "check_pr_body.py"),
    ("changelog-fragment.yml", "fragment", "Require a changelog fragment", "fragments", "check_changelog_fragment.py"),
])
def test_workflow_shell_dispatches_by_head_ref(tmp_path, filename, job, step_name, command, ordinary):
    for executable in ("python", "git"):
        fake = tmp_path / executable
        fake.write_text(
            "#!" + sys.executable + "\nimport json, os, sys\n"
            f"print(json.dumps([{executable!r}, *sys.argv[1:]]))\n"
            + ("sys.exit(int(os.environ.get('CHECKOUT_EXIT', '0')))\n" if executable == "git" else "")
        )
        fake.chmod(0o755)
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    step = next(step for step in workflow["jobs"][job]["steps"] if step["name"] == step_name)
    assert step["env"]["PR_HEAD_REF"] == "${{ github.event.pull_request.head.ref }}"
    assert step["env"]["PR_HEAD_REPO"] == "${{ github.event.pull_request.head.repo.full_name }}"
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    for ref, head_repo, is_batch in (
        ("merge-train/batch-x", "bk-squared/rfx", True),
        ("feature/x", "bk-squared/rfx", False),
        ("xmerge-train/batch-x", "bk-squared/rfx", False),
        ("merge-train/batch", "bk-squared/rfx", False),
        ("merge-train/batch-x", "fork-owner/rfx", False),
        ("merge-train/batch-x", "bk-squared/other", False),
    ):
        for checkout_exit in (0, 1):
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
                env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                     "PR_HEAD_REF": ref, "PR_HEAD_REPO": head_repo, "GITHUB_REPOSITORY": "bk-squared/rfx",
                     "BASE_SHA": "base value", "HEAD_SHA": "head value", "CHECKOUT_EXIT": str(checkout_exit)},
                capture_output=True, text=True,
            )
            assert result.returncode == (checkout_exit if is_batch else 0), result.stderr
            expected = (["scripts/ci/check_batch.py", command] if is_batch else ["scripts/ci/" + ordinary])
            if command == "fragments":
                expected += ["--base", "base value", "--head", "head value"]
            calls = [["python", *expected]] if not is_batch or checkout_exit == 0 else []
            if is_batch:
                calls.insert(0, ["git", "checkout", "--no-overlay", "base value", "--", "scripts/ci", "scripts/changelog"])
            assert [json.loads(line) for line in result.stdout.splitlines()] == calls
