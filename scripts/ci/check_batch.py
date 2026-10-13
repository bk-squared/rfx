#!/usr/bin/env python3
"""Check merge-train batch member bodies, faithful commits, and fragments.

The workflows select this checker for head refs starting with merge-train/batch-.
Member rules remain in check_pr_body and check_changelog_fragment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import check_changelog_fragment
import check_pr_body


REPO_ROOT = Path(__file__).resolve().parents[2]
MEMBER_LINE = re.compile(r"Merge-train-member: #([1-9][0-9]*) ([0-9a-fA-F]{40})")
PR_FIELDS = "state,baseRefName,headRefOid,labels,body,closingIssuesReferences"


def run(repo: Path, *args: str, input: str | None = None) -> str:
    return subprocess.run(
        args, cwd=repo, input=input, text=True, capture_output=True, check=True,
    ).stdout


def members(body: str) -> list[tuple[int, str]]:
    result = []
    seen = set()
    for line in body.splitlines():
        match = MEMBER_LINE.fullmatch(line)
        if match is None:
            if line.lstrip().startswith("Merge-train-member:"):
                raise ValueError("malformed Merge-train-member line")
            continue
        number, sha = int(match[1]), match[2].lower()
        if number in seen:
            raise ValueError(f"#{number}: duplicate Merge-train-member number")
        seen.add(number)
        result.append((number, sha))
    if not result:
        raise ValueError("batch body has no Merge-train-member lines")
    return result


def commits(repo: Path, base: str, head: str) -> list[str]:
    run(repo, "git", "merge-base", "--is-ancestor", base, head)
    rows = run(repo, "git", "rev-list", "--reverse", "--parents", f"{base}..{head}").splitlines()
    result = []
    parent = run(repo, "git", "rev-parse", base).strip()
    for row in rows:
        fields = row.split()
        if len(fields) != 2 or fields[1] != parent:
            raise ValueError("batch commits must be linear, with no merge commits")
        parent = fields[0]
        result.append(parent)
    if not result:
        raise ValueError("batch has no commits")
    return result


def trailers(repo: Path, commit: str) -> tuple[int, str]:
    message = run(repo, "git", "show", "-s", "--format=%B", commit)
    parsed = run(repo, "git", "interpret-trailers", "--parse", input=message)
    numbers, heads = [], []
    for line in parsed.splitlines():
        key, _, value = line.partition(":")
        if key.lower() == "merge-train-member":
            numbers.append(value.strip())
        elif key.lower() == "merge-train-head":
            heads.append(value.strip())
    if (len(numbers) != 1 or len(heads) != 1
            or re.fullmatch(r"#[1-9][0-9]*", numbers[0]) is None
            or re.fullmatch(r"[0-9a-fA-F]{40}", heads[0]) is None):
        raise ValueError(f"{commit}: expected one member trailer and one 40-hex head trailer")
    return int(numbers[0][1:]), heads[0].lower()


def read_pr(repo: Path, number: int, fields: str = PR_FIELDS) -> dict:
    return json.loads(run(repo, "gh", "pr", "view", str(number), "--json", fields))


def labels(payload: dict) -> list[str]:
    return [label["name"] for label in payload["labels"]]


def member_env(repo: Path, pr: dict, env: dict[str, str]) -> dict[str, str]:
    issue_lanes = set()
    for issue in pr["closingIssuesReferences"]:
        repository = issue["repository"]
        name = f"{repository['owner']['login']}/{repository['name']}"
        if name != env["GITHUB_REPOSITORY"]:
            continue
        payload = json.loads(run(
            repo, "gh", "issue", "view", str(issue["number"]), "--json", "labels",
        ))
        issue_lanes.update(label for label in labels(payload) if label.startswith("lane:"))
    return {
        **env,
        "PR_LABELS_JSON": json.dumps(labels(pr)),
        "ISSUE_LANES_JSON": json.dumps(sorted(issue_lanes)),
    }


def paths(repo: Path, base: str, head: str) -> set[str]:
    output = run(repo, "git", "diff", "--no-renames", "--name-only", "-z", base, head)
    return set(output.split("\0")) - {""}


def patch_id(repo: Path, patch: str) -> str:
    output = run(repo, "git", "patch-id", "--stable", input=patch).split()
    return output[0] if output else ""


def check_body(repo: Path, base: str, head: str, env: dict[str, str]) -> list[str]:
    listed = members(env.get("PR_BODY", ""))
    branch = commits(repo, base, head)
    if len(branch) != len(listed):
        return ["batch must have exactly one commit per listed member"]
    failures = []
    for (number, sha), commit in zip(listed, branch):
        try:
            pr = read_pr(repo, number)
            problems = []
            if pr["state"] != "OPEN":
                problems.append("member must be OPEN")
            if pr["baseRefName"] != "main":
                problems.append("member base must be main")
            if pr["headRefOid"] != sha:
                problems.append("member head moved since the batch was built")
            member_labels = labels(pr)
            if "merge-train" not in member_labels:
                problems.append("member is missing merge-train")
            if "merge-train-blocked" in member_labels:
                problems.append("member carries merge-train-blocked")
            problems.extend(check_pr_body.check(pr["body"], member_env(repo, pr, env)))
            if trailers(repo, commit) != (number, sha):
                problems.append("batch commit trailers do not match the listed member and head")
            run(repo, "git", "fetch", "--no-tags", "origin", "main", f"refs/pull/{number}/head")
            merge_base = run(repo, "git", "merge-base", "origin/main", sha).strip()
            member_patch = run(repo, "git", "diff", "--no-renames", merge_base, sha)
            batch_patch = run(repo, "git", "show", "--no-renames", commit)
            if (paths(repo, f"{commit}^", commit) != paths(repo, merge_base, sha)
                    or patch_id(repo, batch_patch) != patch_id(repo, member_patch)):
                problems.append("batch commit differs from the PR's own change")
            failures.extend(f"#{number}: {problem}" for problem in problems)
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            failures.append(f"#{number}: {exc}")
    return failures


def check_fragments(repo: Path, base: str, head: str) -> list[str]:
    failures = []
    for commit in commits(repo, base, head):
        number, _ = trailers(repo, commit)
        try:
            member_labels = labels(read_pr(repo, number, "labels"))
            failures.extend(
                f"#{number}: {failure}"
                for failure in check_changelog_fragment.check(
                    f"{commit}^", commit, member_labels, repo, pr_number=str(number),
                )
            )
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            failures.append(f"#{number}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("body", "fragments"))
    parser.add_argument("--base", default=os.environ.get("BASE_SHA"))
    parser.add_argument("--head", default=os.environ.get("HEAD_SHA"))
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if not args.base or not args.head:
        parser.error("--base/BASE_SHA and --head/HEAD_SHA are required")
    try:
        failures = (
            check_body(args.repo, args.base, args.head, dict(os.environ))
            if args.command == "body"
            else check_fragments(args.repo, args.base, args.head)
        )
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        failures = [str(exc)]
    for failure in failures:
        print(failure, file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
