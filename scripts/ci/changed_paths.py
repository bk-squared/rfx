#!/usr/bin/env python3
"""Decide whether a diff touches anything the simulator fast suite can observe.

Measured on 2026-09-18: three pull requests that changed no file under `rfx/`,
`tests/` or `validation/` still ran the six-shard fast suite on every push --
ten `pr-tests` runs of roughly 35 minutes each for changes confined to
`.github/`, `scripts/` and `docs/agent/`. The shards cannot see those files, so
the runs bought nothing.

This module is the decision, kept away from the workflow yaml so it can be read,
run and unit-tested locally. `scripts/ci/classify_changes.sh` supplies the diff
and writes the answer to `$GITHUB_OUTPUT`; the `changes` job in
`.github/workflows/pr-tests.yml` is one call to that script.

What counts as CODE
-------------------
Anything the fast suite or the guard suite imports, collects, splits on, or is
configured by:

* every file under `rfx/`, `tests/`, `validation/` and `examples/` -- the
  packages under test, the tests themselves, and the example modules the
  tutorial/import tests exercise;
* `pyproject.toml`, `setup.*`, `requirements*.txt`, `MANIFEST.in`, `pytest.ini`
  and `tox.ini` -- the dependency set, the marker filter and the default addopts
  the shards run under. Only `pyproject.toml` exists today; the others are listed
  so that adding one later does not silently open a hole, which is cheaper than
  noticing it after a merge;
* any `conftest.py` -- fixtures and collection hooks;
* any `.test_durations` file -- what `pytest-split` balances the six shards on,
  so a change there changes which test lands in which shard;
* `.github/workflows/pr-tests.yml`, `scripts/ci/changed_paths.py` and
  `scripts/ci/classify_changes.sh` -- the lane and the gatekeeper themselves. A
  change to the thing that decides whether the suite runs has to be exercised by
  the suite it gates, or the first wrong classification lands unobserved.

Everything else -- `docs/`, the other workflows, the rest of `scripts/`,
`changelog.d/`, `README.md` -- is NOT code by this definition.

The classifier never decides to skip on its own: `classify_changes.sh` answers
`true` unconditionally for a push to `main` and for any diff it could not
compute. Skipping is only ever the answer to a pull-request diff that was read
successfully.

Usage
-----
    python scripts/ci/changed_paths.py docs/agent/agent-runbook.mdx   # -> false
    python scripts/ci/changed_paths.py rfx/core/yee.py               # -> true
    git diff -z --name-only main...HEAD | python scripts/ci/changed_paths.py --stdin --null

Prints `true` or `false` on stdout. `--explain` writes the matching paths to
stderr. `--github-output PATH` appends `code_changed=<verdict>` to that file
(the workflow passes `$GITHUB_OUTPUT`).

`--null` reads NUL-separated paths, which is what `git diff -z` writes and what
`classify_changes.sh` pipes in. Without `-z`, git renders a path holding a quote,
a backslash or a newline as a C-quoted string wrapped in double quotes, so
`"rfx/od\"d.py"` would not start with `rfx/` and would read as not-code. Line
splitting cannot recover from that at all when the path contains a newline.

Standard library only, Python 3.10.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from collections.abc import Iterable, Sequence

#: A changed file under any of these directories is code.
CODE_DIRECTORIES: tuple[str, ...] = (
    "rfx/",
    "tests/",
    "validation/",
    "examples/",
)

#: A changed file at exactly one of these paths is code.
CODE_FILES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "conftest.py",
        "MANIFEST.in",
        "pytest.ini",
        "tox.ini",
        ".github/workflows/pr-tests.yml",
        "scripts/ci/changed_paths.py",
        "scripts/ci/classify_changes.sh",
    }
)

#: A changed file whose BASENAME matches one of these globs is code, wherever it
#: sits. `conftest.py` and `.test_durations` are per-directory by design, and a
#: shard reads whichever one is beside it.
CODE_BASENAME_GLOBS: tuple[str, ...] = (
    "conftest.py",
    ".test_durations",
    "*.test_durations",
)

#: A changed file at the repository ROOT matching one of these globs is code.
#: Scoped to the root so that, say, `docs/setup.md` is not mistaken for build
#: configuration. Of these only `pyproject.toml` (above) exists today --
#: `setup.py`, `setup.cfg` and a requirements file are listed in advance so that
#: introducing one does not silently take the dependency set out of the gate.
CODE_ROOT_GLOBS: tuple[str, ...] = ("setup.*", "requirements*.txt")


def normalize(path: str) -> str:
    """Repo-relative, forward-slashed, no `./` prefix, no surrounding space."""
    text = path.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def is_code_path(path: str) -> bool:
    """True when the fast suite or the guard suite can observe this file."""
    target = normalize(path)
    if not target:
        return False
    if target in CODE_FILES:
        return True
    if any(target.startswith(directory) for directory in CODE_DIRECTORIES):
        return True
    basename = target.rsplit("/", 1)[-1]
    if any(fnmatch.fnmatch(basename, glob) for glob in CODE_BASENAME_GLOBS):
        return True
    if "/" not in target and any(
        fnmatch.fnmatch(target, glob) for glob in CODE_ROOT_GLOBS
    ):
        return True
    return False


def code_changed(paths: Iterable[str]) -> bool:
    """True when ANY of `paths` is code the test lanes can observe.

    An empty diff is `False`: nothing changed that the shards could run.
    """
    return any(is_code_path(path) for path in paths)


def matching_paths(paths: Iterable[str]) -> list[str]:
    """The subset of `paths` that made the verdict `True`, in input order."""
    return [path for path in paths if is_code_path(path)]


def _read_stdin(nul_separated: bool) -> list[str]:
    if nul_separated:
        # No strip(): a NUL-separated name is exact, and a path may legally end
        # in a space. Only the trailing empty field after the last NUL goes.
        return [field for field in sys.stdin.read().split("\0") if field]
    return [line for line in (raw.strip() for raw in sys.stdin) if line]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="changed paths, repo-relative")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read the changed paths from stdin, one per line",
    )
    parser.add_argument(
        "--null",
        "-z",
        action="store_true",
        help="with --stdin, paths are NUL-separated (what `git diff -z` writes)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="write the matching paths to stderr",
    )
    parser.add_argument(
        "--github-output",
        default="",
        help="append `code_changed=<verdict>` to this file (pass $GITHUB_OUTPUT)",
    )
    parser.add_argument(
        "--output-name",
        default="code_changed",
        help="the key written by --github-output (default: code_changed)",
    )
    args = parser.parse_args(argv)

    paths = list(args.paths)
    if args.stdin:
        paths += _read_stdin(args.null)

    verdict = code_changed(paths)
    if args.explain:
        matched = matching_paths(paths)
        if matched:
            print(
                f"{len(matched)} of {len(paths)} changed path(s) are code:",
                file=sys.stderr,
            )
            for path in matched:
                print(f"  {path}", file=sys.stderr)
        else:
            print(
                f"none of the {len(paths)} changed path(s) are code",
                file=sys.stderr,
            )

    answer = "true" if verdict else "false"
    print(answer)
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(f"{args.output_name}={answer}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
