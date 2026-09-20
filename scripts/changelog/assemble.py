#!/usr/bin/env python3
"""Assemble ``changelog.d/`` fragments into ``CHANGELOG.md`` (issue #934).

Every PR used to append to the same ``## [Unreleased]`` block of one 5500-line
file, which made ``CHANGELOG.md`` the repo's main merge-conflict source: the
conflicts were pure text adjacency, taught a reviewer nothing, and the rebases
they forced are what escalated #921 into a semantic conflict on a results
record. One file per PR removes the conflict class by construction.

A fragment is ``changelog.d/<number>.<type>.md``. Its content is exactly the
text that used to be pasted under ``## [Unreleased]``: a ``### <Type> — <one
sentence> (#NNN)`` heading and its bullets. This script moves them in.

Stdlib only, on purpose -- a release step that needs an install is a release
step that breaks on a fresh clone.

Usage::

    python scripts/changelog/assemble.py                  # move fragments into Unreleased
    python scripts/changelog/assemble.py --dry-run        # print the diff, touch nothing
    python scripts/changelog/assemble.py --check          # validate names/headings only
    python scripts/changelog/assemble.py --release 2.0.0 [--date 2026-09-18]
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import difflib
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANGELOG = REPO_ROOT / "CHANGELOG.md"
DEFAULT_FRAGMENT_DIR = REPO_ROOT / "changelog.d"

EM_DASH = "—"

#: fragment ``<type>`` -> the heading word used in CHANGELOG.md.
TYPE_HEADINGS = {
    "added": "Added",
    "breaking": "BREAKING",
    "changed": "Changed",
    "deprecated": "Deprecated",
    "fixed": "Fixed",
    "removed": "Removed",
}

FRAGMENT_NAME_RE = re.compile(r"^(?P<number>[1-9][0-9]*)\.(?P<type>[a-z]+)\.md$")
UNRELEASED_HEADING_RE = re.compile(r"^## \[Unreleased[^\]]*\]\s*$")
SECTION_HEADING_RE = re.compile(r"^## ")
#: A fragment body may only carry ``###`` and deeper. A ``#`` or ``##`` line
#: inside a fragment would split the Unreleased block once assembled -- it can
#: forge a released section heading -- which is exactly the #920 failure mode
#: this design is supposed to make impossible.
FORBIDDEN_BODY_HEADING_RE = re.compile(r"^#{1,2} ")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.\-]+)?$")
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
ISSUE_REF_RE = re.compile(r"#([1-9][0-9]*)")

#: Released sections use the Keep-a-Changelog form already in the file:
#: ``## [1.8.0] - 2026-09-06`` (ASCII hyphen). The em dash is the ``###``
#: entry separator, not the version/date one.
RELEASED_HEADING = "## [{version}] - {date}"
FRESH_UNRELEASED_HEADING = "## [Unreleased]"


class FragmentError(ValueError):
    """A fragment filename or first line does not follow the convention."""


def read_text(path: Path) -> Tuple[str, str]:
    """Return ``(text, newline)`` with line endings normalised to ``\\n``.

    ``Path.read_text`` folds CRLF to LF and ``write_text`` would then write the
    whole file back with LF, turning a one-entry insertion into a whole-file
    rewrite on a CRLF checkout. Remember what was there and put it back.
    """
    raw = path.read_bytes().decode("utf-8")
    if "\r\n" in raw:
        return raw.replace("\r\n", "\n"), "\r\n"
    return raw, "\n"


def write_text(path: Path, text: str, newline: str = "\n") -> None:
    """Write *text*, restoring *newline* as the line ending."""
    if newline != "\n":
        text = text.replace("\n", newline)
    path.write_bytes(text.encode("utf-8"))


def parse_fragment_name(name: str) -> Tuple[int, str]:
    """Return ``(number, type)`` for a fragment filename, or raise."""
    match = FRAGMENT_NAME_RE.match(name)
    if match is None:
        raise FragmentError(
            f"{name}: expected changelog.d/<number>.<type>.md, "
            f"<type> one of {', '.join(sorted(TYPE_HEADINGS))}"
        )
    kind = match.group("type")
    if kind not in TYPE_HEADINGS:
        raise FragmentError(
            f"{name}: unknown type {kind!r}; expected one of "
            f"{', '.join(sorted(TYPE_HEADINGS))}"
        )
    return int(match.group("number")), kind


def validate_fragment_text(name: str, text: str) -> None:
    """Raise :class:`FragmentError` unless *text* is a well-formed fragment.

    The number in the filename must appear among the ``#NNN`` references on the
    heading line, so a copy-pasted fragment cannot file itself under someone
    else's number. Extra references (``(#1041, #1055)``) are allowed. Beyond
    the heading the body may only use ``###`` and deeper.
    """
    number, kind = parse_fragment_name(name)
    body = text.replace("\r\n", "\n").strip("\n")
    if not body:
        raise FragmentError(f"{name}: fragment is empty")
    lines = body.split("\n")
    first = lines[0].rstrip()
    expected = f"### {TYPE_HEADINGS[kind]} {EM_DASH} "
    if not first.startswith(expected):
        raise FragmentError(
            f"{name}: first line must start with {expected!r} "
            f"(em dash U+2014, not '-'); got {first!r}"
        )
    headline = first[len(expected):].strip()
    if not headline:
        raise FragmentError(f"{name}: heading has no headline after the em dash")
    if number not in {int(ref) for ref in ISSUE_REF_RE.findall(first)}:
        raise FragmentError(
            f"{name}: heading must reference (#{number}); got {first!r}"
        )
    for lineno, line in enumerate(lines[1:], start=2):
        if FORBIDDEN_BODY_HEADING_RE.match(line):
            raise FragmentError(
                f"{name}: line {lineno} is a '#'/'##' heading ({line.rstrip()!r}). "
                f"A fragment body may only use '###' and deeper -- a '## ' line "
                f"would split the Unreleased block once assembled."
            )


def iter_fragments(fragment_dir: Path) -> List[Path]:
    """Every fragment in *fragment_dir*, newest number first.

    Two fragments may share a number (an issue and its PR); the type and then
    the filename break the tie so the assembled order never depends on the
    order the filesystem happened to list them in. ``README.md`` documents the
    directory and is not a fragment.
    """
    if not fragment_dir.is_dir():
        return []
    paths = [
        path
        for path in fragment_dir.iterdir()
        if path.is_file() and path.suffix == ".md" and path.name != "README.md"
    ]
    for path in paths:
        parse_fragment_name(path.name)

    def sort_key(path: Path) -> Tuple[int, str, str]:
        number, kind = parse_fragment_name(path.name)
        return (-number, kind, path.name)

    return sorted(paths, key=sort_key)


def validate_all(fragment_dir: Path) -> List[str]:
    """Return every fragment problem in *fragment_dir* as a message list."""
    problems: List[str] = []
    if not fragment_dir.is_dir():
        return problems
    for path in sorted(fragment_dir.iterdir()):
        if not path.is_file() or path.name == "README.md":
            continue
        if path.suffix != ".md":
            problems.append(f"{path.name}: fragments must be .md files")
            continue
        try:
            validate_fragment_text(path.name, read_text(path)[0])
        except FragmentError as exc:
            problems.append(str(exc))
    return problems


def find_unreleased_index(lines: Sequence[str]) -> int:
    """Index of the ``## [Unreleased...]`` heading line, or raise."""
    for index, line in enumerate(lines):
        if UNRELEASED_HEADING_RE.match(line):
            return index
    raise FragmentError(
        "CHANGELOG.md has no '## [Unreleased...]' heading; refusing to guess "
        "where entries go"
    )


def unreleased_is_empty(changelog_text: str) -> bool:
    """True when nothing stands between Unreleased and the next ``## ``."""
    lines = changelog_text.split("\n")
    for line in lines[find_unreleased_index(lines) + 1:]:
        if SECTION_HEADING_RE.match(line):
            return True
        if line.strip():
            return False
    return True


def insert_fragments(changelog_text: str, fragments: Iterable[Tuple[str, str]]) -> str:
    """Return *changelog_text* with *fragments* under the Unreleased heading.

    *fragments* is ``(name, text)`` already in the order they should appear.
    Everything outside the inserted block stays byte-identical.
    """
    rendered: List[str] = []
    for name, text in fragments:
        validate_fragment_text(name, text)
        rendered.append("")
        rendered.extend(text.replace("\r\n", "\n").strip("\n").split("\n"))
    lines = changelog_text.split("\n")
    index = find_unreleased_index(lines)
    if not rendered:
        return changelog_text
    # Keep one blank line between the last inserted entry and whatever already
    # followed the heading.
    if index + 1 < len(lines) and lines[index + 1].strip():
        rendered.append("")
    return "\n".join(lines[: index + 1] + rendered + lines[index + 1 :])


def cut_release(changelog_text: str, version: str, date: str) -> str:
    """Rename the Unreleased heading to *version* and open a fresh one above."""
    if not VERSION_RE.match(version):
        raise FragmentError(f"--release {version!r} is not a SemVer X.Y.Z version")
    if not DATE_RE.match(date):
        raise FragmentError(f"--date {date!r} is not YYYY-MM-DD")
    if unreleased_is_empty(changelog_text):
        raise FragmentError(
            f"the Unreleased section is empty; refusing to cut {version} from it. "
            f"Add the fragments this release contains first."
        )
    lines = changelog_text.split("\n")
    index = find_unreleased_index(lines)
    released = RELEASED_HEADING.format(version=version, date=date)
    replacement = [FRESH_UNRELEASED_HEADING, "", released]
    return "\n".join(lines[:index] + replacement + lines[index + 1 :])


def _diff(before: str, after: str, path: Path) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
    )


def run(
    changelog: Path,
    fragment_dir: Path,
    *,
    release: str = "",
    date: str = "",
    dry_run: bool = False,
) -> str:
    """Do the assembly. Returns the diff; writes unless *dry_run*."""
    before, newline = read_text(changelog)
    paths = iter_fragments(fragment_dir)
    loaded = [(path.name, read_text(path)[0]) for path in paths]
    after = insert_fragments(before, loaded)
    if release:
        after = cut_release(after, release, date or _today())
    diff = _diff(before, after, changelog)
    if dry_run:
        return diff
    if after != before:
        write_text(changelog, after, newline)
    for path in paths:
        path.unlink()
    return diff


def _today() -> str:
    return _datetime.date.today().isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
    parser.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--unreleased", action="store_true",
        help="move fragments under the Unreleased heading (the default)",
    )
    mode.add_argument(
        "--release", metavar="X.Y.Z", default="",
        help="also rename Unreleased to this version and open a fresh one",
    )
    parser.add_argument("--date", default="", help="release date, YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="print the diff, change nothing")
    parser.add_argument(
        "--check", action="store_true",
        help="validate fragment names and headings; exit 1 if any is malformed",
    )
    args = parser.parse_args(argv)

    if args.date and not args.release:
        parser.error("--date only means something with --release")

    problems = validate_all(args.fragments)
    if args.check:
        try:
            find_unreleased_index(read_text(args.changelog)[0].split("\n"))
        except FragmentError as exc:
            problems.append(str(exc))
        if problems:
            for problem in problems:
                print(f"changelog fragment: {problem}", file=sys.stderr)
            return 1
        count = len(iter_fragments(args.fragments))
        print(f"changelog fragments OK ({count} pending)")
        return 0
    if problems:
        for problem in problems:
            print(f"changelog fragment: {problem}", file=sys.stderr)
        return 1

    try:
        diff = run(
            args.changelog, args.fragments,
            release=args.release, date=args.date, dry_run=args.dry_run,
        )
    except FragmentError as exc:
        print(f"changelog fragment: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        sys.stdout.write(diff if diff else "(no fragments; CHANGELOG.md unchanged)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
