#!/usr/bin/env python3
"""Enforce the down-only line-count baseline for rfx Python sources.

The structure plan §11 puts file structure in the 2.0 bar.
The only way to add lines to a listed file is to remove at least as many from
it. Split the file instead of raising its baseline; comparisons use numbers,
never source text. A count is the unit "code-lines-without-docstrings": the
terminated lines that are not blank and do not start with "#" after stripping
(the byte rule of code_lines, CR and CRLF read as LF), minus the PEP 257
docstring lines that docstring_lines finds with ast. Other string contents
follow the byte rule. A file that does not parse fails the check.

A base baseline in the old unit "code-lines" starts the unit change: every
bound is the base number minus the docstring lines counted at the base, so no
file gains room (unit_change). A head baseline must carry the new unit. In
either case an entry the head removes while its file exists is refused unless
the cap then governing the file is not above the base entry.

BASE_SHA/HEAD_SHA (or --base/--head) select Git snapshots. Without HEAD_SHA,
check the working tree, including uncommitted edits. An absent baseline at
BASE_SHA permits the initial adoption; an unreadable revision fails closed.
The float64 invariant's PENDING_CEILING follows the same numeric down-only rule.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tokenize

REPO = Path(__file__).resolve().parents[2]
BASELINE = "scripts/ci/file_size_baseline.json"
PENDING = "tests/contracts/float64_pending.py"
SPLIT_ADVICE = "; split the file rather than raise the number."
UNIT = "code-lines-without-docstrings"
OLD_UNIT = "code-lines"
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, input=stdin,
    ).stdout


def code_lines(data: bytes) -> int:
    # Preserve wc -l's unterminated-tail behavior and treat CR like LF.
    lines = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")[:-1]
    return sum(bool(line.strip()) and not line.strip().startswith(b"#") for line in lines)


def docstring_lines(data: bytes, path: str) -> set[int]:
    """Return the 1-based line numbers that PEP 257 docstrings occupy in ``data``.

    A docstring is the first statement of a module, class, def or async def body
    when it is ``Expr(Constant(str))``, at any nesting depth. Its lines run from
    ``lineno`` to ``end_lineno``, except a first line with non-whitespace before
    ``col_offset`` and a last line whose stripped text after ``end_col_offset``
    is non-empty and does not start with "#". The file is decoded with
    ``tokenize.detect_encoding`` (BOM and coding cookie) and split on LF, CRLF
    and CR only, the terminators ``ast`` and ``code_lines`` use; the offsets
    index the UTF-8 bytes of each decoded line. A file that does not parse
    raises ValueError naming ``path``.
    """
    try:
        tree = ast.parse(data)
        encoding = tokenize.detect_encoding(io.BytesIO(data).readline)[0]
        text = data.decode(encoding)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}: does not parse, so its docstrings cannot be found") from exc
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        start, end = first.lineno, first.end_lineno
        if lines[start - 1].encode("utf-8")[:first.col_offset].strip():
            start += 1
        tail = lines[end - 1].encode("utf-8")[first.end_col_offset:].strip()
        if tail and not tail.startswith(b"#"):
            end -= 1
        found.update(range(start, end + 1))
    return found


def ratchet_lines(data: bytes, path: str) -> int:
    """Count the lines ``code_lines`` counts that ``docstring_lines`` does not return.

    The docstring lines mask line numbers, so a docstring line that starts with
    "#" is left out once, never subtracted twice.
    """
    skip = docstring_lines(data, path)
    lines = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")[:-1]
    return sum(bool(line.strip()) and not line.strip().startswith(b"#") and number not in skip
               for number, line in enumerate(lines, 1))


def docstring_code_lines(data: bytes, path: str) -> int:
    """Count the lines ``code_lines`` counts that ``docstring_lines`` returns.

    The ratchet_lines set is a subset of the code_lines set, so the difference
    of the two counts is the size of their intersection.
    """
    return code_lines(data) - ratchet_lines(data, path)


def base_docstring_lines(repo: Path, base: str) -> dict[str, int]:
    """Map every ``rfx/**/*.py`` blob at ``base`` to its docstring_code_lines count.

    The blobs are read in one ``git cat-file --batch`` call.
    """
    entries = []
    for record in git(repo, "ls-tree", "-r", "-z", base, "--", "rfx/").split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        kind, sha = meta.split()[1:]
        if kind == b"blob" and name.endswith(b".py"):
            entries.append((name.decode(), sha))
    if not entries:
        return {}
    out = git(repo, "cat-file", "--batch", stdin=b"".join(sha + b"\n" for _, sha in entries))
    counts, offset = {}, 0
    for name, sha in entries:
        header_end = out.index(b"\n", offset)
        size = int(out[offset:header_end].split()[2])
        start = header_end + 1
        counts[name] = docstring_code_lines(out[start:start + size], f"{base}:{name}")
        offset = start + size + 1
    return counts


def unit_change(repo: Path, base: str, previous: dict, current: dict,
                counts: dict[str, int]) -> list[str]:
    """Check a head baseline in UNIT against a base baseline in OLD_UNIT.

    d(P) is docstring_code_lines of P read at the base; a file absent at the
    base has d = 0. The bounds:
    - T1: an entry listed on both sides is at most its base value minus d(P).
    - T2: the head cap is at most the base cap minus the largest d(P) over the
      base files the base does not list (0 when there is none).
    - T3: a new entry names a file present at the base and at the head, whose
      head count is above the head cap, at most the base cap minus d(P).
      Rename and package-split inheritance does not apply.
    - T4: an entry the head removes needs its file absent at the head, or a head
      cap at most its base value minus d(P).
    """
    docs = base_docstring_lines(repo, base)
    failures = []
    old_cap, cap = previous["cap"], current["cap"]
    unlisted = sorted((value, path) for path, value in docs.items()
                      if path not in previous["files"])
    if unlisted:
        largest = max(value for value, _ in unlisted)
        widest = min(path for value, path in unlisted if value == largest)
        if cap > old_cap - largest:
            failures.append(
                f"{BASELINE}: cap {cap}; it must be at most {old_cap - largest} "
                f"({old_cap} minus {largest}, the docstring lines of {widest} at the base)."
            )
    elif cap > old_cap:
        failures.append(
            f"{BASELINE}: cap {cap}; it must be at most {old_cap} "
            f"({old_cap} minus 0: the base lists every file)."
        )
    for path, value in sorted(current["files"].items()):
        d = docs.get(path, 0)
        if path in previous["files"]:
            old = previous["files"][path]
            if value > old - d:
                failures.append(
                    f"{path}: baseline {value}; with docstrings not counted it must be at most "
                    f"{old - d} ({old} minus {d} docstring lines at the base)."
                )
            continue
        if path not in docs:
            failures.append(f"{path}: new entry {value}, but the file is absent at the base; "
                            "a new entry in the unit change needs a file the base holds.")
            continue
        if path not in counts:
            failures.append(f"{path}: new entry {value}, but the file is absent at the head.")
            continue
        if counts[path] <= cap:
            failures.append(f"{path}: new entry {value}, but its count {counts[path]} is within "
                            f"the cap {cap}; leave it unlisted.")
        if value > old_cap - d:
            failures.append(
                f"{path}: baseline {value}; with docstrings not counted it must be at most "
                f"{old_cap - d} (cap {old_cap} minus {d} docstring lines at the base)."
            )
    for path, old in sorted(previous["files"].items()):
        d = docs.get(path, 0)
        if path not in current["files"] and path in counts and cap > old - d:
            failures.append(
                f"{path}: entry removed while the file exists; with docstrings not counted the "
                f"cap {cap} must be at most {old - d} ({old} minus {d} docstring lines at the "
                "base) to govern it, so keep the entry."
            )
    return failures


def baseline(raw: bytes) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) not in ({"cap", "files"}, {"cap", "files", "counts"}):
        raise ValueError("baseline must contain cap and files, with optional counts")
    if "counts" in data and data["counts"] not in (OLD_UNIT, UNIT):
        raise ValueError(f"counts must be {UNIT} or {OLD_UNIT}")
    if type(data["cap"]) is not int or data["cap"] < 0:
        raise ValueError("cap must be a nonnegative integer")
    if not isinstance(data["files"], dict):
        raise ValueError("files must be a mapping")
    for path, value in data["files"].items():
        if (not path.startswith("rfx/") or not path.endswith(".py")
                or ".." in Path(path).parts
                or type(value) is not int or value < 0):
            raise ValueError(f"invalid baseline entry: {path}")
    return data


def pending_ceiling(repo: Path, revision: str = "") -> int | None:
    if revision:
        if not git(repo, "ls-tree", revision, "--", PENDING):
            return None
        raw = git(repo, "show", f"{revision}:{PENDING}")
    else:
        if not (repo / PENDING).exists():
            return None
        raw = (repo / PENDING).read_bytes()
    try:
        values = [node.value for node in ast.parse(raw).body
                  if isinstance(node, (ast.Assign, ast.AnnAssign))
                  and any(isinstance(target, ast.Name) and target.id == "PENDING_CEILING"
                          for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    except SyntaxError as exc:
        raise ValueError(f"{PENDING}: unreadable PENDING_CEILING") from exc
    if (len(values) != 1 or not isinstance(values[0], ast.Constant)
            or type(values[0].value) is not int or values[0].value < 0):
        raise ValueError(f"{PENDING}: PENDING_CEILING must be a nonnegative integer literal")
    return values[0].value


def check(repo: Path, base: str = "", head: str = "") -> int:
    read = (lambda path: git(repo, "show", f"{head}:{path}")) if head else (
        lambda path: (repo / path).read_bytes()
    )
    current = baseline(read(BASELINE))
    if current.get("counts") != UNIT:
        raise ValueError(f"current baseline counts must be {UNIT}")
    if head:
        paths = [p.decode() for p in git(
            repo, "ls-tree", "-r", "--name-only", "-z", head, "--", "rfx/",
        ).split(b"\0") if p.endswith(b".py")]
    else:
        paths = [p.relative_to(repo).as_posix() for p in (repo / "rfx").rglob("*.py")]
    counts = {path: ratchet_lines(read(path), path) for path in sorted(paths)}
    failures = []
    ceiling = pending_ceiling(repo, head)
    if base:
        old_ceiling = pending_ceiling(repo, base)
        if old_ceiling is not None and (ceiling is None or ceiling > old_ceiling):
            failures.append(
                f"{PENDING}: PENDING_CEILING baseline {old_ceiling}, count {ceiling}; "
                "remove fixed entries rather than raise the ceiling."
            )
    transition = False
    if base:
        # ls-tree fails for a bad revision but returns empty for an absent file.
        if git(repo, "ls-tree", base, "--", BASELINE):
            previous = baseline(git(repo, "show", f"{base}:{BASELINE}"))
            transition = "counts" not in previous
            if previous.get("counts") == OLD_UNIT:
                failures.extend(unit_change(repo, base, previous, current, counts))
            else:
                gone = previous["files"].keys() - set(paths) - current["files"].keys()
                renames = {}
                if gone and not transition and current["files"].keys() - previous["files"].keys():
                    revisions = (base, head) if head else (base,)
                    changes = git(
                        repo, "diff", "-M90%", "--name-status", "--diff-filter=R", "-z",
                        *revisions, "--", "rfx/",
                    ).split(b"\0")
                    for index in range(0, len(changes) - 1, 3):
                        renames[changes[index + 2].decode()] = changes[index + 1].decode()
                if current["cap"] > previous["cap"]:
                    failures.append(
                        f"{BASELINE}: cap baseline {previous['cap']}, count {current['cap']}"
                        + SPLIT_ADVICE
                    )
                for path, value in current["files"].items():
                    old = previous["files"].get(path, previous["cap"] if transition else None)
                    if old is None and path in paths:
                        source = (path.removesuffix("/__init__.py") + ".py"
                                  if path.endswith("/__init__.py") else None)
                        if source not in gone:
                            source = renames.get(path)
                        if source in gone:
                            old = previous["files"][source]
                            gone.remove(source)
                    if old is None or value > old:
                        failures.append(
                            f"{path}: baseline {old if old is not None else 'unlisted'}, "
                            f"count {value} in proposed baseline" + SPLIT_ADVICE
                        )
                for path in sorted(previous["files"].keys() - current["files"].keys()):
                    if path in counts and previous["files"][path] < current["cap"]:
                        failures.append(
                            f"{path}: delisting gives the file the cap of {current['cap']} "
                            f"instead of its baseline {previous['files'][path]}; keep the entry, "
                            "or delete the file."
                        )
    if transition:
        for path in sorted(current["files"].keys() - set(paths)):
            failures.append(f"{path}: transition baseline entry has no source file")
    for path, count in counts.items():
        limit = current["files"].get(path, current["cap"])
        if transition and path in current["files"] and count != limit:
            failures.append(
                f"{path}: transition baseline {limit} must equal actual count {count}; "
                f"set the entry to {count}."
            )
        elif count > limit:
            failures.append(f"{path}: baseline {limit}, count {count}" + SPLIT_ADVICE)
        elif path in current["files"] and count < limit:
            print(f"lower {path} to {count} in {BASELINE}", file=sys.stderr)
    for failure in failures:
        print(failure, file=sys.stderr)
    return int(bool(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=os.environ.get("BASE_SHA", ""))
    parser.add_argument("--head", default=os.environ.get("HEAD_SHA", ""))
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    try:
        return check(args.repo, args.base, args.head)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"file-size ratchet failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
