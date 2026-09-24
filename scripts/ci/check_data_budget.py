#!/usr/bin/env python3
"""CI gate: a pull request carries code, tests and the frozen data a test reads.

The repository is public and its history keeps whatever lands. Measurement
records -- sweep and study JSON, harvested VESSL logs, the measurement of an
option that was not adopted -- go to ``bk-squared/rfx-archive`` under
``rfx/records/<YYYYMMDD>-<topic>/``, and the PR body names that path and the
archive commit (PI, 2026-09-24). Over the 14 days to 2026-09-24, ``rfx/``
changed by +39,849/-37,028 lines on main while JSON alone changed by 3.78
million, and a reviewer cannot read a diff of that shape.

Six rules over the PR's diff against its merge base, all mechanical:

1. Data files added or modified OUTSIDE ``ALLOWLIST`` may add at most
   ``LINE_BUDGET`` lines and ``TEXT_BYTE_BUDGET`` bytes of text in total (a
   whole added file, and the growth of a modified one: ten one-line JSON files
   of 0.95 MB each are ten lines), and binary ones at most ``BINARY_BUDGET``
   bytes. Deletions are free: moving records out is the point.
2. No data file the PR adds or modifies may exceed ``FILE_CAP`` bytes, on the
   allowlist or off it, except a file in a frozen-data home that a reader names
   (rule 3's test), which may reach ``FROZEN_FILE_CAP``. Four fixtures tests
   read, added 2026-09-10..24, were 1.04-1.81 MB; at one cap for everything
   they would have needed the exception label, and a label needed for routine
   work stops meaning anything. ``SIZE_EXEMPT`` names the one exception.
3. A data file newly placed in a frozen-data home -- ``tests/fixtures/``,
   ``tests/data/``, ``tests/crossval/<case>/reference/`` -- must be named by a
   reader (``is_reader``): a tracked ``test_*.py``, ``conftest.py`` or ``_*.py``
   helper under ``tests/``, or any ``.py`` under ``rfx/``; never a file inside a
   frozen-data home, and never a test of this gate. Named means its file name, or its name without its data
   suffixes (``f"{case}.json"`` readers), appears in the reader as a whole
   name; or, when it sits in a subdirectory of its home, that subdirectory's
   name is a path component of a string literal in a reader (glob readers).
   The two shorter names count only when ``_distinctive``. A README or an
   ``_index.py`` beside the records, or this gate's own tests, cannot vouch
   for them.
4. The data files a PR adds or modifies in the frozen-data homes hold at most
   ``FROZEN_BUDGET`` bytes in total. The allowlist lets those homes grow past
   rule 1, and a name is text a record can borrow; this bounds what borrowing
   buys. The largest legitimate PR in the 14-day window was 2,338,905 bytes.
5. A PR that adds or modifies a ``.gitattributes`` fails: ``validation/** -diff``
   turns a 9,000-line CSV into a 196 kB binary that passes rule 1.
6. A PR labelled ``data-budget-exception`` passes whatever it carries, with a
   warning annotation and the would-be failures in the step summary. The
   weekly ``scripts/ci/governance_audit.py`` lists every merged PR that
   carried the label.

What a "data file" is: ``DATA_SUFFIXES`` by the last suffix, Touchstone
``.sNp``, the names in ``DATA_NAMES``, and any file git calls binary that is
not an image (``IMAGE_SUFFIXES``). Git decides text or binary (numstat ``-``),
not the suffix. Markdown and YAML are text and never data here.

Inputs: ``--base``/``--head`` shas (the workflow passes the pull request's
``base.sha`` and ``head.sha``) and ``PR_LABELS_JSON``, the PR's labels as a
JSON array, which the workflow reads from the API at run time so that a label
applied after the push counts on a re-run. With no ``--base`` on a push event
there is nothing to compare and the check says so. Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG_CHECK = REPO_ROOT / "scripts" / "ci" / "check_changelog_fragment.py"

LINE_BUDGET = 500
BINARY_BUDGET = 200_000  # bytes
TEXT_BYTE_BUDGET = 200_000  # bytes
FILE_CAP = 1_000_000  # bytes
FROZEN_FILE_CAP = 5_000_000  # bytes, a named file in a frozen-data home
FROZEN_BUDGET = 5_000_000  # bytes per PR across the frozen-data homes
EXCEPTION_LABEL = "data-budget-exception"
ARCHIVE_REPO = "bk-squared/rfx-archive"
ARCHIVE_PATH = "rfx/records/<YYYYMMDD>-<topic>/"

#: The formats the PI's rule names, plus the ones measured in the tree or in
#: the 222 PRs merged 2026-09-10..24 (``.gz`` VESSL log bundles, ``.xml``
#: pytest reports, ``.started``/``.exit`` run markers, ``.msh`` meshes), plus
#: other spellings of the same formats.
DATA_SUFFIXES = frozenset({
    ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".log", ".txt", ".out",
    ".rc", ".exit", ".started", ".dat", ".xml", ".msh",
    ".npz", ".npy", ".h5", ".hdf5", ".pkl", ".pickle", ".mat",
    ".gz", ".bz2", ".xz", ".zip", ".tar",
})
TOUCHSTONE_RE = re.compile(r"\.s\d+p\Z")
#: Data by content, with no data suffix. pytest-split's JSON file.
DATA_NAMES = frozenset({".test_durations"})
#: Binary files that are figures, not data: never counted.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf"})

#: Data paths that change as part of ordinary code work, and the evidence:
#: PRs among the 222 merged 2026-09-10..24 that touched the path, and the
#: lines they added. Nothing else changed a data path in more than two PRs
#: except records under validation/, scripts/diagnostics/ and docs/.
ALLOWLIST: Tuple[Tuple[str, str], ...] = (
    (".test_durations", "pytest-split durations, regenerated by hand (3 PRs, 18,559 lines)"),
    ("docs/guides/api_symbol_inventory.json", "regenerated API inventory (20 PRs, 160 lines)"),
    ("docs/guides/*support_matrix*.json", "support matrices (14 PRs, 49 lines)"),
    ("validation/crossval/manifest.json", "cross-validation manifest (22 PRs, 170 lines)"),
    ("docs/design_notes/schemas/*.schema.json", "design IR schema tests/interop reads (5 PRs, 161 lines)"),
    ("docs/public/site_map.json", "public docs map scripts/check_public_docs_manifest.py reads (1 PR, 1 line)"),
    ("docs/public/gallery/assets/**", "gallery precompute outputs (1 PR, 1 line)"),
    ("scripts/ops/gpu_suite_shards.json", "rendered GPU shard list (2 PRs, 1 line)"),
    ("tests/contracts/boundary_registry.json", "registry tests/contracts reads (1 PR, 1,041 lines)"),
    ("tests/fixtures/**", "frozen reference data, rule 3 applies (34 PRs, 324,083 lines)"),
    ("tests/data/**", "frozen snapshots, rule 3 applies (34 PRs, 8,094 lines)"),
    ("tests/crossval/*/reference/**", "frozen external-solver data, rule 3 applies (5 PRs, 117,199 lines)"),
)

#: Where frozen reference data lives, and the home root of each path.
FROZEN_HOME_RE = re.compile(
    r"(?P<home>tests/fixtures/|tests/data/|tests/crossval/[^/]+/reference/)(?P<rest>.+)\Z"
)

#: Files the size cap does not apply to, with the reason.
SIZE_EXEMPT = {
    ".test_durations": "1.14 MB; pytest-split reads the whole suite's durations from one file",
}

#: Where the readers for rule 3 are looked for; ``is_reader`` says which files.
READER_ROOTS = ("tests", "rfx")
#: A ``.py`` file that names this gate is a test of it, not a reader of data:
#: its fixtures name ``orphan_case``, ``sweep``, ``point_07`` and so on.
GATE_NAME = "check_data_budget"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Imported, not reimplemented: both gates read PR_LABELS_JSON the same way.
parse_labels = _load(_CHANGELOG_CHECK, "rfx_check_changelog_fragment").parse_labels


def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """``**`` crosses directories, ``*`` does not. Anchored at both ends."""
    out: List[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif pattern[index] == "*":
            out.append("[^/]*")
            index += 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(out) + r"\Z")


_ALLOW_RES = tuple(_glob_regex(pattern) for pattern, _ in ALLOWLIST)


def is_data(path: str) -> bool:
    name = PurePosixPath(path).name
    if name in DATA_NAMES:
        return True
    suffix = PurePosixPath(name).suffix.lower()
    return suffix in DATA_SUFFIXES or TOUCHSTONE_RE.search(suffix) is not None


def counted(path: str, binary: bool) -> bool:
    """Whether the budget sees this file: a data file, or a non-image binary."""
    if is_data(path):
        return True
    return binary and PurePosixPath(path).suffix.lower() not in IMAGE_SUFFIXES


def allowlisted(path: str) -> bool:
    return any(regex.match(path) for regex in _ALLOW_RES)


class Change(NamedTuple):
    """One data file the PR adds or modifies."""

    path: str
    status: str  # A, M, R, C or T (the destination of a rename or copy)
    added: Optional[int]  # None when git calls it binary
    size: int  # bytes at head
    grown: int  # bytes added: the whole file if new, else max(0, head - base)


class Findings(NamedTuple):
    changes: List[Change]
    outside_text: List[Change]  # text data outside the allowlist
    outside_binary: List[Change]  # binary data outside the allowlist
    over_cap: List[Change]
    unreferenced: List[Change]
    new_frozen: List[Change]
    named: FrozenSet[str]  # frozen-data paths a reader names, among those asked about
    frozen: List[Change]  # every data file the PR adds or modifies in a frozen home
    gitattributes: List[str]  # .gitattributes files the PR adds or modifies

    @property
    def line_total(self) -> int:
        return sum(change.added or 0 for change in self.outside_text)

    @property
    def text_byte_total(self) -> int:
        return sum(change.grown for change in self.outside_text)

    @property
    def binary_total(self) -> int:
        return sum(change.size for change in self.outside_binary)

    @property
    def frozen_total(self) -> int:
        return sum(change.size for change in self.frozen)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, check=True
    ).stdout


def _split_z(raw: bytes) -> List[str]:
    return raw.decode("utf-8", "surrogateescape").split("\0")


def _statuses(base: str, head: str, repo: Path) -> Dict[str, Tuple[str, str]]:
    """Post-image path -> (A/M/R/C/T, pre-image path) for everything present at head."""
    tokens = _split_z(_git(repo, "diff", "--name-status", "-z", "--find-renames",
                           f"{base}...{head}"))
    out: Dict[str, Tuple[str, str]] = {}
    index = 0
    while index < len(tokens):
        status = tokens[index]
        if not status:
            index += 1
            continue
        if status[0] in "RC":
            out[tokens[index + 2]] = (status[0], tokens[index + 1])
            index += 3
        else:
            if status[0] != "D":
                out[tokens[index + 1]] = (status[0], tokens[index + 1])
            index += 2
    return out


def _added_lines(base: str, head: str, repo: Path) -> Dict[str, Optional[int]]:
    """Post-image path -> lines added, or None for a binary file."""
    tokens = _split_z(_git(repo, "diff", "--numstat", "-z", "--find-renames",
                           f"{base}...{head}"))
    out: Dict[str, Optional[int]] = {}
    index = 0
    while index < len(tokens):
        record = tokens[index]
        if not record:
            index += 1
            continue
        added, _deleted, path = record.split("\t", 2)
        if path:
            index += 1
        else:  # a rename: "added\tdeleted\t\0old\0new\0"
            path = tokens[index + 2]
            index += 3
        out[path] = None if added == "-" else int(added)
    return out


def _sizes(head: str, repo: Path) -> Dict[str, int]:
    """Blob size in bytes of every file at head."""
    out: Dict[str, int] = {}
    for record in _split_z(_git(repo, "ls-tree", "-r", "-l", "-z", head)):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        size = meta.split()[3]
        if size != "-":
            out[path] = int(size)
    return out


def is_reader(path: str) -> bool:
    """Whether the tracked file at *path* can vouch for frozen data (rule 3).

    A test module, a conftest or a ``_``-prefixed helper under ``tests/``, or
    any module under ``rfx/``. Nothing inside a frozen-data home: a two-line
    ``_index.py`` beside the records named them for a reviewer's stacked
    relocation of #1198.
    """
    if not path.endswith(".py") or FROZEN_HOME_RE.match(path):
        return False
    if path.startswith("rfx/"):
        return True
    if not path.startswith("tests/"):
        return False
    name = PurePosixPath(path).name
    return name.startswith("test_") or name.startswith("_") or name == "conftest.py"


def reader_sources(head: str, repo: Path) -> List[str]:
    """The source of every reader at head (``is_reader``).

    A test of this gate is left out (it names ``GATE_NAME``): its fixtures are
    names for synthetic records, and on the gate's own repository they would
    vouch for any real record that happened to share one.
    """
    oids: List[str] = []
    for record in _split_z(_git(repo, "ls-tree", "-r", "-z", head, "--", *READER_ROOTS)):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        _mode, kind, oid = meta.split()
        if kind == "blob" and is_reader(path):
            oids.append(oid)
    if not oids:
        return []
    raw = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=repo, input="\n".join(oids).encode() + b"\n",
        capture_output=True, check=True,
    ).stdout
    sources: List[str] = []
    offset = 0
    while offset < len(raw):
        header_end = raw.index(b"\n", offset)
        size = int(raw[offset:header_end].split()[2])
        start = header_end + 1
        source = raw[start:start + size].decode("utf-8", "replace")
        if GATE_NAME not in source:
            sources.append(source)
        offset = start + size + 1
    return sources


def literal_components(sources: Iterable[str]) -> FrozenSet[str]:
    """Every path component of every string literal in *sources*, docstrings excepted.

    ``FIX / "sweep" / name`` and ``"tests/fixtures/sweep/"`` both yield
    ``sweep``; a comment, a docstring or ``"my_sweep_dir"`` does not.
    """
    components = set()
    for source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        docstrings = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings:
                components.update(node.value.replace("\\", "/").split("/"))
    return frozenset(components)


def _data_stem(name: str) -> str:
    """*name* without its trailing data suffixes: ``a.json.gz`` -> ``a``.

    Only data suffixes come off, so ``0.01_16.json`` keeps ``0.01_16`` rather
    than collapsing to ``0``, which every file in the repository names.
    """
    stem = name
    while "." in stem.lstrip(".") and is_data(stem):
        stem = stem[:stem.rindex(".")]
    return stem


def _distinctive(token: str) -> bool:
    """A shorter name can vouch for a file only if it could name one thing."""
    return len(token) >= 3 and any(char.isalpha() for char in token)


def reference_tokens(path: str) -> List[str]:
    """The names a reader may use for a frozen-data file (rule 3).

    The file name, then its name without data suffixes; the last entry is the
    subdirectory when ``folder_token`` returns one.
    """
    match = FROZEN_HOME_RE.match(path)
    rest = PurePosixPath(match.group("rest") if match else path)
    tokens = [rest.name]
    stem = _data_stem(rest.name)
    if stem != rest.name and _distinctive(stem):
        tokens.append(stem)
    folder = folder_token(path)
    if folder:
        tokens.append(folder)
    return tokens


def folder_token(path: str) -> str:
    """The subdirectory of its home a frozen file sits in, or ``""``."""
    match = FROZEN_HOME_RE.match(path)
    if not match:
        return ""
    rest = PurePosixPath(match.group("rest"))
    if len(rest.parts) > 1 and _distinctive(rest.parent.name):
        return rest.parent.name
    return ""


class Readers(NamedTuple):
    text: str  # every reader's source, joined
    components: FrozenSet[str]  # path components of their string literals

    @classmethod
    def at(cls, head: str, repo: Path) -> "Readers":
        sources = reader_sources(head, repo)
        return cls("\n\0\n".join(sources), literal_components(sources))

    def name(self, path: str) -> bool:
        """Whether a reader names the frozen-data file at *path* (rule 3)."""
        folder = folder_token(path)
        names = reference_tokens(path)
        if folder:
            names = names[:-1]
        if any(is_named(token, self.text) for token in names):
            return True
        return bool(folder) and folder in self.components


def _name_char(char: str) -> bool:
    return char.isalnum() or char in "_-"


def is_named(token: str, text: str) -> bool:
    """*token* appears in *text* and is not part of a longer name.

    ``str.find`` rather than a regex: the reader text is about 15 MB, and a
    regex scan per token cost 0.3 s where this costs milliseconds.
    """
    start = text.find(token)
    while start != -1:
        end = start + len(token)
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if not (before and (_name_char(before) or before == ".")) and not (
            after and _name_char(after)
        ):
            return True
        start = text.find(token, start + 1)
    return False


def size_cap(path: str, named: FrozenSet[str]) -> int:
    """The largest a data file at *path* may be (rule 2)."""
    if FROZEN_HOME_RE.match(path) and path in named:
        return FROZEN_FILE_CAP
    return FILE_CAP


def data_changes(base: str, head: str, repo: Path,
                 statuses: Optional[Dict[str, Tuple[str, str]]] = None) -> List[Change]:
    """Every counted file the diff ``base...head`` adds or modifies."""
    if statuses is None:
        statuses = _statuses(base, head, repo)
    added = _added_lines(base, head, repo)
    sizes = _sizes(head, repo)
    merge_base = _git(repo, "merge-base", base, head).decode().strip()
    base_sizes = _sizes(merge_base, repo)

    changes = []
    for path, (status, source) in sorted(statuses.items()):
        if path not in sizes or not counted(path, added.get(path, 0) is None):
            continue
        before = base_sizes.get(source, 0) if status != "A" else 0
        changes.append(Change(path, status, added.get(path), sizes[path],
                              max(0, sizes[path] - before)))
    return changes


def frozen_bytes(base: str, head: str, repo: Path) -> int:
    """Bytes of frozen-home data the diff adds or modifies (rule 4's measure)."""
    return sum(change.size for change in data_changes(base, head, repo)
               if FROZEN_HOME_RE.match(change.path))


def frozen_on(rev: str, repo: Path) -> Tuple[int, int]:
    """``(files, bytes)`` of data in the frozen-data homes at *rev*."""
    sizes = _sizes(rev, repo)
    held = [size for path, size in sizes.items()
            if FROZEN_HOME_RE.match(path) and is_data(path)]
    return len(held), sum(held)


def evaluate(base: str, head: str, repo: Path) -> Findings:
    """What the PR's data changes are, sorted into the rules' buckets."""
    statuses = _statuses(base, head, repo)
    changes = data_changes(base, head, repo, statuses)
    outside = [change for change in changes if not allowlisted(change.path)]
    outside_text = [c for c in outside if c.added is not None]
    outside_binary = [c for c in outside if c.added is None]
    frozen = [change for change in changes if FROZEN_HOME_RE.match(change.path)]
    gitattributes = [path for path in sorted(statuses)
                     if PurePosixPath(path).name == ".gitattributes"]
    new_frozen = [
        change for change in changes
        if change.status in ("A", "R", "C") and FROZEN_HOME_RE.match(change.path)
    ]
    # A reader is looked for on every new frozen file (rule 3) and on every
    # frozen file over FILE_CAP, new or edited (rule 2's larger cap).
    asked = new_frozen + [
        change for change in changes
        if change.size > FILE_CAP and FROZEN_HOME_RE.match(change.path)
        and change not in new_frozen
    ]
    named: FrozenSet[str] = frozenset()
    if asked:
        readers = Readers.at(head, repo)
        named = frozenset(change.path for change in asked if readers.name(change.path))
    unreferenced = [change for change in new_frozen if change.path not in named]
    over_cap = [
        change for change in changes
        if change.path not in SIZE_EXEMPT and change.size > size_cap(change.path, named)
    ]
    return Findings(changes, outside_text, outside_binary, over_cap, unreferenced,
                    new_frozen, named, frozen, gitattributes)


def _listing(changes: Iterable[Change], value, unit: str, limit: int = 15) -> List[str]:
    rows = sorted(changes, key=lambda change: -value(change))
    lines = [f"      {value(change):>10,} {unit}  {change.path}" for change in rows[:limit]]
    if len(rows) > limit:
        rest = rows[limit:]
        lines.append(f"      ... and {len(rest)} more files, "
                     f"{sum(value(change) for change in rest):,} {unit}")
    return lines


def failures(findings: Findings) -> List[str]:
    """Return the failure messages; empty means the PR passes."""
    out: List[str] = []
    if findings.line_total > LINE_BUDGET:
        out.append("\n".join([
            f"data files outside the allowlist add {findings.line_total:,} lines "
            f"(budget {LINE_BUDGET}):",
            *_listing(findings.outside_text, lambda change: change.added or 0, "lines"),
        ]))
    if findings.text_byte_total > TEXT_BYTE_BUDGET:
        out.append("\n".join([
            f"text data files outside the allowlist add {findings.text_byte_total:,} "
            f"bytes (budget {TEXT_BYTE_BUDGET:,}):",
            *_listing(findings.outside_text, lambda change: change.grown, "bytes"),
        ]))
    if findings.binary_total > BINARY_BUDGET:
        out.append("\n".join([
            f"binary data files outside the allowlist hold {findings.binary_total:,} "
            f"bytes (budget {BINARY_BUDGET:,}):",
            *_listing(findings.outside_binary, lambda change: change.size, "bytes"),
        ]))
    if findings.over_cap:
        out.append("\n".join([
            f"{len(findings.over_cap)} data file(s) over their size cap ({FILE_CAP:,} "
            f"bytes; {FROZEN_FILE_CAP:,} for a file in a frozen-data home that a "
            f"reader names):",
            *[f"      {change.size:>10,} bytes (cap "
              f"{size_cap(change.path, findings.named):,})  {change.path}"
              for change in sorted(findings.over_cap, key=lambda c: -c.size)[:15]],
            *([f"      ... and {len(findings.over_cap) - 15} more"]
              if len(findings.over_cap) > 15 else []),
        ]))
    if findings.unreferenced:
        out.append("\n".join([
            f"{len(findings.unreferenced)} new frozen-data file(s) that no tracked "
            f"file under {' or '.join(root + '/' for root in READER_ROOTS)} names "
            f"in a .py reader (by file name, name without its data suffixes, or "
            f"subdirectory in a string literal):",
            *[f"      {change.path}" for change in findings.unreferenced[:15]],
            *([f"      ... and {len(findings.unreferenced) - 15} more"]
              if len(findings.unreferenced) > 15 else []),
        ]))
    if findings.frozen_total > FROZEN_BUDGET:
        out.append("\n".join([
            f"data files in the frozen-data homes hold {findings.frozen_total:,} bytes "
            f"(budget {FROZEN_BUDGET:,} per PR):",
            *_listing(findings.frozen, lambda change: change.size, "bytes"),
        ]))
    if findings.gitattributes:
        out.append("\n".join([
            "this PR adds or edits a .gitattributes file. An attribute such as "
            "`-diff` or an LFS filter changes what git counts, so this gate cannot "
            "measure the PR:",
            *[f"      {path}" for path in findings.gitattributes],
        ]))
    return out


WHERE_IT_GOES = f"""\
Where the data goes:
  Measurement records -- sweep and study JSON, harvested VESSL logs, the
  measurement of an option that was not adopted -- go to
  {ARCHIVE_REPO} under {ARCHIVE_PATH}.
  Name that path and the archive commit in the PR body.
  tests/fixtures/, tests/data/ and tests/crossval/<case>/reference/ hold
  only frozen data that a test reads; moving a record there does not make
  it one, and they take at most {FROZEN_BUDGET:,} bytes per PR.
  If this PR has to carry the data anyway, apply the label
  '{EXCEPTION_LABEL}' and re-run the failed job (no push needed: the
  labels are read when the job runs). The weekly governance audit lists
  every merged PR that carried the label."""


def _summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", default="", help="pull request base sha")
    parser.add_argument("--head", default="", help="pull request head sha")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    if not args.base or not args.head:
        event = os.environ.get("GITHUB_EVENT_NAME", "")
        if event and event != "pull_request":
            print(f"data budget: a {event} event has no pull request to measure; "
                  f"nothing to check.")
            return 0
        print("data budget check failed: --base and --head are both required",
              file=sys.stderr)
        return 1
    try:
        labels = parse_labels(os.environ.get("PR_LABELS_JSON", ""))
    except ValueError as exc:
        print(f"data budget check failed: {exc}", file=sys.stderr)
        return 1
    try:
        findings = evaluate(args.base, args.head, args.repo)
    except subprocess.CalledProcessError:
        print(f"data budget check failed: cannot diff {args.base}...{args.head} in "
              f"{args.repo}. Both commits must be present -- check out with "
              f"fetch-depth: 0.", file=sys.stderr)
        return 1

    found = failures(findings)
    if found and EXCEPTION_LABEL in labels:
        notice = (f"This PR carries the '{EXCEPTION_LABEL}' label, so the data "
                  f"budget was NOT enforced. Without the label it would fail:")
        print(f"::warning title=data budget not enforced::{notice} "
              f"{len(found)} rule(s); see the step summary.")
        print(notice)
        for failure in found:
            print(f"- {failure}")
        _summary("\n".join([
            "## Data budget NOT enforced: `" + EXCEPTION_LABEL + "`",
            "",
            notice,
            "",
            "```",
            *[f"- {failure}" for failure in found],
            "```",
        ]))
        return 0
    if found:
        print("data budget check failed:", file=sys.stderr)
        for failure in found:
            print(f"- {failure}", file=sys.stderr)
        print(WHERE_IT_GOES, file=sys.stderr)
        _summary("\n".join([
            "## Data budget failed", "", "```",
            *[f"- {failure}" for failure in found],
            "", WHERE_IT_GOES, "```",
        ]))
        return 1
    print(f"data budget ok: outside the allowlist {findings.line_total:,} lines "
          f"(budget {LINE_BUDGET}) and {findings.text_byte_total:,} bytes of text "
          f"(budget {TEXT_BYTE_BUDGET:,}), {findings.binary_total:,} bytes binary "
          f"(budget {BINARY_BUDGET:,}); frozen-data homes {findings.frozen_total:,} "
          f"bytes (budget {FROZEN_BUDGET:,}), {len(findings.new_frozen)} new file(s), "
          f"each named by a reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
