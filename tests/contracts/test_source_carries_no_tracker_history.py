"""Comments and documentation name the reason; Git carries tracker history."""

import ast
import io
from pathlib import Path
import re
import tokenize

import pytest


HISTORY = re.compile(
    r"https?://github\.com/[^/\s]+/rfx/(?:issues|pull)/\d+\b"
    r"|#\d+\b|\b(?:issue\s+|PR\s+|gh-)\d+\b"
    r"|(?<!\d)\d{4}-\d{2}(?:-\d{2})?(?![\d-])"
    r"|(?<!\d)\d{4}/\d{2}/\d{2}(?!\d)",
    re.IGNORECASE,
)
URL = re.compile(r"https?://[^\s<>`\"']+")
RFX_TRACKER_URL = re.compile(
    r"https?://github\.com/[^/\s]+/rfx/(?:issues|pull)(?:/|$)", re.IGNORECASE,
)
# A date embedded in a cited document's filename is path data, not prose
# chronology. Pin the entire path and require the exception to remain used.
KEEP = {
    ("stepping/kerr.py", "docs/research_notes/2026-07-23_kerr_quantitative_spm_finding.md"):
        "Date is part of the cited document's filename",
}


def _documentation_nodes(tree):
    """Standard, attribute and additional docstrings (PEP 257)."""
    for owner in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(owner, field, None)
            if not isinstance(body, list):
                continue
            previous = None
            for index, node in enumerate(body):
                is_string = (isinstance(node, ast.Expr)
                             and isinstance(node.value, ast.Constant)
                             and isinstance(node.value.value, str))
                standard = index == 0 and isinstance(
                    owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                follows_documentation = isinstance(previous, ast.Expr) and isinstance(
                    previous.value, ast.Constant) and isinstance(previous.value.value, str)
                attribute = isinstance(previous, (ast.Assign, ast.AnnAssign))
                if is_string and (standard or attribute or follows_documentation):
                    yield node
                previous = node


def _history(source, relative):
    """Return locations and used Keep entries without importing the package."""
    tree = ast.parse(source)
    documentation = {(n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)
                     for n in _documentation_nodes(tree)}
    violations = []
    used = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        in_doc = any((start, column) <= token.start and token.end <= (end, end_column)
                     for start, column, end, end_column in documentation)
        if token.type != tokenize.COMMENT and not (token.type == tokenize.STRING and in_doc):
            continue
        text = token.string
        for entry in KEEP:
            if entry[0] == relative and entry[1] in text:
                used.add(entry)
                text = text.replace(entry[1], " " * len(entry[1]))
        # External citations may contain dates or numeric URL fragments.
        # Mask only those URLs; rfx tracker links remain subject to HISTORY.
        text = URL.sub(lambda match: match.group() if RFX_TRACKER_URL.match(match.group())
                       else " " * len(match.group()), text)
        for match in HISTORY.finditer(text):
            line = token.start[0] + text[:match.start()].count("\n")
            violations.append(f"{relative}:{line}: {match.group()}")
    return violations, used


@pytest.mark.source_scan
def test_source_carries_no_tracker_history():
    package = Path(__file__).resolve().parents[2] / "rfx"
    violations = []
    used = set()
    for path in sorted(package.rglob("*.py")):
        found, kept = _history(path.read_text(encoding="utf-8"),
                               path.relative_to(package).as_posix())
        violations.extend(found)
        used.update(kept)
    assert not violations, "Tracker history in source documentation:\n" + "\n".join(violations)
    assert used == set(KEEP), f"Stale Keep entries: {set(KEEP) - used}"


@pytest.mark.parametrize("mutation, expected", [
    ("VALUE = 1  # Fixed in #123\n", "sample.py:1: #123"),
    ('def sample():\n    """Added 2026-05-08."""\n    pass\n',
     "sample.py:2: 2026-05-08"),
    ("# https://github.com/bk-squared/rfx/issues/123\n",
     "sample.py:1: https://github.com/bk-squared/rfx/issues/123"),
    ('"""See https://github.com/bk-squared/rfx/pull/45."""\n',
     "sample.py:1: https://github.com/bk-squared/rfx/pull/45"),
    ('"""Added 2026-04."""\n', "sample.py:1: 2026-04"),
    ("# Fixed in issue 123\n", "sample.py:1: issue 123"),
    ('"""Added in PR 45."""\n', "sample.py:1: PR 45"),
    ("# See gh-123\n", "sample.py:1: gh-123"),
    ('"""Added 2026/05/08."""\n', "sample.py:1: 2026/05/08"),
], ids=["comment-tracker", "docstring-date", "issue-url", "pull-url",
        "year-month", "issue-number", "pr-number", "gh-number", "slash-date"])
def test_history_mutations_are_detected(mutation, expected):
    assert _history(mutation, "sample.py")[0] == [expected]


def test_documentation_scope_and_keep_rules():
    source = '''"""Paper (1997), version 0.12.0, #abcdef, https://example.org/paper."""
VALUE = "runtime #123 and 2026-05-08"
"""Attribute history #456."""
"""Additional history #789."""
def sample():
    """Function history #321."""
    return VALUE
class Sample:
    """Class history #654."""
'''
    assert _history(source, "sample.py")[0] == [
        "sample.py:3: #456", "sample.py:4: #789", "sample.py:6: #321",
        "sample.py:9: #654",
    ]
    path = next(iter(KEEP))
    assert _history(f'"""See {path[1]}."""', path[0]) == ([], {path})
    assert _history(f'"""See {path[1]}."""', "elsewhere.py")[0]


@pytest.mark.parametrize("citation", [
    "https://example.org/2026-04/paper#123",
    "https://github.com/other/project/issues/123#456",
    "issue683_decomposer_flip_predeclaration.md",
    "20261030-record.md",
])
def test_external_citations_and_compact_document_ids_are_kept(citation):
    assert _history(f'"""See {citation}."""', "sample.py") == ([], set())
