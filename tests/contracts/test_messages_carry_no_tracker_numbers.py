"""Runtime messages explain the limitation without tracker numbers or dates."""

import ast
import re
from pathlib import Path

import pytest


TRACKER_PATTERNS = (
    r"#\d{2,5}\b",
    r"\bPR\s*#?\d{3,5}\b",
    r"\btracker\s*#?\d+",
    r"\bPR\s*#?\d+[a-z]?\b",
    r"\bissue[- ]#?\d+",
    r"\b20\d\d-\d\d-\d\d\b",
)

# These constants are record provenance or documentation, never message text.
# Keep prefixes narrow, and require every exception to still be used.
ALLOWLIST = (
    ("measurement/ports.py", "record endpoint: one sample fewer; moved with #1466",
     "Owner record timing provenance, not a warning or refusal"),
    ("sparams/_common.py", "Warn-only complex-reciprocity tolerance for the rectangular-waveguide\n",
     "Attribute docstring documenting the tolerance's measured derivation"),
    ("sparams/msl.py", "PRE-passivity-projection raw extraction referred to the declared feed planes,",
     "Replay dump production_smatrix_stage metadata, not message text"),
    ("sparams/msl.py", "N equally spaced voltage probes plus current at probe 0.",
     "Replay dump deembedding metadata, not message text"),
)


def _non_docstring_constants(tree):
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                docstrings.add(id(node.body[0].value))
    return (node for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings)


@pytest.mark.source_scan
def test_messages_carry_no_tracker_numbers():
    package = Path(__file__).resolve().parents[2] / "rfx"
    violations = []
    used = set()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _non_docstring_constants(tree):
            if not any(re.search(pattern, node.value) for pattern in TRACKER_PATTERNS):
                continue
            exceptions = [entry for entry in ALLOWLIST
                          if relative == entry[0] and node.value.startswith(entry[1])]
            if exceptions:
                used.update(exceptions)
            else:
                violations.append(f"{relative}:{node.lineno}: {node.value!r}")
    assert not violations, "Tracker references in message text:\n" + "\n".join(violations)
    assert used == set(ALLOWLIST), f"Stale non-message exceptions: {set(ALLOWLIST) - used}"


@pytest.mark.parametrize("text", [
    "See #931", "issue #123", "issue-#80", "issue 1218", "PR 123", "PR #123",
    "Measured 2026-10-05", "rfx tracker 1598", "tracker #1598", "PR3b", "PR #3b",
])
def test_tracker_spellings(text):
    assert any(re.search(pattern, text) for pattern in TRACKER_PATTERNS)


def test_scan_includes_constants_message_variables_and_fstring_parts():
    tree = ast.parse('''
"""Module history #931."""
REASON = "issue #123"
def refuse(value):
    """Function history #931."""
    message = "PR 123"
    raise ValueError(f"{message}: {value} (#931)")
class Notice:
    """Class history #931."""
    reason = "2026-10-05"
''')
    assert sorted(node.value for node in _non_docstring_constants(tree)) == [
        " (#931)", "2026-10-05", ": ", "PR 123", "issue #123",
    ]
