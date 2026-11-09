"""S-parameter calculators read preflight checks from their defining modules."""

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.source_scan
def test_sparams_no_api_facade_import():
    violations = []
    for path in sorted((REPO / "rfx" / "sparams").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            forbidden = False
            if isinstance(node, ast.ImportFrom):
                forbidden = node.module == "rfx.api._preflight" or (
                    node.module == "rfx.api"
                    and any(alias.name == "_preflight" for alias in node.names)
                )
            elif isinstance(node, ast.Import):
                forbidden = any(
                    alias.name == "rfx.api._preflight" for alias in node.names
                )
            if forbidden:
                violations.append(
                    f"{path.relative_to(REPO)}:{node.lineno}: {ast.unparse(node)}"
                )
    assert not violations, (
        "S-parameter imports must bypass the API preflight facade:\n"
        + "\n".join(violations)
    )
