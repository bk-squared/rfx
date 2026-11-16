"""Public Pareto exports, navigation, and executable documentation contract."""

import ast
import json
from pathlib import Path

import pytest

import rfx
from tests.contracts.test_doc_python_blocks import python_blocks

REPO = Path(__file__).resolve().parents[2]


def test_pareto_public_exports():
    expected = {
        "ParetoFront", "ParetoPoint", "pareto_mask", "pareto_front",
        "weighted_scalarization", "epsilon_constraint_mask", "select_epsilon_constrained",
    }
    tree = ast.parse((REPO / "rfx/__init__.py").read_text(encoding="utf-8"))
    exports = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__all__"
                           for target in node.targets))
    assert expected <= set(exports)
    namespace = {}
    exec("from rfx import *", namespace)
    for name in expected:
        assert namespace[name] is getattr(rfx, name)


@pytest.mark.docs_consistency
def test_pareto_page_is_connected():
    navigation = json.loads((REPO / "docs/public/site_map.json").read_text())
    design = next(group["items"] for group in navigation["groups"] if group["label"] == "Design")
    assert design[design.index("rfx/design/sweeps") + 1] == "rfx/design/pareto"
    for name in ("sweeps.mdx", "optimization.md"):
        assert "(/rfx/design/pareto/)" in (REPO / "docs/public/design" / name).read_text()


@pytest.mark.docs_consistency
def test_pareto_python_blocks_execute(tmp_path, monkeypatch):
    page = REPO / "docs/public/design/pareto.mdx"
    blocks = list(python_blocks(page.read_text(encoding="utf-8")))
    assert len(blocks) == 6
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for line, source in blocks:
        exec(compile(source, f"{page}:{line}", "exec"), namespace)
    artifact = json.loads((tmp_path / "pareto_front.json").read_text())
    assert artifact["objective_names"] == ["footprint", "bandwidth"]
    assert artifact["minimize"] == [True, False]
    assert artifact["source_count"] == 3
    assert artifact["points"] == [
        {"source_index": 0, "id": "0", "objectives": [1.0, 2.0], "metadata": None},
        {"source_index": 1, "id": "1", "objectives": [2.0, 4.0], "metadata": None},
    ]
