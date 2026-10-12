"""No rfx module or test imports the deprecated ``jax.experimental.shard_map``.

On the pinned JAX it warns ("deprecated in v0.8.0") the first time it is
imported, so a test that asserts no warnings failed or passed by which test
first imported the module that carried it (rfx/stepping/rank.py; found in the
absorber lane's census, 2026-10-12). ``jax.shard_map(check_vma=)`` is the call
the deprecated wrapper made, argument for argument. The scan reads import
statements and dotted attribute chains; it does not follow importlib or getattr.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPRECATED = "jax.experimental.shard_map"


def deprecated_uses(tree):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == DEPRECATED:
            found.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "jax.experimental" and any(
                a.name == "shard_map" for a in node.names):
            found.append(node.lineno)
        elif isinstance(node, ast.Import) and any(a.name == DEPRECATED for a in node.names):
            found.append(node.lineno)
        elif isinstance(node, ast.Attribute) and ast.unparse(node) == DEPRECATED:
            found.append(node.lineno)
    return found


def test_the_scan_sees_each_spelling():
    for source in ("from jax.experimental.shard_map import shard_map",
                   "from jax.experimental import shard_map",
                   "import jax.experimental.shard_map",
                   "f = jax.experimental.shard_map.shard_map"):
        assert deprecated_uses(ast.parse(source)) == [1], source
    assert deprecated_uses(ast.parse("from jax import shard_map")) == []


def test_no_rfx_module_or_test_imports_the_deprecated_shard_map():
    files = sorted([*(ROOT / "rfx").rglob("*.py"), *(ROOT / "tests").rglob("*.py")])
    hits = {str(p.relative_to(ROOT)): lines for p in files
            if (lines := deprecated_uses(ast.parse(p.read_text())))}
    assert hits == {}, hits
