"""The "never flip x64 at module level" rule, as a gate rather than as prose.

`CLAUDE.md` has carried this rule for months:

    Never `jax.config.update('jax_enable_x64', True)` at test module level — it
    is process-global, flips at pytest collection, and reds every same-process
    pytest-split shard. Scope x64 per-test (fixture/context) when flux math
    needs it.

Nothing enforced it. The sanctioned helper `tests/_x64_compat.enable_x64` exists
and several modules carry a docstring promising not to do it, but a module-level
flip in any one of the 600+ test files would have been caught only by a red CI
shard whose failure points somewhere else entirely — the flip happens at
COLLECTION, so the test that fails is whichever unrelated test runs next in the
same process.

This is the repo's own rule applied to its own rule (`AGENTS.md`): every closed
root cause leaves a cheap always-on check, and the check pins the INVARIANT
rather than a value. It is an AST scan, so it costs no JAX import and no solve.

What counts as a violation: a call to `jax.config.update("jax_enable_x64", ...)`
or `jax.config.enable_x64(...)` evaluated when the module is imported — at module
scope, or inside a module-level `if`/`try`/`with`/loop body. A call inside a
function, method, fixture, or context manager is the sanctioned pattern and
passes.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]

# The shim IS the sanctioned scoped flip; its call sits inside a contextmanager
# function, so the walker would pass it anyway. Named here so a reader does not
# have to re-derive that.
_SANCTIONED = {"_x64_compat.py"}

_X64_FLAG = "jax_enable_x64"


def _is_x64_flip(node: ast.Call) -> bool:
    """`jax.config.update("jax_enable_x64", …)` or `…config.enable_x64(…)`."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "update":
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == _X64_FLAG:
                return True
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant) and kw.value.value == _X64_FLAG:
                return True
        return False
    if func.attr == "enable_x64":
        # jax.config.enable_x64() — the old module-level spelling. The scoped
        # context manager `jax.experimental.enable_x64` is a bare Name when
        # imported, not an Attribute, so it is not matched here.
        return isinstance(func.value, ast.Attribute) and func.value.attr == "config"
    return False


def _module_level_flips(tree: ast.Module) -> list[int]:
    """Line numbers of x64 flips that run on import.

    Descends through module-level control flow (`if`, `try`, `with`, loops)
    because their bodies execute at import too; stops at anything that
    introduces a deferred scope (`def`, `async def`, `class`, lambda).
    """
    found: list[int] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call) and _is_x64_flip(node):
            found.append(node.lineno)
        stack.extend(ast.iter_child_nodes(node))
    return sorted(found)


def _python_files() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("*.py") if p.is_file())


def test_the_scan_sees_the_test_tree():
    """A guard that matched nothing would pass forever."""
    files = _python_files()
    assert len(files) > 300, (
        f"expected the whole tests/ tree, found {len(files)} files under {TESTS_ROOT} — "
        "the scan's root is wrong, so a green result here means nothing"
    )


def test_no_test_module_flips_x64_at_import_time():
    offenders: list[str] = []
    for path in _python_files():
        if path.name in _SANCTIONED:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - a broken file is its own failure
            pytest.fail(f"{path.relative_to(TESTS_ROOT)}: {exc}")
        for line in _module_level_flips(tree):
            offenders.append(f"{path.relative_to(TESTS_ROOT)}:{line}")

    assert not offenders, (
        "x64 flipped at module import time in:\n  "
        + "\n  ".join(offenders)
        + "\n\njax.config is process-global and pytest-split runs many modules in one "
        "process, so this flips the flag at COLLECTION and reds whichever unrelated "
        "test runs next in that shard. Scope it instead: `with enable_x64():` from "
        "tests/_x64_compat, or a fixture that restores the previous value."
    )


def test_the_guard_catches_a_planted_flip(tmp_path: Path):
    """Falsifier: the walker must fire on each import-time spelling, and stay
    quiet on the scoped one."""
    module_level = tmp_path / "planted_module.py"
    module_level.write_text(
        "import jax\n"
        "jax.config.update('jax_enable_x64', True)\n",
        encoding="utf-8",
    )
    inside_if = tmp_path / "planted_if.py"
    inside_if.write_text(
        "import os, jax\n"
        "if os.environ.get('X'):\n"
        "    jax.config.update('jax_enable_x64', True)\n",
        encoding="utf-8",
    )
    old_spelling = tmp_path / "planted_old.py"
    old_spelling.write_text("import jax\njax.config.enable_x64()\n", encoding="utf-8")
    scoped = tmp_path / "scoped.py"
    scoped.write_text(
        "import jax\n"
        "def test_thing():\n"
        "    jax.config.update('jax_enable_x64', True)\n",
        encoding="utf-8",
    )

    for planted in (module_level, inside_if, old_spelling):
        tree = ast.parse(planted.read_text(encoding="utf-8"))
        assert _module_level_flips(tree), f"guard missed {planted.name}"
    assert not _module_level_flips(ast.parse(scoped.read_text(encoding="utf-8"))), (
        "guard fired on the sanctioned per-test flip"
    )
