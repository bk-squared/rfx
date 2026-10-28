"""No test module pins the JAX platform in the environment at import time.

The sibling of `test_no_module_level_x64.py`, for `JAX_PLATFORMS` and
`JAX_PLATFORM_NAME`. Under pytest the root `conftest.py` starts JAX before any
test module is collected, so a module-level `os.environ.setdefault(
"JAX_PLATFORMS", "cpu")` cannot move the session itself; it only writes the pin
into `os.environ` during COLLECTION, and every child process a later test starts
from `dict(os.environ)` then runs on the CPU. That is how the GPU suite on main
6b65b3b27 lost `test_n3_gpu_fresh_process` ("N3 requires GPU" in its child):
`tests/locks/test_volume_sheet_cavity_ladder.py` and
`tests/oracle/test_patch_cavity_eps_oracle.py` carried such lines.

The scan, its scope and its stated blind spots are the x64 file's (an AST walk
over `tests/` and the root conftest; a write through a variable or an imported
script is not seen). A write inside a function, fixture or `monkeypatch` passes;
a module-level read passes.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.contracts.test_no_module_level_x64 import (
    REPO_ROOT,
    _ENV_FUNCTIONS,
    _ENV_MUTATORS,
    _is_environ,
    _python_files,
)

_PLATFORM_KEYS = frozenset({"JAX_PLATFORMS", "JAX_PLATFORM_NAME"})


def _names_platform_key(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value in _PLATFORM_KEYS:
            return True
        if isinstance(sub, ast.keyword) and sub.arg in _PLATFORM_KEYS:
            return True
    return False


def _is_platform_subscript(target: ast.AST) -> bool:
    return (isinstance(target, ast.Subscript) and _is_environ(target.value)
            and isinstance(target.slice, ast.Constant) and target.slice.value in _PLATFORM_KEYS)


def _module_level_platform_pins(tree: ast.Module) -> list[int]:
    """Lines of every environment write of a platform key that runs on import."""
    found: set[int] = set()
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            func = node.func
            writes = ((isinstance(func, ast.Attribute)
                       and (func.attr in _ENV_MUTATORS and _is_environ(func.value) or func.attr in _ENV_FUNCTIONS))
                      or (isinstance(func, ast.Name) and func.id in _ENV_FUNCTIONS))
            if writes and _names_platform_key(node):
                found.add(node.lineno)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target])
            if any(_is_platform_subscript(t) for t in targets):
                found.add(node.lineno)
        stack.extend(ast.iter_child_nodes(node))
    return sorted(found)


def test_no_test_module_pins_the_platform_at_import_time():
    offenders = []
    for path in _python_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - a broken file is its own failure
            pytest.fail(f"{relative}: {exc}")
        offenders += [f"{relative}:{line}" for line in _module_level_platform_pins(tree)]
    assert not offenders, (
        "JAX_PLATFORMS / JAX_PLATFORM_NAME written into the environment at module import time in:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe root conftest has already started JAX, so the line cannot choose this session's "
        "platform; it only leaks into every child process a later test starts, which then runs on "
        "the CPU. Set the platform in the child's own env, or with monkeypatch inside the test."
    )


def test_the_guard_catches_a_planted_pin(tmp_path: Path):
    """Falsifier: each import-time spelling fires; scoped writes, reads and other keys do not."""
    planted = {
        "setdefault": "import os\nos.environ.setdefault('JAX_PLATFORMS', 'cpu')\n",
        "platform_name": "import os\nos.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')\n",
        "assign": "import os\nos.environ['JAX_PLATFORMS'] = 'cpu'\n",
        "update_kwarg": "import os\nos.environ.update(JAX_PLATFORMS='cpu')\n",
        "putenv": "import os\nos.putenv('JAX_PLATFORMS', 'cpu')\n",
        "inside_if": "import os\nif True:\n    os.environ['JAX_PLATFORM_NAME'] = 'cpu'\n",
    }
    for name, source in planted.items():
        assert _module_level_platform_pins(ast.parse(source)), f"guard missed the {name} spelling"
    allowed = {
        "in_function": "import os\ndef f():\n    os.environ['JAX_PLATFORMS'] = 'cpu'\n",
        "monkeypatch": "def test_x(monkeypatch):\n    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')\n",
        "read": "import os\nP = os.environ.get('JAX_PLATFORMS')\n",
        "child_env": "import os\nENV = dict(os.environ, JAX_PLATFORMS='cpu')\n",
        "other_key": "import os\nos.environ.setdefault('XLA_FLAGS', '--foo')\n",
    }
    for name, source in allowed.items():
        assert not _module_level_platform_pins(ast.parse(source)), f"guard fired on the allowed {name} pattern"
