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

Two spellings count as a violation, both judged only when they run at IMPORT
time — module scope, or inside a module-level `if`/`try`/`with`/loop body. A
call inside a function, method, fixture, or context manager is the sanctioned
pattern and passes.

1. The config call: `jax.config.update("jax_enable_x64", ...)` or the older
   `jax.config.enable_x64(...)`.
2. The environment spelling: any module-level mutation of the `JAX_ENABLE_X64`
   key — `os.environ.setdefault(...)`, `os.environ[...] = ...`,
   `os.environ.update({...})`, `os.putenv(...)`. JAX reads that variable once,
   when `jax` is first imported, and turns it into the same process-global
   flag. Under pytest the root `conftest.py` imports `jax` before any test
   module is collected, so such a line is either dead (jax already imported) or
   flips the flag for the whole shard; which of the two it is depends on import
   order, which is the worst of both failure modes.

Scanned set: every `*.py` under `tests/`, plus the repository-root
`conftest.py`, which loads before every shard and is the one file outside
`tests/` whose import-time statements run in every test process.

Boundary, stated so a green result is not over-read: a validation script that a
test imports (`validation/crossval/*.py` via `importlib`) is not in the scanned
set, and neither is `rfx/` production code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
ROOT_CONFTEST = REPO_ROOT / "conftest.py"

# Repo-relative paths, not basenames: an `_x64_compat.py` written anywhere else
# in the tree must not inherit the allowance by name alone.
#
# The shim IS the sanctioned scoped flip; its call sits inside a contextmanager
# function, so the walker would pass it anyway. Named here so a reader does not
# have to re-derive that.
_SANCTIONED = frozenset({"tests/_x64_compat.py"})

_X64_FLAG = "jax_enable_x64"
_ENV_KEY = "JAX_ENABLE_X64"

# `os.environ` methods that WRITE. `get`/`__contains__` read and are fine at
# module level; these change the process environment.
_ENV_MUTATORS = frozenset({"setdefault", "update", "pop", "__setitem__"})
# Module-level `os.putenv`/`os.unsetenv`, and `setenv` for a monkeypatch-shaped
# helper that somehow ran at import.
_ENV_FUNCTIONS = frozenset({"putenv", "unsetenv", "setenv"})


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


def _is_environ(node: ast.AST) -> bool:
    """`os.environ` / `os.environb`, or a bare `environ` imported from os."""
    if isinstance(node, ast.Attribute):
        return node.attr in ("environ", "environb")
    if isinstance(node, ast.Name):
        return node.id in ("environ", "environb")
    return False


def _names_env_key(node: ast.AST) -> bool:
    """`JAX_ENABLE_X64` appears as a literal, or as a keyword argument name.

    Anywhere in the statement, which covers the dict form
    `os.environ.update({"JAX_ENABLE_X64": "1"})` and the keyword form
    `os.environ.update(JAX_ENABLE_X64="1")` without enumerating shapes. An
    `update(d)` built from a variable is invisible to any AST scan; that is the
    scan's stated blind spot, not a silent pass.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value == _ENV_KEY:
            return True
        if isinstance(sub, ast.keyword) and sub.arg == _ENV_KEY:
            return True
    return False


def _is_env_x64_call(node: ast.Call) -> bool:
    """A call that writes `JAX_ENABLE_X64` into the process environment."""
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in _ENV_MUTATORS and _is_environ(func.value):
            return _names_env_key(node)
        if func.attr in _ENV_FUNCTIONS:
            return _names_env_key(node)
    elif isinstance(func, ast.Name) and func.id in _ENV_FUNCTIONS:
        return _names_env_key(node)
    return False


def _is_env_x64_subscript(target: ast.AST) -> bool:
    """`os.environ["JAX_ENABLE_X64"]` as an assignment or `del` target."""
    if not isinstance(target, ast.Subscript):
        return False
    if not _is_environ(target.value):
        return False
    key = target.slice
    return isinstance(key, ast.Constant) and key.value == _ENV_KEY


def _module_level_flips(tree: ast.Module) -> list[tuple[int, str]]:
    """`(line, spelling)` for every x64 flip that runs on import.

    Descends through module-level control flow (`if`, `try`, `with`, loops)
    because their bodies execute at import too; stops at anything that
    introduces a deferred scope (`def`, `async def`, `class`, lambda).
    """
    found: dict[int, str] = {}
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            if _is_x64_flip(node):
                found.setdefault(node.lineno, "jax.config")
            elif _is_env_x64_call(node):
                found.setdefault(node.lineno, "os.environ")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_is_env_x64_subscript(t) for t in targets):
                found.setdefault(node.lineno, "os.environ")
        elif isinstance(node, ast.Delete):
            if any(_is_env_x64_subscript(t) for t in node.targets):
                found.setdefault(node.lineno, "os.environ")
        stack.extend(ast.iter_child_nodes(node))
    return sorted(found.items())


def _python_files() -> list[Path]:
    """Every test module, plus the root conftest that loads before them all."""
    files = [p for p in TESTS_ROOT.rglob("*.py") if p.is_file()]
    if ROOT_CONFTEST.is_file():
        files.append(ROOT_CONFTEST)
    return sorted(files)


def test_the_scan_sees_the_test_tree():
    """A guard that matched nothing would pass forever."""
    files = _python_files()
    assert len(files) > 300, (
        f"expected the whole tests/ tree, found {len(files)} files under {TESTS_ROOT} — "
        "the scan's root is wrong, so a green result here means nothing"
    )
    assert ROOT_CONFTEST in files, (
        f"{ROOT_CONFTEST} is not in the scanned set. It loads before every shard, so an "
        "import-time flip there reaches every test in the process — it is the one file "
        "outside tests/ the scan must cover."
    )


def test_no_test_module_flips_x64_at_import_time():
    offenders: list[str] = []
    for path in _python_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in _SANCTIONED:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - a broken file is its own failure
            pytest.fail(f"{relative}: {exc}")
        for line, spelling in _module_level_flips(tree):
            offenders.append(f"{relative}:{line} ({spelling})")

    assert not offenders, (
        "x64 flipped at module import time in:\n  "
        + "\n  ".join(offenders)
        + "\n\njax_enable_x64 is process-global and pytest-split runs many modules in one "
        "process, so this flips the flag at COLLECTION and reds whichever unrelated "
        "test runs next in that shard. The os.environ spelling has the same reach: JAX "
        "reads JAX_ENABLE_X64 when it is first imported. Scope it instead: "
        "`with enable_x64():` from tests/_x64_compat, or a fixture that restores the "
        "previous value."
    )


def test_the_guard_catches_a_planted_flip(tmp_path: Path):
    """Falsifier: the walker must fire on each import-time spelling, and stay
    quiet on the scoped one and on a module-level READ."""
    planted = {
        "config_call": "import jax\njax.config.update('jax_enable_x64', True)\n",
        "inside_if": (
            "import os, jax\n"
            "if os.environ.get('X'):\n"
            "    jax.config.update('jax_enable_x64', True)\n"
        ),
        "old_spelling": "import jax\njax.config.enable_x64()\n",
        "env_setdefault": "import os\nos.environ.setdefault('JAX_ENABLE_X64', '1')\n",
        "env_assign": "import os\nos.environ['JAX_ENABLE_X64'] = '1'\n",
        "env_update_dict": "import os\nos.environ.update({'JAX_ENABLE_X64': '1'})\n",
        "env_update_kwarg": "import os\nos.environ.update(JAX_ENABLE_X64='1')\n",
        "env_putenv": "import os\nos.putenv('JAX_ENABLE_X64', '1')\n",
        "env_bare_import": (
            "from os import environ\nenviron.setdefault('JAX_ENABLE_X64', '1')\n"
        ),
        "env_inside_try": (
            "import os\ntry:\n    os.environ['JAX_ENABLE_X64'] = '1'\nexcept KeyError:\n    pass\n"
        ),
        "env_del": "import os\ndel os.environ['JAX_ENABLE_X64']\n",
    }
    for name, source in planted.items():
        path = tmp_path / f"planted_{name}.py"
        path.write_text(source, encoding="utf-8")
        assert _module_level_flips(ast.parse(path.read_text(encoding="utf-8"))), (
            f"guard missed the {name} spelling"
        )

    allowed = {
        "scoped_config": (
            "import jax\ndef test_thing():\n    jax.config.update('jax_enable_x64', True)\n"
        ),
        "scoped_env": (
            "import os\ndef test_thing(monkeypatch):\n"
            "    monkeypatch.setenv('JAX_ENABLE_X64', '1')\n"
        ),
        "module_level_read": (
            "import os\nWANTS_X64 = os.environ.get('JAX_ENABLE_X64') == '1'\n"
        ),
        "other_env_key": "import os\nos.environ.setdefault('XLA_FLAGS', '--foo')\n",
    }
    for name, source in allowed.items():
        path = tmp_path / f"allowed_{name}.py"
        path.write_text(source, encoding="utf-8")
        assert not _module_level_flips(ast.parse(path.read_text(encoding="utf-8"))), (
            f"guard fired on the sanctioned {name} pattern"
        )


def test_the_sanctioned_list_is_an_exact_path_not_a_basename():
    """A same-named file elsewhere in the tree must not inherit the allowance."""
    for relative in _SANCTIONED:
        assert "/" in relative, (
            f"{relative!r} is a bare basename; the exemption must name the exact "
            "repo-relative path so a copy elsewhere is still scanned"
        )
        assert (REPO_ROOT / relative).is_file(), (
            f"{relative} does not exist — a stale exemption hides whatever takes its name"
        )
