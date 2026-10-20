"""The weekly lane clears JAX's compiled programs at each module end (#1932).

A child pytest runs two small modules with ``tests._module_cache_clear``
loaded; ``jax.clear_caches`` is wrapped to log each call next to the tests and
the module-scoped fixture teardowns, so the order is read from one file.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

_CONFTEST = textwrap.dedent('''
    import os
    import jax

    _real = jax.clear_caches

    def _logged():
        with open(os.environ["EVENTS"], "a") as fh:
            fh.write("clear\\n")
        _real()

    jax.clear_caches = _logged
''')

_MODULE = textwrap.dedent('''
    import os
    import pytest

    def _log(event):
        with open(os.environ["EVENTS"], "a") as fh:
            fh.write(event + "\\n")

    @pytest.fixture(scope="module")
    def held():
        yield
        _log("{name} teardown")

    def test_one(held):
        _log("{name}::one")

    def test_two(held):
        _log("{name}::two")
''')


def _events(tmp_path: Path, enabled: bool) -> list[str]:
    for name in ("a", "b"):
        (tmp_path / f"test_{name}.py").write_text(_MODULE.format(name=name))
    (tmp_path / "conftest.py").write_text(_CONFTEST)
    events = tmp_path / "events.txt"
    # The cluster gate exports PYTEST_ADDOPTS with -n (xdist); the child must
    # run its two modules in one process, in order.
    env = {k: v for k, v in os.environ.items()
           if k not in ("RFX_CLEAR_JAX_CACHES_PER_MODULE", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")}
    env.update(EVENTS=str(events), PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu")
    if enabled:
        env["RFX_CLEAR_JAX_CACHES_PER_MODULE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "tests._module_cache_clear", "--rootdir", str(tmp_path),
         "-c", os.devnull, str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return events.read_text().splitlines()


def test_set_clears_once_per_module_after_its_teardown(tmp_path: Path) -> None:
    assert _events(tmp_path, enabled=True) == [
        "a::one", "a::two", "a teardown", "clear",
        "b::one", "b::two", "b teardown", "clear",
    ]


def test_unset_registers_nothing(tmp_path: Path) -> None:
    assert "clear" not in _events(tmp_path, enabled=False)


def test_the_repo_conftest_loads_the_plugin() -> None:
    # The child runs above load the plugin with -p; the weekly lane gets it
    # only through conftest.py's pytest_plugins.
    import conftest

    assert "tests._module_cache_clear" in conftest.pytest_plugins
