"""Regression coverage for issue #628: import-time binding pollution in
``rfx/runners/uniform.py``.

The defect: ``rfx/runners/uniform.py`` used to do
``from rfx.simulation import run as _run`` at MODULE level. That module is
imported lazily -- the first time some ``Simulation`` call path actually
needs the uniform-grid runner (see ``rfx/api/_execute.py`` /
``rfx/api/_sparams.py``, both of which import it locally, inside functions).
If that first import happens while a test has ``rfx.simulation.run``
monkeypatched (e.g. ``tests/test_coax_two_port_fdtd.py``'s
``test_compute_coaxial_two_port_drive_index_matches_physical_port``),
``rfx.runners.uniform._run`` permanently captures the patched stub -- pytest's
``monkeypatch`` teardown restores ``rfx.simulation.run`` on the SOURCE module,
but has no way to reach the copy already bound into ``rfx.runners.uniform``'s
own namespace. Every later uniform-lane run in that process then silently
calls the stale stub. The concrete symptom was order-dependent:
``pytest tests/test_coax_two_port_fdtd.py tests/locks/test_refplane_port_waves.py``
gave ``1 failed, 63 passed`` (a ``TypeError`` from the leaked stub), while
either file alone, or the suite in its usual collection order, passed clean.

The fix: ``rfx/runners/uniform.py`` now does
``from rfx import simulation as _simulation`` and calls
``_simulation.run(...)`` / ``_simulation.run_until_decay(...)`` -- a module
reference, resolved at CALL time, so it always reflects whatever
``rfx.simulation.run`` currently is (patched or not).
"""

import subprocess
import sys

import pytest

import rfx.simulation as _simulation_mod


def _fake_run(*args, **kwargs):
    raise AssertionError(
        "rfx.runners.uniform called a stale, torn-down monkeypatched "
        "rfx.simulation.run -- the #628 import-binding-pollution regression"
    )


def test_runner_uniform_resolves_run_after_patch_window_closes(monkeypatch):
    """Direct mechanism test: simulate the import-inside-patch-window
    sequence explicitly, without relying on pytest's collection order.

    Forces ``rfx.runners.uniform`` (and its parent package, which itself
    imports it at module level: ``rfx/runners/__init__.py``) out of
    ``sys.modules`` so the next import is a genuine first import, opens a
    monkeypatch window on ``rfx.simulation.run``, imports
    ``rfx.runners.uniform`` for the first time *inside* that window (mirrors
    a monkeypatching test running before any other test has ever touched the
    uniform runner), then closes the window and asserts the runner still
    resolves the CURRENT ``rfx.simulation.run`` rather than a frozen
    reference to the fake.
    """
    for mod_name in ("rfx.runners.uniform", "rfx.runners"):
        sys.modules.pop(mod_name, None)

    real_run = _simulation_mod.run

    with monkeypatch.context() as m:
        m.setattr(_simulation_mod, "run", _fake_run)
        import rfx.runners.uniform as uniform  # first import happens HERE, while patched
        assert "rfx.runners.uniform" in sys.modules
        # Sanity: the patch is live and reachable from the source module.
        assert _simulation_mod.run is _fake_run

    # Patch window closed: monkeypatch has restored the source module.
    assert _simulation_mod.run is real_run

    # The regression under test: the runner module must NOT still be
    # pointing at the torn-down fake. Pre-fix, `uniform._run` would be
    # `_fake_run` here (an import-time copy); post-fix, `uniform._simulation`
    # is the live module object, so `.run` re-reads the current attribute.
    assert uniform._simulation is _simulation_mod
    assert uniform._simulation.run is real_run
    assert uniform._simulation.run is not _fake_run


@pytest.mark.slow
def test_coax_then_refplane_order_does_not_leak_fake_run():
    """Locks the original, concretely-observed symptom: running these two
    files together, in this order, must not fail. Pre-fix this reliably gave
    ``1 failed, 63 passed`` (a ``TypeError`` from the leaked stub in
    ``test_run_short_diagonals_byte_frozen_offdiagonals_move``); each file
    alone always passed, which is what made this an import-ORDER bug rather
    than a bug either test file's author could see on its own.

    Marked ``slow`` (subprocess + two real FDTD test files, ~20s) so it
    doesn't sit on the fast default lane; it runs in the slow-tests CI shard.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_coax_two_port_fdtd.py",
            "tests/locks/test_refplane_port_waves.py",
            "-q",
        ],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "the order-dependent #628 repro reappeared:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "failed" not in result.stdout, result.stdout
