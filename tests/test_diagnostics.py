"""Tests for the onboarding code spine: rfx-diagnose + the hello-world example.

These exercise the two things a brand-new user runs first, so they must stay
fast and headless.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

from rfx import diagnostics


# ---------------------------------------------------------------------------
# rfx-diagnose
# ---------------------------------------------------------------------------

def test_diagnose_passes_in_this_environment(capsys):
    """In a working install (rfx importable, jax + core deps present, smoke
    runs) main([]) returns 0 and prints the expected section labels."""
    rc = diagnostics.main([])
    out = capsys.readouterr().out

    assert rc == 0
    assert "rfx environment diagnostics" in out
    assert "Python version" in out
    assert "import rfx" in out
    assert "jax / jaxlib" in out
    assert "FDTD smoke run" in out
    assert "All critical checks passed" in out


def test_diagnose_critical_failure_returns_nonzero_without_traceback(
    monkeypatch, capsys
):
    """A forced critical failure (the smoke run blows up) returns non-zero,
    prints a FAIL line, and does NOT propagate a traceback."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced smoke failure")

    # Make the FDTD smoke run fail at the Simulation.run() call. The check is
    # wrapped in try/except, so this must surface as a FAIL line, not a raise.
    import rfx.api as rfx_api

    monkeypatch.setattr(rfx_api.Simulation, "run", _boom)

    rc = diagnostics.main([])
    out = capsys.readouterr().out

    assert rc != 0
    assert "[FAIL] FDTD smoke run" in out
    assert "critical check(s) failed" in out


def test_diagnose_missing_core_dep_fails_without_aborting(monkeypatch, capsys):
    """A missing core dependency surfaces as a clean [FAIL] + non-zero exit,
    and does NOT abort the later checks (fault isolation)."""
    import importlib as _il

    real_import = _il.import_module

    def _fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("simulated missing h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(diagnostics.importlib, "import_module", _fake_import)

    rc = diagnostics.main([])
    out = capsys.readouterr().out

    assert rc != 0
    assert "[FAIL]" in out and "h5py" in out
    # fault isolation: a check after the core-dep stage still ran.
    assert "FDTD smoke run" in out


# ---------------------------------------------------------------------------
# hello-world example
# ---------------------------------------------------------------------------

def _example_path() -> Path:
    # tests/ -> repo root -> examples/quickstart/hello_world.py
    return (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "quickstart"
        / "hello_world.py"
    )


def test_hello_world_example_runs(capsys):
    """The hello-world example executes end-to-end and prints a finite
    probe summary. It is already tiny (~50 steps on a ~10-cell box)."""
    path = _example_path()
    assert path.exists(), f"missing example: {path}"

    runpy.run_path(str(path), run_name="__main__")
    out = capsys.readouterr().out

    assert "rfx hello world" in out
    assert "grid size" in out
    assert "peak |Ez|" in out
    assert "all finite     : True" in out


def test_hello_world_main_produces_finite_trace():
    """Directly drive the example's builder via the same public API it uses
    and assert the probe trace is finite (independent of stdout parsing)."""
    from rfx import Simulation
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=2e-3, boundary="pec")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    sim.add_probe((0.014, 0.01, 0.01), "ez")
    result = sim.run(n_steps=50, compute_s_params=False)

    trace = np.asarray(result.time_series)[:, 0]
    assert trace.shape[0] == 50
    assert bool(np.all(np.isfinite(trace)))
