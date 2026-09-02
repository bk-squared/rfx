"""Issue #704 — an NTFF box on an S-matrix path must not be dropped silently.

``add_ntff_box()`` registers a far-field monitor; ``compute_msl_s_matrix()``
(and the sibling paths the issue names: ``compute_waveguide_s_matrix()``,
``compute_coaxial_s_matrix()``) run FDTD solves on that simulation but their
result classes carry no ``ntff_data``/``ntff_box`` fields — the monitor's
recording is discarded. The minimum fix (this file's subject) is ONE
``UserWarning`` per S-matrix call — never per port — naming what was
registered, what is dropped, the ``run()`` workaround, and the STALE-IF
condition. Full per-drive threading stays open under #704.

Mechanics: every test monkeypatches ``Simulation._build_grid`` to raise
``_GridBuildIntercept``. The warning sits after each method's cheap guard
block and before its first ``_build_grid()`` call, so the intercept both
keeps the tests FDTD-free and proves the call got PAST the guards to the
stage a real solve would reach (an early guard raise would fail the
``pytest.raises`` arm, not just the warning arm).

Mutation falsification of the warning gate (both directions, run
2026-08-24 in this worktree; same record kept in the helper's docstring,
``rfx/api/_sparams.py::_warn_ntff_box_dropped``):

- warn DELETED -> 3 failed, 3 passed; the three ``*_warns_with_ntff_box``
  tests each red with verbatim ``Failed: DID NOT WARN. No warnings of
  type (<class 'UserWarning'>,) were emitted.``
- warn UNCONDITIONAL (``if sim._ntff is None: return`` deleted) -> the
  three ``*_silent_without_ntff_box`` tests each red with verbatim
  ``TypeError: cannot unpack non-iterable NoneType object``.
- intact code: 6 passed.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

_NTFF_MATCH = r"add_ntff_box.*far-field"


class _GridBuildIntercept(Exception):
    """Raised by the monkeypatched ``_build_grid`` — no FDTD ever runs."""


def _intercept_grid_build(monkeypatch):
    def _boom(self, **kwargs):
        raise _GridBuildIntercept

    monkeypatch.setattr(Simulation, "_build_grid", _boom)


# --------------------------------------------------------------------------
# fixtures — the committed MSL thru line, a coarse WR-90 two-port, and the
# one-port coax box (geometry copied from tests/unit/runners/test_run_progress_reporting,
# tests/unit/materials/test_sheet_lane_fences, tests/unit/sparams/test_coaxial_s_matrix respectively).
# --------------------------------------------------------------------------


def _msl_thru() -> Simulation:
    domain_y, y_c = 0.008, 0.004
    sim = Simulation(freq_max=20e9, domain=(0.012, domain_y, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, domain_y, 0.0008)), material="sub")
    sim.add(Box((0.0, y_c - 0.0006, 0.0008),
                (0.012, y_c + 0.0006, 0.0010)), material="pec")
    sim.add_msl_port(position=(0.002, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


def _wr90() -> Simulation:
    sim = Simulation(
        freq_max=8e9, domain=(0.06, 0.02286, 0.01016), dx=3e-3,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=6)
    freqs = jnp.linspace(6e9, 7e9, 3)
    for x, d, name in ((0.009, "+x", "l"), (0.051, "-x", "r")):
        sim.add_waveguide_port(x, direction=d, mode=(1, 0), mode_type="TE",
                               freqs=freqs, f0=6.5e9, bandwidth=0.4,
                               name=name)
    return sim


def _coax_box() -> Simulation:
    sim = Simulation(freq_max=10.0e9, domain=(0.020, 0.020, 0.020),
                     boundary="pec")
    sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
    return sim


def _add_box(sim: Simulation) -> Simulation:
    lo = tuple(0.25 * d for d in sim._domain)
    hi = tuple(0.75 * d for d in sim._domain)
    sim.add_ntff_box(lo, hi, n_freqs=5)
    return sim


def _assert_no_ntff_warning(records) -> None:
    dropped = [w for w in records if "add_ntff_box" in str(w.message)]
    assert dropped == [], f"NTFF-drop warning fired without a box: {dropped}"


# --------------------------------------------------------------------------
# MSL — the path #704 names first
# --------------------------------------------------------------------------


def test_msl_warns_with_ntff_box(monkeypatch):
    sim = _add_box(_msl_thru())
    _intercept_grid_build(monkeypatch)
    with pytest.warns(UserWarning, match=_NTFF_MATCH) as rec:
        with pytest.raises(_GridBuildIntercept):
            sim.compute_msl_s_matrix(n_freqs=3)
    hits = [w for w in rec if "add_ntff_box" in str(w.message)]
    # ONE warning per call, not per port (the fixture has two MSL ports).
    assert len(hits) == 1
    msg = str(hits[0].message)
    # Basis clauses: registered / dropped / workaround / stale-if.
    assert "compute_msl_s_matrix()" in msg
    assert "5 frequencies" in msg              # what was registered
    assert "dropped" in msg                    # what happens to it
    assert "run()" in msg                      # the workaround
    assert "STALE-IF" in msg                   # when to delete the warning
    assert "#704" in msg                       # full threading stays open


def test_msl_silent_without_ntff_box(monkeypatch):
    sim = _msl_thru()
    _intercept_grid_build(monkeypatch)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(_GridBuildIntercept):
            sim.compute_msl_s_matrix(n_freqs=3)
    _assert_no_ntff_warning(rec)


# --------------------------------------------------------------------------
# waveguide + coaxial — the audit siblings #704 names
# --------------------------------------------------------------------------


def test_waveguide_warns_with_ntff_box(monkeypatch):
    sim = _add_box(_wr90())
    _intercept_grid_build(monkeypatch)
    with pytest.warns(UserWarning, match=_NTFF_MATCH) as rec:
        with pytest.raises(_GridBuildIntercept):
            sim.compute_waveguide_s_matrix(normalize=True)
    hits = [w for w in rec if "add_ntff_box" in str(w.message)]
    assert len(hits) == 1
    assert "compute_waveguide_s_matrix()" in str(hits[0].message)


def test_waveguide_silent_without_ntff_box(monkeypatch):
    sim = _wr90()
    _intercept_grid_build(monkeypatch)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(_GridBuildIntercept):
            sim.compute_waveguide_s_matrix(normalize=True)
    _assert_no_ntff_warning(rec)


def test_coaxial_warns_with_ntff_box(monkeypatch):
    sim = _add_box(_coax_box())
    _intercept_grid_build(monkeypatch)
    with pytest.warns(UserWarning, match=_NTFF_MATCH) as rec:
        with pytest.raises(_GridBuildIntercept):
            sim.compute_coaxial_s_matrix(n_steps=8, n_freqs=3)
    hits = [w for w in rec if "add_ntff_box" in str(w.message)]
    assert len(hits) == 1
    assert "compute_coaxial_s_matrix()" in str(hits[0].message)


def test_coaxial_silent_without_ntff_box(monkeypatch):
    sim = _coax_box()
    _intercept_grid_build(monkeypatch)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(_GridBuildIntercept):
            sim.compute_coaxial_s_matrix(n_steps=8, n_freqs=3)
    _assert_no_ntff_warning(rec)
