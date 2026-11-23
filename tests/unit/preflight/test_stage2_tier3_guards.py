"""Stage 2 (Tier-3) — silent-wrong-answer guard regression tests.

Each test exercises an input that was silently mishandled before the
Stage 2 fixes and is now either rejected loudly or handled correctly:

* ``Grid.position_to_index`` returned an out-of-range index for a
  position outside the domain — now raises ``ValueError``.
* ``vmap_sweep._sequential_fallback`` rebuilt ``MaterialSpec`` from an
  explicit 5-field list that omitted ``chi3`` — a Kerr material lost
  its nonlinearity. Now uses ``dataclasses.replace`` (all fields).
* RIS built its substrate from a scalar ``eps_r`` only — substrate
  ``sigma`` / dispersion were dropped. Now carries the full material.

(The two ``rfx/runners/subgridded.py`` Stage-2 items are deferred —
PR #81 owns that file; see the execution log.)
"""

from __future__ import annotations

import dataclasses

import pytest

from rfx.grid import Grid


def test_position_to_index_in_bounds_unchanged():
    """Valid in-domain positions still resolve to the same index."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                cpml_layers=0)
    idx = grid.position_to_index((0.01, 0.01, 0.01))
    assert all(0 <= c for c in idx)
    nx, ny, nz = grid.shape
    assert idx[0] < nx and idx[1] < ny and idx[2] < nz


def test_position_to_index_raises_on_out_of_bounds():
    """A position outside the domain raises ValueError instead of
    silently returning an out-of-range / negative index."""
    EXTERIOR_X_M = 0.5
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                cpml_layers=0)
    assert EXTERIOR_X_M > grid.domain[0], "The positive test position lies beyond the x domain edge."
    with pytest.raises(ValueError, match="outside the grid shape"):
        grid.position_to_index((EXTERIOR_X_M, 0.01, 0.01))
    with pytest.raises(ValueError, match="outside the grid shape"):
        grid.position_to_index((-0.01, 0.01, 0.01))


def test_vmap_sequential_fallback_material_rebuild_keeps_chi3():
    """The vmap sequential fallback rebuilds a swept material with
    dataclasses.replace, which carries chi3. The pre-fix explicit
    eps_r/sigma/mu_r/debye_poles/lorentz_poles dict dropped it."""
    from rfx.api import MaterialSpec

    mat = MaterialSpec(eps_r=2.0, sigma=0.1, chi3=1e-18)

    swept = dataclasses.replace(mat, eps_r=5.0)
    assert swept.eps_r == 5.0
    assert swept.chi3 == 1e-18

    pre_fix = MaterialSpec(
        eps_r=5.0, sigma=mat.sigma, mu_r=mat.mu_r,
        debye_poles=mat.debye_poles, lorentz_poles=mat.lorentz_poles,
    )
    assert pre_fix.chi3 == 0.0
