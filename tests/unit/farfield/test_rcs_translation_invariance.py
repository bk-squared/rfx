"""A 3 GHz +x plane wave illuminates a 15.9 mm PEC sphere in a fixed box.

Whole-cell translations must preserve its monostatic RCS magnitude.
The 2026-09-23 audit measured 0.085 dB peak-to-peak over +/-6 cells;
the known-limitations page recorded 2.194 dB before the surface rule in #1159.
The 0.2 dB tolerance leaves room above that measured residual while rejecting
the earlier position dependence. This checks translation, not absolute Mie accuracy.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import rfx.rcs as rcs_module
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import Sphere, rasterize
from rfx.grid import C0, Grid

# Same rig as tests/fixtures/rcs_sphere_mie/generate_fixture.py.
F0 = 3e9
RADIUS = 0.0159
RESOLUTION = 40
DOMAIN_SIZE = 0.10
CPML_LAYERS = 24
N_STEPS = 700
SHIFTS = (-4, 0, 4)


@pytest.mark.slow
def test_monostatic_rcs_is_translation_invariant():
    grid = Grid(
        freq_max=F0 * 1.5,
        domain=(DOMAIN_SIZE,) * 3,
        dx=C0 / F0 / RESOLUTION,
        cpml_layers=CPML_LAYERS,
    )
    occupied = []
    monostatic_dbsm = []
    for shift in SHIFTS:
        center = (
            DOMAIN_SIZE / 2 + shift * float(grid.dx),
            DOMAIN_SIZE / 2,
            DOMAIN_SIZE / 2,
        )
        eps_r, sigma = rasterize(
            grid, [(Sphere(center=center, radius=RADIUS), 1.0, 1e7)]
        )
        occupied.append(int(np.count_nonzero(np.asarray(sigma) > 1.0)))
        materials = MaterialArrays(
            eps_r=eps_r,
            sigma=sigma,
            mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
        )
        result = rcs_module.compute_rcs(
            grid, materials, N_STEPS,
            f0=F0, bandwidth=0.5, theta_inc=0.0, polarization="ez",
            theta_obs=np.array([np.pi / 2]), phi_obs=np.array([0.0, np.pi]),
            freqs=np.array([F0]), boundary="cpml", cpml_layers=CPML_LAYERS,
        )
        monostatic_dbsm.append(float(result.monostatic_rcs[0]))

    assert occupied[0] > 0 and len(set(occupied)) == 1, (
        f"translated sphere rasterization changed: shifts={SHIFTS}, cells={occupied}"
    )
    assert np.all(np.isfinite(monostatic_dbsm)), monostatic_dbsm
    spread_db = float(np.ptp(monostatic_dbsm))
    print(
        f"shifts={SHIFTS}, occupied={occupied}, "
        f"monostatic_dBsm={monostatic_dbsm}, spread_dB={spread_db:.6f}"
    )
    assert spread_db < 0.2, (
        f"monostatic RCS spread {spread_db:.6f} dB exceeds 0.2 dB: "
        f"shifts={SHIFTS}, monostatic_dBsm={monostatic_dbsm}"
    )


def test_translation_check_rejects_a_2_194_db_spread(monkeypatch):
    """A fabricated 2.194 dB position dependence must fail the same comparison."""
    values = iter((-25.0, -22.806, -25.0))

    def position_dependent_rcs(*args, **kwargs):
        return SimpleNamespace(monostatic_rcs=np.array([next(values)]))

    monkeypatch.setattr(rcs_module, "compute_rcs", position_dependent_rcs)
    with pytest.raises(AssertionError, match="monostatic RCS spread"):
        test_monostatic_rcs_is_translation_invariant()
