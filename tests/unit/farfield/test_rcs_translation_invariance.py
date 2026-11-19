"""The far field of one discretized scatterer is independent of its box position.

A 15.9 mm conductivity-filled sphere is moved four whole cells along x on
the lambda/40 mesh at 3 GHz. Across ka = 0.75–1.25, the measured monostatic
difference is at most 0.029 dB with the face-centre surface rule. Node
collocation gives 0.10–0.36 dB at ka = 0.90–1.25; a 2.194 dB position
dependence was recorded before the second-order surface rule.
The per-bin bar is 0.05 dB, except 0.06 dB at ka = 1.25: twice the measured
difference rounded up to 0.01 dB, with a 0.05 dB floor.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import rfx.measurement.rcs as rcs_module
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
SHIFTS = (0, 4)
KA_BINS = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25])
BARS_DB = (0.05,) * 10 + (0.06,)


@pytest.fixture(scope="module")
def translated_sphere_rcs():
    """Share two live runs of the same cell mask across all frequency bins."""
    grid = Grid(
        freq_max=F0 * 1.5,
        domain=(DOMAIN_SIZE,) * 3,
        dx=C0 / F0 / RESOLUTION,
        cpml_layers=CPML_LAYERS,
    )
    masks = []
    materials = []
    for shift in SHIFTS:
        center = (
            DOMAIN_SIZE / 2 + shift * float(grid.dx),
            DOMAIN_SIZE / 2,
            DOMAIN_SIZE / 2,
        )
        eps_r, sigma = rasterize(
            grid, [(Sphere(center=center, radius=RADIUS), 1.0, 1e7)]
        )
        masks.append(np.asarray(sigma) > 1.0)
        materials.append(MaterialArrays(
            eps_r=eps_r,
            sigma=sigma,
            mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
        ))

    occupied = [int(np.count_nonzero(mask)) for mask in masks]
    assert occupied[0] == occupied[1] > 0, (
        f"translated sphere rasterization changed: shifts={SHIFTS}, cells={occupied}"
    )
    assert np.array_equal(masks[1], np.roll(masks[0], 4, axis=0)), (
        "the +4-cell sphere mask is not the centre mask rolled by 4 along x"
    )

    freqs = KA_BINS * C0 / (2 * np.pi * RADIUS)
    monostatic_dbsm = []
    for material in materials:
        result = rcs_module.compute_rcs(
            grid, material, N_STEPS,
            f0=F0, bandwidth=0.5, theta_inc=0.0, polarization="ez",
            theta_obs=np.array([np.pi / 2]), phi_obs=np.array([0.0, np.pi]),
            freqs=freqs, boundary="cpml", cpml_layers=CPML_LAYERS,
        )
        monostatic_dbsm.append(np.asarray(result.monostatic_rcs, dtype=float))

    differences = np.abs(monostatic_dbsm[0] - monostatic_dbsm[1])
    print(
        f"\nRCS translation: shifts={SHIFTS}, occupied={occupied}, "
        f"ka=[{', '.join(f'{ka:.2f}' for ka in KA_BINS)}], "
        f"differences_dB=[{', '.join(f'{delta:.6f}' for delta in differences)}]",
        flush=True,
    )
    return monostatic_dbsm


@pytest.mark.parametrize("index", range(len(KA_BINS)), ids=[f"ka={ka:.2f}" for ka in KA_BINS])
def test_monostatic_rcs_is_translation_invariant(translated_sphere_rcs, index):
    """Bound the monostatic change from translating the same discretized sphere."""
    centre_dbsm = float(translated_sphere_rcs[0][index])
    moved_dbsm = float(translated_sphere_rcs[1][index])
    difference_db = abs(centre_dbsm - moved_dbsm)
    bar_db = BARS_DB[index]
    assert difference_db <= bar_db, (
        f"bin={index}, ka={KA_BINS[index]:.2f}: "
        f"sigma_centre={centre_dbsm:.6f} dBsm, sigma_+4={moved_dbsm:.6f} dBsm, "
        f"|sigma_centre - sigma_+4|={difference_db:.6f} dB exceeds b={bar_db:.2f} dB"
    )
