"""Tests for thin conductor subcell model.

Validates:
1. Effective conductivity computation
2. Selective application via shape mask
3. Integration with Simulation API
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.geometry.csg import Box
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor


def test_thin_conductor_sigma_eff():
    """σ_eff = σ_bulk · (t / Δx) should be correct for lossy conductors.

    For PEC-level conductors (σ_eff >= 1e6), the code routes to pec_mask
    instead of setting sigma. Test both cases.
    """
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
    dx = grid.dx

    # Case 1: Lossy conductor (σ_eff below PEC threshold)
    sigma_bulk_lossy = 1e4
    thickness = 35e-6
    expected_sigma_eff = sigma_bulk_lossy * (thickness / dx)

    shape_box = Box((0.005, 0.005, 0.0), (0.015, 0.015, 0.001))
    tc_lossy = ThinConductor(shape=shape_box, sigma_bulk=sigma_bulk_lossy, thickness=thickness)

    materials = init_materials(grid.shape)
    materials, pec_mask = apply_thin_conductor(grid, tc_lossy, materials)

    mask = shape_box.mask(grid)
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        sigma_val = float(materials.sigma[i, j, k])
        assert abs(sigma_val - expected_sigma_eff) / max(expected_sigma_eff, 1e-30) < 0.01, \
            f"Lossy: σ_eff={sigma_val:.4e}, expected={expected_sigma_eff:.4e}"

    # Case 2: PEC conductor (copper → σ_eff > 1e6 → pec_mask)
    sigma_bulk_pec = 5.8e7  # copper
    tc_pec = ThinConductor(shape=shape_box, sigma_bulk=sigma_bulk_pec, thickness=thickness)
    assert tc_pec.is_pec, "Copper 35µm should be PEC"

    materials2 = init_materials(grid.shape)
    materials2, pec_mask2 = apply_thin_conductor(grid, tc_pec, materials2)

    if pec_mask2 is not None and len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert bool(pec_mask2[i, j, k]), "PEC conductor should set pec_mask"

    # Outside should be unaffected
    outside_idx = np.argwhere(~np.array(mask))
    if len(outside_idx) > 0:
        i, j, k = outside_idx[0]
        assert float(materials.sigma[i, j, k]) == 0.0


def test_thin_conductor_preserves_outside():
    """Thin conductor should not modify material outside its shape."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))

    # Pre-fill with some background
    materials = init_materials(grid.shape)
    materials = materials._replace(
        eps_r=jnp.full(grid.shape, 4.4, dtype=jnp.float32),
        sigma=jnp.full(grid.shape, 0.025, dtype=jnp.float32),
    )

    shape_box = Box((0.005, 0.005, 0.0), (0.01, 0.01, 0.001))
    # Use lossy conductor (below PEC threshold) so sigma is modified, not pec_mask
    tc = ThinConductor(shape=shape_box, sigma_bulk=1e4, thickness=35e-6, eps_r=1.0)
    materials, _ = apply_thin_conductor(grid, tc, materials)

    mask = shape_box.mask(grid)

    # Inside: eps_r should be 1.0 (conductor), sigma should be thin-conductor value
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert float(materials.eps_r[i, j, k]) == 1.0

    # Outside: eps_r should still be 4.4, sigma should still be 0.025
    outside_idx = np.argwhere(~np.array(mask))
    if len(outside_idx) > 0:
        i, j, k = outside_idx[len(outside_idx) // 2]
        assert abs(float(materials.eps_r[i, j, k]) - 4.4) < 1e-4
        assert abs(float(materials.sigma[i, j, k]) - 0.025) < 1e-4


def test_thin_conductor_api_integration():
    """ThinConductor works through the Simulation API.

    Copper (5.8e7 S/m, 35um) exceeds PEC threshold → routed to pec_mask.
    """
    from rfx.api import Simulation

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.002), boundary="pec")
    sim.add_material("substrate", eps_r=4.4, sigma=0.025)
    sim.add(Box((0, 0, 0), (0.02, 0.02, 0.001)), material="substrate")
    sim.add_thin_conductor(
        Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001)),
        sigma_bulk=5.8e7,
        thickness=35e-6,
    )

    # Should build without error
    grid = sim._build_grid()
    materials, debye, lorentz, pec_mask, pec_shapes, *_ = sim._assemble_materials(grid)

    # Copper thin conductor is PEC-level → check pec_mask, not sigma
    tc_box = Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001))
    mask = tc_box.mask(grid)
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0 and pec_mask is not None:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert bool(pec_mask[i, j, k]), \
            "Copper thin conductor should be in pec_mask"

    print(f"\nThin conductor API integration: OK, grid={grid.shape}")


def test_thin_conductor_nonuniform_reflects_like_box():
    """#369: a PEC thin conductor on the non-uniform (dz_profile) path must
    reflect/resonate identically to a geometry PEC Box at the same cell.

    Before the fix, ``assemble_materials_nu`` skipped ALL thin conductors
    ("needs a uniform Grid"), so an ``add_thin_conductor`` PEC sheet on a
    dz_profile grid was silently dropped — no pec_mask, no reflection: the
    box-PEC cavity rang while the thin-sheet cavity decayed. This is an
    end-to-end behavioural lock through the real NU run path.
    """
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    dx = 1.5e-3
    N = 30
    L = N * dx
    dz = [dx] * N
    zc = 15 * dx + 0.5 * dx

    def late_energy(kind):
        sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                         boundary="cpml", cpml_layers=8)
        px_lo, px_hi = L / 2 - 6 * dx, L / 2 + 6 * dx
        py_lo, py_hi = L / 2 - 4 * dx, L / 2 + 4 * dx
        if kind == "box":
            sim.add(Box((px_lo, py_lo, 15 * dx), (px_hi, py_hi, 16 * dx)),
                    material="pec")
        else:
            sim.add_thin_conductor(Box((px_lo, py_lo, zc), (px_hi, py_hi, zc)),
                                   sigma_bulk=5.8e7, thickness=35e-6)
        sim.add_source(position=(px_lo + 3 * dx, L / 2, 10 * dx),
                       component="ez",
                       waveform=GaussianPulse(f0=6e9, bandwidth=1.0))
        sim.add_probe(position=(px_lo + 8 * dx, L / 2 + 2 * dx, 10 * dx),
                      component="ez")
        ts = np.asarray(sim.run(n_steps=250, skip_preflight=True).time_series).ravel()
        return float(np.max(np.abs(ts[-50:])))

    box_e = late_energy("box")
    thin_e = late_energy("thin")
    assert box_e > 0.0
    # Identical nominal patch cell → identical late-time cavity energy.
    # Pre-fix this ratio was <= 0.37 (thin dropped); post-fix it is ~1.0.
    assert abs(box_e - thin_e) / box_e < 0.1, (
        f"#369 regression: PEC thin sheet must ring like a box PEC on the NU "
        f"path (box={box_e:.3e}, thin={thin_e:.3e}, ratio={thin_e / box_e:.3f})")


def test_thin_conductor_lossy_warns_on_nonuniform():
    """A lossy (non-PEC) thin conductor is not yet supported on the NU path;
    it must warn (not silently drop) so the omission is visible."""
    import warnings
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    dx = 1.5e-3
    N = 24
    L = N * dx
    dz = [dx] * N
    sim = Simulation(freq_max=10e9, domain=(L, L, 0), dx=dx, dz_profile=dz,
                     boundary="cpml", cpml_layers=6)
    # sigma_eff below the PEC threshold → lossy branch.
    sim.add_thin_conductor(Box((9 * dx, 9 * dx, 15.5 * dx),
                               (15 * dx, 15 * dx, 15.5 * dx)),
                           sigma_bulk=1.0e3, thickness=35e-6)
    sim.add_source(position=(6 * dx, L / 2, 10 * dx), component="ez",
                   waveform=GaussianPulse(f0=6e9, bandwidth=1.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim.run(n_steps=10, skip_preflight=True)
    assert any("non-uniform" in str(wi.message).lower()
               and "thin conductor" in str(wi.message).lower() for wi in w), \
        "lossy thin conductor on NU path must emit a skip warning"
