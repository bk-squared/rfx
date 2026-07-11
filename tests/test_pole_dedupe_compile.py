"""Pole->mask dedupe semantics in ``_CompileMixin._assemble_materials``.

Issue #274: the pole->mask dicts must key hashable poles BY VALUE (as on
main) so that equal-by-value poles declared by different materials on
overlapping geometry merge their masks and the pole's delta_eps is applied
once per cell.  Keying by ``id(pole)`` instead creates one dict slot per
declaration, and ``init_debye`` then sums one beta array per slot
(``beta_sum``, rfx/materials/debye.py) — silently double-counting delta_eps
on overlap cells (Lorentz analog: two aux-field sets summed).

Poles carrying jax-array fields (traced JVPTracer under ``jax.grad``, or a
concrete ArrayImpl) are unhashable and fall back to identity keying via
``_pole_key`` — the same pole object reused across geometry entries still
merges to one entry, and compile must not raise TypeError.

Small closed PEC boxes, a few cells per side — each test runs in seconds.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation, Box
from rfx.core.yee import EPS_0
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole
from rfx.sources.sources import GaussianPulse


FREQ_MAX = 20e9
DX = 1.5e-3
DOMAIN = (0.012, 0.006, 0.006)   # 8 x 4 x 4 cells

# Two overlapping boxes: overlap region x in [0.004, 0.008).
BOX_A = Box((0.000, 0.000, 0.000), (0.008, 0.006, 0.006))
BOX_B = Box((0.004, 0.000, 0.000), (0.012, 0.006, 0.006))


def _two_material_sim(debye_poles_a=None, debye_poles_b=None,
                      lorentz_poles_a=None, lorentz_poles_b=None):
    """Two dispersive materials on overlapping boxes in a closed PEC box."""
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_material("mat_a", eps_r=2.0,
                     debye_poles=debye_poles_a, lorentz_poles=lorentz_poles_a)
    sim.add_material("mat_b", eps_r=2.0,
                     debye_poles=debye_poles_b, lorentz_poles=lorentz_poles_b)
    sim.add(BOX_A, material="mat_a")
    sim.add(BOX_B, material="mat_b")
    return sim


def _overlap_cell(grid):
    """Index of a cell inside both boxes."""
    both = np.array(BOX_A.mask(grid)) & np.array(BOX_B.mask(grid))
    assert both.any(), "fixture must produce a non-empty overlap"
    return tuple(np.argwhere(both)[0])


def test_equal_concrete_debye_poles_merge_on_overlap():
    """Equal-by-value DebyePoles from two materials -> ONE spec entry.

    Physical witness: on overlap cells beta_sum equals the single-pole
    beta (ratio 1.0).  The id-keyed regression gave two entries and
    ratio 2.0 — delta_eps silently double-counted.
    """
    pole = lambda: DebyePole(delta_eps=74.1, tau=8.3e-12)  # noqa: E731
    sim = _two_material_sim(debye_poles_a=[pole()], debye_poles_b=[pole()])

    grid = sim._build_grid()
    _, debye_spec, _, _, _, _, _ = sim._assemble_materials(grid)

    assert debye_spec is not None
    debye_poles, debye_masks = debye_spec
    assert len(debye_poles) == 1
    # Merged mask covers the union of both boxes.
    union = np.array(BOX_A.mask(grid)) | np.array(BOX_B.mask(grid))
    assert np.array_equal(np.array(debye_masks[0]), union)

    # Physical witness via the coefficients init_debye actually builds.
    _, debye, _ = sim._build_materials(grid)
    coeffs, _ = debye
    beta_sum = np.array(jnp.sum(coeffs.beta, axis=0))
    de, tau, dt = 74.1, 8.3e-12, grid.dt
    single_beta = EPS_0 * de * dt / (2.0 * tau + dt)
    ratio = beta_sum[_overlap_cell(grid)] / single_beta
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-5)


def test_equal_concrete_lorentz_poles_merge_on_overlap():
    """Equal-by-value LorentzPoles from two materials -> ONE spec entry."""
    pole = lambda: lorentz_pole(1.5, 2 * np.pi * 9e9, 2e9)  # noqa: E731
    sim = _two_material_sim(lorentz_poles_a=[pole()], lorentz_poles_b=[pole()])

    grid = sim._build_grid()
    _, _, lorentz_spec, _, _, _, _ = sim._assemble_materials(grid)

    assert lorentz_spec is not None
    lorentz_poles, lorentz_masks = lorentz_spec
    assert len(lorentz_poles) == 1
    union = np.array(BOX_A.mask(grid)) | np.array(BOX_B.mask(grid))
    assert np.array_equal(np.array(lorentz_masks[0]), union)

    # One aux-field set: coeffs.c has exactly one pole slab, active on
    # the overlap cell (two slabs would sum two P contributions).
    _, _, lorentz = sim._build_materials(grid)
    coeffs, _ = lorentz
    assert coeffs.c.shape[0] == 1
    assert float(coeffs.c[0][_overlap_cell(grid)]) > 0.0


def test_unhashable_pole_compiles_via_fallback():
    """A pole with a jax-array field (unhashable) must still compile.

    Main raises TypeError here.  With the #274 identity fallback the SAME
    pole object reused across two overlapping geometry entries merges to
    one spec entry.
    """
    traced_like = DebyePole(delta_eps=jnp.asarray(2.0, dtype=jnp.float32),
                            tau=8.3e-12)
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_material("dut", eps_r=2.0, debye_poles=[traced_like])
    sim.add(BOX_A, material="dut")
    sim.add(BOX_B, material="dut")  # same material -> same pole object

    grid = sim._build_grid()
    _, debye_spec, _, _, _, _, _ = sim._assemble_materials(grid)

    assert debye_spec is not None
    debye_poles, debye_masks = debye_spec
    assert len(debye_poles) == 1
    assert debye_poles[0] is traced_like
    union = np.array(BOX_A.mask(grid)) | np.array(BOX_B.mask(grid))
    assert np.array_equal(np.array(debye_masks[0]), union)


def test_traced_pole_grad_smoke():
    """Tiniest jax.grad through forward() with a traced-pole material.

    Exercises the identity-fallback path with a real JVPTracer field —
    the supported differentiable calibration seam (issue #274; full
    recovery lives in test_calibration_dispersive_recovery.py).
    """
    freqs = jnp.asarray([10e9], dtype=jnp.float32)

    def loss(delta_eps):
        sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX,
                         boundary="pec")
        sim.add_material("dut", eps_r=3.0,
                         debye_poles=[DebyePole(delta_eps=delta_eps,
                                                tau=1.5e-11)])
        sim.add(Box((0.004, 0.0, 0.0), (0.008, 0.006, 0.006)),
                material="dut")
        sim.add_port(
            position=(0.003, 0.003, 0.003),
            component="ez",
            impedance=50.0,
            waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.9,
                                   amplitude=1.0),
        )
        result = sim.forward(
            port_s11_freqs=freqs,
            num_periods=3.0,
            checkpoint=True,
            skip_preflight=True,
        )
        return jnp.sum(jnp.abs(result.s_params) ** 2)

    g = jax.grad(loss)(jnp.asarray(2.0, dtype=jnp.float32))
    assert np.isfinite(float(g))
