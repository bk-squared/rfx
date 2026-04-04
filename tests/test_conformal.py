"""Tests for Dey-Mittra conformal PEC boundaries."""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h, update_e_aniso
from rfx.geometry.csg import Box, Sphere, Cylinder
from rfx.geometry.conformal import (
    compute_conformal_weights,
    compute_conformal_weights_sdf,
    clamp_conformal_weights,
    apply_conformal_pec,
    conformal_eps_correction,
)
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse


def test_conformal_weights_pec_box():
    """Weights should be exactly 0 inside box, 1 well outside."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    box = Box((0.015, 0.015, 0.015), (0.035, 0.035, 0.035))

    w_ex, w_ey, w_ez = compute_conformal_weights(grid, [box])

    mask = np.array(box.mask(grid))

    # Inside PEC: weights = 0
    assert float(jnp.max(w_ex[mask])) == 0.0, "Ex weight inside PEC should be 0"
    assert float(jnp.max(w_ey[mask])) == 0.0, "Ey weight inside PEC should be 0"
    assert float(jnp.max(w_ez[mask])) == 0.0, "Ez weight inside PEC should be 0"

    # Well outside PEC (corner cells far from boundary): weights = 1
    assert float(w_ex[0, 0, 0]) == 1.0, "Far exterior Ex weight should be 1"
    assert float(w_ey[0, 0, 0]) == 1.0, "Far exterior Ey weight should be 1"


def test_conformal_sphere_has_fractional_weights():
    """A PEC sphere should produce fractional conformal weights at its surface."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    sphere = Sphere((0.025, 0.025, 0.025), 0.015)

    w_ex, w_ey, w_ez = compute_conformal_weights(grid, [sphere], n_sub=4)

    w_ex_np = np.array(w_ex)
    n_zero = int(np.sum(w_ex_np == 0))
    n_one = int(np.sum(w_ex_np == 1.0))
    n_frac = int(np.sum((w_ex_np > 0) & (w_ex_np < 1.0)))

    # Sphere should produce fractional weights at surface cells
    assert n_zero > 0, "Should have interior PEC cells"
    assert n_one > 0, "Should have exterior cells"
    assert n_frac > 0, "Sphere should produce fractional boundary weights"

    # Fractional weights should be between 0 and 1
    frac_vals = w_ex_np[(w_ex_np > 0) & (w_ex_np < 1.0)]
    assert np.all(frac_vals > 0) and np.all(frac_vals < 1.0)


def test_conformal_disabled_matches_original():
    """With all weights=1, conformal should give identical results to no conformal."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), dx=0.003, cpml_layers=0)
    w_ones = (jnp.ones(grid.shape), jnp.ones(grid.shape), jnp.ones(grid.shape))

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=3e9)
    src = (grid.nx // 2, grid.ny // 2, grid.nz // 2)

    # Run with conformal weights = 1 (should be identity)
    for n in range(20):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_conformal_pec(state, *w_ones)
        state = state._replace(ez=state.ez.at[src].add(pulse(t)))

    ez_conformal = float(jnp.sum(state.ez ** 2))

    # Run without conformal
    state2 = init_state(grid.shape)
    for n in range(20):
        t = n * grid.dt
        state2 = update_h(state2, materials, grid.dt, grid.dx)
        state2 = update_e(state2, materials, grid.dt, grid.dx)
        state2 = apply_pec(state2)
        state2 = state2._replace(ez=state2.ez.at[src].add(pulse(t)))

    ez_original = float(jnp.sum(state2.ez ** 2))

    diff = abs(ez_conformal - ez_original)
    assert diff < 1e-10, f"Conformal with w=1 should match original: diff={diff}"


# ---------------------------------------------------------------------------
# New tests for SDF-based conformal weights
# ---------------------------------------------------------------------------

def test_conformal_weights_sdf_sphere():
    """SDF-based conformal weights for a sphere should have fractional cells."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    sphere = Sphere((0.025, 0.025, 0.025), 0.015)

    w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [sphere])

    w_np = np.array(w_ex)
    n_zero = int(np.sum(w_np == 0))
    n_one = int(np.sum(w_np == 1.0))
    n_frac = int(np.sum((w_np > 0) & (w_np < 1)))

    assert n_zero > 0, "Should have interior PEC cells (w=0)"
    assert n_one > 0, "Should have exterior cells (w=1)"
    assert n_frac > 0, "Sphere should have fractional boundary cells"


def test_conformal_weights_sdf_cylinder():
    """SDF-based weights for a cylinder should have fractional cells."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    cyl = Cylinder((0.025, 0.025, 0.025), 0.012, 0.04, axis="z")

    w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [cyl])

    w_np = np.array(w_ez)
    n_frac = int(np.sum((w_np > 0) & (w_np < 1)))
    assert n_frac > 0, "Cylinder should have fractional boundary cells"


def test_conformal_weights_sdf_empty():
    """No PEC shapes: all weights should be 1.0."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [])

    assert float(w_ex.min()) == 1.0
    assert float(w_ey.min()) == 1.0
    assert float(w_ez.min()) == 1.0


def test_clamp_preserves_pec_and_free():
    """Clamping: w=0 stays 0, w=1 stays 1, small w -> w_min."""
    w = jnp.array([0.0, 0.05, 0.1, 0.5, 1.0])
    dummy = jnp.ones_like(w)

    cw, _, _ = clamp_conformal_weights(w, dummy, dummy, w_min=0.1)
    cw_np = np.array(cw)

    assert cw_np[0] == 0.0, "w=0 (fully PEC) should remain 0"
    assert abs(cw_np[1] - 0.1) < 1e-6, "w=0.05 should clamp to w_min=0.1"
    assert abs(cw_np[2] - 0.1) < 1e-6, "w=0.1 should remain 0.1"
    assert abs(cw_np[3] - 0.5) < 1e-6, "w=0.5 should stay unchanged"
    assert abs(cw_np[4] - 1.0) < 1e-6, "w=1 (fully free) should remain 1"


def test_conformal_eps_correction_amplification():
    """For w=0.5, eps_eff should be 2x the base eps."""
    eps_r = jnp.ones((3, 3, 3))
    w = 0.5 * jnp.ones((3, 3, 3))
    ones = jnp.ones((3, 3, 3))

    eps_ex, eps_ey, eps_ez = conformal_eps_correction(eps_r, w, ones, ones)

    # eps_ex should be eps_r / 0.5 = 2.0
    assert abs(float(eps_ex[1, 1, 1]) - 2.0) < 1e-6, \
        f"Expected eps=2.0, got {float(eps_ex[1,1,1])}"
    # eps_ey should be eps_r / 1.0 = 1.0
    assert abs(float(eps_ey[1, 1, 1]) - 1.0) < 1e-6


def test_conformal_pec_box_identical_to_staircase():
    """For an axis-aligned box, conformal should give near-identical results to staircase.

    Because the SDF linear approximation may create a thin transition layer
    at box faces, we check that fields match within a small tolerance.
    """
    grid = Grid(freq_max=5e9, domain=(0.04, 0.04, 0.04), dx=0.004, cpml_layers=0)
    box = Box((0.012, 0.012, 0.012), (0.028, 0.028, 0.028))

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=3e9)
    src = (2, 2, 2)

    # Staircase run
    from rfx.boundaries.pec import apply_pec_mask
    pec_mask = box.mask(grid)
    for n in range(30):
        t = n * grid.dt
        state = state._replace(ez=state.ez.at[src].add(pulse(t)))
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_pec_mask(state, pec_mask)

    ez_stair = np.array(state.ez)

    # Conformal run
    w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [box])
    w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez)
    eps_ex, eps_ey, eps_ez = conformal_eps_correction(materials.eps_r, w_ex, w_ey, w_ez)

    state2 = init_state(grid.shape)
    for n in range(30):
        t = n * grid.dt
        state2 = state2._replace(ez=state2.ez.at[src].add(pulse(t)))
        state2 = update_h(state2, materials, grid.dt, grid.dx)
        state2 = update_e_aniso(state2, materials, eps_ex, eps_ey, eps_ez,
                                grid.dt, grid.dx)
        state2 = apply_pec(state2)
        state2 = apply_conformal_pec(state2, w_ex, w_ey, w_ez)

    ez_conf = np.array(state2.ez)

    # Fields should be very close (not bitwise identical due to SDF transition)
    max_diff = np.max(np.abs(ez_stair - ez_conf))
    max_field = max(np.max(np.abs(ez_stair)), 1e-30)
    rel_diff = max_diff / max_field
    assert rel_diff < 0.1, \
        f"Conformal box should be close to staircase, got relative diff={rel_diff:.4f}"


def test_api_conformal_flag():
    """sim.run(conformal_pec=True) should work end to end."""
    from rfx.api import Simulation

    # Place source and probe OUTSIDE the PEC cylinder
    # Cylinder: center=(0.025,0.025,0.025), radius=0.008, height=0.02
    # So it spans z=[0.015, 0.035], r=0.008 from center
    # Source at (0.005, 0.025, 0.025) is outside (x is far from center)
    # Probe at (0.045, 0.025, 0.025) is outside
    sim = Simulation(freq_max=5e9, domain=(0.05, 0.05, 0.05), boundary="pec")
    sim.add(Cylinder((0.025, 0.025, 0.025), 0.008, 0.02), material="pec")
    sim.add_port((0.005, 0.025, 0.025), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.045, 0.025, 0.025), "ez")

    result = sim.run(n_steps=50, conformal_pec=True)
    assert result.time_series.shape == (50, 1)
    # Signal should not be all zeros (source is outside PEC)
    assert float(jnp.max(jnp.abs(result.time_series))) > 0


def test_conformal_stability_10k_steps():
    """Energy must not grow with conformal PEC over 10000 steps."""
    grid = Grid(freq_max=3e9, domain=(0.06, 0.06, 0.06), dx=0.006, cpml_layers=0)
    sphere = Sphere((0.03, 0.03, 0.03), 0.015)

    w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [sphere])
    w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)
    eps_ex, eps_ey, eps_ez = conformal_eps_correction(1.0, w_ex, w_ey, w_ez)

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=1.5e9)
    src = (grid.nx // 2, grid.ny // 2, grid.nz // 4)

    # Inject source for first 200 steps, then let it ring
    max_energy = 0.0
    for n in range(10000):
        t = n * grid.dt
        if n < 200:
            state = state._replace(ez=state.ez.at[src].add(pulse(t)))
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez,
                               grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_conformal_pec(state, w_ex, w_ey, w_ez)

        if n % 500 == 0:
            energy = float(jnp.sum(state.ex**2 + state.ey**2 + state.ez**2))
            if n > 200:
                max_energy = max(max_energy, energy)

    final_energy = float(jnp.sum(state.ex**2 + state.ey**2 + state.ez**2))

    # Energy should not diverge (allow some oscillation but not growth)
    # After source stops, energy should decay or stay bounded
    assert final_energy < max_energy * 10, \
        f"Energy diverged: final={final_energy:.6e}, max_post_source={max_energy:.6e}"


# ---------------------------------------------------------------------------
# Coupled advanced workflow tests (conformal + CPML / dispersive / gradients)
# ---------------------------------------------------------------------------

def test_conformal_with_cpml():
    """Conformal PEC should work with CPML boundary (no NaN, non-zero fields)."""
    from rfx.api import Simulation

    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="cpml")
    sim.add(Cylinder((0.02, 0.02, 0.02), 0.008, 0.03, axis="z"), material="pec")
    sim.add_source((0.005, 0.005, 0.02), component="ez")
    sim.add_probe((0.035, 0.035, 0.02), component="ez")

    result = sim.run(n_steps=300, conformal_pec=True)
    assert result is not None
    ts = np.array(result.time_series)
    # Non-zero fields: source is well outside PEC cylinder, probe should see signal
    assert np.max(np.abs(ts)) > 1e-10, \
        f"Fields are essentially zero: max|ts|={np.max(np.abs(ts)):.2e}"
    # No NaN
    assert not np.any(np.isnan(ts)), "NaN detected in time series"


def test_conformal_with_dispersive():
    """Conformal PEC + Debye dielectric should not conflict."""
    from rfx.api import Simulation
    from rfx.materials.debye import DebyePole

    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec")
    # Fill domain with dispersive Debye medium
    sim.add_material("disp", eps_r=4.0, debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
    sim.add(Box((0, 0, 0), (0.04, 0.04, 0.04)), material="disp")
    # Embed PEC cylinder (conformal)
    sim.add(Cylinder((0.02, 0.02, 0.02), 0.005, 0.03, axis="z"), material="pec")
    sim.add_source((0.005, 0.005, 0.02), component="ez")
    sim.add_probe((0.035, 0.035, 0.02), component="ez")

    result = sim.run(n_steps=200, conformal_pec=True)
    assert result is not None
    ts = np.array(result.time_series)
    assert not np.any(np.isnan(ts)), "NaN detected in conformal+dispersive run"
    assert not np.any(np.isinf(ts)), "Inf detected in conformal+dispersive run"


def test_conformal_gradient_flows():
    """jax.grad should flow through a conformal PEC simulation."""
    import jax
    from rfx.grid import Grid
    from rfx.core.yee import MaterialArrays
    from rfx.simulation import run, make_source, make_probe

    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), dx=0.002, cpml_layers=0)
    cyl = Cylinder((0.01, 0.01, 0.01), 0.004, 0.015, axis="z")

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    n_steps = 50
    src = make_source(grid, (0.002, 0.002, 0.01), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.018, 0.018, 0.01), "ez")

    # Pre-compute conformal weights (outside the AD-traced objective)
    from rfx.geometry.conformal import (
        compute_conformal_weights_sdf,
        clamp_conformal_weights,
        conformal_eps_correction,
    )
    w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, [cyl])
    w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)

    def objective(eps_r):
        eps_ex, eps_ey, eps_ez = conformal_eps_correction(eps_r, w_ex, w_ey, w_ez)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(
            grid, mats, n_steps,
            sources=[src], probes=[prb],
            checkpoint=True,
            aniso_eps=(eps_ex, eps_ey, eps_ez),
            conformal_weights=(w_ex, w_ey, w_ez),
        )
        return jnp.sum(result.time_series ** 2)

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    grad = jax.grad(objective)(eps_r)
    grad_max = float(jnp.max(jnp.abs(grad)))
    assert grad_max > 1e-15, f"Gradient is zero: |grad|_max={grad_max:.2e}"
    assert not np.any(np.isnan(np.array(grad))), "NaN in gradient"
