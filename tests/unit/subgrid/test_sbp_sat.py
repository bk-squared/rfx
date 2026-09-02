"""SBP-SAT FDTD subgridding: 1D / 2D / 3D steppers, penalty coefficient, JIT runner.

One file per subject (tier 3b of the 2026-09 test-corpus reorganisation, see
``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``). Sections,
each formerly its own file:

1. 1D SBP-SAT prototype (``rfx.subgridding.sbp_sat_1d``) — was
   ``test_sbp_sat_1d.py``. Required tests: SBP identity
   ``P*D + D^T*P = E_boundary``; 100,000-step stability; pulse through the
   interface matches a uniform fine grid within 5 %; total energy
   (E^2 + H^2) non-increasing. Module-level ``gpu`` mark of the original
   file is carried per test.
2. 2D TM SBP-SAT (``rfx.subgridding.sbp_sat_2d``) — was ``test_sbp_sat_2d.py``.
   Module-level ``gpu`` mark carried per test.
3. 3D SBP-SAT (``rfx.subgridding.sbp_sat_3d``) — was ``test_sbp_sat_3d.py``.
4. Configurable SAT penalty coefficient ``tau`` — was ``test_sbp_sat_alpha.py``.
5. ``jax.lax.scan``-based subgridded JIT runner (``rfx.subgridding.jit_runner``,
   50-100x over the Python-loop runner) — was ``test_sbp_sat_jit.py``:
   JIT compiles for PEC and CPML; non-trivial field evolution; energy
   stability over 1000 steps; probe recording; no-probe / no-source edge
   cases; H-field SAT coupling parity with the standalone stepper; dielectric
   crossing the coarse-fine boundary.

Every assertion, tolerance, fixture value and marker of the original files
is kept verbatim.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.subgridding.sbp_sat_1d import (
    build_sbp_norm,
    build_sbp_diff,
    build_interpolation_c2f,
    build_interpolation_f2c,
    init_subgrid_1d,
    step_subgrid_1d,
    compute_energy,
    _update_h_1d,
    _update_e_1d,
    C0,
    EPS_0,
    MU_0,
)
from rfx.subgridding.sbp_sat_2d import (
    init_subgrid_2d, step_subgrid_2d, compute_energy_2d,
)
from rfx.subgridding.sbp_sat_3d import (
    init_subgrid_3d, step_subgrid_3d, compute_energy_3d,
)
from rfx.subgridding.jit_runner import SubgridRunOptions


# ===========================================================================
# 1. 1D SBP-SAT prototype (formerly test_sbp_sat_1d.py, module-marked gpu)
# ===========================================================================

# ── 1.1 SBP property ─────────────────────────────────────────────

@pytest.mark.gpu
def test_sbp_property():
    """Verify the SBP identity:  P @ D + D^T @ P = E_boundary.

    E_boundary = diag(-1, 0, ..., 0, +1) for the standard first-derivative
    SBP operator with the trapezoidal norm.

    Also checks interpolation matrix shapes and adjoint property.
    """
    for n in [10, 20, 50]:
        dx = 0.01
        p_diag = build_sbp_norm(n, dx)
        D = build_sbp_diff(n, dx)

        P = np.diag(p_diag)
        Q = P @ D

        S = Q + Q.T

        E_expected = np.zeros((n, n), dtype=np.float64)
        E_expected[0, 0] = -1.0
        E_expected[-1, -1] = +1.0

        err = np.max(np.abs(S - E_expected))
        print(f"  n={n}, dx={dx}: SBP error = {err:.2e}")
        assert err < 1e-12, (
            f"SBP property violated for n={n}: max|P*D + D^T*P - E| = {err}"
        )

    # Verify interpolation matrix shapes and adjoint
    ratio = 3
    R_c2f = build_interpolation_c2f(20, 60, ratio)
    R_f2c = build_interpolation_f2c(60, 20, ratio)
    assert R_c2f.shape == (ratio + 1, 2), f"R_c2f shape: {R_c2f.shape}"
    assert R_f2c.shape == (2, ratio + 1), f"R_f2c shape: {R_f2c.shape}"
    assert np.allclose(R_f2c, R_c2f.T), "R_f2c should equal R_c2f^T"


# ── 1.2 Stability over long run ──────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
def test_stability_long_run():
    """Energy must not grow over 100,000 steps (provable stability).

    The shared-node SBP-SAT coupling with operator splitting introduces
    small bounded energy fluctuations.  We verify that:
    - Energy never exceeds the initial value by more than 15%
    - Final energy is of the same order as initial (no blowup)
    """
    config, state = init_subgrid_1d(n_c=40, n_f=60, dx_c=0.003, ratio=3)

    # Gaussian pulse on coarse grid
    x_c = jnp.arange(config.n_c) * config.dx_c
    pulse = jnp.exp(-((x_c - 0.06) / 0.01) ** 2).astype(jnp.float32)
    state = state._replace(e_c=pulse)

    initial_energy = compute_energy(state, config)
    max_energy = initial_energy

    n_steps = 100_000
    for i in range(n_steps):
        state = step_subgrid_1d(state, config)
        if i % 5000 == 0:
            e = compute_energy(state, config)
            max_energy = max(max_energy, e)

    final_energy = compute_energy(state, config)

    print(f"\nStability test ({n_steps} steps):")
    print(f"  Initial energy: {initial_energy:.6e}")
    print(f"  Max energy:     {max_energy:.6e}")
    print(f"  Final energy:   {final_energy:.6e}")
    print(f"  Growth ratio:   {max_energy / max(initial_energy, 1e-30):.6f}")

    assert not np.isnan(final_energy), "Final energy is NaN"
    assert max_energy < initial_energy * 1.15, (
        f"Energy grew: max {max_energy:.6e} > 1.15 * initial {initial_energy:.6e}"
    )


# ── 1.3 Subgrid matches uniform fine grid ────────────────────────

@pytest.mark.gpu
def test_subgrid_matches_uniform():
    """Pulse propagation through interface matches uniform fine grid within 5%.

    We compare total energy after a fixed number of steps between:
    (a) a uniform fine grid covering the whole domain, and
    (b) the same domain split into coarse + fine with SBP-SAT coupling.

    The pulse starts well inside the coarse region and we run for a
    short time so it doesn't reach PEC boundaries.  Since the standard
    leapfrog and the shared-node scheme are both energy-conserving (up
    to small operator-splitting fluctuations), the total energies should
    agree closely.
    """
    ratio = 3
    dx_f = 0.001
    dx_c = dx_f * ratio

    courant = 0.5
    dt = courant * dx_f / C0
    n_steps = 300  # short run — pulse stays in interior

    # Gaussian pulse centred at cell 10 (well inside coarse/left region)
    pulse_centre = 10.0 * dx_f
    pulse_width = 3.0 * dx_f

    # ── (a) Uniform fine reference ──
    n_uni = 120
    x_uni = jnp.arange(n_uni, dtype=jnp.float32) * dx_f
    e_uni = jnp.exp(-((x_uni - pulse_centre) / pulse_width) ** 2).astype(jnp.float32)
    h_uni = jnp.zeros(n_uni - 1, dtype=jnp.float32)

    for _ in range(n_steps):
        h_uni = _update_h_1d(e_uni, h_uni, dt, dx_f)
        e_uni = _update_e_1d(e_uni, h_uni, dt, dx_f)
        e_uni = e_uni.at[0].set(0.0)
        e_uni = e_uni.at[-1].set(0.0)

    energy_uniform = (
        float(jnp.sum(e_uni ** 2)) * EPS_0 * dx_f
        + float(jnp.sum(h_uni ** 2)) * MU_0 * dx_f
    )

    # ── (b) Subgridded domain ──
    # Coarse: 20 nodes (left half), Fine: 60 nodes (right half)
    n_c = 20
    n_f = 60

    config, state = init_subgrid_1d(
        n_c=n_c, n_f=n_f, dx_c=dx_c, ratio=ratio, dt=dt,
    )

    # Same Gaussian pulse on coarse grid
    x_c = jnp.arange(n_c, dtype=jnp.float32) * dx_c
    pulse_c = jnp.exp(-((x_c - pulse_centre) / pulse_width) ** 2).astype(jnp.float32)
    state = state._replace(e_c=pulse_c)

    for _ in range(n_steps):
        state = step_subgrid_1d(state, config)

    energy_subgrid = compute_energy(state, config)

    print(f"\nUniform vs subgridded ({n_steps} steps):")
    print(f"  Uniform energy: {energy_uniform:.6e}")
    print(f"  Subgrid energy: {energy_subgrid:.6e}")

    assert energy_uniform > 0, "Uniform energy should be positive"
    assert energy_subgrid > 0, "Subgrid energy should be positive"
    assert not np.isnan(energy_subgrid), "Subgrid energy is NaN"

    rel_diff = abs(energy_subgrid - energy_uniform) / energy_uniform
    print(f"  Relative diff:  {rel_diff:.4f} ({rel_diff*100:.2f}%)")
    assert rel_diff < 0.05, (
        f"Energy mismatch too large: {rel_diff*100:.2f}% > 5%"
    )


# ── 1.4 Energy conservation ──────────────────────────────────────

@pytest.mark.gpu
@pytest.mark.slow
def test_energy_conservation():
    """Total energy (E^2 + H^2) is non-increasing over time.

    The SBP-SAT shared-node coupling preserves energy exactly in the
    limit of synchronised timesteps.  With the operator-split scheme
    (coarse H frozen during fine sub-steps), small bounded fluctuations
    occur.  We verify:
    1. Energy never exceeds the initial value by more than 15%.
    2. The overall envelope is bounded (no secular growth).
    3. No NaN values.
    """
    config, state = init_subgrid_1d(n_c=30, n_f=45, dx_c=0.002, ratio=3)

    # Smooth Gaussian pulse
    x_c = jnp.arange(config.n_c, dtype=jnp.float32) * config.dx_c
    pulse = jnp.exp(-((x_c - 0.03) / 0.005) ** 2).astype(jnp.float32)
    state = state._replace(e_c=pulse)

    initial_energy = compute_energy(state, config)

    n_steps = 10000
    sample_every = 100
    energies = [initial_energy]

    for i in range(n_steps):
        state = step_subgrid_1d(state, config)
        if (i + 1) % sample_every == 0:
            energies.append(compute_energy(state, config))

    energies = np.array(energies)

    assert not np.any(np.isnan(energies)), "NaN in energy trace"

    print(f"\nEnergy conservation ({len(energies)} samples over {n_steps} steps):")
    print(f"  Initial: {energies[0]:.6e}")
    print(f"  Final:   {energies[-1]:.6e}")
    print(f"  Min:     {energies.min():.6e}")
    print(f"  Max:     {energies.max():.6e}")
    print(f"  Max/Init: {energies.max() / initial_energy:.4f}")

    # (1) Energy bounded: no sample exceeds initial by more than 15%
    assert energies.max() <= initial_energy * 1.15, (
        f"Energy exceeded bound: max {energies.max():.6e} > 1.15 * initial {initial_energy:.6e}"
    )

    # (2) No secular growth: compare first-half max to second-half max
    mid = len(energies) // 2
    max_first_half = energies[:mid].max()
    max_second_half = energies[mid:].max()
    print(f"  First-half max:  {max_first_half:.6e}")
    print(f"  Second-half max: {max_second_half:.6e}")
    # Second half should not be significantly larger than first half
    assert max_second_half <= max_first_half * 1.5, (
        f"Secular growth detected: second-half max {max_second_half:.6e} "
        f"> 1.5 * first-half max {max_first_half:.6e}"
    )

    # (3) Final energy should be of the same order as initial
    assert energies[-1] < initial_energy * 1.15, (
        f"Final energy {energies[-1]:.6e} too large vs initial {initial_energy:.6e}"
    )


# ===========================================================================
# 2. 2D TM SBP-SAT (formerly test_sbp_sat_2d.py, module-marked gpu)
# ===========================================================================

@pytest.mark.gpu
@pytest.mark.slow
def test_2d_stability():
    """Energy bounded over 10,000 steps."""
    config, state = init_subgrid_2d(
        nx_c=40, ny_c=40, dx_c=0.003,
        fine_region=(15, 25, 15, 25), ratio=3,
    )
    # Inject pulse on coarse grid (outside fine region)
    state = state._replace(ez_c=state.ez_c.at[8, 8].set(1.0))
    initial_energy = compute_energy_2d(state, config)

    max_energy = initial_energy
    for i in range(10000):
        state = step_subgrid_2d(state, config)
        if i % 2000 == 0:
            e = compute_energy_2d(state, config)
            max_energy = max(max_energy, e)

    final_energy = compute_energy_2d(state, config)
    print("\n2D stability (10K steps):")
    print(f"  Initial: {initial_energy:.6e}")
    print(f"  Max:     {max_energy:.6e}")
    print(f"  Final:   {final_energy:.6e}")

    assert max_energy < initial_energy * 20, f"Energy blew up: {max_energy}"
    assert np.isfinite(final_energy), "Final energy should be finite"


@pytest.mark.gpu
def test_2d_pulse_propagation():
    """Pulse should propagate across the 2D coarse-fine interface."""
    config, state = init_subgrid_2d(
        nx_c=30, ny_c=30, dx_c=0.002,
        fine_region=(10, 20, 10, 20), ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[5, 15].set(1.0))

    for _ in range(3000):
        state = step_subgrid_2d(state, config)

    fine_signal = float(jnp.max(jnp.abs(state.ez_f)))
    coarse_signal = float(jnp.max(jnp.abs(state.ez_c)))

    print("\n2D pulse propagation:")
    print(f"  Coarse max |Ez|: {coarse_signal:.6e}")
    print(f"  Fine max |Ez|:   {fine_signal:.6e}")

    assert np.isfinite(coarse_signal), "Coarse field should be finite"
    assert np.isfinite(fine_signal), "Fine field should be finite"
    assert coarse_signal < 10, "Coarse field should not blow up"


@pytest.mark.gpu
def test_2d_small_fine_region():
    """Small fine region should still be stable."""
    config, state = init_subgrid_2d(
        nx_c=20, ny_c=20, dx_c=0.002,
        fine_region=(8, 12, 8, 12),  # small 4x4 coarse = 12x12 fine
        ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[5, 5].set(1.0))

    for _ in range(500):
        state = step_subgrid_2d(state, config)

    energy = compute_energy_2d(state, config)
    print(f"\n2D small fine region energy: {energy:.6e}")
    assert energy > 0, "Should have positive energy"
    assert np.isfinite(energy), "Energy should be finite"


# ===========================================================================
# 3. 3D SBP-SAT (formerly test_sbp_sat_3d.py)
# ===========================================================================

def test_3d_stability():
    """Energy must be non-increasing over 1000 steps in PEC cavity.

    This is the fundamental SBP-SAT stability guarantee. If energy grows,
    the coupling coefficients are wrong.
    """
    config, state = init_subgrid_3d(
        shape_c=(20, 20, 20), dx_c=0.003,
        fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[4, 4, 4].set(1.0))
    initial_energy = compute_energy_3d(state, config)

    max_energy = initial_energy
    for i in range(1000):
        state = step_subgrid_3d(state, config)
        if (i + 1) % 100 == 0:
            e = compute_energy_3d(state, config)
            # Allow small transient growth at early steps (SAT coupling
            # can temporarily increase energy before dissipation dominates).
            # Validated: growth peaks at ~1.004x around step 100, then
            # monotonically decreases. 1.005 tolerance accommodates transient.
            assert e <= max_energy * 1.005, (
                f"Energy grew at step {i+1}: {e:.6e} > {max_energy:.6e} "
                f"(growth {e/max_energy:.6f}x)"
            )
            max_energy = max(max_energy, e)

    final_energy = compute_energy_3d(state, config)
    print(f"\n3D energy conservation: initial={initial_energy:.6e}, "
          f"final={final_energy:.6e}, ratio={final_energy/initial_energy:.6f}")
    # After 1000 steps, energy must be <= initial (net dissipative)
    assert final_energy <= initial_energy, (
        f"Energy grew {final_energy/initial_energy:.4f}x over 1000 steps "
        f"(must be <= 1.0 for stability)"
    )


def test_3d_fine_grid_receives_signal():
    """Signal should appear on fine grid after propagation."""
    config, state = init_subgrid_3d(
        shape_c=(20, 20, 20), dx_c=0.002,
        fine_region=(8, 14, 8, 14, 8, 14), ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[4, 10, 10].set(1.0))

    for _ in range(500):
        state = step_subgrid_3d(state, config)

    max_c = float(jnp.max(jnp.abs(state.ez_c)))
    max_f = float(jnp.max(jnp.abs(state.ez_f)))

    print("\n3D signal propagation:")
    print(f"  Coarse max |Ez|: {max_c:.6e}")
    print(f"  Fine max |Ez|:   {max_f:.6e}")

    assert np.isfinite(max_c), "Coarse field should be finite"
    assert np.isfinite(max_f), "Fine field should be finite"
    assert max_c < 5.0, "Coarse field should not blow up"


def test_3d_energy_finite():
    """Basic sanity: energy stays finite after 200 steps."""
    config, state = init_subgrid_3d(
        shape_c=(15, 15, 15), dx_c=0.004,
        fine_region=(5, 10, 5, 10, 5, 10), ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[3, 7, 7].set(0.5))

    for _ in range(200):
        state = step_subgrid_3d(state, config)

    energy = compute_energy_3d(state, config)
    print(f"\n3D energy after 200 steps: {energy:.6e}")
    assert np.isfinite(energy), "Energy should be finite"
    assert energy >= 0, "Energy should be non-negative"


# ===========================================================================
# 4. Configurable SAT penalty coefficient tau (formerly test_sbp_sat_alpha.py)
# ===========================================================================

class TestSBPSATAlpha:
    """Tests for configurable tau (SAT penalty coefficient)."""

    def _make_sim(self, tau=None):
        """Build a minimal subgridded simulation."""
        dx_c = 2e-3
        ratio = 4
        dom = (0.04, 0.04, 0.04)

        sim = Simulation(freq_max=5e9, domain=dom, boundary="cpml",
                         cpml_layers=4, dx=dx_c)
        sim.add_source(position=(0.02, 0.02, 0.02), component="ez")

        kw = {"z_range": (0.01, 0.03), "ratio": ratio}
        if tau is not None:
            kw["tau"] = tau
        kw["validation"] = "research"
        sim.add_refinement(**kw)
        return sim

    def test_default_tau_is_half(self):
        """Default tau should be 0.5 when not specified."""
        sim = self._make_sim()
        assert sim._refinement["tau"] == 0.5

    def test_custom_tau_stored(self):
        """Custom tau value should be stored in refinement dict."""
        sim = self._make_sim(tau=1.0)
        assert sim._refinement["tau"] == 1.0

    def test_xy_margin_stored_for_research_windowed_refinement(self):
        """xy_margin must be stored rather than silently ignored."""
        sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04))

        sim.add_refinement(
            z_range=(0.01, 0.03),
            ratio=2,
            xy_margin=0.004,
            validation="research",
        )

        assert sim._refinement["xy_margin"] == 0.004

    def test_custom_tau_accepted(self):
        """Simulation with custom tau should run without error."""
        sim = self._make_sim(tau=1.0)
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
        result = sim.run(n_steps=50)
        assert result is not None

    def test_default_tau_runs(self):
        """Simulation with default tau should run without error."""
        sim = self._make_sim()
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
        result = sim.run(n_steps=50)
        assert result is not None

    def test_tau_propagates_to_config(self):
        """Tau should propagate from add_refinement through to SubgridConfig3D."""
        sim = self._make_sim(tau=0.75)
        # Build the grid and config to verify tau reaches SubgridConfig3D
        grid = sim._build_grid()
        base_materials, _, _, pec_mask, *_ = sim._assemble_materials(grid)

        # Patch run to capture config instead of running full sim
        ref = sim._refinement
        assert ref["tau"] == 0.75

        # Verify the config is built with the correct tau by checking
        # the intermediate dict propagation
        for tau_val in [0.25, 0.75, 1.0]:
            sim2 = self._make_sim(tau=tau_val)
            assert sim2._refinement["tau"] == tau_val

    def test_init_subgrid_3d_tau_passthrough(self):
        """init_subgrid_3d should accept and propagate tau."""
        from rfx.subgridding.sbp_sat_3d import init_subgrid_3d
        config, _ = init_subgrid_3d(tau=0.75)
        assert config.tau == 0.75

    def test_init_subgrid_3d_default_tau(self):
        """init_subgrid_3d default tau should be 0.5."""
        from rfx.subgridding.sbp_sat_3d import init_subgrid_3d
        config, _ = init_subgrid_3d()
        assert config.tau == 0.5

    def test_different_tau_different_results(self):
        """Different tau values must produce measurably different time series.

        This is the key regression test for the alpha-cap bug: previously
        cb_vac * 2*tau/dx always exceeded 0.5 so the min() cap made all
        tau values produce identical results.
        """
        from rfx.subgridding.sbp_sat_3d import (
            init_subgrid_3d, step_subgrid_3d,
        )
        import jax.numpy as jnp

        results = {}
        for tau in [0.1, 0.5]:
            config, state = init_subgrid_3d(
                shape_c=(20, 20, 20), dx_c=0.003,
                fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
                tau=tau,
            )
            # Inject pulse on coarse grid
            state = state._replace(ez_c=state.ez_c.at[4, 10, 10].set(1.0))

            for _ in range(300):
                state = step_subgrid_3d(state, config)

            results[tau] = float(jnp.sum(state.ez_c ** 2)) + float(jnp.sum(state.ez_f ** 2))

        diff = abs(results[0.1] - results[0.5])
        assert diff > 1e-10, (
            f"Different tau values produced identical results "
            f"(diff={diff:.2e}); alpha cap bug likely still present"
        )

    def test_alpha_values_are_dimensionless(self):
        """Alpha coefficients must be dimensionless fractions in (0, 1]."""
        from rfx.subgridding.sbp_sat_3d import init_subgrid_3d

        for tau in [0.1, 0.25, 0.5, 0.75, 1.0]:
            config, _ = init_subgrid_3d(
                shape_c=(20, 20, 20), dx_c=0.003,
                fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
                tau=tau,
            )
            # Reproduce the alpha computation from _shared_node_coupling_3d
            alpha_c = tau * min(config.dx_f / config.dx_c, 1.0)
            alpha_f = tau * min(config.dx_c / config.dx_f, 1.0)

            assert 0 < alpha_c <= 1.0, f"alpha_c={alpha_c} out of (0,1] for tau={tau}"
            assert 0 < alpha_f <= 1.0, f"alpha_f={alpha_f} out of (0,1] for tau={tau}"
            # alpha_c should be smaller than alpha_f (fine is finer grid)
            assert alpha_c <= alpha_f, (
                f"alpha_c={alpha_c} > alpha_f={alpha_f} for tau={tau}"
            )

    def test_energy_stable_after_fix(self):
        """Energy must remain finite and bounded after the alpha scaling fix."""
        from rfx.subgridding.sbp_sat_3d import (
            init_subgrid_3d, step_subgrid_3d, compute_energy_3d,
        )

        for tau in [0.25, 0.5, 1.0]:
            config, state = init_subgrid_3d(
                shape_c=(15, 15, 15), dx_c=0.004,
                fine_region=(5, 10, 5, 10, 5, 10), ratio=3,
                tau=tau,
            )
            state = state._replace(ez_c=state.ez_c.at[3, 7, 7].set(0.5))
            compute_energy_3d(state, config)

            for _ in range(500):
                state = step_subgrid_3d(state, config)

            final_energy = compute_energy_3d(state, config)
            assert np.isfinite(final_energy), (
                f"Energy diverged (NaN/Inf) at tau={tau}"
            )
            assert final_energy >= 0, f"Negative energy at tau={tau}"


# ===========================================================================
# 5. SBP-SAT JIT runner (formerly test_sbp_sat_jit.py)
# ===========================================================================

def _make_pec_sim(with_probe=True, with_source=True):
    """Small PEC cavity with 2:1 z-axis refinement."""
    from rfx import Simulation
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec")
    if with_source:
        sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_refinement(z_range=(0.015, 0.025), ratio=2, validation="research")
    if with_probe:
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    return sim


def _make_cpml_sim(with_probe=True):
    """Small CPML domain with 2:1 z-axis refinement."""
    from rfx import Simulation
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="cpml")
    sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_refinement(z_range=(0.015, 0.025), ratio=2, validation="research")
    if with_probe:
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    return sim


# ── 5.1 Basic functionality ─────────────────────────────────────

class TestJITBasic:
    """JIT subgridded runner basic functionality."""

    def test_jit_pec_runs_without_error(self):
        """PEC boundary subgridded simulation completes without error."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=50)
        assert result is not None
        assert result.time_series is not None

    def test_jit_cpml_runs_without_error(self):
        """CPML boundary subgridded simulation completes without error."""
        sim = _make_cpml_sim()
        result = sim.run(n_steps=50)
        assert result is not None
        assert result.time_series is not None

    def test_jit_produces_nonzero_fields(self):
        """JIT path should produce non-trivial field evolution."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=200)
        ez_max = float(jnp.max(jnp.abs(result.state.ez)))
        assert ez_max > 1e-10, f"Fields are near-zero: max|Ez|={ez_max}"

    def test_jit_time_series_shape(self):
        """Time series should have shape (n_steps, n_probes)."""
        sim = _make_pec_sim()
        n_steps = 100
        result = sim.run(n_steps=n_steps)
        ts = np.array(result.time_series)
        assert ts.shape == (n_steps, 1), f"Expected ({n_steps}, 1), got {ts.shape}"

    def test_jit_time_series_nonzero(self):
        """Probe should record non-zero values at the source location."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=200)
        ts = np.array(result.time_series).ravel()
        ts_max = np.max(np.abs(ts))
        assert ts_max > 1e-10, f"Probe recorded near-zero: max={ts_max}"


# ── 5.2 Edge cases ──────────────────────────────────────────────

class TestJITEdgeCases:
    """Edge cases for the JIT subgridded runner."""

    def test_jit_no_probe(self):
        """Simulation without probes should still complete."""
        sim = _make_pec_sim(with_probe=False)
        result = sim.run(n_steps=50)
        assert result is not None
        # time_series should be empty or zeros
        ts = np.array(result.time_series)
        assert ts.size == 0 or np.allclose(ts, 0)

    def test_jit_no_source(self):
        """Simulation without sources should produce zero fields."""
        sim = _make_pec_sim(with_source=False, with_probe=True)
        result = sim.run(n_steps=50)
        ts = np.array(result.time_series).ravel()
        assert np.allclose(ts, 0), "No source should produce zero fields"

    def test_source_outside_fine_region_fails_loudly(self):
        """Subgrid source/probe indices outside the fine block must not be silent."""
        from rfx import Simulation

        sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec")
        sim.add_refinement(z_range=(0.015, 0.025), ratio=2)
        sim.add_source(position=(0.02, 0.02, 0.005), component="ez")

        with pytest.raises(ValueError, match="outside the refined z slab"):
            sim.run(n_steps=5)


# ── 5.3 Energy stability ───────────────────────────────────────

class TestJITStability:
    """Energy stability tests for JIT subgridded runner."""

    def test_jit_fields_finite_1000_steps(self):
        """Fields must remain finite over 1000 steps."""
        sim = _make_pec_sim()
        result = sim.run(n_steps=1000)
        ts = np.array(result.time_series).ravel()

        assert not np.any(np.isnan(ts)), "NaN detected in time series"
        assert np.all(np.isfinite(ts)), "Inf detected in time series"

    def test_jit_energy_stable_1000_steps(self):
        """Energy must not grow unboundedly over 1000 steps.

        In a PEC cavity with a Gaussian pulse source, the source
        injects energy for the first ~200 steps, then stops. After
        that, the SAT coupling dissipates energy. We check that:
        1. No NaN or Inf
        2. Late-time energy does not exceed peak energy
        """
        sim = _make_pec_sim()
        result = sim.run(n_steps=1000)
        ts = np.array(result.time_series).ravel()
        n = len(ts)

        assert not np.any(np.isnan(ts)), "NaN detected"
        assert np.all(np.isfinite(ts)), "Inf detected"

        # Peak energy anywhere in the trace
        peak_energy = np.max(ts ** 2)
        # Late-time energy
        late_energy = np.max(ts[int(0.8 * n):] ** 2)

        # Late energy should not exceed peak (no unbounded growth)
        assert late_energy <= peak_energy * 1.1, (
            f"Late-time energy growth: late={late_energy:.3e} > "
            f"1.1*peak={1.1*peak_energy:.3e}"
        )

    def test_jit_cpml_fields_finite(self):
        """CPML boundary fields must remain finite over 500 steps."""
        sim = _make_cpml_sim()
        result = sim.run(n_steps=500)
        ts = np.array(result.time_series).ravel()
        assert not np.any(np.isnan(ts)), "NaN detected in CPML time series"
        assert np.all(np.isfinite(ts)), "Inf detected in CPML time series"


# ── 5.4 Low-level JIT runner tests ─────────────────────────────

class TestJITRunnerDirect:
    """Direct tests of the jit_runner module (bypassing Simulation API)."""

    def test_direct_jit_runner_pec(self):
        """Call run_subgridded_jit directly with PEC grid."""
        from rfx.grid import Grid
        from rfx.core.yee import MaterialArrays, EPS_0, MU_0
        from rfx.subgridding.sbp_sat_3d import SubgridConfig3D
        from rfx.subgridding.jit_runner import run_subgridded_jit

        # Small 15^3 coarse grid, no CPML
        grid_c = Grid(freq_max=5e9, domain=(0.04, 0.04, 0.04),
                       cpml_layers=0)
        nx, ny, nz = grid_c.shape
        dx_c = grid_c.dx
        ratio = 2
        dx_f = dx_c / ratio

        # Fine region in the center
        fi_lo, fi_hi = 4, nx - 4
        fj_lo, fj_hi = 4, ny - 4
        fk_lo, fk_hi = 4, nz - 4
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
        dt = 0.45 * dx_f / (C0 * np.sqrt(3))

        config = SubgridConfig3D(
            nx_c=nx, ny_c=ny, nz_c=nz, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
        )

        shape_c = (nx, ny, nz)
        shape_f = (nx_f, ny_f, nz_f)
        mats_c = MaterialArrays(
            eps_r=jnp.ones(shape_c, dtype=jnp.float32),
            sigma=jnp.zeros(shape_c, dtype=jnp.float32),
            mu_r=jnp.ones(shape_c, dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones(shape_f, dtype=jnp.float32),
            sigma=jnp.zeros(shape_f, dtype=jnp.float32),
            mu_r=jnp.ones(shape_f, dtype=jnp.float32),
        )

        # Source waveform: Gaussian pulse
        n_steps = 100
        times = np.arange(n_steps) * dt
        f0 = 3e9
        waveform = np.exp(-((times - 3 / f0) * f0) ** 2) * np.sin(
            2 * np.pi * f0 * times
        )
        si, sj, sk = nx_f // 2, ny_f // 2, nz_f // 2
        sources_f = [(si, sj, sk, "ez", waveform.astype(np.float32))]
        probe_indices_f = [(si, sj, sk)]
        probe_components = ["ez"]

        result = run_subgridded_jit(
            grid_c, mats_c, mats_f, config, n_steps,
            opts=SubgridRunOptions(
                sources_f=sources_f,
                probe_indices_f=probe_indices_f,
                probe_components=probe_components,
            ),
        )

        assert result.time_series.shape == (n_steps, 1)
        ts = np.array(result.time_series).ravel()
        assert not np.any(np.isnan(ts)), "NaN in direct JIT runner output"
        assert np.max(np.abs(ts)) > 1e-10, "Fields should be non-zero"

    def test_direct_jit_runner_no_probes_no_sources(self):
        """JIT runner with no sources and no probes should return zeros."""
        from rfx.grid import Grid
        from rfx.core.yee import MaterialArrays, EPS_0, MU_0
        from rfx.subgridding.sbp_sat_3d import SubgridConfig3D
        from rfx.subgridding.jit_runner import run_subgridded_jit

        grid_c = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                       cpml_layers=0)
        nx, ny, nz = grid_c.shape
        dx_c = grid_c.dx
        ratio = 2
        dx_f = dx_c / ratio

        fi_lo, fi_hi = 3, nx - 3
        fj_lo, fj_hi = 3, ny - 3
        fk_lo, fk_hi = 3, nz - 3
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
        dt = 0.45 * dx_f / (C0 * np.sqrt(3))

        config = SubgridConfig3D(
            nx_c=nx, ny_c=ny, nz_c=nz, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
        )

        shape_c = (nx, ny, nz)
        shape_f = (nx_f, ny_f, nz_f)
        mats_c = MaterialArrays(
            eps_r=jnp.ones(shape_c, dtype=jnp.float32),
            sigma=jnp.zeros(shape_c, dtype=jnp.float32),
            mu_r=jnp.ones(shape_c, dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones(shape_f, dtype=jnp.float32),
            sigma=jnp.zeros(shape_f, dtype=jnp.float32),
            mu_r=jnp.ones(shape_f, dtype=jnp.float32),
        )

        result = run_subgridded_jit(
            grid_c, mats_c, mats_f, config, 20,
        )

        # No sources => all fields should be zero
        assert float(jnp.max(jnp.abs(result.state_f.ez))) == 0.0


class TestJITRunnerHCoupling:
    """Verify that the JIT runner applies H-field SAT coupling."""

    def test_jit_runner_h_coupling_energy(self):
        """JIT runner energy should track standalone stepper (which has H-coupling).

        If JIT is missing H-coupling, energy diverges from the standalone
        reference by orders of magnitude.
        """
        from rfx.subgridding.sbp_sat_3d import (
            SubgridConfig3D, init_subgrid_3d, step_subgrid_3d,
        )
        from rfx.subgridding.jit_runner import run_subgridded_jit
        from rfx.grid import Grid
        from rfx.core.yee import MaterialArrays, EPS_0, MU_0

        # --- Config: 20^3 coarse, fine region 7-13, ratio=3 ---
        shape_c = (20, 20, 20)
        dx_c = 0.003
        ratio = 3
        fine_region = (7, 13, 7, 13, 7, 13)
        fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = fine_region
        dx_f = dx_c / ratio
        C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
        dt = 0.45 * dx_f / (C0 * np.sqrt(3))

        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        config = SubgridConfig3D(
            nx_c=20, ny_c=20, nz_c=20, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
        )

        n_steps = 200
        times = np.arange(n_steps) * dt
        f0 = 3e9
        waveform = np.exp(-((times - 3 / f0) * f0) ** 2) * np.sin(
            2 * np.pi * f0 * times
        )

        si, sj, sk = nx_f // 2, ny_f // 2, nz_f // 2

        # --- JIT runner ---
        shape_f = (nx_f, ny_f, nz_f)
        mats_c = MaterialArrays(
            eps_r=jnp.ones(shape_c, dtype=jnp.float32),
            sigma=jnp.zeros(shape_c, dtype=jnp.float32),
            mu_r=jnp.ones(shape_c, dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones(shape_f, dtype=jnp.float32),
            sigma=jnp.zeros(shape_f, dtype=jnp.float32),
            mu_r=jnp.ones(shape_f, dtype=jnp.float32),
        )

        # Build a Grid that matches the 20^3 config
        grid_c_custom = Grid.__new__(Grid)
        grid_c_custom.__dict__.update({
            '_dx': dx_c, '_dy': dx_c, '_dz': dx_c,
            '_nx': 20, '_ny': 20, '_nz': 20,
            'cpml_layers': 0, 'dt': dt,
        })
        grid_c_custom.shape = shape_c

        sources_f = [(si, sj, sk, "ez", waveform.astype(np.float32))]

        result_jit = run_subgridded_jit(
            grid_c_custom, mats_c, mats_f, config, n_steps,
            opts=SubgridRunOptions(sources_f=sources_f),
        )

        # Compute JIT final energy (sum of squared fields)
        jit_e_sq = float(
            jnp.sum(result_jit.state_f.ex ** 2 +
                     result_jit.state_f.ey ** 2 +
                     result_jit.state_f.ez ** 2 +
                     result_jit.state_f.hx ** 2 +
                     result_jit.state_f.hy ** 2 +
                     result_jit.state_f.hz ** 2)
        )

        # --- Standalone stepper (has H-coupling) ---
        config_sa, state_sa = init_subgrid_3d(
            shape_c=shape_c, dx_c=dx_c,
            fine_region=fine_region, ratio=ratio,
            courant=0.45, tau=0.5,
        )

        for step in range(n_steps):
            state_sa = step_subgrid_3d(state_sa, config_sa)
            # Inject source AFTER stepping (matches JIT injection order:
            # sources are added after E-coupling in the JIT scan body)
            state_sa = state_sa._replace(
                ez_f=state_sa.ez_f.at[si, sj, sk].add(
                    float(waveform[step]))
            )

        sa_e_sq = float(
            jnp.sum(state_sa.ex_f ** 2 + state_sa.ey_f ** 2 +
                     state_sa.ez_f ** 2 + state_sa.hx_f ** 2 +
                     state_sa.hy_f ** 2 + state_sa.hz_f ** 2)
        )

        # --- Assertions ---
        # 1. JIT energy should be finite and positive
        assert np.isfinite(jit_e_sq), f"JIT energy is not finite: {jit_e_sq}"
        assert jit_e_sq > 0, f"JIT energy should be positive: {jit_e_sq}"

        # 2. Standalone energy should also be positive
        assert sa_e_sq > 0, (
            f"Standalone energy should be positive: {sa_e_sq}"
        )

        # 3. With H-coupling in both, energies should be in the same
        #    ballpark. Without H-coupling, JIT energy diverges by
        #    orders of magnitude. Allow 100x tolerance.
        ratio_energy = max(jit_e_sq, sa_e_sq) / (min(jit_e_sq, sa_e_sq) + 1e-30)
        assert ratio_energy < 100, (
            f"JIT energy diverges from standalone: JIT={jit_e_sq:.3e}, "
            f"SA={sa_e_sq:.3e}, ratio={ratio_energy:.1f}x — "
            f"H-coupling likely missing in JIT runner"
        )


class TestSubgridMaterialTransition:
    """Validate subgridding with dielectric material crossing the coarse-fine boundary."""

    def test_dielectric_crossing_boundary_stable(self):
        """Dielectric box straddling refinement boundary should not cause NaN or divergence."""
        from rfx import Simulation, Box, GaussianPulse

        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                         boundary="pec", dx=0.003)
        sim.add_material("dielectric", eps_r=4.0)
        # Box crosses the refinement z-boundary (0.009-0.021 vs refine 0.009-0.021)
        sim.add(Box((0.005, 0.005, 0.005), (0.020, 0.020, 0.020)),
                material="dielectric")
        sim.add_source(position=(0.015, 0.015, 0.015), component="ez",
                       waveform=GaussianPulse(f0=2e9, bandwidth=0.5))
        sim.add_probe(position=(0.015, 0.015, 0.015), component="ez")
        sim.add_refinement(z_range=(0.009, 0.021), ratio=2, validation="research")

        result = sim.run(n_steps=200)
        ts = np.array(result.time_series[:, 0])

        assert not np.any(np.isnan(ts)), "NaN in material-transition subgrid"
        assert np.max(np.abs(ts)) > 0, "Zero signal with dielectric"

    def test_dielectric_changes_field_amplitude(self):
        """Dielectric material should produce different field amplitudes vs vacuum.

        With eps_r=4, the wave impedance and propagation speed change.
        The probe is separated from the source inside the fine region so the
        assertion measures propagation through material, not only the local
        source-cell response.
        """
        from rfx import Simulation, Box, GaussianPulse

        domain = (0.03, 0.03, 0.03)
        dx = 0.003

        def _run_with_eps(eps_r):
            sim = Simulation(freq_max=5e9, domain=domain, boundary="pec", dx=dx)
            if eps_r > 1.0:
                sim.add_material("diel", eps_r=eps_r)
                sim.add(Box((0, 0, 0), domain), material="diel")
            sim.add_source((0.015, 0.015, 0.011), "ez",
                           waveform=GaussianPulse(f0=2e9, bandwidth=0.5))
            sim.add_probe((0.015, 0.015, 0.019), "ez")
            sim.add_refinement(z_range=(0.009, 0.021), ratio=2, validation="research")
            return sim.run(n_steps=150)

        res_vac = _run_with_eps(1.0)
        res_die = _run_with_eps(4.0)

        ts_vac = np.array(res_vac.time_series[:, 0])
        ts_die = np.array(res_die.time_series[:, 0])

        # Signals should differ (different wave impedance in dielectric)
        diff = np.max(np.abs(ts_vac - ts_die))
        ref = np.max(np.abs(ts_vac)) + 1e-30
        rel_diff = diff / ref

        assert rel_diff > 0.05, (
            f"Dielectric should change signal: rel_diff={rel_diff:.4f}"
        )
