"""Oblique TFSF plane wave tests: 2D auxiliary grid with periodic y.

Tests:
1. 45-degree oblique incidence leakage < 1%
2. 30-degree oblique incidence leakage < 1%
3. Normal incidence (0 deg) no regression (< 0.1%)
4. 2D auxiliary grid basic sanity (fields propagate, no NaN)
5. Dispatch: oblique angle returns TFSF2DConfig, normal returns TFSFConfig
"""

import numpy as np
import jax.numpy as jnp

from rfx.core.yee import init_state, init_materials, update_h, update_e


# ---------------------------------------------------------------------------
# Helper: run a TFSF simulation and measure leakage
# ---------------------------------------------------------------------------

def _run_tfsf_leakage(angle_deg, nx=80, ny=80, nz=3, n_steps=500,
                       f0=5e9, bandwidth=0.5, cpml_layers=10,
                       tfsf_margin=3):
    """Run TFSF simulation and return (leakage_ratio, tf_energy, sf_energy)."""
    from rfx.sources.tfsf import (
        init_tfsf, apply_tfsf_e, apply_tfsf_h,
        update_tfsf_1d_h, update_tfsf_1d_e,
        is_tfsf_2d,
    )
    from rfx.sources.tfsf_2d import update_tfsf_2d_h, update_tfsf_2d_e

    dx = 1e-3
    dt = 0.5 * dx / 3e8

    cfg, state = init_tfsf(
        nx, dx, dt,
        cpml_layers=cpml_layers,
        tfsf_margin=tfsf_margin,
        f0=f0,
        bandwidth=bandwidth,
        polarization="ez",
        direction="+x",
        angle_deg=angle_deg,
        ny=ny,
    )

    materials = init_materials((nx, ny, nz))
    sim_state = init_state((nx, ny, nz))
    periodic = (False, True, True)
    is_2d = is_tfsf_2d(cfg)

    for step in range(n_steps):
        t = step * dt

        # H update
        sim_state = update_h(sim_state, materials, dt, dx, periodic=periodic)
        sim_state = apply_tfsf_h(sim_state, cfg, state, dx, dt)

        # Advance aux grid H
        if is_2d:
            state = update_tfsf_2d_h(cfg, state, dx, dt)
        else:
            state = update_tfsf_1d_h(cfg, state, dx, dt)

        # E update
        sim_state = update_e(sim_state, materials, dt, dx, periodic=periodic)
        sim_state = apply_tfsf_e(sim_state, cfg, state, dx, dt)

        # Advance aux grid E + source
        if is_2d:
            state = update_tfsf_2d_e(cfg, state, dx, dt, t)
        else:
            state = update_tfsf_1d_e(cfg, state, dx, dt, t)

    # Measure leakage
    ez = sim_state.ez
    sf_energy = float(jnp.sum(ez[:cfg.x_lo - 2, :, :] ** 2))
    tf_energy = float(jnp.sum(ez[cfg.x_lo:cfg.x_hi, :, :] ** 2))

    leakage = sf_energy / tf_energy if tf_energy > 0 else float('inf')
    return leakage, tf_energy, sf_energy


class TestObliqueTFSF:
    """Tests for oblique-incidence TFSF using 2D auxiliary grid."""

    def test_2d_aux_grid_basic_sanity(self):
        """2D auxiliary grid produces non-zero fields without NaN."""
        from rfx.sources.tfsf_2d import (
            init_tfsf_2d, update_tfsf_2d_h, update_tfsf_2d_e, TFSF2DConfig,
        )

        dx = 1e-3
        dt = 0.5 * dx / 3e8

        cfg, state = init_tfsf_2d(
            60, 60, dx, dt, cpml_layers=8, f0=5e9,
            bandwidth=0.5, theta_deg=45.0,
        )

        assert isinstance(cfg, TFSF2DConfig)
        assert cfg.n2y == 60, "2D grid y should match 3D ny (periodic)"

        for step in range(200):
            t = step * dt
            state = update_tfsf_2d_h(cfg, state, dx, dt)
            state = update_tfsf_2d_e(cfg, state, dx, dt, t)

        assert not np.any(np.isnan(np.array(state.ez_2d))), "NaN in 2D aux grid"
        max_ez = float(jnp.max(jnp.abs(state.ez_2d)))
        assert max_ez > 1e-6, f"2D aux grid has no signal: max |Ez| = {max_ez}"

    def test_oblique_45deg_leakage_below_1pct(self):
        """45-degree oblique incidence leakage < 1%."""
        leakage, tf_energy, sf_energy = _run_tfsf_leakage(
            angle_deg=45.0, nx=100, ny=100, nz=3, n_steps=600,
        )
        print(f"\n45-deg oblique TFSF: leakage={leakage:.6f}, "
              f"tf_energy={tf_energy:.6e}, sf_energy={sf_energy:.6e}")
        assert tf_energy > 0, "No total-field energy detected"
        assert leakage < 0.01, f"Oblique TFSF leakage {leakage:.4f} > 1%"

    def test_oblique_30deg_leakage_below_1pct(self):
        """30-degree oblique incidence leakage < 1%."""
        leakage, tf_energy, sf_energy = _run_tfsf_leakage(
            angle_deg=30.0, nx=100, ny=100, nz=3, n_steps=600,
        )
        print(f"\n30-deg oblique TFSF: leakage={leakage:.6f}, "
              f"tf_energy={tf_energy:.6e}, sf_energy={sf_energy:.6e}")
        assert tf_energy > 0, "No total-field energy detected"
        assert leakage < 0.01, f"Oblique TFSF leakage {leakage:.4f} > 1%"

    def test_normal_incidence_no_regression(self):
        """Normal incidence (0 deg) must still work -- no regression."""
        leakage, tf_energy, sf_energy = _run_tfsf_leakage(
            angle_deg=0.0, nx=60, ny=60, nz=3, n_steps=300,
        )
        print(f"\n0-deg normal TFSF: leakage={leakage:.6f}, "
              f"tf_energy={tf_energy:.6e}, sf_energy={sf_energy:.6e}")
        assert tf_energy > 0, "No total-field energy detected"
        assert leakage < 0.001, f"Normal TFSF leakage {leakage:.4f} > 0.1%"

    def test_init_tfsf_dispatches_correctly(self):
        """init_tfsf dispatches to 2D for oblique, 1D for normal."""
        from rfx.sources.tfsf import init_tfsf, TFSFConfig
        from rfx.sources.tfsf_2d import TFSF2DConfig

        dx = 1e-3
        dt = 0.5 * dx / 3e8

        cfg_1d, _ = init_tfsf(60, dx, dt, cpml_layers=8, angle_deg=0.0)
        assert isinstance(cfg_1d, TFSFConfig)

        cfg_2d, _ = init_tfsf(60, dx, dt, cpml_layers=8, angle_deg=30.0, ny=60)
        assert isinstance(cfg_2d, TFSF2DConfig)

    def test_oblique_negative_angle(self):
        """Negative oblique angle (-30 deg) should also have low leakage."""
        leakage, tf_energy, sf_energy = _run_tfsf_leakage(
            angle_deg=-30.0, nx=100, ny=100, nz=3, n_steps=600,
        )
        print(f"\n-30-deg oblique TFSF: leakage={leakage:.6f}, "
              f"tf_energy={tf_energy:.6e}, sf_energy={sf_energy:.6e}")
        assert tf_energy > 0, "No total-field energy detected"
        assert leakage < 0.01, f"Oblique TFSF leakage {leakage:.4f} > 1%"
