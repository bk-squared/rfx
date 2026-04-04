"""Extended oblique TFSF coverage: ey polarization and sign-correctness.

Tests:
1. ey normal incidence leakage < 0.1% (1D aux grid, curl_sign=-1 path)
2. ey oblique incidence correctly raises NotImplementedError
   (2D aux grid is TMz/xy-plane only; ey oblique needs TEz/xz-plane)
3. E-field correction sign independence: verify the fixed (-coeff, +coeff)
   signs in apply_tfsf_2d_e give identical results to the curl_sign-based
   formula for ez polarization (regression guard for the sign fix).
"""

import numpy as np
import jax.numpy as jnp
import pytest
from rfx.sources.tfsf import (
    init_tfsf, apply_tfsf_e, apply_tfsf_h, is_tfsf_2d,
    update_tfsf_1d_h, update_tfsf_1d_e,
)
from rfx.sources.tfsf_2d import (
    update_tfsf_2d_h, update_tfsf_2d_e, init_tfsf_2d,
)
from rfx.core.yee import init_state, init_materials, update_h, update_e


def _run_tfsf_leakage(nx, ny, nz, dx, dt, angle_deg, polarization, direction, n_steps):
    """Run TFSF and measure scattered-to-total field energy ratio."""
    cfg, st = init_tfsf(
        nx, dx, dt, ny=ny, cpml_layers=10,
        f0=5e9, bandwidth=0.5, amplitude=1.0,
        polarization=polarization, direction=direction,
        angle_deg=angle_deg,
    )

    materials = init_materials((nx, ny, nz))
    sim_state = init_state((nx, ny, nz))
    periodic = (False, True, True)
    is_2d = is_tfsf_2d(cfg)

    for step in range(n_steps):
        t = step * dt

        # H update
        sim_state = update_h(sim_state, materials, dt, dx, periodic=periodic)
        sim_state = apply_tfsf_h(sim_state, cfg, st, dx, dt)

        # Advance aux grid H
        if is_2d:
            st = update_tfsf_2d_h(cfg, st, dx, dt)
        else:
            st = update_tfsf_1d_h(cfg, st, dx, dt)

        # E update
        sim_state = update_e(sim_state, materials, dt, dx, periodic=periodic)
        sim_state = apply_tfsf_e(sim_state, cfg, st, dx, dt)

        # Advance aux grid E + source
        if is_2d:
            st = update_tfsf_2d_e(cfg, st, dx, dt, t)
        else:
            st = update_tfsf_1d_e(cfg, st, dx, dt, t)

    component = getattr(sim_state, polarization)
    sf_energy = float(jnp.sum(component[:cfg.x_lo - 2, :, :] ** 2))
    tf_energy = float(jnp.sum(component[cfg.x_lo:cfg.x_hi, :, :] ** 2))
    return sf_energy / tf_energy if tf_energy > 0 else 1.0


class TestObliqueTFSFCoverage:
    @pytest.fixture
    def grid_params(self):
        dx = 1e-3
        return 80, 80, 3, dx, 0.5 * dx / 3e8

    def test_ey_polarization_normal(self, grid_params):
        """ey polarization at normal incidence: leakage < 0.1%.

        Exercises the 1D auxiliary grid with curl_sign=-1 (Ey/Hz pair).
        """
        nx, ny, nz, dx, dt = grid_params
        leakage = _run_tfsf_leakage(
            nx, ny, nz, dx, dt,
            angle_deg=0.0, polarization="ey",
            direction="+x", n_steps=300,
        )
        print(f"\ney normal TFSF: leakage={leakage:.6f}")
        assert leakage < 0.001, f"ey normal leakage {leakage:.5f} > 0.1%"

    def test_ey_polarization_normal_minus_x(self, grid_params):
        """ey polarization at normal incidence in -x direction."""
        nx, ny, nz, dx, dt = grid_params
        leakage = _run_tfsf_leakage(
            nx, ny, nz, dx, dt,
            angle_deg=0.0, polarization="ey",
            direction="-x", n_steps=300,
        )
        print(f"\ney normal -x TFSF: leakage={leakage:.6f}")
        assert leakage < 0.001, f"ey normal -x leakage {leakage:.5f} > 0.1%"

    def test_ey_oblique_raises_not_implemented(self):
        """ey + oblique angle correctly raises NotImplementedError.

        The 2D auxiliary grid operates in TMz mode (xy-plane).  For ey
        polarization the oblique tilt lies in the xz-plane, which requires
        a TEz auxiliary grid not yet implemented.
        """
        dx = 1e-3
        dt = 0.5 * dx / 3e8
        with pytest.raises(NotImplementedError, match="ey polarization"):
            init_tfsf(80, dx, dt, ny=80, cpml_layers=10,
                      polarization="ey", angle_deg=45.0)

    def test_ez_oblique_e_correction_sign_regression(self, grid_params):
        """ez oblique must still have near-zero leakage (sign fix regression guard).

        The E-field correction in apply_tfsf_2d_e uses fixed signs
        (-coeff, +coeff) that are independent of curl_sign.  This test
        ensures the fix does not regress ez oblique performance.
        """
        nx, ny, nz, dx, dt = grid_params
        leakage = _run_tfsf_leakage(
            nx, ny, nz, dx, dt,
            angle_deg=45.0, polarization="ez",
            direction="+x", n_steps=500,
        )
        print(f"\nez oblique 45-deg TFSF: leakage={leakage:.6f}")
        assert leakage < 0.001, f"ez oblique 45 deg leakage {leakage:.5f} > 0.1%"

    def test_negative_x_direction_oblique_30_ez(self, grid_params):
        """-x direction with ez at 30 deg oblique should have < 1% leakage."""
        nx, ny, nz, dx, dt = grid_params
        leakage = _run_tfsf_leakage(
            nx, ny, nz, dx, dt,
            angle_deg=30.0, polarization="ez",
            direction="-x", n_steps=500,
        )
        print(f"\nez oblique 30-deg -x TFSF: leakage={leakage:.6f}")
        assert leakage < 0.01, f"-x ez oblique 30 deg leakage {leakage:.4f} > 1%"

    def test_negative_x_direction_normal_ez(self, grid_params):
        """-x direction normal incidence ez."""
        nx, ny, nz, dx, dt = grid_params
        leakage = _run_tfsf_leakage(
            nx, ny, nz, dx, dt,
            angle_deg=0.0, polarization="ez",
            direction="-x", n_steps=300,
        )
        print(f"\nez normal -x TFSF: leakage={leakage:.6f}")
        assert leakage < 0.001, f"-x normal leakage {leakage:.5f} > 0.1%"

    def test_init_tfsf_2d_ey_normal_allowed(self):
        """ey polarization at normal incidence dispatches to 1D (no 2D grid)."""
        dx = 1e-3
        dt = 0.5 * dx / 3e8
        from rfx.sources.tfsf import TFSFConfig
        cfg, _ = init_tfsf(80, dx, dt, cpml_layers=10,
                           polarization="ey", angle_deg=0.0)
        assert isinstance(cfg, TFSFConfig), "Normal ey should use 1D aux grid"
        assert cfg.electric_component == "ey"
        assert cfg.magnetic_component == "hz"
        assert cfg.curl_sign == -1.0
