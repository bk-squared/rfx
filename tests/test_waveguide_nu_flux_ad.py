"""NU waveguide ``normalize='flux'`` S-matrix on the AD tape (rung 5).

Mirror of ``tests/test_waveguide_flux_ad.py`` (the uniform PR #172 flux-AD
gate) for the NON-UNIFORM mesh path. Before the fix the NU dispatch
rejected ``eps_override`` outright on every normalize mode, and the NU
flux branch concretized the flux/phase/mag assembly through ``np.*``
(plus ``run_nonuniform_path`` wrapped ``waveguide_port_flux`` in
``np.array(...)``), so ``compute_waveguide_s_matrix(normalize='flux',
eps_override=<traced>)`` on a graded mesh could not be optimized through.

After the fix the device run threads the traced ``eps_override`` into the
Yee update, the flux extraction is jnp-native end-to-end (double-where at
sqrt(0)/angle(0) per #171/#172/#148), and the reference run stays vacuum.

Gates (composition-level, per the G2 lesson — unit AD tests do not
protect compositions):
  1+2. grad(|S21(f0)|^2) w.r.t. a substrate-eps scalar through the FULL
       NU flux extraction is finite AND matches central finite
       differences to <=5% rel (fixture has real sensitivity: FD != 0).
  3.   Forward S-matrix with a no-op (deps=0) eps_override == the
       no-override NU flux forward (rtol<=1e-5) — proves the np->jnp
       rewrite + threading did not change forward values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rfx import Simulation
from rfx.auto_config import smooth_grading
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

NUM_PERIODS = 8.0

_A_WG = 0.02286
_B_WG = 0.01016
_F_MAX = 12e9
_FREQS = jnp.linspace(8.2e9, 12.4e9, 4)
_SLAB_X_LO = 0.045
_SLAB_X_HI = 0.049
_SLAB_EPS_R = 4.0


def _wr90_nu_sim():
    """Graded-mesh (dx_profile) WR-90 with a dielectric slab mid-guide.

    Same geometry as ``tests/test_waveguide_nu_flux.py`` so the fixture is
    a known-good NU flux witness; the slab gives the eps_override a real
    |S21| sensitivity for the FD gate.
    """
    dx_coarse = 1.5e-3
    dx_fine = 0.75e-3
    n_pre = int(round(0.030 / dx_coarse))
    n_fine = int(round(0.040 / dx_fine))
    n_post = int(round(0.030 / dx_coarse))
    raw = np.concatenate([
        np.full(n_pre, dx_coarse),
        np.full(n_fine, dx_fine),
        np.full(n_post, dx_coarse),
    ])
    dx_profile = smooth_grading(raw, max_ratio=1.3)
    domain_x = float(np.sum(dx_profile))

    sim = Simulation(
        freq_max=_F_MAX,
        domain=(domain_x, _A_WG, _B_WG),
        dx=dx_coarse,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
        dx_profile=dx_profile,
    )
    sim.add_material("diel_slab", eps_r=_SLAB_EPS_R, sigma=0.0)
    sim.add(
        Box((_SLAB_X_LO, 0.0, 0.0), (_SLAB_X_HI, _A_WG, _B_WG)),
        material="diel_slab",
    )
    sim.add_waveguide_port(
        0.015, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.020, name="left",
    )
    sim.add_waveguide_port(
        domain_x - 0.015, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5,
        reference_plane=domain_x - 0.020, name="right",
    )
    return sim, domain_x


def _eps_override_for(sim, domain_x, deps):
    """NU-grid eps override 1.0 + deps inside the slab region (traced).

    Sized to the NU grid the flux extractor builds (``_build_nonuniform_grid``
    synthesises the same scalar-dx dz_profile and the same x-cpml pad the
    NU S-matrix path uses, so the shape matches the device materials array).
    """
    from rfx.runners.nonuniform import pos_to_nu_index
    grid = sim._build_nonuniform_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    i_lo = pos_to_nu_index(grid, (_SLAB_X_LO, _A_WG / 2, _B_WG / 2))[0]
    i_hi = pos_to_nu_index(grid, (_SLAB_X_HI, _A_WG / 2, _B_WG / 2))[0]
    # Perturb the slab region: base slab eps (=4) + deps so deps=0 is a
    # no-op vs the assembled materials (gate 3) and deps>0 changes S21.
    return eps.at[i_lo:i_hi, :, :].set(_SLAB_EPS_R + deps)


def _s21_mag2(deps):
    sim, domain_x = _wr90_nu_sim()
    eps = _eps_override_for(sim, domain_x, deps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_waveguide_s_matrix(
            num_periods=NUM_PERIODS, normalize="flux", eps_override=eps,
        )
    # |S21|^2 at the band-center bin (index 2 of the 4-point grid)
    return jnp.abs(res.s_params[1, 0, 2]) ** 2


@pytest.mark.slow
def test_nu_flux_smatrix_grad_finite_and_fd_consistent():
    """Gates 1+2: traced NU flux extraction yields a finite, FD-consistent
    grad w.r.t. a substrate-eps scalar."""
    deps0 = jnp.asarray(0.5, dtype=jnp.float32)
    val, g = jax.value_and_grad(_s21_mag2)(deps0)
    assert np.isfinite(float(val)), f"value is {val}"
    assert np.isfinite(float(g)), f"grad is {g}"

    h = 0.1
    fd = (float(_s21_mag2(deps0 + h)) - float(_s21_mag2(deps0 - h))) / (2 * h)
    assert fd != 0.0, "FD slope is zero — fixture has no sensitivity; rebuild it"
    rel = abs(float(g) - fd) / max(abs(fd), 1e-12)
    print(f"\n[NU-FLUX-AD] AD={float(g):+.6e} FD={fd:+.6e} rel={rel:.4f}")
    assert rel <= 0.05, (
        f"AD={float(g):+.6e} vs FD={fd:+.6e} (rel diff {rel:.3f} > 5%)"
    )


@pytest.mark.slow
def test_nu_flux_smatrix_forward_matches_untraced():
    """Gate 3: the np->jnp rewrite + eps_override threading did not change
    forward values — compare normalize='flux' against itself with a no-op
    (deps=0) override that reproduces the assembled slab eps."""
    sim_a, _ = _wr90_nu_sim()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_a = sim_a.compute_waveguide_s_matrix(
            num_periods=NUM_PERIODS, normalize="flux")
    sim_b, domain_x_b = _wr90_nu_sim()
    eps = _eps_override_for(sim_b, domain_x_b, jnp.asarray(0.0, dtype=jnp.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_b = sim_b.compute_waveguide_s_matrix(
            num_periods=NUM_PERIODS, normalize="flux", eps_override=eps)
    sa = np.asarray(res_a.s_params)
    sb = np.asarray(res_b.s_params)
    assert np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))
    np.testing.assert_allclose(sb, sa, rtol=1e-5, atol=1e-7)


def _s11_mag2_at_vacuum(deps):
    """|S11|^2 with a pure-vacuum device override (eps = 1 + deps everywhere).

    At deps=0 the device run is identical to the vacuum reference run, so
    P_refl = |F_ref - F_dev| = 0 exactly -> the diagonal sqrt(P_refl/P_inc)
    (and the abs() feeding it) hit their zero argument. This is the DEAD
    branch the FD gate's reflective slab never reaches.
    """
    sim, _ = _wr90_nu_sim()
    grid = sim._build_nonuniform_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32) + deps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_waveguide_s_matrix(
            num_periods=NUM_PERIODS, normalize="flux", eps_override=eps,
        )
    return jnp.abs(res.s_params[0, 0, 2]) ** 2


@pytest.mark.slow
def test_nu_flux_smatrix_grad_finite_at_perfect_null():
    """Dead-branch NaN-safety (the double-where guard the live FD gate misses).

    At deps=0 the traced device override is pure vacuum == the reference run,
    so the diagonal reflected power P_refl = |F_ref - F_dev| = 0. A single
    output-only ``jnp.where`` would leak a NaN gradient through ``sqrt(0)`` /
    ``abs(0)`` here; the input-guarded double-where must keep the grad finite.
    Locks the protection in CI so a future single-where simplification trips.
    """
    deps0 = jnp.asarray(0.0, dtype=jnp.float32)
    val, g = jax.value_and_grad(_s11_mag2_at_vacuum)(deps0)
    assert np.isfinite(float(val)), f"value is {val}"
    assert np.isfinite(float(g)), (
        f"grad is {g} at the perfect-null (P_refl=0) — dead-branch NaN leak; "
        "the sqrt(0)/abs(0) double-where guard has regressed"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
