"""NU normalize='flux' branch — CPU sanity (issue #88 Step B).

The NU normalize=True diagonal blows up at the band edges because
``S11 = (b_dev - b_ref) / a_inc_ref`` divides by the source-spectrum-
weighted modal incident amplitude, which collapses at band edges. The
flux branch assembles magnitudes from Poynting power ratios
``|S_ii|² = |F_ref - F_dev| / |F_ref|`` with phase from the modal V/I
decomposition.

Scope of this CPU test: assert the flux branch RUNS on the NU path and
that at a SETTLED num_periods it produces a passive, physical S-matrix
(passivity ≤ 1.10, real reflection registered). Absolute-accuracy /
broad-E5-envelope validation against analytic Airy is a GPU task
(docs/research_notes/20260529_flux_nu_wiring_design.md).

Caveat (documented, not asserted): at transition num_periods the flux
extractor — like any DFT-based extractor — is not yet settled and can
overshoot passivity. The flux advantage over normalize=True is that the
SETTLED result preserves passivity (~1.0 vs ~1.18) and tracks the uniform
|S21|; it does NOT make the band-edge denominator fully immune (P_inc is
still source-spectrum weighted).
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

_A_WG = 0.02286
_B_WG = 0.01016
_F_MAX = 12e9
_FREQS = jnp.linspace(8.2e9, 12.4e9, 5)
_SLAB_X_LO = 0.045
_SLAB_X_HI = 0.049
_SLAB_EPS_R = 4.0


def _make_wr90_nu_sim_with_slab():
    from rfx.api import Simulation
    from rfx.auto_config import smooth_grading
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

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
    return sim


def test_nu_flux_branch_runs_and_returns_shape():
    """The flux branch executes on the NU path and returns a (2,2,nf)
    finite S-matrix (no NotImplementedError, no NaN/Inf)."""
    sim = _make_wr90_nu_sim_with_slab()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_waveguide_s_matrix(num_periods=8, normalize="flux")
    s = np.array(res.s_params)
    assert s.shape == (2, 2, len(_FREQS)), f"shape {s.shape}"
    assert np.all(np.isfinite(s)), "flux S-matrix has NaN/Inf"


def test_nu_flux_settled_is_passive_and_reflects():
    """At a settled num_periods the flux S-matrix is passive
    (|S11|²+|S21|² ≤ 1.10) and registers real reflection (|S11|>0.02).

    This is the property normalize=True FAILS (passivity ~1.18 at the
    same settling, band-edge blow-up at intermediate num_periods)."""
    sim = _make_wr90_nu_sim_with_slab()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_waveguide_s_matrix(num_periods=40, normalize="flux")
    s = np.array(res.s_params)
    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    passivity = s11**2 + s21**2

    print("\n[NU-FLUX] settled (num_periods=40):")
    print(f"  |S11|: {np.array2string(s11, precision=4)}")
    print(f"  |S21|: {np.array2string(s21, precision=4)}")
    print(f"  passivity: {np.array2string(passivity, precision=4)} "
          f"max={passivity.max():.4f}")

    assert passivity.max() <= 1.10, (
        f"flux passivity {passivity.max():.4f} > 1.10 at settled num_periods"
    )
    assert s11.max() > 0.02, (
        f"flux |S11| max {s11.max():.4f} — slab reflection not registered"
    )
