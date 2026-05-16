"""Tier-1 reproduction battery for the 2026-05-16 rfx code review.

Each test makes one CRITICAL finding from
``docs/research_notes/2026-05-16_rfx_review_and_refactor_roadmap.md``
(Part A) executable, so the finding is either *confirmed* (the test
reproduces the defect) or *refuted* (the test passes and the finding
was overstated).

Convention:
- A confirmed defect ⇒ the test asserts CORRECT physics and is marked
  ``xfail(strict=True)``. It fails today (bug present) and will
  XPASS once fixed, forcing the marker to be removed.
- A factual-but-not-defect finding ⇒ a plain characterization test
  that documents the actual behavior.

Confirmed items here are candidates for promotion to
``docs/agent-memory/rfx-known-issues.md`` once a fix lands.

Covered: CORE-C2, GEO-C1, PROBE-C1, RIS-C2. (OPT-C1 — adi_step_3d
3D-cavity eigenfrequency — lives in its own slower test.)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import (
    FDTDState, MaterialArrays, update_h_nu, update_e_nu, EPS_0, MU_0,
)
from rfx.materials.nonlinear import apply_kerr_ade
from rfx.probes.probes import init_flux_monitor, flux_spectrum
from rfx.adi import run_adi_3d
from rfx.grid import C0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros(shape):
    return jnp.zeros(shape, dtype=jnp.float32)


def _state(shape, **fields):
    base = dict(ex=_zeros(shape), ey=_zeros(shape), ez=_zeros(shape),
                hx=_zeros(shape), hy=_zeros(shape), hz=_zeros(shape))
    base.update(fields)
    return FDTDState(step=jnp.array(0, dtype=jnp.int32), **base)


def _free_space(shape):
    ones = jnp.ones(shape, dtype=jnp.float32)
    return MaterialArrays(eps_r=ones, sigma=_zeros(shape), mu_r=ones)


# A deliberately non-uniform 1-D z profile with 3:1 cell-size jumps so
# the mean-vs-local spacing discrepancy is ~1.5x — far above float32 noise.
_DZ = jnp.array([1., 1., 3., 1., 1., 3., 1., 1.], dtype=jnp.float32) * 1.0e-3


# ===========================================================================
# CORE-C2 — non-uniform Yee H/E spacing metrics  (FIXED 2026-05-16)
# ===========================================================================
#
# First-principles Yee staggering (1-D, z-axis):
#   * H sits at the cell centre. Its update differences two E *nodes*
#     straddling one cell ⇒ denominator = the LOCAL cell width d[k].
#   * E sits at a node. Its update differences two H cell-centres ⇒
#     denominator = the MEAN spacing (d[k-1]+d[k])/2.
#
# Before the fix rfx had these swapped. `_profile_to_inv_arrays` now
# emits inv_d_e (E: mean) and inv_d_h (H: local); the runner feeds the
# second to update_h_nu and the first to update_e_nu. These tests drive
# the production array-construction path and assert the discrete curl
# of a linear field recovers the exact analytic derivative on a graded
# mesh. They were xfail(strict) pre-fix; they pass post-fix.

def test_corec2_h_update_uses_local_cell_width():
    """update_h_nu, fed the production inv_d_h, recovers the exact
    derivative of a z-linear E field on a 3:1-graded mesh."""
    from rfx.nonuniform import _profile_to_inv_arrays

    dz = _DZ
    nz = int(dz.shape[0])
    nx = ny = 3
    dx = 1.0e-3
    a = 1.0

    # E_y linear in z: ey = a*z_node, z_node[k] = cumsum of dz.
    z_node = jnp.concatenate([jnp.zeros(1, jnp.float32), jnp.cumsum(dz)])[:nz]
    ey = jnp.broadcast_to(a * z_node[None, None, :], (nx, ny, nz)).astype(jnp.float32)
    state = _state((nx, ny, nz), ey=ey)
    mats = _free_space((nx, ny, nz))
    dt = 1.0e-13

    # Production wiring: _profile_to_inv_arrays -> (E array, H array);
    # the runner passes the H array (2nd) to update_h_nu.
    _, inv_dz_h = _profile_to_inv_arrays(np.asarray(dz))
    _, inv_xy_h = _profile_to_inv_arrays(np.full(nx, dx, dtype=np.float64))
    new = update_h_nu(state, mats, dt, inv_xy_h, inv_xy_h, inv_dz_h)

    # curl_x = dEz/dy - dEy/dz = -discrete(dEy/dz); hx = -(dt/mu)*curl_x
    # => discrete(dEy/dz) = hx * mu / dt
    discrete = np.asarray(new.hx)[1, 1, :] * MU_0 / dt
    # interior z cells (exclude k=nz-1: forward-diff boundary, inv=0)
    np.testing.assert_allclose(discrete[:nz - 1], a, rtol=0.02)


def test_corec2_e_update_uses_mean_spacing():
    """update_e_nu, fed the production inv_d_e, recovers the exact
    derivative of a z-linear H field on a 3:1-graded mesh."""
    from rfx.nonuniform import _profile_to_inv_arrays

    dz = _DZ
    nz = int(dz.shape[0])
    nx = ny = 3
    dx = 1.0e-3
    a = 1.0

    # H_y linear in z at cell centres: hy = a*z_half, z_half[k] = z_node[k]+dz[k]/2.
    z_node = jnp.concatenate([jnp.zeros(1, jnp.float32), jnp.cumsum(dz)])[:nz]
    z_half = z_node + dz / 2.0
    hy = jnp.broadcast_to(a * z_half[None, None, :], (nx, ny, nz)).astype(jnp.float32)
    state = _state((nx, ny, nz), hy=hy)
    mats = _free_space((nx, ny, nz))
    dt = 1.0e-13

    # Production wiring: the runner passes the E array (1st) to update_e_nu.
    inv_dz_e, _ = _profile_to_inv_arrays(np.asarray(dz))
    inv_xy_e, _ = _profile_to_inv_arrays(np.full(nx, dx, dtype=np.float64))
    new = update_e_nu(state, mats, dt, inv_xy_e, inv_xy_e, inv_dz_e)

    # curl_x = dHz/dy - dHy/dz = -discrete(dHy/dz); ex = (dt/eps)*curl_x
    # => discrete(dHy/dz) = -ex * eps / dt
    discrete = -np.asarray(new.ex)[1, 1, :] * EPS_0 / dt
    # interior z cells (exclude k=0: backward-diff boundary)
    np.testing.assert_allclose(discrete[1:], a, rtol=0.02)


# ===========================================================================
# GEO-C1 — Kerr ADE uses an explicit linearization
# ===========================================================================
#
# apply_kerr_ade implements  E <- E*(1-factor)  (explicit forward-Euler).
# The docstring's stated discrete relation E^{n+1} -= f*E^{n+1} has the
# exact solution E/(1+f). The two agree to O(f); they diverge for f≳0.1.
# Verdict: the finding is factually correct but the code+docstring
# already label this "explicit" — severity MEDIUM (use the exact /(1+f)
# form, same cost), not CRITICAL.

def _kerr_setup():
    shape = (2, 2, 2)
    E0 = 3.0
    dt = 1.0e-12
    target_factor = 0.30  # large enough that explicit vs implicit differ
    chi3 = target_factor * EPS_0 / (dt * E0 ** 2)
    state = _state(shape, ex=jnp.full(shape, E0, jnp.float32))
    chi3_arr = jnp.full(shape, chi3, jnp.float32)
    factor = (dt / EPS_0) * chi3 * E0 ** 2
    return state, chi3_arr, dt, E0, factor


def test_geoc1_kerr_ade_is_explicit_linearization():
    """Characterization: confirms apply_kerr_ade returns E*(1-factor)."""
    state, chi3_arr, dt, E0, factor = _kerr_setup()
    out = apply_kerr_ade(state, chi3_arr, dt)
    got = float(np.asarray(out.ex)[0, 0, 0])

    explicit = E0 * (1.0 - factor)
    implicit = E0 / (1.0 + factor)
    assert got == pytest.approx(explicit, rel=1e-4)
    # Explicit and exact-implicit are genuinely distinct at factor=0.3.
    assert abs(got - implicit) > 0.01 * E0


@pytest.mark.xfail(strict=True, reason=(
    "GEO-C1: apply_kerr_ade uses the explicit E*(1-f); the exact solve "
    "of its own documented implicit relation is E/(1+f) — same cost, "
    "unconditionally stable for this term. nonlinear.py:82-84."))
def test_geoc1_kerr_ade_should_use_exact_implicit_solve():
    state, chi3_arr, dt, E0, factor = _kerr_setup()
    out = apply_kerr_ade(state, chi3_arr, dt)
    got = float(np.asarray(out.ex)[0, 0, 0])
    assert got == pytest.approx(E0 / (1.0 + factor), rel=1e-4)


# ===========================================================================
# PROBE-C1 — FluxMonitor axis-aware area weighting  (FIXED 2026-05-16)
# ===========================================================================
#
# The pre-fix FluxMonitor stored a single scalar ``dx`` and flux_spectrum
# used dA = dx*dx for every face orientation — correct only for a cubic
# cell. init_flux_monitor now takes the two tangential cell sizes
# (d1, d2) and flux_spectrum uses the axis-aware dA = d1*d2. (Currently
# all flux monitors run on a cubic Grid, so this was a latent bug; the
# fix makes the path correct for non-cubic / non-uniform meshes too.)

def test_probec1_flux_area_is_axis_aware():
    """flux_spectrum weights an x-normal face by dy*dz, not dx*dx."""
    DX, DY, DZ = 1.0e-3, 2.0e-3, 4.0e-3  # deliberately non-cubic
    grid_shape = (5, 5, 5)
    freqs = jnp.array([1.0e9])

    # x-normal face (axis=0): the two tangential axes are y and z.
    mon = init_flux_monitor(axis=0, index=2, freqs=freqs,
                            grid_shape=grid_shape, d1=DY, d2=DZ)
    n1, n2 = grid_shape[1], grid_shape[2]
    ones = jnp.ones((1, n1, n2), dtype=jnp.complex64)
    zeros = jnp.zeros((1, n1, n2), dtype=jnp.complex64)
    # integrand = e1*conj(h2) - e2*conj(h1) = 1 over the whole face
    mon = mon._replace(e1_dft=ones, h2_dft=ones, e2_dft=zeros, h1_dft=zeros)

    flux = float(np.asarray(flux_spectrum(mon))[0])
    correct = n1 * n2 * DY * DZ  # x-normal face: area element is dy*dz
    assert flux == pytest.approx(correct, rel=1e-5)
    # ...and NOT the pre-fix cubic-cell dx*dx weighting.
    assert flux != pytest.approx(n1 * n2 * DX * DX, rel=1e-3)


# ===========================================================================
# RIS-C2 — does np.interp discard the imaginary part of a complex fp?
# ===========================================================================
#
# ris._extract_reflection (probes.py fallback) runs np.interp over a
# complex rfft spectrum. The review claimed np.interp drops the
# imaginary part. This test adjudicates against the installed numpy.

def test_risc2_np_interp_complex_part_behavior():
    xp = np.array([0.0, 1.0])
    fp = np.array([1.0 + 0.0j, 1.0 + 10.0j])
    x = np.array([0.5])

    r = np.interp(x, xp, fp)
    proper = np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)

    # If numpy supports complex fp, r preserves the imaginary part and
    # equals `proper` ⇒ RIS-C2 is REFUTED. If r.imag == 0, RIS-C2 holds.
    assert np.allclose(r, proper), (
        f"np.interp complex result {r} != proper complex interp {proper} "
        "— RIS-C2 confirmed (imaginary part discarded).")


# ===========================================================================
# OPT-C1 — adi_step_3d 3D PEC-cavity eigenfrequency
# ===========================================================================
#
# adi_step_3d uses an LOD split that, by its own code comment
# (adi.py:781-787), "applies the tridiagonal solve to ALL E components
# along each axis, which adds artificial diffusion for components whose
# curl has no derivative along that axis". The only 3D ADI tests check
# bounded/oscillating fields — none checks a physical eigenfrequency,
# while the 2D path has test_adi_cavity_resonance (2% gate).
#
# This test runs a lossless 3D PEC cubic cavity at a modest CFL factor
# (where splitting error is small) and compares the dominant resonance
# to the analytic mode  f_mnp = (C0/2)*sqrt((m/Lx)^2+(n/Ly)^2+(p/Lz)^2).
# A modest CFL is the FAIR setting: a structural scheme error shows up
# even there, so a large miss is strong evidence for OPT-C1.

@pytest.mark.xfail(strict=True, reason=(
    "OPT-C1: adi_step_3d LOD applies the implicit solve to off-axis E "
    "components (artificial diffusion, adi.py:781-787). 3D PEC-cavity "
    "eigenfrequency is expected to miss the analytic mode beyond the 2% "
    "gate the 2D ADI path holds. Adjudication test — see roadmap Part A."))
def test_optc1_adi_3d_cavity_eigenfrequency():
    L = 0.06           # 60 mm cube
    dx = dy = dz = 0.005   # 12 cells per side
    N = int(round(L / dx))

    # Degenerate fundamental: TE_101 / TE_011 / TE_110 of a cube.
    f_analytic = (C0 / 2.0) * np.sqrt((1.0 / L) ** 2 + (1.0 / L) ** 2)

    # 3D CFL limit; ADI is run at a modest factor so splitting error is small.
    dt_cfl = dx / (C0 * np.sqrt(3.0))
    dt = 2.0 * dt_cfl
    n_steps = 1500

    shape = (N, N, N)
    z = jnp.zeros(shape, dtype=jnp.float32)
    eps_r = jnp.ones(shape, dtype=jnp.float32)
    sigma = jnp.zeros(shape, dtype=jnp.float32)   # lossless — isolates the scheme

    # Short Gaussian pulse on Ey at an off-centre interior cell.
    t_arr = jnp.arange(n_steps) * dt
    tau = 8.0 * dt
    t0 = 4.0 * tau
    waveform = jnp.exp(-((t_arr - t0) / tau) ** 2).astype(jnp.float32)
    src = (N // 3, N // 2, N // 3, "ey", waveform)
    prb = (2 * N // 3, N // 2, 2 * N // 3, "ey")

    out = run_adi_3d(z, z, z, z, z, z, eps_r, sigma,
                     dt, dx, dy, dz, n_steps,
                     sources=[src], probes=[prb])
    probe_data = out[-1]
    signal = np.asarray(probe_data[:, 0], dtype=np.float64)

    skip = int(n_steps * 0.2)               # drop the excitation transient
    late = signal[skip:]
    freqs = np.fft.rfftfreq(len(late), d=float(dt))
    spectrum = np.abs(np.fft.rfft(late))
    spectrum[0] = 0.0                       # ignore DC
    f_peak = freqs[int(np.argmax(spectrum))]

    rel_err = abs(f_peak - f_analytic) / f_analytic
    assert rel_err < 0.02, (
        f"3D ADI cavity resonance {f_peak/1e9:.4f} GHz vs analytic "
        f"{f_analytic/1e9:.4f} GHz — error {rel_err*100:.2f}% > 2%")
