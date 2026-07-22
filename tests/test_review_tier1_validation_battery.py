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

Covered: CORE-C2, GEO-C1, PROBE-C1, RIS-C2, OPT-C1 (adi_step_3d
3D-cavity eigenfrequency — FIXED 2026-07-13 by the ZCZ full 3D ADI
scheme; the former strict-xfail marker was removed on XPASS per the
convention above).
"""

from __future__ import annotations

import numpy as np
import jax
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
# GEO-C1 — Kerr ADE exact implicit solve  (FIXED 2026-05-17, Stage 1)
# ===========================================================================
#
# apply_kerr_ade previously implemented  E <- E*(1-factor)  (explicit
# linearization). The docstring's stated discrete relation
# E^{n+1} -= factor*E^{n+1} has the exact solution E/(1+factor); the two
# agree only to O(factor) and diverge for factor >~ 0.1. Stage 1 (GEO-C1)
# switched apply_kerr_ade to the exact implicit form E/(1+factor) — same
# cost, unconditionally stable for this term. This is an intentional,
# reference-gated numerical change (NOT bit-identical to the old output).

def _kerr_setup():
    shape = (2, 2, 2)
    E0 = 3.0        # E^n (pre-update, sets the intensity |E^n|^2)
    E_lin = 5.0     # E_lin (post-linear-update field)
    eps_r = 2.0
    target_factor = 0.30                     # factor = chi3*E0^2/eps_r
    chi3 = target_factor * eps_r / E0 ** 2
    e_prev = (jnp.full(shape, E0, jnp.float32),
              jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
    state = _state(shape, ex=jnp.full(shape, E_lin, jnp.float32))
    chi3_arr = jnp.full(shape, chi3, jnp.float32)
    eps_r_arr = jnp.full(shape, eps_r, jnp.float32)
    return state, e_prev, chi3_arr, eps_r_arr, E0, E_lin, target_factor


def test_geoc1_kerr_ade_reactive_increment_solve():
    """GEO-C1 (#437 redesign) — apply_kerr_ade scales the INCREMENT (reactive ε_eff),
    NOT the whole field.

    Reactive Kerr is ε_eff = ε_r + χ³|E|² (lossless index change).  The correct update
    scales only the newly-integrated increment::

        E^{n+1} = E^n + (E_lin - E^n) / (1 + χ³·|E^n|²/ε_r)

    The pre-#437 code scaled the WHOLE field (E_lin/(1+factor)) — a nonlinear absorber
    that reduced |E| with zero phase shift (docs/research_notes/2026-07-22_kerr_operator_defect.md).
    This pins the reactive increment form and that it differs from the old dissipative one.
    """
    state, e_prev, chi3_arr, eps_r_arr, E0, E_lin, factor = _kerr_setup()
    out = apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr)
    got = float(np.asarray(out.ex)[0, 0, 0])

    reactive = E0 + (E_lin - E0) / (1.0 + factor)   # increment-scaled (correct)
    dissipative = E_lin / (1.0 + factor)            # whole-field (the old bug)
    assert got == pytest.approx(reactive, rel=1e-4)
    # genuinely different from the old dissipative operator at factor=0.3
    assert abs(got - dissipative) > 0.01 * E_lin


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
# OPT-C1 — adi_step_3d 3D PEC-cavity eigenfrequency  (FIXED 2026-07-13)
# ===========================================================================
#
# HISTORY: adi_step_3d used an LOD split that, by its own code comment,
# applied the tridiagonal solve to ALL E components along each axis —
# artificial diffusion on components whose curl has no derivative along
# that axis. Under the honest comparator (below) the 12^3-cavity peak at
# 2x CFL was 2.077 GHz, a 46% miss, and the mode energy decayed within
# ~4 ns. This test was xfail(strict) against that scheme.
#
# VERDICT (2026-07-13, marker removed per module convention): adi_step_3d
# now implements the full Zheng–Chen–Zhang two-sub-step 3D ADI (issue
# #338 follow-up — the "fully unconditionally-stable 3D ADI" the
# 2026-05-30 R2-STOP verdict sanctioned). Measured on this test's
# committed settings: f_peak = 3.8079 GHz vs analytic 3.8543 GHz ->
# rel_err = 1.204% < 2% (PASS), late/early probe RMS 0.999 (the mode
# rings; no artificial dissipation), and the CFL-5x stability rung in
# tests/test_adi.py stays green (bounded at 10x/50x as free witnesses).
# Evidence: scripts/diagnostics/gate_honesty_20260713/lane_338_adi_redesign/.
#
# This test runs a lossless 3D PEC cubic cavity at a modest CFL factor
# (where splitting error is small) and compares the dominant resonance
# to the analytic mode  f_mnp = (C0/2)*sqrt((m/Lx)^2+(n/Ly)^2+(p/Lz)^2).
#
# COMPARATOR CONVENTION (issue #338): _apply_pec_3d zeroes tangential E
# at node indices 0 and N-1, so the PEC walls sit L_eff = (N-1)*dx apart
# — the fence-post convention Grid documents ("PEC walls at index 0 and
# index N span exactly N*dx": a physical span of L needs N+1 nodes).
# Computing f_analytic from L = N*dx biased it 9.1% low — larger than
# the 2% gate itself, so the strict xfail could never legitimately
# XPASS. The falsifier test below pins the corrected convention: the
# known-good explicit Yee scheme on the SAME 12^3 node cavity lands
# 0.21% from the corrected analytic and 8.9% from the biased one.

def test_optc1_adi_3d_cavity_eigenfrequency():
    L = 0.06           # nominal 60 mm cube
    dx = dy = dz = 0.005   # 12 cells per side
    N = int(round(L / dx))
    # Walls at nodes 0 and N-1 (node-indexed PEC, see block comment above):
    # the honest analytic length is one cell short of the nominal L.
    L_eff = (N - 1) * dx

    # Degenerate fundamental: TE_101 / TE_011 / TE_110 of the L_eff cube.
    f_analytic = (C0 / 2.0) * np.sqrt((1.0 / L_eff) ** 2 + (1.0 / L_eff) ** 2)

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


def test_optc1_falsifier_yee_matches_corrected_analytic():
    """Falsifier for the OPT-C1 comparator length (issue #338).

    The SAME 12^3 node cavity as test_optc1_adi_3d_cavity_eigenfrequency,
    advanced by the known-good explicit Yee scheme (the test_cavity.py
    rung: update_h / update_e / apply_pec — production PEC also zeroes
    tangential E at nodes 0 and N-1), must resonate within the same 2%
    gate of the CORRECTED analytic built from L_eff = (N-1)*dx.

    Discrimination: had the honest cavity length been N*dx instead, the
    Yee peak would sit ~8.9% from that analytic — asserted below, so
    this rung fails if the comparator convention is wrong EITHER way.
    Measured 2026-07-13: f_peak = 3.846 GHz vs corrected 3.854 GHz
    -> 0.21% (FFT half-bin 0.28%); vs biased 3.533 GHz -> 8.87%.
    """
    from rfx.core.yee import init_state, init_materials, update_e, update_h
    from rfx.boundaries.pec import apply_pec

    N = 12
    dx = 0.005
    L_eff = (N - 1) * dx
    f_corrected = (C0 / 2.0) * np.sqrt(2.0) / L_eff   # TE101 of L_eff cube
    f_biased = (C0 / 2.0) * np.sqrt(2.0) / (N * dx)   # the pre-#338 analytic

    dt = 0.99 * dx / (C0 * np.sqrt(3.0))
    n_steps = 6000

    state = init_state((N, N, N))
    materials = init_materials((N, N, N))

    # Same pulse width and source/probe cells as the ADI rung above.
    tau = 16.0 * dx / (C0 * np.sqrt(3.0))
    t0 = 4.0 * tau
    si, sj, sk = N // 3, N // 2, N // 3
    pi, pj, pk = 2 * N // 3, N // 2, 2 * N // 3

    @jax.jit
    def step(s):
        s = update_h(s, materials, dt, dx)
        s = update_e(s, materials, dt, dx)
        return apply_pec(s)

    signal = np.zeros(n_steps)
    for n in range(n_steps):
        t = n * dt
        state = step(state)
        ey = state.ey.at[si, sj, sk].add(float(np.exp(-((t - t0) / tau) ** 2)))
        state = state._replace(ey=ey)
        signal[n] = float(state.ey[pi, pj, pk])

    assert np.max(np.abs(signal)) > 1e-9, "no probe energy — dead run"

    skip = int(n_steps * 0.2)
    late = signal[skip:]
    freqs = np.fft.rfftfreq(len(late), d=float(dt))
    spectrum = np.abs(np.fft.rfft(late))
    # Search near the mode (soft-J source leaves a static Ey offset in a
    # closed cavity — exclude DC and far-off bins, as test_cavity.py does).
    mask = (freqs >= 0.5 * f_corrected) & (freqs <= 1.5 * f_corrected)
    f_peak = freqs[int(np.argmax(np.where(mask, spectrum, 0.0)))]

    err_corrected = abs(f_peak - f_corrected) / f_corrected
    err_biased = abs(f_peak - f_biased) / f_biased
    assert err_corrected < 0.02, (
        f"Yee on the 12^3 node cavity: {f_peak/1e9:.4f} GHz vs corrected "
        f"analytic {f_corrected/1e9:.4f} GHz — {err_corrected*100:.2f}% > 2%; "
        "the (N-1)*dx comparator convention is wrong")
    assert err_biased > 0.05, (
        f"Yee peak {f_peak/1e9:.4f} GHz sits within 5% of the OLD N*dx "
        f"analytic {f_biased/1e9:.4f} GHz — the pre-#338 comparator would "
        "have been right after all; re-adjudicate the convention")
