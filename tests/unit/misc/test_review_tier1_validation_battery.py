"""Tier-1 reproduction battery for the 2026-05-16 rfx code review.

Each test makes one CRITICAL finding executable, so the finding is either *confirmed* (the test
reproduces the defect) or *refuted* (the test passes and the finding
was overstated).

Convention:
- A confirmed defect ⇒ the test asserts CORRECT physics and is marked
  ``xfail(strict=True)``. It fails today (bug present) and will
  XPASS once fixed, forcing the marker to be removed.
- A factual-but-not-defect finding ⇒ a plain characterization test
  that documents the actual behavior.
For the one-dimensional z-axis tests, Yee staggering places Ey at nodes and Hx at cell centres.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.core.yee import FDTDState, MaterialArrays, EPS_0, MU_0
from rfx.stepping.yee import update_h_nu, update_e_nu
from rfx.stepping.kerr import apply_kerr_ade
from rfx.probes.probes import init_flux_monitor, flux_spectrum
from rfx.grid import C0


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


_DZ = jnp.array([1., 1., 3., 1., 1., 3., 1., 1.], dtype=jnp.float32) * 1.0e-3


def test_corec2_h_update_uses_local_cell_width():
    """update_h_nu, fed the production inv_d_h, recovers the exact
    derivative of a z-linear E field on a 3:1-graded mesh.
    Interior z cells exclude k=nz-1, the forward-difference boundary with zero inverse spacing.
    """
    from rfx.model.graded_grid import _profile_to_inv_arrays

    dz = _DZ
    assert np.isclose(float(np.max(np.asarray(dz))) / float(np.min(np.asarray(dz))), 3.0), "Graded z widths need 3:1 jumps to separate mean and local spacing above float32 noise"
    nz = int(dz.shape[0])
    nx = ny = 3
    dx = 1.0e-3
    a = 1.0

    z_node = jnp.concatenate([jnp.zeros(1, jnp.float32), jnp.cumsum(dz)])[:nz]
    ey = jnp.broadcast_to(a * z_node[None, None, :], (nx, ny, nz)).astype(jnp.float32)
    state = _state((nx, ny, nz), ey=ey)
    mats = _free_space((nx, ny, nz))
    dt = 1.0e-13

    _, inv_dz_h = _profile_to_inv_arrays(np.asarray(dz), dtype="float32")
    _, inv_xy_h = _profile_to_inv_arrays(np.full(nx, dx, dtype=np.float64), dtype="float32")
    new = update_h_nu(state, mats, dt, inv_xy_h, inv_xy_h, inv_dz_h, high_wall=(False, False, False))

    discrete = np.asarray(new.hx)[1, 1, :] * MU_0 / dt
    np.testing.assert_allclose(discrete[:nz - 1], a, rtol=0.02)


def test_corec2_e_update_uses_mean_spacing():
    """update_e_nu, fed the production inv_d_e, recovers the exact
    derivative of a z-linear H field on a 3:1-graded mesh.
    Interior z cells exclude k=0, the backward-difference boundary.
    """
    from rfx.model.graded_grid import _profile_to_inv_arrays

    dz = _DZ
    nz = int(dz.shape[0])
    nx = ny = 3
    dx = 1.0e-3
    a = 1.0

    z_node = jnp.concatenate([jnp.zeros(1, jnp.float32), jnp.cumsum(dz)])[:nz]
    z_half = z_node + dz / 2.0
    hy = jnp.broadcast_to(a * z_half[None, None, :], (nx, ny, nz)).astype(jnp.float32)
    state = _state((nx, ny, nz), hy=hy)
    mats = _free_space((nx, ny, nz))
    dt = 1.0e-13

    inv_dz_e, _ = _profile_to_inv_arrays(np.asarray(dz), dtype="float32")
    inv_xy_e, _ = _profile_to_inv_arrays(np.full(nx, dx, dtype=np.float64), dtype="float32")
    new = update_e_nu(state, mats, dt, inv_xy_e, inv_xy_e, inv_dz_e,
                      cell_sizes=(None, None, dz), high_wall=(False, False, False))

    discrete = -np.asarray(new.ex)[1, 1, :] * EPS_0 / dt
    np.testing.assert_allclose(discrete[1:], a, rtol=0.02)


def _kerr_setup():
    shape = (2, 2, 2)
    PRE_UPDATE_FIELD = 3.0
    E0 = PRE_UPDATE_FIELD
    POST_LINEAR_FIELD = 5.0
    E_lin = POST_LINEAR_FIELD
    eps_r = 2.0
    target_factor = 0.30
    chi3 = target_factor * eps_r / E0 ** 2
    e_prev = (jnp.full(shape, E0, jnp.float32),
              jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
    state = _state(shape, ex=jnp.full(shape, E_lin, jnp.float32))
    chi3_arr = jnp.full(shape, chi3, jnp.float32)
    eps_r_arr = jnp.full(shape, eps_r, jnp.float32)
    return state, e_prev, chi3_arr, eps_r_arr, E0, E_lin, target_factor


def test_geoc1_kerr_ade_selfconsistent_solve():
    """GEO-C1 (#446 D-based) — apply_kerr_ade solves the reactive-Kerr constitutive relation
    SELF-CONSISTENTLY, not by scaling the increment.

    Reactive Kerr is D = ε0(ε_r + χ³|E|²)E.  The D-based update advances the reconstructed
    displacement linearly and recovers E from the cubic::

        D_target/ε0 = ε_r·E_lin + χ³·|E^n|²·E^n
        solve  ε_r·E^{n+1} + χ³·|E^{n+1}|²·E^{n+1} = D_target      (fixed-point)

    This is the rigorous reactive Kerr; the earlier increment scaling
    E^n + (E_lin-E^n)/(1+factor) UNDERESTIMATES the SPM (~2.8×; #446). This pins the
    self-consistent value and that it differs from both the increment and the pre-#437
    dissipative (E_lin/(1+factor)) forms.
    """
    state, e_prev, chi3_arr, eps_r_arr, E0, E_lin, factor = _kerr_setup()
    out = apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr)
    got = float(np.asarray(out.ex)[0, 0, 0])

    chi3 = float(np.asarray(chi3_arr)[0, 0, 0])
    eps_r = float(np.asarray(eps_r_arr)[0, 0, 0])
    D_target = eps_r * E_lin + chi3 * E0 ** 3
    E = E_lin
    for _ in range(4):
        E = D_target / (eps_r + chi3 * E ** 2)
    assert got == pytest.approx(E, rel=1e-3)
    increment = E0 + (E_lin - E0) / (1.0 + factor)
    assert abs(got - increment) > 0.01 * E_lin


def test_probec1_flux_area_is_axis_aware():
    """flux_spectrum weights an x-normal face by dy*dz, not dx*dx.
    Five stored nodes bound four cells on each non-periodic axis.
    """
    DX, DY, DZ = 1.0e-3, 2.0e-3, 4.0e-3
    assert DY * DZ != DX * DX, "Non-cubic cells must distinguish tangential face area from dx squared"
    grid_shape = (5, 5, 5)
    freqs = jnp.array([1.0e9])

    mon = init_flux_monitor(axis=0, index=2, freqs=freqs,
                            grid_shape=grid_shape, d1=DY, d2=DZ)
    n1, n2 = grid_shape[1], grid_shape[2]
    ones = jnp.ones((1, n1, n2), dtype=jnp.complex64)
    zeros = jnp.zeros((1, n1, n2), dtype=jnp.complex64)
    mon = mon._replace(e1_dft=ones, h2_dft=ones, e2_dft=zeros, h1_dft=zeros)

    flux = float(np.asarray(flux_spectrum(mon))[0])
    correct = (n1-1) * (n2-1) * DY * DZ
    assert flux == pytest.approx(correct, rel=1e-5)
    assert flux != pytest.approx((n1-1) * (n2-1) * DX * DX, rel=1e-3)


def test_risc2_np_interp_complex_part_behavior():
    xp = np.array([0.0, 1.0])
    fp = np.array([1.0 + 0.0j, 1.0 + 10.0j])
    x = np.array([0.5])

    r = np.interp(x, xp, fp)
    proper = np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)

    assert np.allclose(r, proper), (
        f"np.interp complex result {r} != proper complex interp {proper} "
        "— RIS-C2 confirmed (imaginary part discarded).")


def test_optc1_falsifier_yee_matches_corrected_analytic():
    """Falsifier for the OPT-C1 comparator length (issue #338).

    Discrimination: had the honest cavity length been N*dx instead, the
    Yee peak would sit ~8.9% from that analytic — asserted below, so
    this rung fails if the comparator convention is wrong EITHER way.
    Measured 2026-07-13: f_peak = 3.846 GHz vs corrected 3.854 GHz
    -> 0.21% (FFT half-bin 0.28%); vs biased 3.533 GHz -> 8.87%.
    Search near the TE101 mode because a soft-J source leaves a static Ey offset in a closed cavity.
    """
    from rfx.core.yee import init_state, init_materials
    from rfx.stepping.yee import update_e, update_h
    from rfx.boundaries.pec import apply_pec

    N = 12
    dx = 0.005
    L_eff = (N - 1) * dx
    f_corrected = (C0 / 2.0) * np.sqrt(2.0) / L_eff
    f_biased = (C0 / 2.0) * np.sqrt(2.0) / (N * dx)

    dt = 0.99 * dx / (C0 * np.sqrt(3.0))
    n_steps = 6000

    state = init_state((N, N, N))
    materials = init_materials((N, N, N))

    tau = 16.0 * dx / (C0 * np.sqrt(3.0))
    t0 = 4.0 * tau
    si, sj, sk = N // 3, N // 2, N // 3
    pi, pj, pk = 2 * N // 3, N // 2, 2 * N // 3

    @jax.jit
    def step(s):
        s = update_h(s, materials, dt, dx, high_wall=(False, False, False))
        s = update_e(s, materials, dt, dx, high_wall=(False, False, False))
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
