"""Exact discrete 1D scattering model for W2/W3 window derivation.

Model class: geometry/first-principles (SPEC-00 0.2-2 window-source
rule). It is derived from the SAME update-equation family as the solver
(Christ 2002 Eqs.(1)-(2) == rfx update_e_nu/update_h_nu restricted to
the x-invariant TE10 fixture) but is computed in the FREQUENCY domain
by direct linear solve — it shares no FDTD time-stepping data with the
measurements it will judge.

Reduction (derived in the pre-declaration note): for the fixture's
x-invariant TE10-like mode Ex = E_k sin(pi y / b) the rfx NU update
equations reduce, per z-node k, to the scalar recurrence

    inv_e[k] * ( inv_h[k](E_{k+1}-E_k) - inv_h[k-1](E_k-E_{k-1}) )
        + (S0^2 - Sy^2) E_k = 0

    S0 = 2 sin(w dt/2) / (c0 dt),   Sy = 2 sin(ky dy/2) / dy,
    ky = pi/b,  inv_e[k] = 2/(d[k-1]+d[k]),  inv_h[k] = 1/d[k].

On a uniform run of cells d this gives the standard discrete relation
S0^2 = Sy^2 + Sz^2 with Sz = 2 sin(kz d/2)/d, which pins the Bloch
wavenumbers kz1 (fine) and kz2 (coarse). The scattering solve pins the
incident/reflected decomposition on the fine side and the transmitted
Bloch wave on the coarse side of an arbitrary explicit profile.
"""

from __future__ import annotations

import numpy as np

C0 = 299792458.0


def s0_sy(freq: float, dt: float, dy: float, b: float):
    w = 2 * np.pi * freq
    s0 = 2 * np.sin(w * dt / 2) / (C0 * dt)
    ky = np.pi / b
    sy = 2 * np.sin(ky * dy / 2) / dy
    return s0, sy


def bloch_kz(freq: float, dt: float, dy: float, b: float, d: float) -> float:
    """Discrete propagation constant kz on a uniform-d region (real,
    propagating; raises if evanescent)."""
    s0, sy = s0_sy(freq, dt, dy, b)
    arg = d * np.sqrt(s0 ** 2 - sy ** 2) / 2
    if not (0 < arg <= 1):
        raise ValueError(f"evanescent/aliased: arg={arg}")
    return 2 * np.arcsin(arg) / d


def step_reflection(d0: float, d1: float, freq: float, dt: float,
                    dy: float, b: float) -> float:
    """|R| of a SINGLE d0 -> d1 cell-size step, in closed form.

    Same model as ``scattering`` on ``[d0]*n_lead + [d1]*n_tail`` — this
    is that problem solved on paper instead of by ``np.linalg.solve``,
    and it is the formulation the F7 per-step numbers must use.

    Why the solve is not good enough here. A single step has exactly ONE
    non-uniform recurrence row (the junction node k = L, the only k whose
    two cells differ); every other row is the uniform Bloch relation the
    asymptotic forms already satisfy. Eliminating them leaves a 2x2
    system, and in it the REAL part of both numerator and denominator
    cancels identically — that cancellation IS the smallness of |R|. A
    dense float64 solve of the full (n+2)x(n+2) system performs the same
    cancellation on numbers of order 1/d^2 ~ 1e10 to produce a residue of
    order 1e-7, so its relative accuracy on |R| is about 1e-5, and its
    last bits depend on the LAPACK build: the committed W6 JSON (macOS
    Accelerate, arm64) and any Linux/OpenBLAS x86 run of the same code
    disagree by up to 3e-4 per step and 4e-6 on the 46-step sum.

    Derivation. With E_k = e^{i q1 k} + R e^{-i q1 k} on the fine side,
    E_k = T e^{i q2 (k - (n-1))} on the coarse side, q_j = kz_j d_j, and
    the junction row written with a = inv_e[L] inv_h[L], b_ = inv_e[L]
    inv_h[L-1], gam = S0^2 - Sy^2:

        |R| = |Y1 - Y2| / (Y1 + Y2),   Y_j = sin(q_j) / d_j

    a discrete admittance mismatch. The real parts of both sides vanish
    exactly because the Bloch relation gives gam = (2 sin(q_j/2)/d_j)^2
    on EACH side, so 2 a sin^2(q2/2) + 2 b_ sin^2(q1/2) = gam identically.
    Then sin(q_j)/d_j = sqrt(gam) sqrt(1 - gam d_j^2 / 4), and rationalising
    the difference of the two square roots removes the last cancellation:

        |R| = [gam (d1^2 - d0^2) / 4] / (sqrt(A) + sqrt(B))^2
        A = 1 - gam d0^2 / 4,  B = 1 - gam d1^2 / 4

    Every term is now evaluated without subtracting nearly equal
    quantities, so |R| is accurate to a few ulp and is the same number on
    every platform. Checked against a 60-decimal-digit Gaussian
    elimination on the ORIGINAL (n+2)x(n+2) system (the same rows
    ``scattering`` builds): agreement to 5e-51 relative on the F7 steps
    10.667->8.333, 32.0->21.333 and 8.333->16.667 um, i.e. the closed form
    is the exact value of the solved system and the float64 solve is the
    approximation. ``tests/unit/nonuniform/test_band_builder_chain_model.py``
    keeps a float64 version of that check.

    Independent of the runway lengths, as the derivation says and the
    60-digit check confirms at n_lead = n_tail = 12.
    """
    s0, sy = s0_sy(freq, dt, dy, b)
    gam = s0 ** 2 - sy ** 2
    d0 = float(d0)
    d1 = float(d1)
    a = 1.0 - gam * d0 * d0 / 4.0
    b_ = 1.0 - gam * d1 * d1 / 4.0
    if not (a > 0 and b_ > 0):
        raise ValueError(f"evanescent/aliased step: d0={d0}, d1={d1}")
    return abs(gam * (d1 * d1 - d0 * d0) / 4.0) / (np.sqrt(a) + np.sqrt(b_)) ** 2


def _inv_arrays(profile: np.ndarray):
    d = np.asarray(profile, dtype=np.float64)
    inv_h = 1.0 / d
    inv_e = np.empty_like(d)
    inv_e[0] = 1.0 / d[0]
    inv_e[1:] = 2.0 / (d[:-1] + d[1:])
    return inv_e, inv_h


def scattering(profile: np.ndarray, n_lead: int, n_tail: int,
               freq: float, dt: float, dy: float, b: float):
    """Solve the discrete scattering problem for an explicit z profile.

    `profile` must start with >= n_lead uniform cells (size d1) and end
    with >= n_tail uniform cells (size d2). Incident Bloch wave of unit
    amplitude enters from the d1 side. Returns (R, T) complex amplitude
    coefficients of the reflected (fine-side) and transmitted
    (coarse-side) Bloch waves.
    """
    d = np.asarray(profile, dtype=np.float64)
    n = len(d)
    d1, d2 = d[0], d[-1]
    assert np.allclose(d[:n_lead], d1) and np.allclose(d[-n_tail:], d2)
    inv_e, inv_h = _inv_arrays(d)
    s0, sy = s0_sy(freq, dt, dy, b)
    kz1 = bloch_kz(freq, dt, dy, b, d1)
    kz2 = bloch_kz(freq, dt, dy, b, d2)
    q1 = kz1 * d1
    q2 = kz2 * d2

    # Unknowns: E_0..E_{n-1} (node values), R, T  -> n+2 unknowns.
    # Rows: recurrence at k=1..n-2 (n-2 rows) + 4 asymptotic-form rows
    # pinning the two leading nodes to  e^{i q1 k} + R e^{-i q1 k}
    # and the two trailing nodes to  T e^{i q2 (k-(n-1))}.
    A = np.zeros((n + 2, n + 2), dtype=complex)
    rhs = np.zeros(n + 2, dtype=complex)
    gam = s0 ** 2 - sy ** 2
    for k in range(1, n - 1):
        A[k - 1, k + 1] += inv_e[k] * inv_h[k]
        A[k - 1, k] += -inv_e[k] * (inv_h[k] + inv_h[k - 1]) + gam
        A[k - 1, k - 1] += inv_e[k] * inv_h[k - 1]
    r = n - 2
    iR, iT = n, n + 1
    for k in (0, 1):
        A[r, k] = 1.0
        A[r, iR] = -np.exp(-1j * q1 * k)
        rhs[r] = np.exp(1j * q1 * k)
        r += 1
    for k in (n - 2, n - 1):
        A[r, k] = 1.0
        A[r, iT] = -np.exp(1j * q2 * (k - (n - 1)))
        r += 1
    # Interior rows scale as 1/d^2, while Bloch boundary rows are O(1).
    # Balance equations before elimination: the small reflected amplitude
    # otherwise depends on the BLAS kernel (E1 N60/r1.2 failed its unchanged
    # 1e-9 replay window on Haswell). This preserves the exact R/T problem.
    row_scale = np.max(np.abs(A), axis=1)
    sol = np.linalg.solve(A / row_scale[:, None], rhs / row_scale)
    return sol[iR], sol[iT]


def power_rt(profile, n_lead, n_tail, freq, dt, dy, b):
    """(|R|, |T|, group-velocity-corrected |T|_power_amp) — the
    energy-normalized transmitted amplitude |T|*sqrt(vg2 dd2 /(vg1 dd1))
    is what should satisfy R^2 + T_pw^2 = 1 for a lossless junction;
    we report raw amplitudes and the flux-normalized one."""
    R, T = scattering(profile, n_lead, n_tail, freq, dt, dy, b)
    d1 = profile[0]
    d2 = profile[-1]
    # discrete group velocity dw/dkz on each uniform side
    def vg(dcell):
        f0 = freq
        df = freq * 1e-6
        k_p = bloch_kz(f0 + df, dt, dy, b, dcell)
        k_m = bloch_kz(f0 - df, dt, dy, b, dcell)
        return 2 * np.pi * 2 * df / (k_p - k_m)
    t_pw = abs(T) * np.sqrt(vg(d2) / vg(d1))
    return abs(R), abs(T), t_pw
