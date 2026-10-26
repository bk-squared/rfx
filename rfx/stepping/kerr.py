"""Instantaneous reactive Kerr correction for the E-field step."""

from __future__ import annotations

from rfx.materials.nonlinear import _KERR_FIXED_POINT_ITERS


def apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr):
    """Apply the instantaneous **reactive** Kerr correction to the E-field (#437, #446).

    A Kerr χ³ nonlinearity is a REACTIVE, intensity-dependent permittivity
    (LOSSLESS — it changes the phase velocity, not the amplitude):

        D = ε0·(ε_r + χ³·|E|²)·E     ⇔     P_NL = ε0·χ³·|E|²·E

    This solves that constitutive relation SELF-CONSISTENTLY (the "D-based" update),
    which is the rigorous reactive Kerr.  The displacement is reconstructed from the
    pre-update field and advanced linearly, then E is recovered by solving the cubic::

        D_target/ε0 = ε_r·E_lin + χ³·|E^n|²·E^n           (advance D linearly)
        solve  ε_r·E^{n+1} + χ³·|E^{n+1}|²·E^{n+1} = D_target   (fixed-point)

    where ``E_lin`` is the linear (ε_r) E-update the scan already produced.  This is
    STATELESS — D^n is exactly reconstructable from E^n since E^n itself solved the
    constitutive relation — so no auxiliary D-field is carried.

    **Why the self-consistent solve, not the earlier increment scaling** (``E^n +
    ΔE/(1+χ³|E^n|²/ε_r)``): the increment form applies ε_eff to the E-update increment
    ``ΔE ∝ ∂ₓH``, which is 90° out of phase with E, so the χ³ correction hits the
    increment where it is smallest ⇒ it UNDERESTIMATES the self-phase-modulation index
    shift. The true-CW absolute oracle (tests/oracle/test_kerr_spm_absolute_oracle.py, #446, using
    ``waveform='continuous_wave'`` so ⟨E²⟩=A²/2 is unambiguous) pins the D-based operator at
    ratio ≈0.95 of the (3/8)χ³A² textbook (residual ~5% = Yee dispersion), while the increment
    form scores ~0.33 there — the gate cleanly discriminates them. (An earlier pulsed-source
    3D estimate read the pair as ~0.21× increment vs ~0.59× D-based — a comparator-independent
    2.8× relative gain, but its absolute scale was biased low by the pulsed-⟨E²⟩ ambiguity that #446
    resolves.) See docs/research_notes/2026-07-23_kerr_quantitative_spm_finding.md.
    The pre-#437 form ``E → E/(1+factor)`` was worse still — a nonlinear
    ABSORBER (phase-neutral to first order). χ³=0 ⇒ D_target=ε_r·E_lin, denom=ε_r ⇒
    E=E_lin (byte-identical). Stable and lossless up to strong χ³ (verified χ³≤4).

    Parameters
    ----------
    state : FDTDState
        State **after** the linear ε_r E-update (contains ``E_lin``).
    e_prev : tuple(jnp.ndarray, jnp.ndarray, jnp.ndarray)
        The E-field components (ex, ey, ez) **before** the linear update (``E^n``),
        used to reconstruct D^n and form the co-located pre-update intensity.
    chi3_arr : jnp.ndarray, shape (Nx, Ny, Nz)
        Third-order susceptibility at each cell (m^2/V^2). Zero where linear ⇒ the
        cubic reduces to E=E_lin (byte-identical).
    eps_r_arr : jnp.ndarray, shape (Nx, Ny, Nz)
        Relative permittivity per cell (ε_r ≥ 1).

    Returns
    -------
    FDTDState with the reactive-Kerr-corrected E-field components.
    """
    ex_p, ey_p, ez_p = e_prev
    e_sq_p = ex_p ** 2 + ey_p ** 2 + ez_p ** 2       # |E^n|^2 (co-located, pre-update)
    # D_target/ε0 = ε_r·E_lin + χ³·|E^n|²·E^n  (advance the reconstructed D linearly)
    dtx = eps_r_arr * state.ex + chi3_arr * e_sq_p * ex_p
    dty = eps_r_arr * state.ey + chi3_arr * e_sq_p * ey_p
    dtz = eps_r_arr * state.ez + chi3_arr * e_sq_p * ez_p
    # solve  ε_r·E + χ³·|E|²·E = D_target  by fixed-point  E = D_target/(ε_r + χ³|E|²)
    ex, ey, ez = state.ex, state.ey, state.ez         # initial guess = E_lin
    for _ in range(_KERR_FIXED_POINT_ITERS):
        denom = eps_r_arr + chi3_arr * (ex ** 2 + ey ** 2 + ez ** 2)   # = ε_eff (>= ε_r >= 1)
        ex = dtx / denom
        ey = dty / denom
        ez = dtz / denom

    return state._replace(ex=ex, ey=ey, ez=ez)
