"""tests/_transverse_resonance_o3.py

Exact transverse-resonance comparator for the O3 parallel-plate fixture
(issue #700) — the 4-conductor supermode model the O3 gate is paired with.

Why this exists (issue #700 scout, 2026-08-24): the O3 fixture in
``test_leontovich_alpha_oracle.py`` backs each Rs sheet with a PEC wall one
cell (g = 0.5 mm) behind it — the #642-avoidance choice. That makes the
z-stack

    PEC | g air | Rs sheet | b air | Rs sheet | g air | PEC

a FOUR-conductor line, not the closed form's two-plate guide. Its exact
(non-discretized) x-propagating TM spectrum contains

  * a strictly lossless z-uniform TEM supermode at kx = k0 exactly:
    uniform Hy across the whole stack -> zero Hy jump at each sheet ->
    zero sheet current -> zero dissipation;
  * a symmetric lossy supermode (alpha = 6.326 Np/m at 10 GHz for
    b = 5 mm, g = 0.5 mm) and an antisymmetric one (5.273 Np/m);
  * NO mode at the closed form alpha = Rs/(eta0*b) = 1.05482 Np/m.

A gap-only launch excites the lossless + symmetric-lossy pair, and the
fitted "alpha" of the fixture is the transient slope of that two-mode
beat — pure fixture physics, resolution-independent (dx 0.5 -> 0.25 mm
moved the envelope 0.3381 -> 0.3323 only). The closed form re-emerges as
the stub influence vanishes: in the small-kz*g regime the exact roots
follow

    alpha_antisym = Rs/(2*eta0*g)
    alpha_sym     = Rs/(2*eta0*g) + Rs/(eta0*b)

(verified against the exact solve to <= 0.4% relative over g = 0.5 ->
32 mm), so the symmetric lossy supermode converges to the closed form
Rs/(eta0*b) exactly as the stub term Rs/(2*eta0*g) -> 0 with deepening
stub — rel-to-closed-form error b/(2g), halving per g-doubling. That
convergence is gated in ``test_o3_model_limit_reduces_to_closed_form``
(the ladder stops at g = 32 mm: by g = 64 mm the stub's transverse
seed for the symmetric lossy root stops landing in its basin — the
adversarial verifier tracked the branch itself to g >= 80 mm with the
closed-form residual still shrinking monotonically, 0.078 -> 0.036, and
kz*g(64 mm) = 0.98+0.96j, nowhere near the stub resonance pi/2 — so the
ladder's end is a SEEDING limitation of find_symmetric_lossy_mode, not
a property of the physics).

Method: impedance transformation up the stack. For the TM field
(Ex, Ez, Hy; y-invariant, exp(-j kx x) propagation) each air layer is a
transmission line with transverse wavenumber kz = sqrt(k0^2 - kx^2) and
characteristic impedance Zc = eta0 * kz / k0; a resistive sheet of
surface impedance Rs is a shunt admittance 1/Rs (tangential Ex
continuous, Hy jump = Ex/Rs). Transform the bottom-PEC short up through
g | sheet | b | sheet | g and require Z_top = 0 (top PEC). Roots kx of
that scalar condition are the supermodes; alpha = -Im(kx).

House comparator rule (research/CLAUDE.md): a new comparator must first
reproduce a known-good limit as a checkable artifact in its own tests.
The committed self-checks live in ``test_leontovich_alpha_oracle.py``:

  * ``test_o3_model_limit_reduces_to_closed_form`` — LIMIT REDUCTION:
    as the stub deepens (g = 0.5 -> 32 mm) the symmetric lossy-mode
    alpha converges (gated: log-space monotone, rate b/(2g), final
    residual) to the closed form 1.05482 Np/m;
  * ``test_o3_model_fits_measured_field`` — FIELD FIT: on the committed
    O3 fixture the 3-supermode model fits the measured 2-D Hy(x, z) to
    < 1% relative rms at every frequency bin (scout measured
    0.26–0.56%).

Scout provenance (numbers reproduced by the self-checks):
scratchpad wf-backlog/s700.md, scripts s700_e_modesolver.py /
s700_f_modeclosure.py; model-vs-measured fitted-alpha ladder
0.219/0.441/0.667/0.852/0.993 vs 0.231/0.456/0.679/0.868/1.008 Np/m at
8..12 GHz.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize

C0 = 299792458.0


def _kz_zc(kx, k0, eta0):
    kz = np.sqrt(complex(k0 * k0 - kx * kx))
    return kz, eta0 * kz / k0


def z_top(kx, k0, b, g, rs, eta0):
    """Transverse-resonance residual: impedance seen at the top PEC after
    transforming the bottom-PEC short up through g | sheet | b | sheet | g.
    A supermode is a root z_top(kx) = 0."""
    kz, zc = _kz_zc(kx, k0, eta0)

    def through(Z, d):
        t = np.tan(kz * d)
        return zc * (Z + 1j * zc * t) / (zc + 1j * Z * t)

    Z = 0.0 + 0.0j                      # bottom PEC short
    Z = through(Z, g)
    Z = 1.0 / (1.0 / Z + 1.0 / rs) if Z != 0 else 0.0j
    Z = through(Z, b)
    Z = 1.0 / (1.0 / Z + 1.0 / rs)
    Z = through(Z, g)
    return Z


def find_modes(f, b, g, rs, eta0, *, n_seed=60, max_modes=3):
    """Supermodes kx (complex, Re > 0, sorted least-lossy first) of the
    PEC|g|sheet|b|sheet|g|PEC stack near kx ~ k0.

    The z-uniform lossless TEM (kx = k0 exactly; uniform Hy, Ex = 0, zero
    sheet current) is appended analytically when Newton misses it — kz -> 0
    makes that root numerically awkward but it is exact by construction.
    """
    k0 = 2.0 * np.pi * f / C0
    with np.errstate(all="ignore"):     # Newton walks through poles of tan
        return _find_modes_inner(k0, b, g, rs, eta0, n_seed, max_modes)


def _find_modes_inner(k0, b, g, rs, eta0, n_seed, max_modes):
    roots: list[complex] = []
    for re in np.linspace(0.985, 1.02, n_seed):
        for im in (-1e-6, -3e-5, -1e-4, -1e-3, -3e-3, -1e-2, -3e-2):
            try:
                r = scipy.optimize.newton(
                    lambda kx: z_top(kx, k0, b, g, rs, eta0),
                    k0 * (re + 1j * im), tol=1e-12, maxiter=100)
            except (RuntimeError, ValueError):
                continue
            if not np.isfinite(r):
                continue
            if abs(z_top(r, k0, b, g, rs, eta0)) > 1e-7 * eta0:
                continue
            if r.real < 0:
                r = -r
            if not any(abs(r - q) < 5e-3 for q in roots):
                roots.append(complex(r))
    if not any(abs(r - k0) < 1e-3 for r in roots):
        roots.append(complex(k0))
    return sorted(roots, key=lambda r: -r.imag)[:max_modes]


def find_symmetric_lossy_mode(f, b, g, rs, eta0):
    """Exact kx of the SYMMETRIC lossy supermode, tracked by Newton from
    the small-kz*g perturbative estimate

        alpha_pert = rs/(eta0*b) + rs/(2*eta0*g)

    (gap-Leontovich term + stub-current term). Used by the limit-reduction
    self-check: as g deepens the stub term vanishes and the exact root's
    alpha must converge to the closed form rs/(eta0*b). Returns complex kx
    or raises RuntimeError if no root is found in the expected window
    (measured: a seed outside the root's basin — the branch itself
    continues past g = 80 mm; see the module docstring)."""
    k0 = 2.0 * np.pi * f / C0
    a_pert = rs / (eta0 * b) + rs / (2.0 * eta0 * g)
    for re_s in (1 + 2e-5, 1 + 1e-4, 1 + 4e-4, 1 + 1e-3):
        for sc in (1.0, 0.8, 1.2):
            seed = k0 * re_s - 1j * a_pert * sc
            try:
                with np.errstate(all="ignore"):
                    r = scipy.optimize.newton(
                        lambda kx: z_top(kx, k0, b, g, rs, eta0),
                        seed, tol=1e-12, maxiter=100)
            except (RuntimeError, ValueError):
                continue
            if not np.isfinite(r):
                continue
            if abs(z_top(r, k0, b, g, rs, eta0)) > 1e-7 * eta0:
                continue
            if r.real < 0:
                r = -r
            if 0.3 * a_pert < -r.imag < 3.0 * a_pert:
                return complex(r)
    raise RuntimeError(
        f"symmetric lossy supermode not found near alpha ~ {a_pert:.4f} "
        f"(f={f:.3g}, b={b:.3g}, g={g:.3g})")


def hy_profile(kx, f, b, g, rs, eta0, z_pts):
    """Complex Hy(z) of one supermode across the stack (z = 0 at the
    bottom PEC), by piecewise ABCD propagation with the sheet-current
    jump Hy -> Hy + Ex/Rs at each sheet. The exact kx = k0 lossless TEM
    is the uniform profile."""
    k0 = 2.0 * np.pi * f / C0
    if abs(kx - k0) < 1e-6 * k0:
        return np.ones(len(z_pts), complex)
    kz, zc = _kz_zc(kx, k0, eta0)
    sheets = (g, g + b)
    prof = np.zeros(len(z_pts), complex)
    Ex0, Hy0 = 0.0 + 0.0j, 1.0 + 0.0j   # bottom PEC: Ex = 0
    zb, si = 0.0, 0
    for m, zp in enumerate(np.asarray(z_pts, float)):
        while si < 2 and zp > sheets[si] + 1e-12:
            d = sheets[si] - zb
            Ex = Ex0 * np.cos(kz * d) + 1j * zc * Hy0 * np.sin(kz * d)
            Hy = Hy0 * np.cos(kz * d) + 1j * Ex0 / zc * np.sin(kz * d)
            Hy = Hy + Ex / rs           # sheet current jump J = Ex/Rs
            Ex0, Hy0, zb = Ex, Hy, sheets[si]
            si += 1
        d = zp - zb
        prof[m] = Hy0 * np.cos(kz * d) + 1j * Ex0 / zc * np.sin(kz * d)
    return prof


def fit_hy_field(xs, z_nodes, hy_meas, f, b, g, rs, eta0):
    """Fit the supermode expansion Hy(x, z) = sum_i c_i phi_i(z)
    exp(-j kx_i (x - x0)) to a measured complex Hy sampled on
    (len(xs), len(z_nodes)); only the complex amplitudes c_i are fitted.

    Returns dict with:
      modes      — the supermode kx list,
      amps       — fitted complex amplitudes c_i,
      rel_resid  — relative rms residual of the fit over the whole plane,
      alpha_model — the model's own log-linear fitted alpha of |Hy| at
                    the z node nearest the gap midplane over xs (the
                    two-mode-transient prediction for this probe span),
      alpha_meas — the same fit applied to the measured column.
    """
    xs = np.asarray(xs, float)
    z_nodes = np.asarray(z_nodes, float)
    hy_meas = np.asarray(hy_meas)
    modes = find_modes(f, b, g, rs, eta0)
    X = np.zeros((hy_meas.size, len(modes)), complex)
    for mi, m in enumerate(modes):
        prof = hy_profile(m, f, b, g, rs, eta0, z_nodes)
        X[:, mi] = (np.exp(-1j * m * (xs - xs[0]))[:, None]
                    * prof[None, :]).ravel()
    y = hy_meas.ravel()
    amps, *_ = np.linalg.lstsq(X, y, rcond=None)
    model = (X @ amps).reshape(hy_meas.shape)
    rel_resid = float(np.linalg.norm(model - hy_meas)
                      / np.linalg.norm(hy_meas))
    kmid = int(np.argmin(np.abs(z_nodes - (g + b / 2.0))))
    alpha_model = _fit_alpha_loglin(xs, np.abs(model[:, kmid]))
    alpha_meas = _fit_alpha_loglin(xs, np.abs(hy_meas[:, kmid]))
    return {"modes": modes, "amps": amps, "rel_resid": rel_resid,
            "alpha_model": alpha_model, "alpha_meas": alpha_meas}


def _fit_alpha_loglin(xs, mag):
    y = np.log(np.asarray(mag, float))
    A = np.vstack([np.ones_like(xs), -xs]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1])
