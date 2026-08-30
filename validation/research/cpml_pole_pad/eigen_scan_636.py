"""#636 M1 — discrete frozen-coefficient eigenvalue scan (root-cause probe).

Predeclared in docs/design_notes/i636_cpml_pole_pad_predeclaration.md
(commit 841dcc2) BEFORE this script was first run. Summary of the
declaration:

  H1: the shipped composition of the ADE pole recurrence with the CPML
  recursive-convolution correction is not unconditionally stable per
  cell: the frozen-coefficient one-step operator (E, H, P, P_prev,
  psi_e, psi_h) has spectral radius > 1 for some spatial frequency when
  a high-Q pole coexists with CPML sigma > 0.

  CONFIRMED if max|lambda| > 1 + 1e-6 over (layer, k) for pole+CPML,
  with both controls (pole alone; CPML alone) <= 1 + 1e-9.
  FALSIFIER F1: pole+CPML staying <= 1 + 1e-6 everywhere (1D face AND
  2D corner) falsifies H1 — record and stop; no tweaking.

The matrices below transcribe the SHIPPED update order and signs:

  rfx/core/yee.py::update_h          H^{n+1/2} = H^{n-1/2} - (dt/mu) curl_f(E^n)
  rfx/boundaries/cpml.py::apply_cpml_h   psi_h' = b psi_h + c D_f E;
                                         H += ch (psi_h' + (1/kappa - 1) D_f E)
  rfx/materials/lorentz.py::update_e_lorentz
                                     P' = a_p P + b_p P_prev + c_p E^n
                                     E' = Ca E + Cb curl_b(H^{n+1/2}) - Cc (P'-P)
  rfx/boundaries/cpml.py::apply_cpml_e   psi_e' = b psi_e + c D_b H^{n+1/2};
                                         E += ce (psi_e' + (1/kappa - 1) D_b H)

with D_f = (e^{ik dx} - 1)/dx, D_b = (1 - e^{-ik dx})/dx on the staggered
grid, ce = dt/(eps_r eps0) (equals Cb here: sigma = 0), ch = dt/mu0.
CPML profile coefficients come from the shipped
rfx.boundaries.cpml._cpml_profile (order=3, R=1e-15, kappa_max=1.0), and
the transcription of the profile for the alpha-rule variant is asserted
against the shipped function at the shipped alpha before use.

Run:  .venv/bin/python validation/research/cpml_pole_pad/eigen_scan_636.py
"""

from __future__ import annotations

import numpy as np

EPS_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
C0 = 1.0 / np.sqrt(EPS_0 * MU_0)

DX = 1e-3
F0 = 3e9
W0 = 2 * np.pi * F0
FREQ_MAX = 2.5 * F0  # fixture f_top = 7.5e9
ALPHA_SHIPPED = 0.05
ALPHA_RULE = 1.2 * 2 * np.pi * FREQ_MAX * EPS_0  # ~0.5007 S/m


def fixture_dt() -> float:
    """dt of the actual lock-test grid (importing rfx only for the grid)."""
    from rfx import Simulation
    sim = Simulation(freq_max=FREQ_MAX, domain=(45 * DX, 39 * DX, 12 * DX),
                     dx=DX, boundary="cpml", cpml_layers=8)
    return float(sim._build_grid().dt)


def profile(n_layers: int, dt: float, dx: float, alpha_max: float,
            order: int = 3, kappa_max: float = 1.0, R: float = 1e-15):
    """Transcription of rfx.boundaries.cpml._cpml_profile with alpha_max
    exposed. Asserted against the shipped function at ALPHA_SHIPPED."""
    eta = float(np.sqrt(MU_0 / EPS_0))
    d = n_layers * dx
    sigma_max = -float(np.log(R)) * (order + 1) / (2.0 * eta * d)
    sigma_max = sigma_max * kappa_max
    rho = 1.0 - np.arange(n_layers, dtype=np.float64) / max(n_layers - 1, 1)
    sigma = sigma_max * rho ** order
    kappa = 1.0 + (kappa_max - 1.0) * rho ** order
    alpha = alpha_max * (1.0 - rho)
    denom = sigma * kappa + kappa ** 2 * alpha
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS_0)
    c = np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
    return sigma, kappa, alpha, b, c


def check_profile_transcription(dt: float) -> None:
    from rfx.boundaries.cpml import _cpml_profile
    ship = _cpml_profile(8, dt, DX)
    s, k, a, b, c = profile(8, dt, DX, ALPHA_SHIPPED)
    for name, mine, theirs in (("sigma", s, ship.sigma), ("kappa", k, ship.kappa),
                               ("alpha", a, ship.alpha), ("b", b, ship.b),
                               ("c", c, ship.c)):
        theirs = np.asarray(theirs, dtype=np.float64)
        assert np.allclose(mine, theirs, rtol=2e-6), (name, mine, theirs)
    print("[transcription] profile matches shipped _cpml_profile (rtol 2e-6)")


class Material:
    def __init__(self, label, eps_inf, omega_0, delta, kappa_p):
        self.label = label
        self.eps_inf = eps_inf
        self.omega_0 = omega_0
        self.delta = delta
        self.kappa_p = kappa_p

    def ade(self, dt):
        w0, d, kp = self.omega_0, self.delta, self.kappa_p
        den = 1.0 + d * dt
        a_p = (2.0 - w0 ** 2 * dt ** 2) / den
        b_p = -(1.0 - d * dt) / den
        c_p = EPS_0 * kp * dt ** 2 / den
        gamma = self.eps_inf * EPS_0  # sigma = 0
        return a_p, b_p, c_p, 1.0, dt / gamma, 1.0 / gamma  # a,b,c,Ca,Cb,Cc


MAT_C1 = Material("C1 Lorentz Q60 eps_inf=4", 4.0, W0, W0 / 120.0, 3.0 * W0 ** 2)
MAT_C3 = Material("C3 Lorentz Q5 eps_inf=1", 1.0, W0, W0 / 10.0, 3.0 * W0 ** 2)
# Drude: omega_0 = 0, delta = gamma/2, kappa = omega_p^2 (rfx drude_pole)
MAT_C4 = Material("C4 Drude eps_inf=1", 1.0, 0.0, (W0 / 100.0) / 2.0, W0 ** 2)


def one_step_1d(zk, layer, mat_coeffs, dt, with_pole=True, with_cpml=True):
    """One-step matrix, state [Ez, Hy, P, Pprev, psi_e, psi_h]."""
    a_p, b_p, c_p, Ca, Cb, Cc = mat_coeffs
    if with_cpml:
        b, c, kap = layer
    else:
        b, c, kap = 0.0, 0.0, 1.0
    ch = dt / MU_0
    ce = Cb  # sigma = 0 -> ce == Cb
    Df = (zk - 1.0) / DX
    Db = (1.0 - 1.0 / zk) / DX

    n = 6
    M = np.zeros((n, n), dtype=complex)
    E, H, P, PP, PE, PH = range(n)

    # psi_h' = b psi_h + c Df E
    M[PH, PH] = b
    M[PH, E] = c * Df
    # H' = H + ch (1/kap) Df E + ch psi_h'
    M[H, H] = 1.0
    M[H, E] = ch * (1.0 / kap) * Df + ch * M[PH, E]
    M[H, PH] = ch * M[PH, PH]
    # P' = a_p P + b_p PP + c_p E
    M[P, P] = a_p
    M[P, PP] = b_p
    M[P, E] = c_p
    M[PP, P] = 1.0
    # psi_e' = b psi_e + c Db H'
    M[PE, PE] = b
    for col in range(n):
        M[PE, col] += c * Db * M[H, col]
    # E' = Ca E + Cb (1/kap) Db H' + ce psi_e' - Cc (P' - P)
    M[E, E] = Ca
    for col in range(n):
        M[E, col] += Cb * (1.0 / kap) * Db * M[H, col] + ce * M[PE, col]
        M[E, col] -= Cc * M[P, col]
    M[E, P] += Cc  # +Cc P^n
    if not with_pole:
        keep = [E, H, PE, PH]
        M = M[np.ix_(keep, keep)]
    return M


def one_step_2d(zx, zy, layer_x, layer_y, mat_coeffs, dt, with_pole=True):
    """2D TM corner: state [Ez, Hx, Hy, P, Pprev, pex, pey, phx, phy]."""
    a_p, b_p, c_p, Ca, Cb, Cc = mat_coeffs
    bx, cx, kx = layer_x
    by, cy, ky = layer_y
    ch = dt / MU_0
    ce = Cb
    Dfx = (zx - 1.0) / DX
    Dbx = (1.0 - 1.0 / zx) / DX
    Dfy = (zy - 1.0) / DX
    Dby = (1.0 - 1.0 / zy) / DX

    n = 9
    M = np.zeros((n, n), dtype=complex)
    E, HX, HY, P, PP, PEX, PEY, PHX, PHY = range(n)

    # psi_hy' = bx psi_hy + cx Dfx E ; Hy' = Hy + ch(1/kx)Dfx E + ch psi_hy'
    M[PHY, PHY] = bx
    M[PHY, E] = cx * Dfx
    M[HY, HY] = 1.0
    M[HY, E] = ch * (1.0 / kx) * Dfx + ch * M[PHY, E]
    M[HY, PHY] = ch * M[PHY, PHY]
    # psi_hx' = by psi_hx + cy Dfy E ; Hx' = Hx - ch(1/ky)Dfy E - ch psi_hx'
    M[PHX, PHX] = by
    M[PHX, E] = cy * Dfy
    M[HX, HX] = 1.0
    M[HX, E] = -ch * (1.0 / ky) * Dfy - ch * M[PHX, E]
    M[HX, PHX] = -ch * M[PHX, PHX]
    # P
    M[P, P] = a_p
    M[P, PP] = b_p
    M[P, E] = c_p
    M[PP, P] = 1.0
    # psi_ex' = bx psi_ex + cx Dbx Hy'
    M[PEX, PEX] = bx
    for col in range(n):
        M[PEX, col] += cx * Dbx * M[HY, col]
    # psi_ey' = by psi_ey + cy Dby Hx'
    M[PEY, PEY] = by
    for col in range(n):
        M[PEY, col] += cy * Dby * M[HX, col]
    # Ez' = Ca E + Cb[(1/kx)Dbx Hy' - (1/ky)Dby Hx'] + ce pex' - ce pey' - Cc dP
    M[E, E] = Ca
    for col in range(n):
        M[E, col] += Cb * ((1.0 / kx) * Dbx * M[HY, col]
                           - (1.0 / ky) * Dby * M[HX, col])
        M[E, col] += ce * M[PEX, col] - ce * M[PEY, col]
        M[E, col] -= Cc * M[P, col]
    M[E, P] += Cc
    if not with_pole:
        keep = [E, HX, HY, PEX, PEY, PHX, PHY]
        M = M[np.ix_(keep, keep)]
    return M


def scan_1d(mat, layers, dt, with_pole=True, with_cpml=True, nk=481):
    mc = mat.ade(dt)
    kds = np.linspace(1e-4, np.pi, nk)
    worst = (0.0, None, None)
    for li, layer in enumerate(layers if with_cpml else [None]):
        for kd in kds:
            zk = np.exp(1j * kd)
            M = one_step_1d(zk, layer, mc, dt, with_pole, with_cpml)
            r = float(np.abs(np.linalg.eigvals(M)).max())
            if r > worst[0]:
                worst = (r, li, kd)
    return worst


def scan_2d(mat, layers, dt, with_pole=True, nk=25):
    """Corner scan: all (layer_x, layer_y) pairs; nk x nk k-grid."""
    mc = mat.ade(dt)
    kds = np.linspace(1e-4, np.pi, nk)
    worst = (0.0, None, None)
    zs = np.exp(1j * kds)
    for lx in range(len(layers)):
        for ly in range(len(layers)):
            for zx in zs:
                for zy in zs:
                    M = one_step_2d(zx, zy, layers[lx], layers[ly], mc, dt,
                                    with_pole)
                    r = float(np.abs(np.linalg.eigvals(M)).max())
                    if r > worst[0]:
                        worst = (r, (lx, ly), (zx, zy))
    return worst


def main():
    dt = fixture_dt()
    print(f"dt = {dt:.6e} s  (dx={DX}, c0*dt/dx = {C0 * dt / DX:.4f})")
    check_profile_transcription(dt)

    results = {}
    for alpha_label, alpha_max in (("shipped-alpha", ALPHA_SHIPPED),
                                   ("rule-alpha", ALPHA_RULE)):
        for n_layers in (8, 12):
            _, kap, _, b, c = profile(n_layers, dt, DX, alpha_max)
            layers = list(zip(b, c, kap))
            tag = f"{alpha_label}/L{n_layers}"

            # Controls (shipped alpha, 8 layers only — declared)
            if alpha_label == "shipped-alpha" and n_layers == 8:
                r, _, _ = scan_1d(MAT_C1, layers, dt, with_pole=True,
                                  with_cpml=False)
                results["control pole-only (no CPML)"] = r
                r, li, kd = scan_1d(MAT_C1, layers, dt, with_pole=False,
                                    with_cpml=True)
                results["control CPML-only (no pole, eps=4)"] = r

            for mat in (MAT_C1, MAT_C3, MAT_C4):
                r, li, kd = scan_1d(mat, layers, dt)
                results[f"1D face {tag} {mat.label}"] = r
                print(f"[1D {tag}] {mat.label}: max|lambda| = {r:.9f} "
                      f"(layer {li}, k*dx = {kd:.3f})")

            r2, lp, _ = scan_2d(MAT_C1, layers, dt)
            results[f"2D corner {tag} {MAT_C1.label}"] = r2
            print(f"[2D {tag}] {MAT_C1.label}: max|lambda| = {r2:.9f} "
                  f"(layer pair {lp})")

    print()
    print("=== summary ===")
    for k, v in results.items():
        print(f"  {k}: {v:.9f}")

    c1 = max(v for k, v in results.items()
             if "C1" in k and "shipped-alpha" in k)
    ctrl = max(results["control pole-only (no CPML)"],
               results["control CPML-only (no pole, eps=4)"])
    print()
    if c1 > 1 + 1e-6 and ctrl <= 1 + 1e-9:
        print(f"H1 CONFIRMED: pole+CPML max|lambda| = {c1:.9f} > 1+1e-6, "
              f"controls max = {ctrl:.12f} <= 1+1e-9")
    elif c1 <= 1 + 1e-6:
        print(f"FALSIFIER F1 FIRED: pole+CPML max|lambda| = {c1:.9f} "
              f"<= 1+1e-6 — H1 (per-cell composed instability) falsified")
    else:
        print(f"INCONCLUSIVE: controls not clean (max {ctrl:.12f})")


if __name__ == "__main__":
    main()
