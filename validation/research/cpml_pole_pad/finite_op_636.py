"""#636 M1b/M1c — exact finite one-step operators (grading + interfaces).

Declared in docs/design_notes/i636_cpml_pole_pad_predeclaration.md BEFORE
first run (M1b/M1c section). M1's frozen-coefficient falsifier F1 fired,
so this stage tests whether the instability is reachable by the FINITE
1D (M1b) or 2D TM corner (M1c) operator, with the shipped graded CPML
profile, the interior/pad interfaces, and the PEC outer wall included.

Variant A = pole mask interior-only (shipped). Variant B = pole mask
extended into the pads (what #627b tried). Statics (eps_inf) are extended
in BOTH variants, as shipped.

Predictions/falsifiers (verbatim from the note):
  M1b: B rho > 1+1e-6 and A <= 1+1e-9, else F1b fires -> M1c.
  M1c: B rho > 1+1e-6 with pad-localized eigenvector, A <= 1+1e-9,
       else F1c fires -> root cause rests on M2's empirical measurement.

Update order and signs transcribed from rfx/core/yee.py,
rfx/materials/lorentz.py, rfx/boundaries/cpml.py, rfx/simulation.py —
the same transcription eigen_scan_636.py asserts against the shipped
profile function.

Run:  .venv/bin/python validation/research/cpml_pole_pad/finite_op_636.py
"""

from __future__ import annotations

import numpy as np

EPS_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

DX = 1e-3
F0 = 3e9
W0 = 2 * np.pi * F0
FREQ_MAX = 2.5 * F0
N_PAD = 8
NX_INT = 45
NY_INT = 39

EPS_INF = 4.0
DELTA = W0 / 120.0
KAPPA_P = 3.0 * W0 ** 2


def fixture_dt() -> float:
    from rfx import Simulation
    sim = Simulation(freq_max=FREQ_MAX,
                     domain=(NX_INT * DX, NY_INT * DX, 12 * DX),
                     dx=DX, boundary="cpml", cpml_layers=N_PAD)
    return float(sim._build_grid().dt)


def shipped_profile(dt):
    from rfx.boundaries.cpml import _cpml_profile
    p = _cpml_profile(N_PAD, dt, DX)
    return (np.asarray(p.b, dtype=np.float64),
            np.asarray(p.c, dtype=np.float64),
            np.asarray(p.kappa, dtype=np.float64))


def per_node_cpml(n_nodes, dt):
    """Node-indexed (b, c, kappa) along one axis: lo pad ascending from the
    outer boundary (profile index 0 = outermost, as _extend/clip use), hi
    pad mirrored, interior no-op. Matches apply_cpml_e/h slicing:
    lo slice [:n] takes profile[0..n-1] (index 0 = outermost node), hi
    slice [-n:] uses the flipped profile (outermost node = last index)."""
    b_prof, c_prof, k_prof = per_node_cpml.profile
    # b = 0 outside the pads: the real psi buffers exist only in the pad
    # slices, so interior psi dofs must not persist (b=1 there would add
    # artificial defective unit eigenvalues to this reduced model).
    b = np.zeros(n_nodes)
    c = np.zeros(n_nodes)
    k = np.ones(n_nodes)
    b[:N_PAD] = b_prof
    c[:N_PAD] = c_prof
    k[:N_PAD] = k_prof
    b[-N_PAD:] = b_prof[::-1]
    c[-N_PAD:] = c_prof[::-1]
    k[-N_PAD:] = k_prof[::-1]
    return b, c, k


def ade_coeffs(dt):
    den = 1.0 + DELTA * dt
    a_p = (2.0 - W0 ** 2 * dt ** 2) / den
    b_p = -(1.0 - DELTA * dt) / den
    c_p = EPS_0 * KAPPA_P * dt ** 2 / den
    gamma = EPS_INF * EPS_0
    return a_p, b_p, c_p, dt / gamma, 1.0 / gamma  # a, b, c, Cb, Cc


# ---------------------------------------------------------------- M1b (1D)

def build_1d_step(dt, extend_pole: bool):
    """Dense one-step matrix for the 1D x-line.

    Nodes 0..nx-1; Ez at node i, Hy at i+1/2. PEC: Ez[0] = Ez[nx-1] = 0
    (apply_pec zeroes tangential E at both faces). eps_inf everywhere
    (statics extended); pole mask = interior nodes only (A) or all (B).
    """
    nx = NX_INT + 2 * N_PAD
    b_n, c_n, k_n = per_node_cpml(nx, dt)
    a_p, b_p, c_p, Cb, Cc = ade_coeffs(dt)
    ch = dt / MU_0
    ce = Cb

    pole = np.zeros(nx, dtype=bool)
    if extend_pole:
        pole[:] = True
    else:
        pole[N_PAD:N_PAD + NX_INT] = True

    # state layout: Ez[0:nx], Hy[nx:2nx], P[2nx:3nx], Pp[3nx:4nx],
    # psi_e[4nx:5nx], psi_h[5nx:6nx]  (psi stored full-length; zero
    # coefficients outside pads keep them inert exactly like the real
    # sliced buffers).
    n = 6 * nx
    E0, H0, P0, PP0, PE0, PH0 = (i * nx for i in range(6))
    M = np.zeros((n, n))

    def rows(base):
        return slice(base, base + nx)

    # Work on explicit per-row construction.
    # Df E at H-node i: (Ez[i+1] - Ez[i])/dx  (last node: shift_fwd pads 0)
    DfE = np.zeros((nx, nx))
    for i in range(nx):
        DfE[i, i] -= 1.0 / DX
        if i + 1 < nx:
            DfE[i, i + 1] += 1.0 / DX
    # Db H at E-node i: (Hy[i] - Hy[i-1])/dx  (first node: shift_bwd pads 0)
    DbH = np.zeros((nx, nx))
    for i in range(nx):
        DbH[i, i] += 1.0 / DX
        if i - 1 >= 0:
            DbH[i, i - 1] -= 1.0 / DX

    I = np.eye(nx)
    bd = np.diag(b_n)
    cd = np.diag(c_n)
    kinv = np.diag(1.0 / k_n)
    pm = np.diag(pole.astype(float))

    # psi_h' = b psi_h + c DfE Ez
    T_ph_ph = bd
    T_ph_e = cd @ DfE
    # Hy' = Hy + ch kinv DfE Ez + ch psi_h'
    T_h_h = I
    T_h_e = ch * (kinv @ DfE) + ch * T_ph_e
    T_h_ph = ch * T_ph_ph
    # P' = a P + b Pp + c_p*pole Ez ; ADE coefficients are masked arrays in
    # the shipped code (a=b=c=0 outside the mask), matching init_lorentz.
    T_p_p = a_p * pm
    T_p_pp = b_p * pm
    T_p_e = c_p * pm
    # psi_e' = b psi_e + c DbH Hy'
    T_pe_pe = bd
    T_pe_h = cd @ DbH @ T_h_h
    T_pe_e = cd @ DbH @ T_h_e
    T_pe_ph = cd @ DbH @ T_h_ph
    # Ez' = Ez + Cb kinv DbH Hy' + ce psi_e' - Cc (P' - P)
    T_e_e = I + Cb * (kinv @ DbH @ T_h_e) + ce * T_pe_e - Cc * T_p_e
    T_e_h = Cb * (kinv @ DbH @ T_h_h) + ce * T_pe_h
    T_e_ph = Cb * (kinv @ DbH @ T_h_ph) + ce * T_pe_ph
    T_e_pe = ce * T_pe_pe
    T_e_p = -Cc * (T_p_p - I)
    T_e_pp = -Cc * T_p_pp

    # PEC: zero Ez rows at the outer nodes after the update.
    pec = np.ones(nx)
    pec[0] = 0.0
    pec[-1] = 0.0
    pecd = np.diag(pec)
    for blk in (T_e_e, T_e_h, T_e_ph, T_e_pe, T_e_p, T_e_pp):
        blk[:] = pecd @ blk

    M[rows(E0), rows(E0)] = T_e_e
    M[rows(E0), rows(H0)] = T_e_h
    M[rows(E0), rows(P0)] = T_e_p
    M[rows(E0), rows(PP0)] = T_e_pp
    M[rows(E0), rows(PE0)] = T_e_pe
    M[rows(E0), rows(PH0)] = T_e_ph
    M[rows(H0), rows(H0)] = T_h_h
    M[rows(H0), rows(E0)] = T_h_e
    M[rows(H0), rows(PH0)] = T_h_ph
    M[rows(P0), rows(P0)] = T_p_p
    M[rows(P0), rows(PP0)] = T_p_pp
    M[rows(P0), rows(E0)] = T_p_e
    M[rows(PP0), rows(P0)] = I
    M[rows(PE0), rows(PE0)] = T_pe_pe
    M[rows(PE0), rows(E0)] = T_pe_e
    M[rows(PE0), rows(H0)] = T_pe_h
    M[rows(PE0), rows(PH0)] = T_pe_ph
    M[rows(PH0), rows(PH0)] = T_ph_ph
    M[rows(PH0), rows(E0)] = T_ph_e
    return M


def m1b(dt):
    print("=== M1b: finite 1D operator ===")
    out = {}
    for label, ext in (("A shipped (interior-only pole)", False),
                       ("B extended (pole into pads)", True)):
        M = build_1d_step(dt, ext)
        ev = np.linalg.eigvals(M)
        rho = float(np.abs(ev).max())
        out[label] = rho
        print(f"  {label}: rho = {rho:.12f} (rho-1 = {rho-1:+.3e})")
    return out


# ---------------------------------------------------------------- M1c (2D)

def build_2d_masks(extend_pole: bool):
    nx = NX_INT + 2 * N_PAD
    ny = NY_INT + 2 * N_PAD
    pole = np.zeros((nx, ny), dtype=bool)
    pole[N_PAD:N_PAD + NX_INT, N_PAD:N_PAD + NY_INT] = True
    if extend_pole:
        # statics-style replication: x pads copy the interior-edge column
        # (all True there), then y pads copy the row (corners included) —
        # for a full-extent slab this is simply all-True.
        pole[:, :] = True
    return pole


def m1c(dt, n_eigs=6):
    import scipy.sparse.linalg as spla

    print("=== M1c: finite 2D TM corner operator ===")
    nx = NX_INT + 2 * N_PAD
    ny = NY_INT + 2 * N_PAD
    bx, cx, kx = per_node_cpml(nx, dt)
    by, cy, ky = per_node_cpml(ny, dt)
    a_p, b_p, c_p, Cb, Cc = ade_coeffs(dt)
    ch = dt / MU_0
    ce = Cb
    kxi = (1.0 / kx)[:, None]
    kyi = (1.0 / ky)[None, :]
    bxc = bx[:, None]
    cxc = cx[:, None]
    byc = by[None, :]
    cyc = cy[None, :]

    def dfx(f):
        out = np.zeros_like(f)
        out[:-1, :] = (f[1:, :] - f[:-1, :]) / DX
        out[-1, :] = (0.0 - f[-1, :]) / DX
        return out

    def dbx(f):
        out = np.zeros_like(f)
        out[1:, :] = (f[1:, :] - f[:-1, :]) / DX
        out[0, :] = (f[0, :] - 0.0) / DX
        return out

    def dfy(f):
        out = np.zeros_like(f)
        out[:, :-1] = (f[:, 1:] - f[:, :-1]) / DX
        out[:, -1] = (0.0 - f[:, -1]) / DX
        return out

    def dby(f):
        out = np.zeros_like(f)
        out[:, 1:] = (f[:, 1:] - f[:, :-1]) / DX
        out[:, 0] = (f[:, 0] - 0.0) / DX
        return out

    shp = (nx, ny)
    nfield = nx * ny
    n_state = 9 * nfield

    results = {}
    for label, ext in (("A shipped (interior-only pole)", False),
                       ("B extended (pole into pads)", True)):
        pole = build_2d_masks(ext).astype(float)
        ap = a_p * pole
        bp = b_p * pole
        cp = c_p * pole

        def step(v):
            (ez, hx, hy, p, pp, pex, pey, phx, phy) = (
                v[i * nfield:(i + 1) * nfield].reshape(shp) for i in range(9))
            # H updates
            dfe_x = dfx(ez)
            dfe_y = dfy(ez)
            phy_n = bxc * phy + cxc * dfe_x
            hy_n = hy + ch * kxi * dfe_x + ch * phy_n
            phx_n = byc * phx + cyc * dfe_y
            hx_n = hx - ch * kyi * dfe_y - ch * phx_n
            # P update
            p_n = ap * p + bp * pp + cp * ez
            # E update
            dbh_y = dbx(hy_n)
            dbh_x = dby(hx_n)
            pex_n = bxc * pex + cxc * dbh_y
            pey_n = byc * pey + cyc * dbh_x
            ez_n = (ez + Cb * (kxi * dbh_y - kyi * dbh_x)
                    + ce * pex_n - ce * pey_n - Cc * (p_n - p))
            # PEC outer ring: Ez tangential at all four faces -> zero
            ez_n[0, :] = 0.0
            ez_n[-1, :] = 0.0
            ez_n[:, 0] = 0.0
            ez_n[:, -1] = 0.0
            return np.concatenate([f.ravel() for f in
                                   (ez_n, hx_n, hy_n, p_n, p, pex_n, pey_n,
                                    phx_n, phy_n)])

        op = spla.LinearOperator((n_state, n_state), matvec=step,
                                 dtype=np.float64)
        rng = np.random.default_rng(636)
        v0 = rng.standard_normal(n_state)
        vals = spla.eigs(op, k=n_eigs, which="LM", v0=v0,
                         maxiter=200000, tol=1e-10,
                         return_eigenvectors=False)
        rho = float(np.abs(vals).max())
        results[label] = (rho, np.sort(np.abs(vals))[::-1])
        print(f"  {label}: rho = {rho:.12f} (rho-1 = {rho-1:+.3e}); "
              f"|top eigs| = {[f'{x:.9f}' for x in results[label][1]]}")

    # localization witness for B if unstable: dominant eigenvector
    if results["B extended (pole into pads)"][0] > 1 + 1e-6:
        pole = build_2d_masks(True).astype(float)
        ap = a_p * pole
        bp = b_p * pole
        cp = c_p * pole
        # power iteration to get the mode shape
        rng = np.random.default_rng(1636)
        v = rng.standard_normal(n_state)
        v /= np.linalg.norm(v)
        growth = None
        for it in range(6000):
            def step_b(v):
                (ez, hx, hy, p, pp, pex, pey, phx, phy) = (
                    v[i * nfield:(i + 1) * nfield].reshape(shp)
                    for i in range(9))
                dfe_x = dfx(ez)
                dfe_y = dfy(ez)
                phy_n = bxc * phy + cxc * dfe_x
                hy_n = hy + ch * kxi * dfe_x + ch * phy_n
                phx_n = byc * phx + cyc * dfe_y
                hx_n = hx - ch * kyi * dfe_y - ch * phx_n
                p_n = ap * p + bp * pp + cp * ez
                dbh_y = dbx(hy_n)
                dbh_x = dby(hx_n)
                pex_n = bxc * pex + cxc * dbh_y
                pey_n = byc * pey + cyc * dbh_x
                ez_n = (ez + Cb * (kxi * dbh_y - kyi * dbh_x)
                        + ce * pex_n - ce * pey_n - Cc * (p_n - p))
                ez_n[0, :] = 0.0
                ez_n[-1, :] = 0.0
                ez_n[:, 0] = 0.0
                ez_n[:, -1] = 0.0
                return np.concatenate([f.ravel() for f in
                                       (ez_n, hx_n, hy_n, p_n, p, pex_n,
                                        pey_n, phx_n, phy_n)])
            v2 = step_b(v)
            growth = np.linalg.norm(v2)
            v = v2 / growth
        ez_mode = np.abs(v[:nfield].reshape(shp))
        pad_x = np.zeros(shp, dtype=bool)
        pad_x[:N_PAD, :] = True
        pad_x[-N_PAD:, :] = True
        pad_y = np.zeros(shp, dtype=bool)
        pad_y[:, :N_PAD] = True
        pad_y[:, -N_PAD:] = True
        corner = pad_x & pad_y
        face = (pad_x | pad_y) & ~corner
        interior = ~(pad_x | pad_y)
        tot = ez_mode.sum() + 1e-300
        print(f"  B dominant-mode |Ez| mass: interior "
              f"{ez_mode[interior].sum()/tot:.3f}, face "
              f"{ez_mode[face].sum()/tot:.3f}, corner "
              f"{ez_mode[corner].sum()/tot:.3f}; per-step growth "
              f"{growth:.9f}")
        imax = np.unravel_index(np.argmax(ez_mode), shp)
        print(f"  B dominant-mode |Ez| peak at (i,j) = {imax} "
              f"(nx={nx}, ny={ny}, pad={N_PAD})")
    return results


def main():
    dt = fixture_dt()
    print(f"dt = {dt:.6e}")
    per_node_cpml.profile = shipped_profile(dt)
    r1 = m1b(dt)
    a = r1["A shipped (interior-only pole)"]
    b = r1["B extended (pole into pads)"]
    if b > 1 + 1e-6 and a <= 1 + 1e-9:
        print("M1b: prediction CONFIRMED — 1D interface/grading instability")
    else:
        print("M1b: FALSIFIER F1b FIRED (B <= 1+1e-6 or A not clean) — "
              "proceeding to M1c as declared")
        r2 = m1c(dt)
        a2 = r2["A shipped (interior-only pole)"][0]
        b2 = r2["B extended (pole into pads)"][0]
        if b2 > 1 + 1e-6 and a2 <= 1 + 1e-9:
            print("M1c: prediction CONFIRMED — 2D corner instability")
        else:
            print("M1c: FALSIFIER F1c FIRED — linear mechanism not "
                  "reachable below 3D by these reduced models; root cause "
                  "rests on M2's empirical measurement")


if __name__ == "__main__":
    main()
