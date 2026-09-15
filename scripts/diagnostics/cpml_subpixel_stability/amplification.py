"""#1043 Stage A, gate G-C — the amplification factor of one CPML cell.

Pre-declaration: ``docs/design_notes/issue1043_cpml_subpixel_coefficient_predeclaration.md``
section 3, gate G-C, frozen before this ran.

WHY AN EIGENVALUE AND NOT ANOTHER FDTD RUN
------------------------------------------
G-A shows a negative instantaneous curl coefficient and G-B shows the field
breaking at exactly those cells, but neither says the update is *unstable* —
only that one coefficient changed sign. This builds the actual one-cell
leapfrog + psi amplification matrix and reads its spectral radius, which is
what "unstable" means.

THE MODEL
---------
A 1-D CPML slice, ``n`` cells, Ez / Hy / psi_ez, with the rfx assembly:

    H^{n+1/2}  = H^{n-1/2} - (dt/mu0) * dEz/dx
    psi^{n+1}  = b*psi^n + c*dHy/dx
    E^{n+1}    = E^n + (dt/(eps_a*eps0)) * dHy/dx + (dt/(eps_b*eps0)) * psi^{n+1}

``eps_a`` is what ``update_e_aniso`` uses (``aniso_eps``); ``eps_b`` is what
``apply_cpml_e`` uses (``materials.eps_r``). rfx today has them free to differ;
the fix makes ``eps_b = eps_a``.

Note the psi ORDER: ``apply_cpml_e`` writes the new psi and uses it in the same
step (cpml.py ``new_psi_* = b*psi + c*curl`` then ``+= ce*new_psi``), so the
matrix below does the same.

Assembled as one linear operator over the state ``(E, H, psi)`` and its
spectral radius taken. H1 predicts ``rho`` depends on ``eps_a/eps_b`` and ``c``
alone; H2 predicts it depends on ``dt/dx`` and the ABSOLUTE permittivity.
Both are swept, which is what discriminates them.

Usage::

    PYTHONPATH=$(git rev-parse --show-toplevel) python3 \
        scripts/diagnostics/cpml_subpixel_stability/amplification.py \
        --output scripts/diagnostics/_artifacts/cpml_subpixel_stability/amplification.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]

EPS_0 = 8.8541878128e-12
MU_0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _assert_rfx_is_this_repo() -> dict:
    import rfx
    f = pathlib.Path(rfx.__file__).resolve()
    try:
        f.relative_to(REPO)
    except ValueError:
        raise SystemExit(f"rfx provenance check FAILED: {f} is not under {REPO}")
    return {"rfx_file": str(f), "repo_root": str(REPO),
            "commit": _git("rev-parse", "HEAD")}


def amplification_matrix(n: int, dx: float, dt: float,
                         eps_a: np.ndarray, eps_b: np.ndarray,
                         b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """One-step operator on ``(Ez[n], Hy[n], psi[n])``, PEC-terminated."""
    m = 3 * n
    A = np.eye(m)
    iE = slice(0, n)
    iH = slice(n, 2 * n)
    iP = slice(2 * n, 3 * n)

    # dEz/dx at H nodes: (E[i+1] - E[i]) / dx, last row 0 (PEC)
    DE = np.zeros((n, n))
    for i in range(n - 1):
        DE[i, i + 1] = 1.0 / dx
        DE[i, i] = -1.0 / dx
    # dHy/dx at E nodes: (H[i] - H[i-1]) / dx, first row H[0]/dx
    DH = np.zeros((n, n))
    for i in range(n):
        DH[i, i] = 1.0 / dx
        if i > 0:
            DH[i, i - 1] = -1.0 / dx

    # H^{n+1/2} = H^{n-1/2} + (dt/mu0) * dEz/dx.
    # rfx writes ``hy -= (dt/mu)*curl_y`` with ``curl_y = dEx/dz - dEz/dx``
    # (yee.py update_h); in this 1-D Ez/Hy slice that is ``+ (dt/mu)*dEz/dx``.
    # Taking the minus sign literally turns the leapfrog into positive
    # feedback and makes EVERY case report rho > 1, vacuum included -- which
    # is how this sign was caught.
    H_new = np.zeros((n, m))
    H_new[:, iH] = np.eye(n)
    H_new[:, iE] = (dt / MU_0) * DE

    # psi^{n+1} = b*psi^n + c*(DH @ H^{n+1/2})
    P_new = np.zeros((n, m))
    P_new[:, iP] = np.diag(b)
    P_new += np.diag(c) @ DH @ H_new

    # E^{n+1} = E^n + (dt/(eps_a*eps0))*(DH @ H^{n+1/2})
    #                + (dt/(eps_b*eps0))*psi^{n+1}
    E_new = np.zeros((n, m))
    E_new[:, iE] = np.eye(n)
    E_new += np.diag(dt / (eps_a * EPS_0)) @ DH @ H_new
    E_new += np.diag(dt / (eps_b * EPS_0)) @ P_new

    A[iE, :] = E_new
    A[iH, :] = H_new
    A[iP, :] = P_new
    return A


def rho(*args, **kw) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(amplification_matrix(*args, **kw)))))


def cpml_profile(n_layers: int, dt: float, dx: float, order: int = 3,
                 R: float = 1e-15):
    """Re-derives rfx's own profile (cpml.py:_cpml_profile) in plain numpy."""
    eta = float(np.sqrt(MU_0 / EPS_0))
    d = n_layers * dx
    sigma_max = -float(np.log(R)) * (order + 1) / (2.0 * eta * d)
    rho_p = 1.0 - np.arange(n_layers) / max(n_layers - 1, 1)
    sigma = sigma_max * rho_p ** order
    kappa = np.ones(n_layers)
    alpha = 0.05 * (1.0 - rho_p)
    denom = sigma * kappa + kappa ** 2 * alpha
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS_0)
    c = np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
    return sigma, kappa, alpha, b, c


def _assert_profile_matches_rfx(n_layers, dt, dx) -> dict:
    """The numpy profile above must equal rfx's, or nothing here is about rfx."""
    from rfx.boundaries.cpml import _cpml_profile
    p = _cpml_profile(n_layers, dt, dx)
    _s, _k, _a, b, c = cpml_profile(n_layers, dt, dx)
    db = float(np.max(np.abs(np.asarray(p.b, dtype=float) - b)))
    dc = float(np.max(np.abs(np.asarray(p.c, dtype=float) - c)))
    if db > 1e-6 or dc > 1e-6:
        raise SystemExit(f"profile mismatch vs rfx: db={db} dc={dc}")
    return {"max_abs_db_vs_rfx": db, "max_abs_dc_vs_rfx": dc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prov = _assert_rfx_is_this_repo()

    # The failing cells of the FDTD rig: a 10-layer pad, dx = 0.1 um, the
    # rig's own CFL dt, eps_a = the Kottke interface value at the guide wall,
    # eps_b = the staircase value the pad replication left beside it.
    n_layers = 10
    dx = 1.0e-7
    courant = 0.5
    dt = courant * dx / C0
    _s, kap, _al, b, c = cpml_profile(n_layers, dt, dx)
    chk = _assert_profile_matches_rfx(n_layers, dt, dx)

    n = 24  # pad cells + interior tail
    bb = np.zeros(n)
    cc = np.zeros(n)
    bb[:n_layers] = b
    cc[:n_layers] = c
    bb[n_layers:] = 1.0

    EPS_A_FAIL = 6.5   # Kottke half-cell carried into the pad by the fix
    EPS_B_FAIL = 1.0   # what materials.eps_r reads at the same cell

    def homog(val):
        return np.full(n, float(val))

    # Comparator first: the model has to be right before its verdict is
    # readable. A lossless vacuum slice with NO absorber must sit on the unit
    # circle, and a consistent CPML must sit on or inside it. The leapfrog
    # sign error that made every case report rho = 1.3-2.6 was caught here.
    no_pml_b = np.ones(n)
    no_pml_c = np.zeros(n)
    rho_vac_nopml = rho(n, dx, dt, homog(1.0), homog(1.0), no_pml_b, no_pml_c)
    rho_vac_cpml = rho(n, dx, dt, homog(1.0), homog(1.0), bb, cc)
    if not (abs(rho_vac_nopml - 1.0) < 1e-6 and rho_vac_cpml <= 1.0 + 1e-6):
        raise SystemExit(
            "amplification model SELF-CHECK FAILED — a lossless vacuum slice "
            f"must give rho = 1 (got {rho_vac_nopml}) and a consistent CPML "
            f"must give rho <= 1 (got {rho_vac_cpml}). The model is wrong; its "
            "verdict on the defect is not readable.")

    rows = []

    def add(label, eps_a, eps_b, **kw):
        r = rho(n, dx, dt, eps_a, eps_b, bb, cc)
        rows.append({"case": label, "rho": r, "unstable": bool(r > 1 + 1e-6),
                     "eps_a": float(eps_a[0]), "eps_b": float(eps_b[0]), **kw})
        return r

    # --- the defect and the fix, at the failing cell ---
    rho_old = add("today: eps_a from aniso_eps, eps_b from materials.eps_r",
                  homog(EPS_A_FAIL), homog(EPS_B_FAIL))
    rho_new = add("fixed: eps_b = eps_a", homog(EPS_A_FAIL), homog(EPS_A_FAIL))

    # --- controls that must stay stable ---
    add("vacuum pad, consistent", homog(1.0), homog(1.0))
    add("eps 12 pad, consistent (= subpixel OFF row)", homog(12.0), homog(12.0))
    add("today's cv01: eps_a = 1, eps_b = 12 (under-damped, stable)",
        homog(1.0), homog(12.0))

    # --- H1 vs H2 discriminator -------------------------------------------
    # eps_b enters ONLY the psi coefficient; it is not a property of the
    # medium the wave travels in. So:
    #   H1 (epsilon disagreement) -> rho crosses 1 as eps_b is swept past
    #      eps_a * |c|, with eps_a and dt/dx held.
    #   H2 (local CFL / stability bound) -> rho is FLAT in eps_b, because a
    #      CFL bound cannot see a coefficient the wave speed does not contain.
    # Predicted crossing: |c_outermost| * eps_a / eps_b = 1.
    ratio_scan = []
    for eb_val in (1.0, 2.0, 4.0, 5.0, 6.0, 6.294, 6.35, 6.4, 6.45, 6.49,
                   6.5, 8.0, 12.0, 20.0):
        ea = homog(6.5)
        eb = homog(eb_val)
        ratio_scan.append({
            "eps_a": 6.5, "eps_b": eb_val, "eps_a_over_eps_b": 6.5 / eb_val,
            "K_eff_norm_outermost": 1.0 / 6.5 + float(cc[0]) / eb_val,
            "rho": rho(n, dx, dt, ea, eb, bb, cc)})

    # Consistent (eps_a == eps_b) across a wide absolute range: if H2 were the
    # mechanism, a high absolute permittivity in the graded region would break
    # stability on its own. It must not.
    scale_scan = []
    for eps in (1.0, 2.0, 4.0, 6.5, 12.0, 30.0, 80.0):
        e = homog(eps)
        scale_scan.append({"eps_consistent": eps,
                           "rho": rho(n, dx, dt, e, e, bb, cc)})

    courant_scan = []
    for cf in (0.2, 0.35, 0.5, 0.7):
        dt_c = cf * dx / C0
        _s2, _k2, _a2, b2, c2 = cpml_profile(n_layers, dt_c, dx)
        bb2 = np.zeros(n); cc2 = np.zeros(n)
        bb2[:n_layers] = b2; cc2[:n_layers] = c2; bb2[n_layers:] = 1.0
        courant_scan.append({
            "courant": cf,
            "rho_defect": rho(n, dx, dt_c, homog(EPS_A_FAIL),
                              homog(EPS_B_FAIL), bb2, cc2),
            "rho_fixed": rho(n, dx, dt_c, homog(EPS_A_FAIL),
                             homog(EPS_A_FAIL), bb2, cc2),
        })

    rec = {
        "provenance": prov,
        "profile_check_vs_rfx": chk,
        "model": {"n_cells": n, "n_cpml_layers": n_layers, "dx": dx,
                  "dt": dt, "courant": courant, "kappa_max": float(kap.max()),
                  "c_outermost": float(cc[0]), "b_outermost": float(bb[0])},
        "self_check": {"rho_vacuum_no_pml": rho_vac_nopml,
                       "rho_vacuum_consistent_cpml": rho_vac_cpml},
        "cases": rows,
        "ratio_scan": ratio_scan,
        "absolute_scale_scan": scale_scan,
        "courant_scan": courant_scan,
        "gates": {"G_C": {
            "rho_old": rho_old, "rho_new": rho_new,
            "r_C": max(0.0, 1.0 + 1e-6 - rho_old) + max(0.0, rho_new - (1.0 + 1e-6)),
            "verdict": ("H1 SUPPORTED" if rho_old > 1 + 1e-6 >= rho_new
                        else "NON-CLOSING"),
        }},
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    for r in rows:
        print(f"  rho={r['rho']:.6f} {'UNSTABLE' if r['unstable'] else 'stable  '}"
              f"  {r['case']}")
    print(f"self-check: rho(vacuum, no PML) = {rho_vac_nopml:.9f}, "
          f"rho(vacuum, consistent CPML) = {rho_vac_cpml:.9f}")
    print("eps_b scan at eps_a = 6.5 (eps_b, K_eff_norm, rho):",
          [(r["eps_b"], round(r["K_eff_norm_outermost"], 4), round(r["rho"], 6))
           for r in ratio_scan])
    print("consistent-eps scan (eps, rho):",
          [(r["eps_consistent"], round(r["rho"], 6)) for r in scale_scan])
    print("courant scan:",
          [(r["courant"], round(r["rho_defect"], 6), round(r["rho_fixed"], 6))
           for r in courant_scan])
    print(json.dumps(rec["gates"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
