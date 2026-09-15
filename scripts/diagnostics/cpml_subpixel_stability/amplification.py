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

    H^{n+1/2}  = H^{n-1/2} + (dt/mu0) * dEz/dx
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
            "driver_commit": _git("rev-parse", "HEAD")}


def _rho_fn():
    """The ONE copy of the amplification model lives in the committed gate.

    ``tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py``
    owns ``amplification_rho``; this driver sweeps it. Keeping a second
    hand-maintained copy here is how the repo's grep map drifted, so there
    isn't one.
    """
    import importlib
    mod = importlib.import_module(
        "tests.unit.boundaries.test_cpml_subpixel_coefficient_consistency")
    return mod.amplification_rho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prov = _assert_rfx_is_this_repo()
    amplification_rho = _rho_fn()

    # The failing cells of the FDTD rig: a 10-layer pad, dx = 0.1 um, the
    # rig's own CFL dt, eps_a = the Kottke interface value at the guide wall,
    # eps_b = the staircase value the pad replication left beside it.
    from rfx.boundaries.cpml import _cpml_profile
    n_layers = 10
    dx = 1.0e-7
    courant = 0.5
    dt = courant * dx / C0
    prof = _cpml_profile(n_layers, dt, dx)
    c_out = float(np.asarray(prof.c, dtype=float)[0])
    b_out = float(np.asarray(prof.b, dtype=float)[0])
    kappa_max = float(np.asarray(prof.kappa, dtype=float).max())

    n = 24  # pad cells + interior tail

    EPS_A_FAIL = 6.5   # Kottke half-cell carried into the pad by the fix
    EPS_B_FAIL = 1.0   # what materials.eps_r reads at the same cell

    def rho(eps_a, eps_b, dt_=None, absorber=True):
        return amplification_rho(n_layers, dx, dt if dt_ is None else dt_,
                                 eps_a, eps_b, n=n, absorber=absorber)

    # Comparator first: the model has to be right before its verdict is
    # readable. A lossless vacuum slice with NO absorber must sit on the unit
    # circle, and a consistent CPML must sit on or inside it. The leapfrog
    # sign error that made every case report rho = 1.3-2.6 was caught here.
    rho_vac_nopml = rho(1.0, 1.0, absorber=False)
    rho_vac_cpml = rho(1.0, 1.0)
    if not (abs(rho_vac_nopml - 1.0) < 1e-6 and rho_vac_cpml <= 1.0 + 1e-6):
        raise SystemExit(
            "amplification model SELF-CHECK FAILED — a lossless vacuum slice "
            f"must give rho = 1 (got {rho_vac_nopml}) and a consistent CPML "
            f"must give rho <= 1 (got {rho_vac_cpml}). The model is wrong; its "
            "verdict on the defect is not readable.")

    rows = []

    def add(label, eps_a, eps_b, **kw):
        r = rho(eps_a, eps_b)
        rows.append({"case": label, "rho": r, "unstable": bool(r > 1 + 1e-6),
                     "eps_a": float(eps_a), "eps_b": float(eps_b), **kw})
        return r

    # --- the defect and the fix, at the failing cell ---
    rho_old = add("today: eps_a from aniso_eps, eps_b from materials.eps_r",
                  EPS_A_FAIL, EPS_B_FAIL)
    rho_new = add("fixed: eps_b = eps_a", EPS_A_FAIL, EPS_A_FAIL)

    # --- controls that must stay stable ---
    add("vacuum pad, consistent", 1.0, 1.0)
    add("eps 12 pad, consistent (= subpixel OFF row)", 12.0, 12.0)
    add("today's cv01: eps_a = 1, eps_b = 12 (under-damped, stable)", 1.0, 12.0)

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
        ratio_scan.append({
            "eps_a": 6.5, "eps_b": eb_val, "eps_a_over_eps_b": 6.5 / eb_val,
            "K_eff_norm_outermost": 1.0 / 6.5 + c_out / eb_val,
            "rho": rho(6.5, eb_val)})

    # Consistent (eps_a == eps_b) across a wide absolute range: if H2 were the
    # mechanism, a high absolute permittivity in the graded region would break
    # stability on its own. It must not.
    scale_scan = []
    for eps in (1.0, 2.0, 4.0, 6.5, 12.0, 30.0, 80.0):
        scale_scan.append({"eps_consistent": eps, "rho": rho(eps, eps)})

    courant_scan = []
    for cf in (0.2, 0.35, 0.5, 0.7):
        dt_c = cf * dx / C0
        courant_scan.append({
            "courant": cf,
            "rho_defect": rho(EPS_A_FAIL, EPS_B_FAIL, dt_=dt_c),
            "rho_fixed": rho(EPS_A_FAIL, EPS_A_FAIL, dt_=dt_c),
        })

    rec = {
        "provenance": prov,
        "model_owner": ("tests/unit/boundaries/"
                        "test_cpml_subpixel_coefficient_consistency.py"
                        "::amplification_rho"),
        "model": {"n_cells": n, "n_cpml_layers": n_layers, "dx": dx,
                  "dt": dt, "courant": courant, "kappa_max": kappa_max,
                  "c_outermost": c_out, "b_outermost": b_out},
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
