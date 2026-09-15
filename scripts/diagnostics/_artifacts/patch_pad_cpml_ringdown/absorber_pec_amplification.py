#!/usr/bin/env python3
"""H3: does rfx's own CPML slice amplify at FEW layers when a conductor terminates it?

No FDTD.  The thin-absorber ladder says the governing variable is the absorber's layer COUNT
(4 and 6 cells grow, 8 and 16 settle, at fixed dx and fixed geometry).  ``_cpml_profile`` scales
``sigma_max`` as ``1/d`` with ``d = n_layers*dx`` so the total optical depth is nominally held,
but ``alpha = 0.05*(1 - rho)`` is an ABSOLUTE CFS value that does not scale with the layer
count at all.  This asks whether that combination, with a conductor terminating the slice, puts
an eigenvalue outside the unit circle at 4-6 layers and not at 8-16.

THE MODEL IS NOT NEW.  It is the one that owns the #1047 analysis --
``tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py::amplification_rho`` --
extended by ONE thing: a PEC node, imposed by zeroing that node's E update row so ``E[k] = 0``
for all time.  The extension is checked against the owned model before it is read: with
``pec_node=None`` it must reproduce ``amplification_rho`` to 1e-12 on every case tried.

COMPARATOR FIRST, twice, because a model that cannot do the trivial cases cannot be read on the
interesting one:
  * a lossless slice with NO absorber must sit exactly on the unit circle;
  * the same slice with a PEC end must ALSO sit on the unit circle -- a PEC-terminated lossless
    cavity is marginally stable, and a model that amplifies there is wrong about PEC, not about
    absorbers.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "unit", "boundaries"))

from test_cpml_subpixel_coefficient_consistency import (  # noqa: E402
    EPS_0, MU_0, _cpml_profile, amplification_rho,
)

H = 0.787e-3
EPS_R = 3.38


def rho_with_pec(n_layers, dx, dt, eps_a, eps_b, n=24, absorber=True, pec_node=None):
    """``amplification_rho`` plus an optional PEC node.  Identical to it when pec_node is None."""
    b = np.ones(n)
    c = np.zeros(n)
    if absorber:
        p = _cpml_profile(n_layers, dt, dx)
        b[:n_layers] = np.asarray(p.b, dtype=float)
        c[:n_layers] = np.asarray(p.c, dtype=float)
    ea = np.full(n, float(eps_a))
    eb = np.full(n, float(eps_b))

    DE = np.zeros((n, n))
    for i in range(n - 1):
        DE[i, i + 1] = 1.0 / dx
        DE[i, i] = -1.0 / dx
    DH = np.zeros((n, n))
    for i in range(n):
        DH[i, i] = 1.0 / dx
        if i > 0:
            DH[i, i - 1] = -1.0 / dx

    m = 3 * n
    iE, iH, iP = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
    H_new = np.zeros((n, m))
    H_new[:, iH] = np.eye(n)
    H_new[:, iE] = (dt / MU_0) * DE
    P_new = np.zeros((n, m))
    P_new[:, iP] = np.diag(b)
    P_new += np.diag(c) @ DH @ H_new
    E_new = np.zeros((n, m))
    E_new[:, iE] = np.eye(n)
    E_new += np.diag(dt / (ea * EPS_0)) @ DH @ H_new
    E_new += np.diag(dt / (eb * EPS_0)) @ P_new
    if pec_node is not None:
        E_new[int(pec_node), :] = 0.0          # E at the conductor is zero for all time
    Aop = np.zeros((m, m))
    Aop[iE, :] = E_new
    Aop[iH, :] = H_new
    Aop[iP, :] = P_new
    return float(np.max(np.abs(np.linalg.eigvals(Aop))))


def main():
    n_cells_per_h = 3                       # the reference arm
    dx = H / n_cells_per_h
    dt = 5.001581407736339e-13              # this arm's own dt, from its recorded run
    out = {"arm": "n = 3, dx = h/3, dt from the recorded n3 run", "dx_m": dx, "dt_s": dt,
           "model_owner": ("tests/unit/boundaries/"
                           "test_cpml_subpixel_coefficient_consistency.py::amplification_rho"),
           "extension_check": {}, "comparator": {}, "cases": []}

    # --- the extension reproduces the owned model when no PEC node is imposed ---------
    worst = 0.0
    for layers in (4, 6, 8, 16):
        for ea, eb in ((1.0, 1.0), (EPS_R, EPS_R), (EPS_R, 1.0)):
            nn = max(24, 3 * layers)
            a = amplification_rho(layers, dx, dt, ea, eb, n=nn)
            b_ = rho_with_pec(layers, dx, dt, ea, eb, n=nn, pec_node=None)
            worst = max(worst, abs(a - b_))
    out["extension_check"]["max_abs_diff_vs_owned_model"] = worst
    assert worst < 1e-12, f"the PEC extension does not reproduce the owned model: {worst}"

    # --- comparators ------------------------------------------------------------------
    out["comparator"]["lossless_no_absorber"] = rho_with_pec(
        6, dx, dt, 1.0, 1.0, absorber=False, pec_node=None)
    out["comparator"]["lossless_no_absorber_pec_end"] = rho_with_pec(
        6, dx, dt, 1.0, 1.0, absorber=False, pec_node=20)
    for k, v in out["comparator"].items():
        assert abs(v - 1.0) <= 1e-6, f"comparator {k} gives rho={v}, must be 1"

    # --- the question ------------------------------------------------------------------
    for layers in (4, 6, 8, 16):
        nn = max(24, 3 * layers)
        for placement, node in (("no PEC (control)", None),
                                ("PEC inside the absorber (n_layers-1)", layers - 1),
                                ("PEC at the absorber seam (n_layers)", layers),
                                ("PEC one cell inside the interior", layers + 1)):
            for ea in (1.0, EPS_R):
                rho = rho_with_pec(layers, dx, dt, ea, ea, n=nn, pec_node=node)
                out["cases"].append(dict(n_layers=layers, pec_placement=placement,
                                         pec_node=node, eps=ea, rho=rho,
                                         unstable=bool(rho > 1 + 1e-6)))

    # --- POST-HOC, NOT PART OF THE PRE-DECLARED GATE -----------------------------------
    # The gate above was declared "at this arm's dx and dt", i.e. n = 3's, for every layer
    # count.  But the ladder's 4-layer and 8/16-layer arms are at n = 2 and its 8-layer
    # sibling at n = 4, each with its OWN dx and dt -- so the rows above compare layer counts
    # at one resolution, while the ladder compares arms that each carry their own.  This
    # re-evaluates each ladder arm at ITS OWN dx and dt.  It is recorded as an extension and
    # is NOT used to claim the pre-declared verdict either way.
    arms = [(2, 4, 7.502372111604508e-13, "GROWS"), (3, 6, 5.001581407736339e-13, "GROWS"),
            (2, 8, 7.502372111604508e-13, "settles"), (2, 16, 7.502372111604508e-13, "settles"),
            (4, 8, 3.751186055802254e-13, "settles")]
    out["post_hoc_per_arm"] = []
    for n_h, layers, arm_dt, fdtd in arms:
        arm_dx = H / n_h
        nn = max(24, 3 * layers)
        r_ctl = rho_with_pec(layers, arm_dx, arm_dt, EPS_R, EPS_R, n=nn, pec_node=None)
        r_pec = rho_with_pec(layers, arm_dx, arm_dt, EPS_R, EPS_R, n=nn, pec_node=layers)
        out["post_hoc_per_arm"].append(dict(
            n_cells_per_h=n_h, n_layers=layers, dt_s=arm_dt, fdtd_verdict=fdtd,
            rho_no_pec=r_ctl, rho_pec_at_seam=r_pec,
            model_unstable=bool(r_pec > 1 + 1e-6),
            implied_log_rate_per_step=float(np.log(max(r_pec, 1e-300)))))

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "absorber_pec_amplification.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    chk = out["extension_check"]["max_abs_diff_vs_owned_model"]
    print(f"extension vs owned model: max|diff| = {chk:.3e}")
    print(f"comparator lossless slice        rho = {out['comparator']['lossless_no_absorber']!r}")
    print(f"comparator lossless + PEC end    rho = "
          f"{out['comparator']['lossless_no_absorber_pec_end']!r}")
    print()
    print(f"{'layers':>6s} {'eps':>5s}  {'PEC placement':38s} {'rho':>18s}  verdict")
    for c in out["cases"]:
        print(f"{c['n_layers']:6d} {c['eps']:5.2f}  {c['pec_placement']:38s} "
              f"{c['rho']:18.12f}  {'UNSTABLE' if c['unstable'] else 'bounded'}")
    print()
    print("POST-HOC (not the pre-declared gate): each ladder arm at ITS OWN dx and dt")
    print(f"{'arm':>16s} {'FDTD':>8s}  {'rho no PEC':>18s} {'rho PEC at seam':>18s}  "
          f"{'ln(rho)/step':>14s}  model")
    for c in out["post_hoc_per_arm"]:
        tag = f"n={c['n_cells_per_h']}/cpml{c['n_layers']}"
        print(f"{tag:>16s} {c['fdtd_verdict']:>8s}  "
              f"{c['rho_no_pec']:18.12f} {c['rho_pec_at_seam']:18.12f}  "
              f"{c['implied_log_rate_per_step']:+14.3e}  "
              f"{'UNSTABLE' if c['model_unstable'] else 'bounded'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
