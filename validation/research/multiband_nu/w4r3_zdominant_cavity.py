"""W4R3 - F-S4 on a fixture whose error budget the GRADED axis dominates.

Why this fixture exists (adversarial-review finding BL2; note section
W4R3). The W4R2 analytic cavity (27 x 18 x 64 mm, TE101, dx = 1.5 dzf)
carries only ~1 % of its total frequency error on the graded z axis. The
frozen analytic decomposition (``analytic_dispersion.py``) reproduces every
committed W4R2 arm to <= 0.041 MHz and splits the error as
|e_z| / (|e_z|+|e_x|+|e_t|) = 0.0106 (uniform) / 0.0114 (multiband); the
other ~99 % is transverse + time dispersion, IDENTICAL in a multiband arm
and in its matched uniform control. That fixture's order gate therefore
could not have fired for a grading reason. W4R2 remains a valid measurement
of what it measured; it is not a witness for the graded axis.

W4R3 keeps the fixture CLASS that made W4R2 the right instrument (empty PEC
box, smooth-field eigenmode, exact analytic target, no rasterization, no
dielectric, no reference ladder, no Richardson) and re-proportions it so the
graded axis carries the error budget:

* mode TE_{1,0,4} instead of TE_{1,0,1}: k_z = 4 pi / L_z is 3.75x k_x and
  an axis error scales as k^4 d^2, so the transverse term drops ~200x;
* dx = dy = dzf / 2 instead of 1.5 dzf: the transverse term drops another
  9x, and the CFL step (hence the time-dispersion term) drops with it.

Derived budget, frozen by ``analytic_dispersion`` BEFORE any arm ran, at
every scale:

    |e_z| / (|e_z|+|e_x|+|e_t|)      uniform 0.888   multiband 0.922
    |e_z(mb) - e_z(uc)| / |e_tot(mb)|                          0.357
    predicted fitted order           p_uc 2.000      p_mb 2.000

i.e. the graded axis carries ~89-92 % of the error and the grading-SPECIFIC
part is 36 % of the multiband arm total (W4R2: 0.17 %). The matched-scale
contrast f_mb - f_uc is -28.4 / -7.09 / -1.77 / -0.44 MHz at
s = 2 / 1 / 0.5 / 0.25 - above the fit floor at every scale, so the quantity
under test is resolved by the instrument rather than buried in it.

Fixture: PEC box 60 x 3 x 64 mm, empty (vacuum, cpml_layers = 0). z profile
fine(12 mm) | coarse(14 mm) | fine(12) | coarse(14) | fine(12), ABRUPT
r = 1.4 (the envelope cap, worst case); dzf = s mm, coarse = 1.4 s mm;
transverse uniform dx = dy = 0.5 s mm; uniform control dzf everywhere.
Scales s in {0.25, 0.5, 1, 2} - every band length, L_z and both transverse
extents are exact multiples at every scale, and both source/probe planes
land on exact nodes at every scale.

Target: TE_{1,0,4} = (c/2) sqrt((1/a)^2 + (4/L)^2) = 9.6958969 GHz. Mode
selection: Ey source pair at x = a/3 and x = 2a/3 (equal sign, same z) and
Ey probe at x = a/3. Those planes are exact nodes of the m = 3 discrete
eigenvector and an exactly mirror-symmetric pair about x = a/2, so the
m = 2, 3, 4 families are neither driven nor observed and only the (1,0,p)
family survives in band: nearest neighbours 7.456 and 11.973 GHz.
Source at z = 8 mm (an antinode, |sin| = 1) and probe at z = 28 mm
(|sin| = 0.924), both exact nodes of BOTH profiles at every scale - see
correction WP6R.6 in the note: the pre-declaration's z = 40 mm probe is an
exact node of the uniform profile but lands 2 mm into a 1.4-dzf coarse
band on the multiband profile, where it is not a node at any scale.
T = 15 ns.

Judge - the frozen W4R.3 pass/fail structure, UNCHANGED:
    fit floor 0.3 MHz, >= 3 surviving points per arm;
    p = LS slope of log|e| vs log dzf;
    fixture gate p_uc in [1.7, 2.6];
    F-S4 FIRES iff fixture valid AND (p_mb < 1.5 OR p_mb < p_uc - 0.4);
    anomaly A4 iff p_mb > p_uc + 0.4.

Fixture-validity gates, re-derived for THIS fixture (note W4R3). Failing any
of them is FIXTURE-INVALID / INCONCLUSIVE - no envelope support, and not a
multiband fault:
    G1  z_fraction >= 0.80 at every arm and scale        (derived 0.888)
    G2  |e_z(mb) - e_z(uc)| / |e_tot(mb)| >= 0.20        (derived 0.357)
    G3  |f_model - f_meas| <= 0.15 MHz at every arm      (prior-provenance
        class: <= 0.041 MHz = 6.8e-6 relative over the eight committed
        W4R2 arms, scaled to 9.70 GHz and x2.2)

Revert-proof (``--revert-proof``): the CORE-C2-class metric defect
(E-update dual 2/(d[k-1]+d[k]) -> primal 1/d[k]) on ONE multiband
transition node, the coarse->fine transition at z = 38 mm. The defect is
identically null on a uniform mesh - it is a purely grading-side defect.
Predicted by the same frozen model: per-scale shift -47.3 / -24.0 / -12.0 /
-6.0 MHz (11-20x the fit floor at every scale) and p_mb -> 1.374, which
fires BOTH clauses of the frozen rule.

Usage:
    PYTHONPATH=. python -m validation.research.multiband_nu.w4r3_zdominant_cavity
    PYTHONPATH=. python -m validation.research.multiband_nu.w4r3_zdominant_cavity --revert-proof
    PYTHONPATH=. python -m validation.research.multiband_nu.w4r3_zdominant_cavity --bringup
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import jax.numpy as jnp

from rfx.harminv import harminv
from rfx.nonuniform import run_nonuniform

from .analytic_dispersion import C0, decompose, inv_arrays
from .harness import build_pec_fixture

A_X = 60e-3
B_Y = 3e-3
FINE_LEN = 12e-3
COARSE_LEN = 14e-3
R_CAP = 1.4
L_Z = 3 * FINE_LEN + 2 * COARSE_LEN                 # 64 mm
M_X = 1
P_Z = 4
F_TARGET = (C0 / 2) * np.sqrt((M_X / A_X) ** 2 + (P_Z / L_Z) ** 2)
DZF0 = 1e-3                                          # dzf = DZF0 * s
DXY0 = 0.5e-3                                        # dx = dy = DXY0 * s
SCALES = (0.25, 0.5, 1.0, 2.0)
T_TOTAL = 15e-9
MATCH_GUARD = 0.03
E_FLOOR_HZ = 0.3e6
BAND = (8.8e9, 10.6e9)
HARMINV_BAND = (6.5e9, 13.5e9)
SIGMA_T = 200e-12
Z_SRC = 8e-3
Z_PRB = 28e-3
# fixture-validity gates (note W4R3)
G1_Z_FRACTION_MIN = 0.80
G2_GRADING_SHARE_MIN = 0.20
G3_MODEL_RESIDUAL_HZ = 0.15e6
# revert-proof: CORE-C2-class metric defect on ONE transition node of the
# multiband arm (dual 2/(d[k-1]+d[k]) -> primal 1/d[k]). Index 2 is the
# coarse->fine transition at z = 38 mm. The defect is identically null on a
# uniform mesh, so it is a purely grading-side defect.
DEFECT_TRANSITION = 2


def mb_profile(s: float) -> np.ndarray:
    dzf = s * DZF0
    dzc = R_CAP * dzf
    nf = int(round(FINE_LEN / dzf))
    nc = int(round(COARSE_LEN / dzc))
    assert abs(nf * dzf - FINE_LEN) < 1e-12 and abs(nc * dzc - COARSE_LEN) < 1e-12
    return np.asarray([dzf] * nf + [dzc] * nc + [dzf] * nf
                      + [dzc] * nc + [dzf] * nf, np.float64)


def uc_profile(s: float) -> np.ndarray:
    dzf = s * DZF0
    n = int(round(L_Z / dzf))
    assert abs(n * dzf - L_Z) < 1e-12
    return np.full(n, dzf, np.float64)


def assert_planes_realizable() -> None:
    """Both source and probe planes must be exact E nodes of BOTH profiles
    at EVERY scale (WP6R.6). Cheap, so it runs before the ladder."""
    for s in SCALES:
        for prof in (mb_profile(s), uc_profile(s)):
            zn = np.concatenate([[0.0], np.cumsum(prof)])
            for z in (Z_SRC, Z_PRB):
                assert np.min(np.abs(zn - z)) < 1e-12, (s, z, len(prof))


def transition_nodes(s: float) -> list[int]:
    dzf = s * DZF0
    nf = int(round(FINE_LEN / dzf))
    nc = int(round(COARSE_LEN / (R_CAP * dzf)))
    return [nf, nf + nc, 2 * nf + nc, 2 * nf + 2 * nc]


def _defect_inv_e(profile: np.ndarray, k_bad: int) -> np.ndarray:
    inv_e, _ = inv_arrays(profile)
    dfull = np.concatenate([profile, profile[-1:]])
    inv_e = inv_e.copy()
    inv_e[k_bad] = 1.0 / dfull[k_bad]
    return inv_e


def measure(s: float, multiband: bool, defect: bool = False) -> dict:
    prof = mb_profile(s) if multiband else uc_profile(s)
    dx = DXY0 * s
    grid, mats = build_pec_fixture(prof, (A_X, B_Y), dx)
    inv_e_override = None
    if defect:
        assert multiband, "the grading defect is null on a uniform mesh"
        k_bad = transition_nodes(s)[DEFECT_TRANSITION]
        inv_e_override = _defect_inv_e(prof, k_bad)
        grid = grid._replace(inv_dz=jnp.asarray(inv_e_override, jnp.float32))
    dt = float(grid.dt)
    n_steps = int(round(T_TOTAL / dt))
    t = np.arange(n_steps) * dt
    t0 = 5 * SIGMA_T
    wf = (np.exp(-((t - t0) / SIGMA_T) ** 2 / 2.0)
          * np.sin(2 * np.pi * F_TARGET * (t - t0))).astype(np.float32)
    zn = np.concatenate([[0.0], np.cumsum(prof)])
    k_src = int(np.argmin(np.abs(zn - Z_SRC)))
    k_prb = int(np.argmin(np.abs(zn - Z_PRB)))
    assert abs(zn[k_src] - Z_SRC) < 1e-12 and abs(zn[k_prb] - Z_PRB) < 1e-12
    # x = a/3 and x = 2a/3, equal sign: exact nodes of the m=3 discrete
    # eigenvector and an exactly mirror-symmetric pair, so the m=2, 3 and 4
    # families are neither driven nor observed. Only the (1,0,p) family is
    # left in band (nearest neighbours 7.46 and 11.97 GHz).
    i_a = int(round(A_X / 3 / dx))
    i_b = int(round(2 * A_X / 3 / dx))
    assert abs(i_a * dx - A_X / 3) < 1e-12 and abs(i_b * dx - 2 * A_X / 3) < 1e-12
    j_c = grid.ny // 2
    t_start = time.time()
    out = run_nonuniform(grid, mats, n_steps,
                         sources=[(i_a, j_c, k_src, "ey", wf),
                                  (i_b, j_c, k_src, "ey", wf)],
                         probes=[(i_a, j_c, k_prb, "ey")])
    ts = np.asarray(out["time_series"][:, 0], np.float64)
    wall = time.time() - t_start
    n_skip = int((2 * t0) / dt)
    sig = ts[n_skip:] - ts[n_skip:].mean()
    modes = sorted(harminv(sig, dt, *HARMINV_BAND), key=lambda m: m.freq)
    in_band = [m for m in modes if BAND[0] <= m.freq <= BAND[1]]
    if in_band:
        f_meas = float(min(in_band, key=lambda m: abs(m.freq - F_TARGET)).freq)
    else:
        f_meas = float("nan")
    valid = bool(np.isfinite(f_meas)
                 and abs(f_meas - F_TARGET) <= MATCH_GUARD * F_TARGET)
    model = decompose(prof, dx, A_X, dt, M_X, P_Z, L_Z,
                      inv_e_z_override=inv_e_override)
    return {
        "scale": s, "multiband": multiband, "defect": defect,
        "nz": len(prof), "n_steps": n_steps,
        "cells": int(grid.nx * grid.ny * grid.nz),
        "dt": dt,
        "modes": [(float(m.freq), float(m.Q), float(abs(m.amplitude)))
                  for m in modes[:6]],
        "f_meas": f_meas, "valid": valid,
        "err_hz": abs(f_meas - F_TARGET) if valid else float("nan"),
        "f_model": model["f_model"],
        "model_residual_hz": (f_meas - model["f_model"]) if valid else float("nan"),
        "e_z": model["e_z"], "e_x": model["e_x"], "e_t": model["e_t"],
        "e_total_model": model["e_total"], "z_fraction": model["z_fraction"],
        "wallclock_s": wall,
    }


def fit_order(rows: dict, multiband: bool, scales=SCALES):
    pts = [(DZF0 * s, rows[(multiband, s)]["err_hz"]) for s in scales
           if rows[(multiband, s)]["valid"]
           and rows[(multiband, s)]["err_hz"] >= E_FLOOR_HZ]
    if len(pts) < 3:
        return None, pts
    h = np.log10([p[0] for p in pts])
    e = np.log10([p[1] for p in pts])
    return float(np.polyfit(h, e, 1)[0]), pts


def judge(rows: dict, defect: bool = False) -> dict:
    """The frozen W4R.3 pass/fail rule + the W4R3 fixture-validity gates."""
    out = {}
    p_uc, pts_uc = fit_order(rows, False)
    p_mb, pts_mb = fit_order(rows, True)
    out["p_uc"], out["p_mb"] = p_uc, p_mb
    out["n_fit_points"] = {"uc": len(pts_uc), "mb": len(pts_mb)}

    # --- fixture-validity gates (note W4R3) ---
    z_fracs = [r["z_fraction"] for r in rows.values()]
    g1 = float(min(z_fracs))
    shares, resids = [], []
    for s in SCALES:
        mb, uc = rows[(True, s)], rows[(False, s)]
        shares.append(abs(mb["e_z"] - uc["e_z"]) / abs(mb["e_total_model"]))
        for r in (mb, uc):
            if r["valid"]:
                resids.append(abs(r["model_residual_hz"]))
    g2 = float(min(shares))
    g3 = float(max(resids)) if resids else float("nan")
    gates = {
        "g1_min_z_fraction": g1, "g1_target": G1_Z_FRACTION_MIN,
        "g1_pass": bool(g1 >= G1_Z_FRACTION_MIN),
        "g2_min_grading_share": g2, "g2_target": G2_GRADING_SHARE_MIN,
        "g2_pass": bool(g2 >= G2_GRADING_SHARE_MIN),
        "g3_max_model_residual_hz": g3, "g3_target": G3_MODEL_RESIDUAL_HZ,
        "g3_pass": bool(np.isfinite(g3) and g3 <= G3_MODEL_RESIDUAL_HZ),
    }
    out["fixture_gates"] = gates
    fixture_ok = gates["g1_pass"] and gates["g2_pass"] and gates["g3_pass"]

    if not fixture_ok:
        failed = [k for k in ("g1", "g2", "g3") if not gates[f"{k}_pass"]]
        out["verdict"] = ("FIXTURE-INVALID (validity gate(s) %s failed)"
                          % ", ".join(failed))
        out["fs4_fired"] = None
        out["anomaly_a4"] = None
    elif p_uc is None or p_mb is None:
        out["verdict"] = "INCONCLUSIVE (fewer than 3 valid fit points >= floor)"
        out["fs4_fired"] = None
        out["anomaly_a4"] = None
    elif p_uc < 1.7 or p_uc > 2.6:
        out["verdict"] = "FIXTURE-INVALID (p_uc %.2f outside [1.7, 2.6])" % p_uc
        out["fs4_fired"] = None
        out["anomaly_a4"] = None
    else:
        fired = bool(p_mb < 1.5 or p_mb < p_uc - 0.4)
        anomaly = bool(p_mb > p_uc + 0.4)
        out["fs4_fired"] = fired
        out["anomaly_a4"] = anomaly
        out["verdict"] = (f"p_uc={p_uc:.2f} p_mb={p_mb:.2f} fired={fired}"
                          + (" ANOMALY(p_mb>p_uc+0.4)" if anomaly else ""))
    if defect:
        out["revert_proof_pass"] = bool(out["fs4_fired"] is True)
    return out


def _print_row(e):
    tag = ("MB*" if e["defect"] else "MB") if e["multiband"] else "UC"
    print(f"{tag} s={e['scale']}: f={e['f_meas']/1e9:.6f} GHz "
          f"err={e['err_hz']/1e6:.4f} MHz model={e['f_model']/1e9:.6f} "
          f"resid={e['model_residual_hz']/1e6:+.4f} MHz "
          f"zfrac={e['z_fraction']:.4f} valid={e['valid']} "
          f"cells={e['cells']} wall={e['wallclock_s']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert-proof", action="store_true",
                    help="re-run the multiband arms with the grading defect")
    ap.add_argument("--bringup", action="store_true",
                    help="uniform-control instrument check only, writes nothing")
    args = ap.parse_args()

    assert_planes_realizable()
    if args.bringup:
        for s in (2.0, 1.0):
            _print_row(measure(s, False))
        return

    out = {"f_target_hz": float(F_TARGET), "mode": f"TE_{M_X}0{P_Z}",
           "a_x_m": A_X, "b_y_m": B_Y, "l_z_m": L_Z, "r_cap": R_CAP,
           "scales": list(SCALES), "e_floor_hz": E_FLOOR_HZ, "arms": []}
    rows = {}
    for mb in (False, True):
        for s in SCALES:
            e = measure(s, mb)
            rows[(mb, s)] = e
            out["arms"].append(e)
            _print_row(e)
    out.update(judge(rows))
    print("F-S4 (W4R3):", out["verdict"], flush=True)
    print("  fixture gates:", json.dumps(out["fixture_gates"]), flush=True)
    path = ("validation/research/multiband_nu/results/"
            "w4r3_zdominant_cavity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)

    if args.revert_proof:
        rp = {"f_target_hz": float(F_TARGET),
              "defect": ("CORE-C2 class: inv_dz (E-update dual metric) -> "
                         "primal 1/d[k] at ONE multiband transition node "
                         f"(transition index {DEFECT_TRANSITION}, z = 38 mm); "
                         "identically null on a uniform mesh"),
              "clean_p_uc": out["p_uc"], "clean_p_mb": out["p_mb"],
              "clean_verdict": out["verdict"], "arms": []}
        drows = {(False, s): rows[(False, s)] for s in SCALES}
        for s in SCALES:
            e = measure(s, True, defect=True)
            drows[(True, s)] = e
            rp["arms"].append(e)
            _print_row(e)
        rp.update(judge(drows, defect=True))
        rp["per_scale_shift_hz"] = {
            str(s): drows[(True, s)]["f_meas"] - rows[(True, s)]["f_meas"]
            for s in SCALES}
        print("REVERT-PROOF (defect arm):", rp["verdict"],
              "-> fires:", rp["fs4_fired"], flush=True)
        dpath = ("validation/research/multiband_nu/results/"
                 "w4r3_revert_proof.json")
        with open(dpath, "w") as fh:
            json.dump(rp, fh, indent=1)
        print("wrote", dpath)


if __name__ == "__main__":
    main()
