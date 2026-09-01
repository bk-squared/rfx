"""D4 (issue #786) — reference quality, in three parts.

D4a  exact-reference instrument twin: an empty vacuum PEC box whose dt,
     n_steps, analysis band and record length are IDENTICAL to the W4R
     uniform rung at the same scale, and whose exact discrete leapfrog
     eigenfrequency is computable in closed form -- so the extraction
     instrument's OWN error is isolated at every rung, reference rung
     included.
D4b  independent reference: f(h) = f_inf - C h^p fitted to the five
     ladder rungs, with s = 0.25 held OUT of the fit and then judged.
D4c  independent estimators on the identical stored records.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d4_reference [--part a|b|c|all]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from rfx import Simulation, GaussianPulse
from validation.research.convergence_floor import fixture as fx
from validation.research.convergence_floor import estimators as est

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
WINDOWS = os.path.join(RES, "predeclared_windows_786.json")

C0 = 299792458.0
# --- D4a twin geometry (frozen in the pre-declaration) ----------------
TWIN_A = 38.25e-3          # 17 * 2.25 mm -> integer cells at every scale
TWIN_B = 38.25e-3
TWIN_LZ = 1.5e-3           # 6 * 0.25 mm -> integer cells at every scale
TWIN_BAND = (4.0e9, 6.5e9)
TWIN_SRC = (6.75e-3, 9.0e-3, TWIN_LZ / 2)
TWIN_PRB = (13.5e-3, 20.25e-3, TWIN_LZ / 2)


def twin_f_exact() -> float:
    return C0 / 2 * np.sqrt((1 / TWIN_A) ** 2 + (1 / TWIN_B) ** 2)


def _mu(profile: np.ndarray, m: int) -> float:
    """The rfx 1-D difference operator's m-th eigenvalue (Dirichlet ends),
    float64 mirror of analytic_dispersion.operator_eigenvalues."""
    d = np.asarray(profile, float)
    dfull = np.concatenate([d, d[-1:]])           # #562 bounding-node dup
    inv = 1.0 / dfull
    inv_h = np.concatenate([inv[:-1], [0.0]])
    inv_e = np.concatenate([inv[:1], 2.0 / (dfull[:-1] + dfull[1:])])
    ks = np.arange(1, len(d))
    s = np.sqrt(inv_e[ks])
    diag = s * (inv_h[ks] + inv_h[ks - 1]) * s
    off = -inv_h[ks[:-1]] * s[:-1] * s[1:]
    M = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return float(np.sort(np.linalg.eigvalsh(M))[m - 1])


def twin_f_discrete(scale: float, dt: float) -> float:
    dx = fx.PC_DX0 * scale
    mu_x = _mu(np.full(int(round(TWIN_A / dx)), dx), 1)
    mu_y = _mu(np.full(int(round(TWIN_B / dx)), dx), 1)
    arg = C0 * dt * np.sqrt(mu_x + mu_y) / 2.0
    return float(np.arcsin(arg) / (np.pi * dt))


def twin_rung(scale: float) -> dict:
    dx = fx.PC_DX0 * scale
    dz = fx.PC_DZF0 * scale
    prof = np.full(int(round(TWIN_LZ / dz)), dz)
    sim = Simulation(freq_max=fx.F_MAX, domain=(TWIN_A, TWIN_B, TWIN_LZ),
                     dx=dx, boundary="pec", dz_profile=prof)
    sim.add_source(TWIN_SRC, "ez",
                   waveform=GaussianPulse(amplitude=1.0, **fx.WAVEFORM),
                   amplitude_kind="current")
    sim.add_probe(TWIN_PRB, "ez")
    n_steps = fx.n_steps_for(scale, dz)
    t0 = time.time()
    res = sim.run(n_steps=n_steps, subpixel_smoothing=fx.SUBPIXEL)
    wall = time.time() - t0
    dt = float(res.dt)
    modes = fx.modes_of(res)
    f_e1, dom, info = fx.target_line(modes, TWIN_BAND)
    ts = np.asarray(res.time_series).ravel().astype(np.float64)
    cons = est.consensus(ts, dt, TWIN_BAND)
    f_disc = twin_f_discrete(scale, dt)
    return {
        "scale": scale, "dt": dt, "n_steps": n_steps,
        "cells": int(round(TWIN_A / dx)) * int(round(TWIN_B / dx)) * len(prof),
        "f_exact_continuum_hz": twin_f_exact(),
        "f_discrete_exact_hz": f_disc,
        "f_E1_hz": f_e1, "dominance": dom, "in_band": info["in_band"],
        "eps_instr_E1_hz": abs(f_e1 - f_disc),
        "consensus": {k: v for k, v in cons.items()},
        "eps_instr_consensus_hz": abs(cons["mean_hz"] - f_disc),
        "e_disc_hz": f_disc - twin_f_exact(),
        "wallclock_s": wall,
        "_series": ts,
    }


def part_a(win) -> dict:
    sound = float(win["D4_reference_quality"]
                  ["instrument_resolution_derivation"]["sound_hz"])
    limited = float(win["D4_reference_quality"]
                    ["instrument_resolution_derivation"]["limited_hz"])
    rows, records = [], {}
    for s in sorted(list(fx.SCALES) + [fx.REF_SCALE], reverse=True):
        r = twin_rung(s)
        records["twin_%s" % s] = r.pop("_series")
        records["twin_%s__dt" % s] = np.array([r["dt"]])
        rows.append(r)
        print("twin s=%-5s f_disc=%.6f  E1=%.6f  eps_instr(E1)=%9.4f kHz  "
              "consensus spread=%.1f kHz  eps_instr(cons)=%9.4f kHz  "
              "e_disc=%+.4f MHz  dom=%s  wall=%.0fs"
              % (s, r["f_discrete_exact_hz"] / 1e9, r["f_E1_hz"] / 1e9,
                 r["eps_instr_E1_hz"] / 1e3, r["consensus"]["spread_hz"] / 1e3,
                 r["eps_instr_consensus_hz"] / 1e3, r["e_disc_hz"] / 1e6,
                 r["dominance"], r["wallclock_s"]), flush=True)
    np.savez_compressed(os.path.join(RES, "d4a_twin_records.npz"), **records)

    admissible = all(r["dominance"] >= 10.0 for r in rows)
    eps = {r["scale"]: r["eps_instr_E1_hz"] for r in rows}
    verdicts = {}
    for s, e in eps.items():
        verdicts[str(s)] = ("SOUND" if e <= sound else
                            ("INSTRUMENT-LIMITED" if e >= limited
                             else "INCONCLUSIVE"))
    # The twin is also the primary SMOOTH-FIELD control (D2): its error
    # against the exact continuum frequency is a pure discretization
    # sequence with no metal edge anywhere.
    h = np.array([fx.PC_DZF0 * r["scale"] for r in rows if r["scale"] in fx.SCALES])
    e_d = np.array([abs(r["e_disc_hz"]) for r in rows if r["scale"] in fx.SCALES])
    p_smooth_analytic = est.fit_order_loglog(h, e_d)
    e_meas = np.array([abs(r["f_E1_hz"] - r["f_exact_continuum_hz"])
                       for r in rows if r["scale"] in fx.SCALES])
    p_smooth_measured = est.fit_order_loglog(h, e_meas)
    return {"rows": rows, "admissible": bool(admissible),
            "eps_instr_hz": {str(k): v for k, v in eps.items()},
            "per_rung_verdict": verdicts,
            "p_smooth_analytic": p_smooth_analytic,
            "p_smooth_measured": p_smooth_measured,
            "verdict": ("INSTRUMENT-LIMITED at rung(s) "
                        + ",".join(k for k, v in verdicts.items()
                                   if v == "INSTRUMENT-LIMITED")
                        if any(v == "INSTRUMENT-LIMITED"
                               for v in verdicts.values())
                        else ("SOUND at every rung"
                              if all(v == "SOUND" for v in verdicts.values())
                              else "MIXED/INCONCLUSIVE"))}


def part_b(win, d0) -> dict:
    w = win["D4_reference_quality"]["D4b_independent_reference"]
    k_out = float(w["outlier_k"])
    k_vin = float(w["vindicated_k"])
    uc = {r["scale"]: r["f_target"] for r in d0["rows"] if not r["multiband"]}
    mb = {r["scale"]: r["f_target"] for r in d0["rows"] if r["multiband"]}
    out = {}
    for name, tbl in (("uniform", uc), ("multiband", mb)):
        ss = sorted([s for s in tbl if s in fx.SCALES], reverse=True)
        if len(ss) < 4:
            continue
        h = np.array([fx.PC_DZF0 * s for s in ss])
        f = np.array([tbl[s] for s in ss])
        fit = est.fit_power_law(h, f)
        pred_ref = fit["predict"](fx.PC_DZF0 * fx.REF_SCALE)
        f_ref = uc[fx.REF_SCALE]
        dev = abs(f_ref - pred_ref)
        rms = fit["rms_residual_hz"]
        trend_broken = (np.sign(f_ref - tbl[ss[-1]])
                        == -np.sign(tbl[ss[-1]] - tbl[ss[-2]]))
        verdict = ("OUTLIER (mechanism 4 attributed)"
                   if dev >= k_out * rms and trend_broken else
                   ("VINDICATED" if dev <= k_vin * rms else "INCONCLUSIVE"))
        out[name] = {
            "scales": ss, "h_m": h.tolist(), "f_hz": f.tolist(),
            "f_inf_hz": fit["f_inf_hz"], "p": fit["p"], "C": fit["C"],
            "rms_residual_hz": rms, "residuals_hz": fit["residuals_hz"],
            "f_pred_at_ref_hz": pred_ref,
            "f_E1_at_ref_hz": f_ref,
            "deviation_hz": dev, "deviation_over_rms": dev / rms if rms else None,
            "trend_broken": bool(trend_broken),
            "verdict": verdict,
            "err_vs_f_inf_mhz": {str(s): abs(tbl[s] - fit["f_inf_hz"]) / 1e6
                                 for s in ss},
            "p_from_loglog_vs_f_inf": est.fit_order_loglog(
                h, np.array([abs(tbl[s] - fit["f_inf_hz"]) for s in ss])),
        }
    return out


def part_c(win) -> dict:
    w = win["D4_reference_quality"]["D4c_independent_estimators"]
    spread_ok = float(w["spread_hz"])
    attribute = float(w["attribute_hz"])
    exonerate = float(w["exonerate_hz"])
    d0 = json.load(open(os.path.join(RES, "d0_reproduction.json")))
    recs = np.load(os.path.join(RES, "d0_records.npz"))
    rows = []
    for r in d0["rows"]:
        key = "%s_%s" % ("MB" if r["multiband"] else "UC", r["scale"])
        if key not in recs:
            continue
        ts = recs[key]
        dt = float(recs[key + "__dt"][0])
        cons = est.consensus(ts, dt, fx.BAND)
        e1 = r["f_target"]
        agree = cons["spread_hz"] <= spread_ok
        delta = abs(e1 - cons["mean_hz"])
        rows.append({
            "arm": "MB" if r["multiband"] else "UC", "scale": r["scale"],
            "n_samples": int(len(ts)), "dt": dt,
            "E1_hz": e1, **cons, "E1_minus_consensus_hz": e1 - cons["mean_hz"],
            "estimators_agree": bool(agree),
            "verdict": (("ATTRIBUTE-4a" if delta >= attribute else
                         ("EXONERATE-4a" if delta <= exonerate
                          else "INCONCLUSIVE")) if agree
                        else "CONSENSUS-UNAVAILABLE"),
        })
        print("%s s=%-5s N=%6d  E1=%.6f  E2=%.6f E3=%.6f E4=%.6f  "
              "spread=%.1f kHz  E1-cons=%+9.3f MHz  %s"
              % (rows[-1]["arm"], r["scale"], len(ts), e1 / 1e9,
                 cons["values_hz"]["E2"] / 1e9, cons["values_hz"]["E3"] / 1e9,
                 cons["values_hz"]["E4"] / 1e9, cons["spread_hz"] / 1e3,
                 rows[-1]["E1_minus_consensus_hz"] / 1e6,
                 rows[-1]["verdict"]), flush=True)
    return {"rows": rows}


def main():
    fx.quiet_third_party_warnings()
    part = "all"
    if "--part" in sys.argv:
        part = sys.argv[sys.argv.index("--part") + 1]
    win = json.load(open(WINDOWS))
    out = {"issue": 786, "discriminator": "D4"}
    if part in ("a", "all"):
        print("=== D4a: exact-reference instrument twin ===", flush=True)
        out["D4a"] = part_a(win)
        print("D4a:", out["D4a"]["verdict"])
    if part in ("b", "all"):
        print("=== D4b: independent (Richardson) reference ===", flush=True)
        d0 = json.load(open(os.path.join(RES, "d0_reproduction.json")))
        out["D4b"] = part_b(win, d0)
        for k, v in out["D4b"].items():
            print(" %s: f_inf=%.6f GHz p=%.3f rms=%.3f MHz  dev(ref)=%.3f MHz"
                  " = %.1f x rms -> %s"
                  % (k, v["f_inf_hz"] / 1e9, v["p"],
                     v["rms_residual_hz"] / 1e6, v["deviation_hz"] / 1e6,
                     v["deviation_over_rms"], v["verdict"]))
    if part in ("c", "all"):
        print("=== D4c: independent estimators on the identical records ===",
              flush=True)
        out["D4c"] = part_c(win)
    path = os.path.join(RES, "d4_reference_%s.json" % part)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
