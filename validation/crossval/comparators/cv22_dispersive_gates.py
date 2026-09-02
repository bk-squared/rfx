"""cv22 dispersive slab -- declared arms, derived windows, gates, falsifiers.

Pure numpy (plus ``tests/_gate_policy.py`` for the repo-wide envelope rule).
Shared by the case script (``validation/crossval/22_dispersive_slab_fresnel.py``),
the Meep leg (``scripts/crossval/meep_cv22_dispersive_slab.py``) and the gate
test (``tests/test_cv22_dispersive_slab_gates.py``) so that the pre-declared
numbers exist in exactly one place.

Everything numeric here is fixed by
``docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md``; change
the note (append-only) before changing a number here.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dispersive_eps as de  # noqa: E402
from tests._gate_policy import gate_from_envelope  # noqa: E402

TWO_PI = 2.0 * math.pi
C0 = de.C0

# ---------------------------------------------------------------------------
# Rig (cv04's; pre-declaration §1)
# ---------------------------------------------------------------------------
DX_M = 1.0e-3
D_SLAB_M = 10.0e-3
NX_INTERIOR = 600
N_CPML = 20
TFSF_F0_HZ = 10.0e9
TFSF_BW = 0.5
NFFT_OVERSAMPLE = 8
MASK_F_LO_HZ = 3.0e9
MASK_F_HI_HZ = 15.0e9
MASK_AMP_FRAC = 0.02
# cv04 witness constants (04_multilayer_fresnel.py:208-210, :314), unchanged.
TAIL_WINDOW = 50
TAIL_PURITY_LIMIT = 1e-3
TAIL_LIMIT = 0.10
CONS_MAX_LIMIT = 0.06

# Gated band (§5) and the rig-sanity floor on incident amplitude inside it.
BAND_GATED_HZ = (4.0e9, 10.0e9)
GATED_BAND_MIN_INC_AMP_FRAC = 0.05

# Meep leg (§7): a = 1 cm as in cv04; Meep default Courant.
MEEP_A_M = 0.01
MEEP_COURANT = 0.5
DEBYE_MEEP_MAP_FN_HZ = 100.0e9
DEBYE_MAP_RESIDUAL_REL_BOUND = 3.0e-3

# ---------------------------------------------------------------------------
# Material arms (§2)
# ---------------------------------------------------------------------------
ARMS = {
    "debye": {
        "model": "debye",
        "params": {"eps_inf": 2.0, "delta_eps": 4.0, "tau": 1.0 / (TWO_PI * 5.0e9)},
    },
    "lorentz": {
        "model": "lorentz",
        "params": {"eps_inf": 2.0, "delta_eps": 1.5, "f0": 7.0e9,
                   "delta": TWO_PI * 7.0e9 / 6.0},  # Q = w0/(2 delta) = 3
    },
    "drude": {
        "model": "drude",
        "params": {"eps_inf": 3.0, "fp": 7.0e9, "gamma": TWO_PI * 3.0e9},
    },
}
ARM_ORDER = ("debye", "lorentz", "drude")

# ---------------------------------------------------------------------------
# Committed cv04 envelope on this rig (§4) and the derived rig windows
# ---------------------------------------------------------------------------
CV04_ENVELOPE = {
    # tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[..].observed_baseline
    "mean_dR": 0.0066,
    "mean_dT": 0.011,
    # validation/crossval/04_multilayer_fresnel.py:309 (code comment, rung C4,
    # job 369367246779) -- the only committed per-bin number on this rig.
    "per_bin_max_RT_closure": 0.0487,
}
W_BIN = gate_from_envelope(CV04_ENVELOPE["per_bin_max_RT_closure"], quantum=1000)   # 0.074
W_MEAN_R = gate_from_envelope(CV04_ENVELOPE["mean_dR"], quantum=1000)               # 0.010
W_MEAN_T = gate_from_envelope(CV04_ENVELOPE["mean_dT"], quantum=1000)               # 0.017

# ---------------------------------------------------------------------------
# Falsifiers (§6)
# ---------------------------------------------------------------------------
FALSIFIERS = {
    # name: (arm, description, param transform)
    "debye_tau_x2": ("debye", "F1: tau -> 2 tau",
                     lambda p: {**p, "tau": 2.0 * p["tau"]}),
    "debye_deps_zero": ("debye", "F2: delta_eps -> 0 (dispersionless eps_inf slab)",
                        lambda p: {**p, "delta_eps": 0.0}),
    "lorentz_f0_x1p3": ("lorentz", "F1: f0 -> 1.3 f0",
                        lambda p: {**p, "f0": 1.3 * p["f0"]}),
    "lorentz_deps_zero": ("lorentz", "F2: delta_eps -> 0",
                          lambda p: {**p, "delta_eps": 0.0}),
    "drude_fp_x1p3": ("drude", "F1: fp -> 1.3 fp",
                      lambda p: {**p, "fp": 1.3 * p["fp"]}),
    "drude_wp_zero": ("drude", "F2: omega_p -> 0",
                      lambda p: {**p, "fp": 0.0}),
}
# Meep-leg falsifiers (F3), applied to the to_meep(...) dict of the Lorentz arm.
MEEP_FALSIFIER_ARM = "lorentz"
MEEP_FALSIFIERS = {
    "no_2pi": "F3: frequency = omega_n a/c (2 pi not divided out)",
    "gamma_half": "F3: gamma built from delta instead of 2 delta",
}
# Case-script names for the Meep falsifier arms (read meep_lorentz__falsifier_<x>.json).
MEEP_FALSIFIER_CASE_NAMES = {f"meep_lorentz_{k}": k for k in MEEP_FALSIFIERS}


def apply_falsifier(name: str):
    """Return (arm, model, defective params) for an rfx-side falsifier."""
    arm, _desc, fn = FALSIFIERS[name]
    base = ARMS[arm]
    return arm, base["model"], fn(dict(base["params"]))


def apply_meep_falsifier(meep_params: dict, name: str) -> dict:
    bad = dict(meep_params)
    if name == "no_2pi":
        bad["frequency"] = meep_params["frequency"] * TWO_PI
    elif name == "gamma_half":
        bad["gamma"] = meep_params["gamma"] / 2.0
    else:
        raise ValueError(name)
    return bad


# ---------------------------------------------------------------------------
# Windows (§3, §4)
# ---------------------------------------------------------------------------

def gated_mask(freqs_hz) -> np.ndarray:
    f = np.asarray(freqs_hz, dtype=float)
    return (f >= BAND_GATED_HZ[0]) & (f <= BAND_GATED_HZ[1])


def analytic_rt(freqs_hz, model: str, params: dict):
    eps = de.eps_analytic(freqs_hz, model, params)
    return de.tmm_slab_rt(freqs_hz, eps, D_SLAB_M)


def ade_window(freqs_hz, model: str, params: dict, dt: float):
    """W_ADE,R(f), W_ADE,T(f) = |TMM(eps_num) - TMM(eps)| at timestep dt."""
    R, T = analytic_rt(freqs_hz, model, params)
    eps_n = de.eps_numerical_ade(freqs_hz, model, params, dt)
    Rn, Tn = de.tmm_slab_rt(freqs_hz, eps_n, D_SLAB_M)
    return np.abs(Rn - R), np.abs(Tn - T), Rn, Tn


def meep_windows(freqs_hz, model: str, params: dict, meep_params: dict, dt_meep_s: float):
    """(W_ADE,meep,R, W_ADE,meep,T, W_map,R, W_map,T) per bin.

    W_map is the Debye -> overdamped-Lorentz mapping residual through the
    TMM (zero for Lorentz/Drude); W_ADE,meep is Meep's own discrete-time
    error relative to the (mapped) continuous target it is asked to realize.
    """
    R, T = analytic_rt(freqs_hz, model, params)
    if model == "debye":
        fn = meep_params["debye_map"]["fn_hz"]
        eps_target = de.eps_debye_mapped_target(freqs_hz, params, fn_debye_map_hz=fn)
    else:
        eps_target = de.eps_analytic(freqs_hz, model, params)
    Rt, Tt = de.tmm_slab_rt(freqs_hz, eps_target, D_SLAB_M)
    eps_num = de.eps_numerical_meep(freqs_hz, meep_params, dt_meep_s)
    Rn, Tn = de.tmm_slab_rt(freqs_hz, eps_num, D_SLAB_M)
    return np.abs(Rn - Rt), np.abs(Tn - Tt), np.abs(Rt - R), np.abs(Tt - T)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def _f(x):
    return float(x)


def evaluate_e2(freqs_hz, R_rfx, T_rfx, model: str, params: dict, dt: float,
                *, tail: dict | None = None) -> dict:
    """E2 gates G1 (per-bin), G2 (band-mean), G3 (witnesses) for one arm.

    ``freqs_hz`` are the masked rfx bins (cv04's mask); the gated subset is
    the 4-10 GHz band. Returns a JSON-ready dict; arrays as lists.
    """
    f = np.asarray(freqs_hz, dtype=float)
    R_rfx = np.asarray(R_rfx, dtype=float)
    T_rfx = np.asarray(T_rfx, dtype=float)
    g = gated_mask(f)
    R_an, T_an = analytic_rt(f, model, params)
    w_ade_R, w_ade_T, R_ade, T_ade = ade_window(f, model, params, dt)
    dR = np.abs(R_rfx - R_an)
    dT = np.abs(T_rfx - T_an)
    win_R = W_BIN + w_ade_R
    win_T = W_BIN + w_ade_T
    g1_R = bool(np.all(dR[g] <= win_R[g]))
    g1_T = bool(np.all(dT[g] <= win_T[g]))
    mean_win_R = W_MEAN_R + _f(np.mean(w_ade_R[g]))
    mean_win_T = W_MEAN_T + _f(np.mean(w_ade_T[g]))
    g2_R = bool(np.mean(dR[g]) <= mean_win_R)
    g2_T = bool(np.mean(dT[g]) <= mean_win_T)
    closure = R_rfx + T_rfx - 1.0
    g3_pass = bool(np.all(closure <= CONS_MAX_LIMIT))
    g3_tail = bool(tail["ok"]) if tail is not None else None
    out = {
        "model": model, "params": dict(params), "dt_s": dt,
        "freqs_hz": f.tolist(), "gated": g.tolist(),
        "R_rfx": R_rfx.tolist(), "T_rfx": T_rfx.tolist(),
        "R_tmm": R_an.tolist(), "T_tmm": T_an.tolist(),
        "R_tmm_ade": R_ade.tolist(), "T_tmm_ade": T_ade.tolist(),
        "dR": dR.tolist(), "dT": dT.tolist(),
        "window_R": win_R.tolist(), "window_T": win_T.tolist(),
        "w_ade_R": w_ade_R.tolist(), "w_ade_T": w_ade_T.tolist(),
        "max_dR_gated": _f(dR[g].max()), "max_dT_gated": _f(dT[g].max()),
        "mean_dR_gated": _f(dR[g].mean()), "mean_dT_gated": _f(dT[g].mean()),
        "mean_window_R": mean_win_R, "mean_window_T": mean_win_T,
        "worst_bin_R_hz": _f(f[g][np.argmax(dR[g])]),
        "worst_bin_T_hz": _f(f[g][np.argmax(dT[g])]),
        "n_bins_gated": int(g.sum()),
        "n_bins_R_over_window": int(np.sum(dR[g] > win_R[g])),
        "n_bins_T_over_window": int(np.sum(dT[g] > win_T[g])),
        "max_RT_closure_masked": _f(closure.max()),
        "tail": tail,
        "gates": {"G1_R": g1_R, "G1_T": g1_T, "G2_R": g2_R, "G2_T": g2_T,
                  "G3_passivity": g3_pass, "G3_tail": g3_tail},
    }
    out["e2_ok"] = bool(all(v for v in out["gates"].values() if v is not None))
    return out


def evaluate_e4(e2: dict, meep_doc: dict) -> dict:
    """E4 gates G4 (Meep vs TMM) and G5 (rfx vs Meep) on the rfx bin grid.

    ``meep_doc`` is the Meep leg JSON (``freqs_hz``, ``R``, ``T``,
    ``dt_meep_s``, ``meep_params``).
    """
    f = np.asarray(e2["freqs_hz"], dtype=float)
    g = np.asarray(e2["gated"], dtype=bool)
    model, params = e2["model"], e2["params"]
    fm = np.asarray(meep_doc["freqs_hz"], dtype=float)
    covers = bool(fm.min() <= BAND_GATED_HZ[0] and fm.max() >= BAND_GATED_HZ[1])
    R_m = np.interp(f, fm, np.asarray(meep_doc["R"], dtype=float))
    T_m = np.interp(f, fm, np.asarray(meep_doc["T"], dtype=float))
    R_an = np.asarray(e2["R_tmm"]); T_an = np.asarray(e2["T_tmm"])
    R_x = np.asarray(e2["R_rfx"]); T_x = np.asarray(e2["T_rfx"])
    w_ade_R = np.asarray(e2["w_ade_R"]); w_ade_T = np.asarray(e2["w_ade_T"])
    # The Meep-side windows are derived from the DECLARED material mapped by
    # to_meep(...), never from the parameters the Meep JSON reports: a wrong
    # mapping (F3) must not be allowed to widen its own window. The JSON's
    # own meep_params are recorded for audit only.
    doc_mp = meep_doc["meep_params"]
    fn_map = (doc_mp.get("debye_map") or {}).get("fn_hz", DEBYE_MEEP_MAP_FN_HZ)
    declared_mp = de.to_meep(model, params, a_m=float(doc_mp.get("a_m", MEEP_A_M)),
                             fn_debye_map_hz=float(fn_map))
    wm_ade_R, wm_ade_T, w_map_R, w_map_T = meep_windows(
        f, model, params, declared_mp, float(meep_doc["dt_meep_s"]))
    dR_mt = np.abs(R_m - R_an); dT_mt = np.abs(T_m - T_an)
    win4_R = W_BIN + wm_ade_R + w_map_R
    win4_T = W_BIN + wm_ade_T + w_map_T
    mean4_R = W_MEAN_R + _f(np.mean(wm_ade_R[g] + w_map_R[g]))
    mean4_T = W_MEAN_T + _f(np.mean(wm_ade_T[g] + w_map_T[g]))
    dR_xm = np.abs(R_x - R_m); dT_xm = np.abs(T_x - T_m)
    win5_R = 2 * W_BIN + w_ade_R + wm_ade_R + w_map_R
    win5_T = 2 * W_BIN + w_ade_T + wm_ade_T + w_map_T
    mean5_R = 2 * W_MEAN_R + _f(np.mean(w_ade_R[g] + wm_ade_R[g] + w_map_R[g]))
    mean5_T = 2 * W_MEAN_T + _f(np.mean(w_ade_T[g] + wm_ade_T[g] + w_map_T[g]))
    gates = {
        "band_covered": covers,
        "G4_R": bool(covers and np.all(dR_mt[g] <= win4_R[g])),
        "G4_T": bool(covers and np.all(dT_mt[g] <= win4_T[g])),
        "G4_mean_R": bool(covers and np.mean(dR_mt[g]) <= mean4_R),
        "G4_mean_T": bool(covers and np.mean(dT_mt[g]) <= mean4_T),
        "G5_R": bool(covers and np.all(dR_xm[g] <= win5_R[g])),
        "G5_T": bool(covers and np.all(dT_xm[g] <= win5_T[g])),
        "G5_mean_R": bool(covers and np.mean(dR_xm[g]) <= mean5_R),
        "G5_mean_T": bool(covers and np.mean(dT_xm[g]) <= mean5_T),
    }
    return {
        "present": True,
        "source": meep_doc.get("_source"),
        "dt_meep_s": float(meep_doc["dt_meep_s"]),
        "meep_params_reported": doc_mp,
        "meep_params_declared": declared_mp,
        "precheck": meep_doc.get("precheck"),
        "R_meep": R_m.tolist(), "T_meep": T_m.tolist(),
        "dR_meep_tmm": dR_mt.tolist(), "dT_meep_tmm": dT_mt.tolist(),
        "dR_rfx_meep": dR_xm.tolist(), "dT_rfx_meep": dT_xm.tolist(),
        "window4_R": win4_R.tolist(), "window4_T": win4_T.tolist(),
        "window5_R": win5_R.tolist(), "window5_T": win5_T.tolist(),
        "w_ade_meep_R": wm_ade_R.tolist(), "w_ade_meep_T": wm_ade_T.tolist(),
        "w_map_R": w_map_R.tolist(), "w_map_T": w_map_T.tolist(),
        "max_dR_meep_tmm_gated": _f(dR_mt[g].max()), "max_dT_meep_tmm_gated": _f(dT_mt[g].max()),
        "mean_dR_meep_tmm_gated": _f(dR_mt[g].mean()), "mean_dT_meep_tmm_gated": _f(dT_mt[g].mean()),
        "max_dR_rfx_meep_gated": _f(dR_xm[g].max()), "max_dT_rfx_meep_gated": _f(dT_xm[g].max()),
        "mean_dR_rfx_meep_gated": _f(dR_xm[g].mean()), "mean_dT_rfx_meep_gated": _f(dT_xm[g].mean()),
        "mean_window4_R": mean4_R, "mean_window4_T": mean4_T,
        "mean_window5_R": mean5_R, "mean_window5_T": mean5_T,
        "gates": gates,
        "e4_ok": bool(all(gates.values())),
    }


def meep_json_name(arm: str, falsifier: str | None = None) -> str:
    return f"meep_{arm}.json" if falsifier is None else f"meep_{arm}__falsifier_{falsifier}.json"


def rfx_json_name(falsifier: str | None = None) -> str:
    return "rfx.json" if falsifier is None else f"rfx__falsifier_{falsifier}.json"
