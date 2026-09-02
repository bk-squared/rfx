"""cv23 lossy slab -- declared arms, derived windows (R, T and the absorption A),
gates, falsifiers.

Pure numpy on top of ``cv22_dispersive_gates`` (the rig, the cv04-derived
windows, the record-length derivation, the tail fit) and ``dispersive_eps``
(the "conductive" model, its discrete-time form and the Meep
``D_conductivity`` mapping). Shared by the case script
(``validation/crossval/23_lossy_slab_fresnel.py``), the Meep leg
(``scripts/crossval/meep_cv23_lossy_slab.py``) and the gate/mapping tests so
the pre-declared numbers exist in exactly one place.

Everything numeric here is fixed by
``docs/design_notes/20260902_cv23_lossy_slab_predeclaration.md``; change the
note (append-only) before changing a number here.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv22_dispersive_gates as G  # noqa: E402
import dispersive_eps as de  # noqa: E402
from tests._gate_policy import gate_from_envelope  # noqa: E402

MODEL = "conductive"
CASE_ID = "23_lossy_slab_fresnel"
SCHEMA = "cv23-lossy-slab/v1"
RESULTS_DIRNAME = "_23_lossy_results"

# Rig constants are cv22's (note section 1); re-exported, never restated.
DX_M = G.DX_M
D_SLAB_M = G.D_SLAB_M
MEEP_A_M = G.MEEP_A_M
MEEP_COURANT = G.MEEP_COURANT
MEEP_PRIMARY_RESOLUTION = G.MEEP_PRIMARY_RESOLUTION
# Note section 13 (round 3): the Meep primary reference per arm -- 80 px/cm on
# tand0p1 (its first-order thickness term is inside the window by 1.28x
# there, one rung under, stated), 40 px/cm on tand1 / tand3.
MEEP_PRIMARY_RESOLUTION_BY_ARM = {"tand0p1": 80, "tand1": 40, "tand3": 40}
# Note section 13: the rfx primary recipe per arm -- dx/2 for tand3, whose
# |n| k0 dx = 0.64 at the band top puts the lattice term outside a window
# derived at |n| = 2; dx for tand0p1 and tand1 (inside). Resolution, not
# tolerance: the windows are unchanged.
ARM_DX_DIV = {"tand0p1": 1, "tand1": 1, "tand3": 2}
MEEP_LADDER_RESOLUTIONS = G.MEEP_LADDER_RESOLUTIONS
MEEP_LADDER_RESOLUTIONS_R2 = (10, 20, 40, 80)   # note section 12: the res-80 rung
RFX_DX_LADDER = (1, 2, 4)                        # note section 12: --dx-div K per arm
MEEP_DIAGNOSTIC_TAGS = ("res40_thin1", "res40_thin_half", "res40_shift_half")   # note section 13.2
MEEP_EPS_AVERAGING = False     # note section 1: conductivity is "not compatible with
                               # subpixel averaging" (Meep docs); faces sit on pixel
                               # boundaries at 10/20/40 px/cm anyway

# ---------------------------------------------------------------------------
# Material arms (note section 2): eps' = 4 (cv04's slab), sigma from tan delta
# at the band centre; tan delta = sigma/(omega eps0 eps') is then ~ 1/f.
# ---------------------------------------------------------------------------
EPS_R_SLAB = 4.0
F_CENTRE_HZ = 7.0e9
ARM_TAN_DELTA = {"tand0p1": 0.1, "tand1": 1.0, "tand3": 3.0}
ARM_ORDER = ("tand0p1", "tand1", "tand3")
# Which construction feeds materials.sigma (note section 2, "material paths"):
#   direct : init_materials(grid.shape) + .at[slab].set (cv22/cv04's construction)
#   api    : Simulation.add_material(..., sigma=) + Simulation.add(Box, material=)
#            + Simulation._assemble_materials(grid) -- the documented user path
MATERIALS_PATH = {"tand0p1": "direct", "tand1": "api", "tand3": "api"}
ARMS = {
    arm: {"model": MODEL,
          "params": {"eps_inf": EPS_R_SLAB,
                     "sigma": de.sigma_from_tan_delta(td, F_CENTRE_HZ, EPS_R_SLAB)},
          "materials_path": MATERIALS_PATH[arm]}
    for arm, td in ARM_TAN_DELTA.items()
}
API_MATERIAL_NAME = "lossy_slab"

# ---------------------------------------------------------------------------
# Windows (note section 4): the R/T windows are cv22's (same rig); A's are
# the triangle-inequality sums (DECLARED gate). The closure-derived tighter A
# window is REPORTED (A_tight_ok), never gated.
# ---------------------------------------------------------------------------
W_BIN = G.W_BIN                    # 0.074
W_MEAN_R = G.W_MEAN_R              # 0.010
W_MEAN_T = G.W_MEAN_T              # 0.017
W_BIN_A = 2.0 * G.W_BIN            # 0.148   |dA| <= |dR| + |dT|
W_MEAN_A = G.W_MEAN_R + G.W_MEAN_T  # 0.027
# tests/fixtures/golden_workflows/multilayer_fresnel.json::expected_metrics[id=mean_energy_closure_error].observed_baseline
CV04_MEAN_CLOSURE = 0.0091
W_BIN_A_TIGHT = gate_from_envelope(G.CV04_ENVELOPE["per_bin_max_RT_closure"], quantum=1000)   # 0.074
W_MEAN_A_TIGHT = gate_from_envelope(CV04_MEAN_CLOSURE, quantum=1000)                         # 0.014

# ---------------------------------------------------------------------------
# Falsifiers (note section 6)
# ---------------------------------------------------------------------------
FALSIFIERS = {}
for _arm in ARM_ORDER:
    FALSIFIERS[f"{_arm}_sigma_x1p5"] = (_arm, "F1: sigma -> 1.5 sigma",
                                        lambda p: {**p, "sigma": 1.5 * p["sigma"]})
    FALSIFIERS[f"{_arm}_sigma_zero"] = (_arm, "F2: sigma -> 0 (cv04's lossless slab, judged as lossy)",
                                        lambda p: {**p, "sigma": 0.0})
FALSIFIERS["tand0p1_sigma_neg"] = ("tand0p1", "F4 passivity: sigma -> -sigma (gain slab, R + T > 1)",
                                   lambda p: {**p, "sigma": -p["sigma"]})
PASSIVITY_FALSIFIER = "tand0p1_sigma_neg"
# Meep-leg falsifiers (F3): the wrong sigma_D scaling, on the tand1 arm.
MEEP_FALSIFIER_ARM = "tand1"
MEEP_FALSIFIERS = {
    "sigma_2pi": "F3: sigma_D x 2 pi (sigma_D taken in units of 2 pi c/a)",
    "sigma_no_eps": "F3: eps' division dropped (sigma applied to E, not D)",
}
MEEP_FALSIFIER_CASE_NAMES = {f"meep_{MEEP_FALSIFIER_ARM}_{k}": k for k in MEEP_FALSIFIERS}


def apply_falsifier(name: str):
    """Return (arm, defective params) for an rfx-side falsifier."""
    arm, _desc, fn = FALSIFIERS[name]
    return arm, fn(dict(ARMS[arm]["params"]))


def apply_meep_falsifier(meep_params: dict, name: str) -> dict:
    bad = dict(meep_params)
    if name == "sigma_2pi":
        bad["D_conductivity"] = meep_params["D_conductivity"] * G.TWO_PI
    elif name == "sigma_no_eps":
        bad["D_conductivity"] = meep_params["D_conductivity"] * meep_params["eps_inf"]
    else:
        raise ValueError(name)
    return bad


# ---------------------------------------------------------------------------
# Analytic R, T, A and the sigma-update window term (note section 3)
# ---------------------------------------------------------------------------

def analytic_rta(freqs_hz, params: dict):
    R, T = G.analytic_rt(freqs_hz, MODEL, params)
    return R, T, 1.0 - R - T


def sigma_window(freqs_hz, params: dict, dt: float):
    """(W_sig,R, W_sig,T, W_sig,A, R_num, T_num, A_num): |TMM(eps_num) - TMM(eps)|
    at timestep dt, eps_num = eps' - j sigma (x/tan x)/(omega eps0)."""
    wR, wT, Rn, Tn = G.ade_window(freqs_hz, MODEL, params, dt)
    R, T, A = analytic_rta(freqs_hz, params)
    An = 1.0 - Rn - Tn
    return wR, wT, np.abs(An - A), Rn, Tn, An


def meep_sigma_window_A(freqs_hz, params: dict, meep_params: dict, dt_meep_s: float):
    """W_sig,meep,A(f): Meep's own discrete-time term in A (no mapping residual:
    D_conductivity realizes eps'(1 + i sigma_D/omega) exactly)."""
    _R, _T, A = analytic_rta(freqs_hz, params)
    eps_num = de.eps_numerical_meep(freqs_hz, meep_params, dt_meep_s)
    Rn, Tn = de.tmm_slab_rt(freqs_hz, eps_num, D_SLAB_M)
    return np.abs((1.0 - Rn - Tn) - A)


def derive_record_length(params: dict, dt: float, **kw) -> dict:
    return G.derive_record_length(MODEL, params, dt, **kw)


# ---------------------------------------------------------------------------
# Gate evaluation: cv22's R/T evaluators plus the absorption observable
# ---------------------------------------------------------------------------

def _f(x):
    return float(x)


def evaluate_e2(freqs_hz, R_rfx, T_rfx, params: dict, dt: float, *, tail: dict | None = None,
                dx: float | None = None) -> dict:
    """E2 gates G1 (per-bin R, T, A), G2 (band-mean R, T, A), G3 (witnesses).
    With ``dx`` given, the exact Yee-lattice solution at (dx, dt) is added as
    a REPORTED witness (``lattice``: W_lat per bin and |rfx - lattice|; note
    section 13) -- it enters no gate."""
    out = G.evaluate_e2(freqs_hz, R_rfx, T_rfx, MODEL, params, dt, tail=tail)
    f = np.asarray(freqs_hz, dtype=float)
    g = np.asarray(out["gated"], dtype=bool)
    R_x = np.asarray(out["R_rfx"]); T_x = np.asarray(out["T_rfx"])
    R_an = np.asarray(out["R_tmm"]); T_an = np.asarray(out["T_tmm"])
    A_x = 1.0 - R_x - T_x
    A_an = 1.0 - R_an - T_an
    A_ade = 1.0 - np.asarray(out["R_tmm_ade"]) - np.asarray(out["T_tmm_ade"])
    w_A = np.abs(A_ade - A_an)
    dA = np.abs(A_x - A_an)
    win_A = W_BIN_A + w_A
    mean_win_A = W_MEAN_A + _f(np.mean(w_A[g]))
    g1_A = bool(np.all(dA[g] <= win_A[g]))
    g2_A = bool(np.mean(dA[g]) <= mean_win_A)
    # Reported, not gated (note section 4): the closure-derived tighter window.
    tight = bool(np.all(dA[g] <= W_BIN_A_TIGHT + w_A[g]) and np.mean(dA[g]) <= W_MEAN_A_TIGHT + np.mean(w_A[g]))
    gates = dict(out["gates"])
    out["gates"] = {"G1_R": gates["G1_R"], "G1_T": gates["G1_T"], "G1_A": g1_A,
                    "G2_R": gates["G2_R"], "G2_T": gates["G2_T"], "G2_A": g2_A,
                    "G3_passivity": gates["G3_passivity"], "G3_tail": gates["G3_tail"]}
    out.update({
        "A_rfx": A_x.tolist(), "A_tmm": A_an.tolist(), "A_tmm_ade": A_ade.tolist(),
        "dA": dA.tolist(), "window_A": win_A.tolist(), "w_ade_A": w_A.tolist(),
        "max_dA_gated": _f(dA[g].max()), "mean_dA_gated": _f(dA[g].mean()),
        "mean_window_A": mean_win_A, "worst_bin_A_hz": _f(f[g][np.argmax(dA[g])]),
        "n_bins_A_over_window": int(np.sum(dA[g] > win_A[g])),
        "A_tight_ok": tight, "A_tight_windows": {"per_bin": W_BIN_A_TIGHT, "mean": W_MEAN_A_TIGHT},
        "tan_delta_gated": de.tan_delta_of(f[g], params).tolist(),
    })
    if dx is not None:
        Rl, Tl, Al = lattice_rta(f, params, float(dx), dt)
        wl_R, wl_T, wl_A = np.abs(Rl - R_an), np.abs(Tl - T_an), np.abs(Al - A_an)
        rl_R, rl_T, rl_A = np.abs(R_x - Rl), np.abs(T_x - Tl), np.abs(A_x - Al)
        out["lattice"] = {
            "dx_m": float(dx), "dt_s": float(dt), "gated": False,
            "R_lattice": Rl.tolist(), "T_lattice": Tl.tolist(), "A_lattice": Al.tolist(),
            "W_lat_R": wl_R.tolist(), "W_lat_T": wl_T.tolist(), "W_lat_A": wl_A.tolist(),
            "mean_W_lat_R_gated": _f(wl_R[g].mean()), "mean_W_lat_T_gated": _f(wl_T[g].mean()),
            "mean_W_lat_A_gated": _f(wl_A[g].mean()),
            "mean_dR_lattice_gated": _f(rl_R[g].mean()), "max_dR_lattice_gated": _f(rl_R[g].max()),
            "mean_dT_lattice_gated": _f(rl_T[g].mean()), "max_dT_lattice_gated": _f(rl_T[g].max()),
            "mean_dA_lattice_gated": _f(rl_A[g].mean()), "max_dA_lattice_gated": _f(rl_A[g].max()),
        }
    out["e2_ok"] = bool(all(v for v in out["gates"].values() if v is not None))
    return out


def evaluate_e4(e2: dict, meep_doc: dict) -> dict:
    """E4 gates G4 (Meep vs TMM) and G5 (rfx vs Meep) on R, T and A."""
    out = G.evaluate_e4(e2, meep_doc)
    f = np.asarray(e2["freqs_hz"], dtype=float)
    g = np.asarray(e2["gated"], dtype=bool)
    params = e2["params"]
    R_m = np.asarray(out["R_meep"]); T_m = np.asarray(out["T_meep"])
    A_m = 1.0 - R_m - T_m
    A_an = np.asarray(e2["A_tmm"]); A_x = np.asarray(e2["A_rfx"])
    w_A = np.asarray(e2["w_ade_A"])
    declared_mp = de.to_meep(MODEL, params, a_m=MEEP_A_M)
    wm_A = meep_sigma_window_A(f, params, declared_mp, float(meep_doc["dt_meep_s"]))
    dA_mt = np.abs(A_m - A_an)
    dA_xm = np.abs(A_x - A_m)
    win4_A = W_BIN_A + wm_A
    mean4_A = W_MEAN_A + _f(np.mean(wm_A[g]))
    win5_A = 2 * W_BIN_A + w_A + wm_A
    mean5_A = 2 * W_MEAN_A + _f(np.mean(w_A[g] + wm_A[g]))
    covers = bool(out["gates"]["band_covered"])
    gates = dict(out["gates"])
    gates.update({
        "G4_A": bool(covers and np.all(dA_mt[g] <= win4_A[g])),
        "G4_mean_A": bool(covers and np.mean(dA_mt[g]) <= mean4_A),
        "G5_A": bool(covers and np.all(dA_xm[g] <= win5_A[g])),
        "G5_mean_A": bool(covers and np.mean(dA_xm[g]) <= mean5_A),
    })
    out.update({
        "A_meep": A_m.tolist(), "dA_meep_tmm": dA_mt.tolist(), "dA_rfx_meep": dA_xm.tolist(),
        "window4_A": win4_A.tolist(), "window5_A": win5_A.tolist(), "w_ade_meep_A": wm_A.tolist(),
        "max_dA_meep_tmm_gated": _f(dA_mt[g].max()), "mean_dA_meep_tmm_gated": _f(dA_mt[g].mean()),
        "max_dA_rfx_meep_gated": _f(dA_xm[g].max()), "mean_dA_rfx_meep_gated": _f(dA_xm[g].mean()),
        "mean_window4_A": mean4_A, "mean_window5_A": mean5_A,
        "eps_averaging_reported": meep_doc.get("eps_averaging"),
        "gates": gates,
    })
    out["e4_ok"] = bool(all(gates.values()))
    return out


def meep_json_name(arm: str, falsifier: str | None = None) -> str:
    return G.meep_json_name(arm, falsifier)


def rfx_json_name(falsifier: str | None = None) -> str:
    return G.rfx_json_name(falsifier)




# ---------------------------------------------------------------------------
# Round 2 (note section 12): the derived lattice term and Meep's thickness excess
# ---------------------------------------------------------------------------

def lattice_rta(freqs_hz, params: dict, dx: float, dt: float):
    """R, T, A of the exact 1-D Yee lattice for the staircase slab at (dx, dt)."""
    R, T = de.yee_lattice_slab_rt(freqs_hz, float(params["eps_inf"]), float(params["sigma"]), D_SLAB_M, dx, dt)
    return R, T, 1.0 - R - T


def lattice_window(freqs_hz, params: dict, dx: float, dt: float):
    """W_lat,{R,T,A}(f) = |lattice - TMM|: the rig's own discretization of the
    slab (bulk dispersion + node interface + sigma warp), a priori. NOT a gate
    term in round 2 (note section 12); evaluated and reported."""
    R, T, A = analytic_rta(freqs_hz, params)
    Rl, Tl, Al = lattice_rta(freqs_hz, params, dx, dt)
    return np.abs(Rl - R), np.abs(Tl - T), np.abs(Al - A)


def meep_thickness_excess_rta(freqs_hz, params: dict, resolution: int, cells: float = 1.0):
    """The note-section-12 hypothesis for Meep's first-order term: the block
    realizes d + cells * a/res (E nodes inclusive at both faces). R, T, A of
    the transfer matrix at that thickness."""
    eps = de.eps_analytic(freqs_hz, MODEL, params)
    R, T = de.tmm_slab_rt(freqs_hz, eps, D_SLAB_M + cells * MEEP_A_M / resolution)
    return R, T, 1.0 - R - T


def meep_ladder_summary(results_dir: str, rfx_doc: dict, resolutions=MEEP_LADDER_RESOLUTIONS_R2) -> dict:
    """Meep-vs-TMM per resolution and the measured order per doubling (R, T).
    cv22's summary on cv23's artifacts (rungs 10/20/40 and, from round 2, 80);
    evidence of Meep's convergence, not a window term."""
    summ = G.meep_ladder_summary(results_dir, rfx_doc, resolutions=resolutions)
    summ["schema"] = "cv23-meep-ladder/v1"
    # Note sections 12-14: the three Meep node-count discriminators on tand0p1
    # at 40 px/cm (thin by one cell, thin by half a cell, centre shifted half
    # a pixel), evaluated with the same E4 evaluator so their Meep-vs-TMM
    # means are citable by key.
    import json as _json
    diag = {}
    arm = "tand0p1"
    if arm in rfx_doc["arms"]:
        ad = rfx_doc["arms"][arm]
        e2 = evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], ad["params"], ad["dt_s"], tail=ad["tail"])
        for tag in MEEP_DIAGNOSTIC_TAGS:
            p = os.path.join(results_dir, f"meep_{arm}__{tag}.json")
            if not os.path.isfile(p):
                continue
            with open(p) as fh:
                md = _json.load(fh)
            e4 = evaluate_e4(e2, md)
            diag[tag] = {"resolution": md["resolution"], "d_slab_m": md.get("d_slab_m"),
                         "thickness_offset_cells": md.get("thickness_offset_cells", 0),
                         "center_offset_cells": md.get("center_offset_cells", 0),
                         "mean_dR_meep_tmm_gated": e4["mean_dR_meep_tmm_gated"],
                         "mean_dT_meep_tmm_gated": e4["mean_dT_meep_tmm_gated"],
                         "mean_dA_meep_tmm_gated": e4["mean_dA_meep_tmm_gated"],
                         "G4_mean_R": e4["gates"]["G4_mean_R"]}
    summ["diagnostics"] = {arm: diag}
    return summ

