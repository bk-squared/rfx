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

# ---------------------------------------------------------------------------
# Round-3 recipe (§12): the record length comes from the slab's OWN ring-down,
# not from cv04's 719-step CPML rule. Settling witness bar = -40 dB of the
# incident peak (amplitude 1e-2), the bar the other cases use.
# ---------------------------------------------------------------------------
RECIPE_R3 = "r3"
RECIPE_CV04 = "cv04"
NX_INTERIOR_R3 = 1000          # derived in derive_record_length(): the smallest
                               # round number whose CPML round-trip gate exceeds
                               # the longest derived record (Lorentz)
SETTLING_LIMIT = 1e-2          # -40 dB, amplitude, last TAIL_WINDOW steps
TFSF_MARGIN = 5
PROBE_OFFSET_CELLS = 30
SRC_T0_OVER_TAU = 3.0          # rfx.sources.tfsf: src_t0 = 3 tau for differentiated_gaussian
PULSE_END_ARG_40DB = 2.5255070008312575   # 2a e^{-a^2} = 1e-2 x peak (peak at a = 1/sqrt2)
MEEP_PRIMARY_RESOLUTION = 40   # §12: the converged Meep reference (first-order ladder measured in r2)
# §13 (round 4): the ring-down search covers the incident band where the
# differentiated-Gaussian amplitude is >= RING_W_MIN of its peak (not only the
# gated band): the r3 Debye tail decayed at <= 1.36e10/s, the etalon rate of the
# 1-2.5 GHz content where Debye's per-pass absorption is weakest. Each component
# starts at most at its incident weight w(f), so it needs ln(100 w)/rate, not
# ln(100)/rate. The witness is then ADAPTIVE: extend the record in
# RECORD_EXTEND_STEPS while the -40 dB bar is not met, and grow the box by
# NX_GROW_CELLS when the CPML gate is reached (never clip).
RING_W_MIN = 0.5
RING_F_MAX_HZ = MASK_F_HI_HZ
RECORD_EXTEND_STEPS = 100
NX_GROW_CELLS = 200
TAIL_ENVELOPE_STEPS = 300      # stored in the artifact so the decay can be fitted offline
MEEP_LADDER_RESOLUTIONS = (10, 20, 40)

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
    # Every Meep-side window input is the DECLARED constant; the reported
    # values are checked against it, never used (review finding 2: the first
    # version read fn_hz from the JSON's debye_map to size W_map).
    if model == "debye":
        fn_reported = (doc_mp.get("debye_map") or {}).get("fn_hz")
        if fn_reported is None or abs(float(fn_reported) - DEBYE_MEEP_MAP_FN_HZ) > 1e-6 * DEBYE_MEEP_MAP_FN_HZ:
            raise ValueError(f"Meep Debye leg reports fn_hz={fn_reported!r}; the declared mapping is "
                             f"{DEBYE_MEEP_MAP_FN_HZ} (window inputs are never taken from the report)")
    if abs(float(doc_mp.get("a_m", MEEP_A_M)) - MEEP_A_M) > 1e-12:
        raise ValueError(f"Meep leg reports a_m={doc_mp.get('a_m')!r}; declared {MEEP_A_M}")
    declared_mp = de.to_meep(model, params, a_m=MEEP_A_M, fn_debye_map_hz=DEBYE_MEEP_MAP_FN_HZ)
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
    # The leg's own 1e-9 pre-run mapping check is a gate, not a record
    # (review finding 2). True on every committed primary; the two Meep
    # falsifier legs carry passed=false by design and fail here as well.
    precheck_passed = bool((meep_doc.get("precheck") or {}).get("passed", False))
    gates = {
        "precheck_passed": precheck_passed,
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


# ---------------------------------------------------------------------------
# Round-3 record-length derivation (§12) -- physics, no measurement enters.
# ---------------------------------------------------------------------------

def rig_cells(nx_interior: int, dx_div: int = 1):
    """Cell bookkeeping of the cv04 rig (Grid adds 2*n_cpml + 1 cells)."""
    K = int(dx_div)
    n_cpml = N_CPML * K
    nx = int(nx_interior) * K + 2 * n_cpml + 1
    half = int(D_SLAB_M / (2 * DX_M / K))
    slab_lo = nx // 2 - half
    slab_hi = nx // 2 + half
    x_lo = n_cpml + TFSF_MARGIN * K
    probe_refl = slab_lo - PROBE_OFFSET_CELLS * K
    probe_trans = slab_hi + PROBE_OFFSET_CELLS * K
    return {"nx": nx, "n_cpml": n_cpml, "slab_lo": slab_lo, "slab_hi": slab_hi,
            "x_lo": x_lo, "probe_refl": probe_refl, "probe_trans": probe_trans}


def incident_amplitude_rel(f_hz):
    """Amplitude spectrum of the rig's differentiated-Gaussian incident pulse,
    relative to its peak: |S(f)| ∝ f exp(-(pi f tau)^2), tau = 1/(pi f0 bw)."""
    tau = 1.0 / (math.pi * TFSF_F0_HZ * TFSF_BW)
    f = np.asarray(f_hz, dtype=float)
    s = f * np.exp(-(math.pi * f * tau) ** 2)
    peak = (1.0 / (math.sqrt(2.0) * math.pi * tau)) * math.exp(-0.5)
    return s / peak


def ring_band_hz():
    """[f_lo, RING_F_MAX_HZ]: the incident band with amplitude >= RING_W_MIN of peak."""
    f = np.linspace(1e7, TFSF_F0_HZ, 20000)
    w = incident_amplitude_rel(f)
    f_lo = float(f[np.argmax(w >= RING_W_MIN)])
    return f_lo, RING_F_MAX_HZ


def slab_ringdown_rates(model: str, params: dict):
    """Amplitude decay rates (1/s) of the slab's own ring-down over the incident
    ring band: the material pole (Debye 1/tau, Lorentz delta, Drude gamma/2)
    and the etalon round-trip, rho = |r|^2 exp(-2 k0 Im(n) d) per
    t_rt = 2 Re(n) d / c, each component weighted by its incident amplitude
    w(f): a component starting at w needs ln(100 w)/rate to reach -40 dB. The
    slowest entry is the one with the largest ln(100 w)/rate."""
    f_lo, f_hi = ring_band_hz()
    f = np.linspace(f_lo, f_hi, 1401)
    eps = de.eps_analytic(f, model, params)
    n = np.sqrt(eps)
    n = np.where(n.imag > 0, -n, n)         # decaying branch in e^{+jwt}
    k0 = TWO_PI * f / C0
    r = (1 - n) / (1 + n)
    rho = np.abs(r) ** 2 * np.exp(-2 * k0 * (-n.imag) * D_SLAB_M)
    t_rt = 2 * np.abs(n.real) * D_SLAB_M / C0
    rate_et = -np.log(rho) / t_rt
    if model == "debye":
        rate_mat = 1.0 / params["tau"]
    elif model == "lorentz":
        rate_mat = float(params["delta"])
    else:
        rate_mat = float(params["gamma"]) / 2.0
    w = incident_amplitude_rel(f)
    rate = np.minimum(rate_et, rate_mat)
    t_need = np.log(100.0 * w) / rate       # seconds to -40 dB of the incident peak
    i = int(np.argmax(t_need))
    return {"rate_material_1_s": float(rate_mat), "rate_etalon_slowest_1_s": float(rate_et.min()),
            "f_etalon_slowest_hz": float(f[int(np.argmin(rate_et))]),
            "ring_band_hz": [f_lo, f_hi], "ring_w_min": RING_W_MIN,
            "t_ring_s": float(t_need[i]), "f_ring_hz": float(f[i]), "w_ring": float(w[i]),
            "rate_ring_1_s": float(rate[i]), "rho_etalon": float(rho[i]), "t_rt_s": float(t_rt[i])}


def derive_record_length(model: str, params: dict, dt: float, *, nx_interior: int = NX_INTERIOR_R3,
                         dx_div: int = 1) -> dict:
    """n_steps_min = n_pulse_end + n_ring + TAIL_WINDOW (all in steps):
      n_pulse_end : the differentiated-Gaussian incident has fallen to -40 dB at
                    the transmission probe: t0 + a40 tau, plus propagation from the
                    TFSF injection plane at the Courant speed;
      n_ring      : max over the incident ring band of ln(100 w(f)) / rate(f) --
                    each spectral component starts at most at its incident weight
                    w(f) and must reach 1e-2 = -40 dB of the incident peak
                    (section 13; rounds 1-3 used ln(100)/rate over the gated band);
      TAIL_WINDOW : the witness window itself sits after the ring-down.
    The CPML round-trip gate of the rig must exceed n_steps (asserted). The case
    script then EXTENDS the record adaptively while the witness is above the bar."""
    K = int(dx_div)
    dx = DX_M / K
    cells = rig_cells(nx_interior, dx_div)
    tau = 1.0 / (math.pi * TFSF_F0_HZ * TFSF_BW)
    t0 = SRC_T0_OVER_TAU * tau
    v_cells = C0 * dt / dx
    n_pulse_end = int(math.ceil((t0 + PULSE_END_ARG_40DB * tau) / dt
                                + (cells["probe_trans"] - cells["x_lo"]) / v_cells))
    rates = slab_ringdown_rates(model, params)
    rate_slow = rates["rate_ring_1_s"]
    n_ring = int(math.ceil(rates["t_ring_s"] / dt))
    tail_window = TAIL_WINDOW * K
    n_steps = n_pulse_end + n_ring + tail_window
    dist_hi = cells["nx"] - cells["n_cpml"] - cells["probe_trans"]
    dist_lo = cells["probe_refl"] - cells["n_cpml"]
    t_safe = int(min(2 * dist_hi, 2 * dist_lo) / v_cells * 0.95)
    return {"recipe": RECIPE_R3, "nx_interior": int(nx_interior) * K, "dx_div": K,
            "n_pulse_end": n_pulse_end, "n_ring": n_ring, "tail_window": tail_window,
            "n_steps": n_steps, "t_safe_cpml_steps": t_safe, "cpml_gate_ok": bool(t_safe >= n_steps),
            "settling_limit": SETTLING_LIMIT, "rate_slowest_1_s": float(rate_slow), **rates,
            "src_t0_s": t0, "src_tau_s": tau, "v_cells": float(v_cells), **cells}


def meep_ladder_summary(results_dir: str, rfx_doc: dict) -> dict:
    """Meep-vs-TMM deviation per resolution (from meep_<arm>__res<N>.json) and the
    measured convergence order per doubling. Measured-in-r2 evidence, not a
    pre-declared window term."""
    import json as _json
    out = {"schema": "cv22-meep-ladder/v1", "resolutions": list(MEEP_LADDER_RESOLUTIONS), "arms": {}}
    for arm, ad in rfx_doc["arms"].items():
        e2 = evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], ad["model"], ad["params"], ad["dt_s"],
                         tail=ad["tail"])
        rungs = {}
        for res in MEEP_LADDER_RESOLUTIONS:
            p = os.path.join(results_dir, f"meep_{arm}__res{res}.json")
            if not os.path.isfile(p):
                continue
            with open(p) as fh:
                md = _json.load(fh)
            if not md["run"]["finite"]:
                rungs[str(res)] = {"finite": False}
                continue
            e4 = evaluate_e4(e2, md)
            rungs[str(res)] = {"finite": True, "dt_meep_s": md["dt_meep_s"],
                               "mean_dR_meep_tmm_gated": e4["mean_dR_meep_tmm_gated"],
                               "mean_dT_meep_tmm_gated": e4["mean_dT_meep_tmm_gated"],
                               "max_dR_meep_tmm_gated": e4["max_dR_meep_tmm_gated"],
                               "max_dT_meep_tmm_gated": e4["max_dT_meep_tmm_gated"]}
        orders = {}
        for lo, hi in ((10, 20), (20, 40)):
            a, b = rungs.get(str(lo)), rungs.get(str(hi))
            if a and b and a.get("finite") and b.get("finite"):
                for q in ("mean_dR_meep_tmm_gated", "mean_dT_meep_tmm_gated"):
                    if b[q] > 0:
                        orders[f"order_{q[5:7]}_{lo}_{hi}"] = float(math.log2(a[q] / b[q]))
        out["arms"][arm] = {"rungs": rungs, "orders": orders}
    return out


def fit_tail_rate(env_rel, dt: float, start: int = 0):
    """Amplitude decay rate (1/s) of a stored tail envelope (relative to the
    incident peak): log-linear least squares through the running maxima of
    |signal| over TAIL_WINDOW-sized blocks (the etalon is oscillatory; the block
    maxima trace the envelope). ``start`` is the envelope index at which the
    incident pulse has ended (n_pulse_end - envelope_start_step); the fit uses
    only whole blocks that begin at or after start + TAIL_WINDOW -- one window
    of margin for the slab's own group delay of the transmitted pulse (review
    finding 1: the first r4 fit started 100 steps inside the Debye pulse, and
    the block right after n_pulse_end still holds the delayed transmitted
    pulse on the Drude arm). Returns (rate, n_blocks)."""
    e = np.asarray(env_rel, dtype=float)
    first = int(math.ceil(max(0, start + TAIL_WINDOW) / TAIL_WINDOW)) * TAIL_WINDOW
    e = e[first:]
    nb = e.size // TAIL_WINDOW
    if nb < 3:
        return float("nan"), int(nb)
    blocks = e[: nb * TAIL_WINDOW].reshape(nb, TAIL_WINDOW).max(axis=1)
    if np.any(blocks <= 0):
        return float("nan"), int(nb)
    t = (np.arange(nb) + 0.5) * TAIL_WINDOW * dt
    slope, _ = np.polyfit(t, np.log(blocks), 1)
    return float(-slope), int(nb)


def refit_tail(tail: dict, dt: float, n_steps: int, n_pulse_end: int) -> dict:
    """Recompute the fitted tail rates of an artifact's ``tail`` dict from its
    STORED envelopes, starting the fit after the incident pulse. Used both by
    the case script at run time and as a post-processing step on the committed
    r4 artifacts (review finding 1; no rerun)."""
    n_env = int(tail["envelope_steps"])
    env_start = int(n_steps) - n_env
    start = int(n_pulse_end) - env_start
    r_s, nb_s = fit_tail_rate(tail["envelope_scat_refl_rel"], dt, start=start)
    r_t, nb_t = fit_tail_rate(tail["envelope_total_trans_rel"], dt, start=start)
    return dict(tail, envelope_start_step=env_start, fit_start_step=int(n_pulse_end) + TAIL_WINDOW,
                fitted_rate_scat_refl_1_s=r_s, fitted_rate_total_trans_1_s=r_t,
                fitted_rate_blocks=int(min(nb_s, nb_t)))
