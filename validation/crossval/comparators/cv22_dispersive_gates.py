"""cv22 dispersive slab -- declared arms, derived windows, gates, falsifiers.

Pure numpy (plus ``tests/_gate_policy.py`` for the repo-wide envelope rule).
Shared by the case script (``validation/crossval/22_dispersive_slab_fresnel.py``),
the Meep leg (``scripts/crossval/meep_cv22_dispersive_slab.py``) and the gate
test (``tests/crossval/test_cv22_dispersive_slab_gates.py``) so that the pre-declared
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
import slab_family  # noqa: E402
from tests._gate_policy import gate_from_envelope  # noqa: E402

TWO_PI = 2.0 * math.pi
C0 = de.C0

# ---------------------------------------------------------------------------
# Rig (cv04's; pre-declaration §1) -- DECLARED in ``slab_family``, re-exported
# here so the modules that import it from this one keep working (#928). The rig
# belongs to the producer's case, not to this consumer; the values are
# unchanged and bit-identical (tests/fixtures/slab_family_windows_baseline.json).
# ---------------------------------------------------------------------------
DX_M = slab_family.DX_M
D_SLAB_M = slab_family.D_SLAB_M
NX_INTERIOR = slab_family.NX_INTERIOR
N_CPML = slab_family.N_CPML
TFSF_F0_HZ = slab_family.TFSF_F0_HZ
TFSF_BW = slab_family.TFSF_BW
NFFT_OVERSAMPLE = slab_family.NFFT_OVERSAMPLE
MASK_F_LO_HZ = slab_family.MASK_F_LO_HZ
MASK_F_HI_HZ = slab_family.MASK_F_HI_HZ
MASK_AMP_FRAC = slab_family.MASK_AMP_FRAC
# cv04's own settling-tail witness constants and per-bin closure ceiling.
TAIL_WINDOW = slab_family.TAIL_WINDOW
TAIL_PURITY_LIMIT = slab_family.TAIL_PURITY_LIMIT
TAIL_LIMIT = slab_family.TAIL_LIMIT
CONS_MAX_LIMIT = slab_family.CONS_MAX_LIMIT

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
# Rig-level, declared in slab_family and re-exported (#928): the settling bar
# and the cell placements are properties of the shared rig, not of this case's
# gate policy, and the producer's own witness needs them.
SETTLING_LIMIT = slab_family.SETTLING_LIMIT
TFSF_MARGIN = slab_family.TFSF_MARGIN
PROBE_OFFSET_CELLS = slab_family.PROBE_OFFSET_CELLS
SRC_T0_OVER_TAU = slab_family.SRC_T0_OVER_TAU
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
RING_W_MIN = slab_family.RING_W_MIN
RING_F_MAX_HZ = slab_family.RING_F_MAX_HZ
RECORD_EXTEND_STEPS = 100
NX_GROW_CELLS = 200
TAIL_ENVELOPE_STEPS = 300      # stored in the artifact so the decay can be fitted offline
MEEP_LADDER_RESOLUTIONS = (10, 20, 40)

# Gated band (§5; declared in slab_family so the producer can mark its own
# gated bins) and the rig-sanity floor on incident amplitude inside it.
BAND_GATED_HZ = slab_family.BAND_GATED_HZ
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
# The cv04 envelope this case ADOPTED (§4), and the windows derived from it
#
# The values are NOT written here. They live in the producer's own artifact,
# `validation/crossval/_04_fresnel_results/envelope.json`, which is where cv04
# records what it measured; before #928 they were copied into this module from
# a UI fixture and a source comment, so a consumer-named module was the home of
# a producer's evidence.
#
# This declaration is the CALIBRATION: which revision of that evidence this
# case adopted, that revision's hash, the realized rig's hash, and the gate
# policy in force when it was adopted. Appending a new revision to the
# producer's artifact does not move these windows -- only editing this record
# does, and that edit is reviewed under the repo's no-silent-gate-loosening
# rule. The windows below are the EXPECTATION, derived from the two and stated
# nowhere else.
# ---------------------------------------------------------------------------
CV04_ADOPTION = {
    "envelope": slab_family.CV04_ENVELOPE_REL,
    "adopted_revision": "r1",
    "revision_sha256": "sha256:456a657a06be3b8d483701b126befd1e06ada99c3f9b5575185ed94810150ef2",
    "rig_hash": "sha256:24164f616573af51b91f5c596e7b79e521005c4a872218fede25d009ed9dd211",
    "gate_policy": {"multiplier": 1.5, "quantum": 1000},
    "adopted_in": "docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md",
    "adopted_by_reviewer": "cv22 pre-declaration review, 2026-09-02",
}
CV04_ADOPTED = slab_family.load_adopted_envelope(CV04_ADOPTION)
CV04_ENVELOPE = CV04_ADOPTED["values"]
_QUANTUM = CV04_ADOPTION["gate_policy"]["quantum"]
W_BIN = gate_from_envelope(CV04_ENVELOPE["per_bin_max_RT_closure"], quantum=_QUANTUM)   # 0.074
W_MEAN_R = gate_from_envelope(CV04_ENVELOPE["mean_dR"], quantum=_QUANTUM)               # 0.010
W_MEAN_T = gate_from_envelope(CV04_ENVELOPE["mean_dT"], quantum=_QUANTUM)               # 0.017

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

# Rig-level, declared in slab_family (#928); re-exported for this module's
# importers and for the pre-declaration's §5 wording.
gated_mask = slab_family.gated_mask


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
                *, tail: dict | None = None, require_complete: bool = False) -> dict:
    """E2 gates G1 (per-bin), G2 (band-mean), G3 (witnesses) for one arm.

    ``freqs_hz`` are the masked rfx bins (cv04's mask); the gated subset is
    the 4-10 GHz band. Returns a JSON-ready dict; arrays as lists.

    ``require_complete`` (issue #928): a gate whose evidence is ABSENT is None
    here -- ``G3_tail`` is None when no tail witness is handed in -- and the
    default aggregate skips it, which is right for the analytic falsifier
    checks that legitimately have no run behind them. For a CLAIMS-BEARING
    invocation that is wrong: a missing required witness would silently make a
    two-of-three verdict read PASS. Pass ``require_complete=True`` there (the
    case scripts do) and a None gate makes the verdict False, with the missing
    names listed in ``incomplete_gates``.
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
    out.update(aggregate_gates(out["gates"], require_complete=require_complete))
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

# The rig's cell bookkeeping and its auxiliary-grid echo are geometry of the
# SHARED rig -- the producer's witness (cv04 --lattice-witness, through
# lattice_witness) needs them as much as this case does -- so they are declared
# in slab_family and re-exported here (#928). Bodies unchanged.
rig_cells = slab_family.rig_cells
aux_echo_arrival = slab_family.aux_echo_arrival
slab_aux_echo = slab_family.slab_aux_echo
aux_echo_verdict = slab_family.aux_echo_verdict
aux_echo_failure_message = slab_family.aux_echo_failure_message
AUX_N_CPML_1D = slab_family.AUX_N_CPML_1D
AUX_N_MARGIN_1D = slab_family.AUX_N_MARGIN_1D
AUX_SRC_OFFSET_1D = slab_family.AUX_SRC_OFFSET_1D
AUX_I0_1D = slab_family.AUX_I0_1D
AUX_REFLECTOR_DEPTH_CELLS = slab_family.AUX_REFLECTOR_DEPTH_CELLS
AUX_ECHO_RATIO_LIMIT = slab_family.AUX_ECHO_RATIO_LIMIT
AUX_ECHO_SCHEMA = slab_family.AUX_ECHO_SCHEMA


# The incident pulse's spectrum, the ring band it defines and the slab's own
# ring-down rates are rig + material physics, not this case's gate policy:
# cv04's witness and cv23's record recipe need them too, so they are DECLARED
# in slab_family and re-exported here (#928). Bodies unchanged.
incident_amplitude_rel = slab_family.incident_amplitude_rel
ring_band_hz = slab_family.ring_band_hz
slab_ringdown_rates = slab_family.slab_ringdown_rates
# The verdict aggregate, including the completeness rule (#928).
aggregate_gates = slab_family.aggregate_gates


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


def meep_ladder_summary(results_dir: str, rfx_doc: dict, resolutions=MEEP_LADDER_RESOLUTIONS) -> dict:
    """Meep-vs-TMM deviation per resolution (from meep_<arm>__res<N>.json) and the
    measured convergence order per doubling. Measured-in-r2 evidence, not a
    pre-declared window term."""
    import json as _json
    resolutions = tuple(int(r) for r in resolutions)
    out = {"schema": "cv22-meep-ladder/v1", "resolutions": list(resolutions), "arms": {}}
    for arm, ad in rfx_doc["arms"].items():
        e2 = evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], ad["model"], ad["params"], ad["dt_s"],
                         tail=ad["tail"])
        rungs = {}
        for res in resolutions:
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
        for lo, hi in zip(resolutions[:-1], resolutions[1:]):
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
    # Two blocks give a two-point slope: reported, flagged unreliable by
    # refit_tail (fit_reliable = nb >= 3; cv23 note section 14 -- the
    # shortest cv23 record has 109 post-window steps = 2 blocks).
    if nb < 2:
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
    nb = int(min(nb_s, nb_t))
    return dict(tail, envelope_start_step=env_start, fit_start_step=int(n_pulse_end) + TAIL_WINDOW,
                fitted_rate_scat_refl_1_s=r_s, fitted_rate_total_trans_1_s=r_t,
                fitted_rate_blocks=nb, fit_reliable=bool(nb >= 3))
