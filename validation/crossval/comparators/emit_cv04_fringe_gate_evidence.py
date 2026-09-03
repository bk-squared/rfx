"""Emit cv04's fringe-gate evidence as machine-readable JSON (issue #812).

WHY THIS EXISTS (issue #812 round-2 discipline). Round 1 of this issue shipped a
wrong number into a durable document four times, and cv04's own round-1 blocker
was a source comment asserting a detector property the implementation does not
have. The fix for that class is mechanical, not editorial: every quantity the
cv04 fringe gate's prose asserts is computed HERE, committed as
``validation/crossval/_04_fresnel_results/fringe_gate_geometry.json``, and
re-derived on every test run. Prose references an artifact key; it does not
restate digits.

Everything in this module is closed-form or a 3-line synthetic array — no FDTD,
no solver, runs in well under a second. The two things it records:

1. ``windows`` -- the gate geometry derived from first principles (spectral bin,
   exact discrete-Yee dispersion, free spectral range). This includes
   ``non_entailment_ratio``, the property that actually keeps the verdict from
   being entailed by its own search window, which is what the withdrawn
   "reference-blind" claim was reaching for.
2. ``falsifiers`` -- criterion (A) on an exact analytic slab and criterion (B)
   on each defect the audit measured the pre-#812 gates blind to, run through
   the SAME ``fringe_gate.compare_fringes`` the crossval script calls.

Regenerate with:
    PYTHONPATH=<worktree> python validation/crossval/comparators/emit_cv04_fringe_gate_evidence.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
OUTPUT_PATH = (
    _HERE.parent / "_04_fresnel_results" / "fringe_gate_geometry.json"
)

# --- the committed cv04 configuration --------------------------------------
# Every value here is read off validation/crossval/04_multilayer_fresnel.py at
# the line cited; DT is Grid(...).dt for that configuration and BAND is the
# contiguous band the committed spectral mask selects (a MEASURED property of
# the committed run, not a gate threshold -- see "band_provenance" in the JSON).
EPS_R = 4.0             # 04_multilayer_fresnel.py:63
D_M = 10.0e-3           # 04_multilayer_fresnel.py:65
N_INDEX = 2.0           # 04_multilayer_fresnel.py:64  (sqrt(4.0))
DX_M = 1.0e-3           # 04_multilayer_fresnel.py:67
C0 = 2.998e8            # 04_multilayer_fresnel.py:43
DT_S = 2.335067793382187e-12   # Grid(freq_max=20e9, domain=(0.6,0.004,1e-3),
#                                dx=1e-3, cpml_layers=10, mode="2d_tmz").dt
NFFT = 8192             # 2**ceil(log2(719)) * 8   (04_multilayer_fresnel.py:290)
DF_BIN_HZ = 1.0 / (NFFT * DT_S)
BAND_LO_HZ = 3.0321e9
BAND_HI_HZ = 11.8666e9

# The audit's three measured defects (issue #812 cv04 row). Each is a
# perturbation of the REFERENCE holding the measurement fixed, which is the
# direction the audit measured the pre-#812 mean gates blind to.
AUDIT_EPS_TRUE = 4.4933         # +12.33% on eps
AUDIT_D_SCALE = 1.08            # +8.0% on thickness
AUDIT_RMAX_SCALE = 0.2797 / 0.3600   # measured R_max reads 22.3% low


def _load_fringe_gate():
    spec = importlib.util.spec_from_file_location(
        "_cv04_fringe_gate_evidence", _HERE / "fringe_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    # registered before exec so the frozen dataclasses can resolve their own
    # module namespace
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def band_axis() -> np.ndarray:
    n = int(round((BAND_HI_HZ - BAND_LO_HZ) / DF_BIN_HZ)) + 1
    return BAND_LO_HZ + DF_BIN_HZ * np.arange(n)


def ideal_slab_R(freqs: np.ndarray, eps_r: float = EPS_R,
                 d: float = D_M) -> np.ndarray:
    """Exact lossless-slab R(f) at normal incidence (Airy form)."""
    n = math.sqrt(eps_r)
    delta = 2.0 * np.pi * freqs * n * d / C0
    r = ((n - 1.0) / (n + 1.0)) ** 2
    finesse = 4.0 * r / (1.0 - r) ** 2
    s2 = np.sin(delta) ** 2
    return finesse * s2 / (1.0 + finesse * s2)


def withdrawn_reference_blind_probe(fg, freqs: np.ndarray,
                                    measured: np.ndarray) -> dict:
    """Re-derive why the reference-BLIND detector was withdrawn (note section 9).

    The withdrawn design located extrema with a prominence floor equal to the
    value window ``V`` and no reference to the analytic positions. ``find_peaks``
    measures prominence against the array boundary when no higher peak lies on
    that side, and this band truncates the outer shoulder of both maxima -- so
    the floor rejects them and the detector fails criterion (A) on correct code.
    Measured here on the exact analytic slab so the claim is an artifact key,
    not a sentence.
    """
    floor = fg.FRINGE_VALUE_LIMIT
    out = {"prominence_floor": floor}
    survivors = {}
    for kind, signed in (("max", measured), ("min", -measured)):
        idx, props = find_peaks(signed, prominence=0.0)
        prom = props["prominences"]
        out[f"{kind}_candidates_hz"] = [float(v) for v in freqs[idx]]
        out[f"{kind}_prominences"] = [float(v) for v in prom]
        keep = freqs[idx][prom >= floor]
        survivors[kind] = [float(v) for v in keep]
        out[f"{kind}_surviving_the_floor_hz"] = survivors[kind]
    analytic = fg.analytic_slab_extrema(
        EPS_R, D_M, float(freqs[0]), float(freqs[-1]), c0=C0
    )
    n_expected = {"max": sum(1 for k, _, _ in analytic if k == "max"),
                  "min": sum(1 for k, _, _ in analytic if k == "min")}
    out["n_analytic"] = n_expected
    out["n_detected"] = {k: len(v) for k, v in survivors.items()}
    out["criterion_A_ok"] = bool(
        out["n_detected"] == n_expected
    )
    return out


def _verdict_summary(fg, verdict) -> dict:
    """Worst position/value utilisation of a verdict, plus its reason classes."""
    classes = sorted({
        "CONTAINMENT" if "CONTAINMENT" in r
        else "POSITION" if "fringe POSITION" in r
        else "VALUE" if "fringe VALUE" in r
        else "OTHER"
        for r in verdict.reasons
    })
    pos = [abs(row.df_hz) / row.f_window_hz for row in verdict.rows]
    val = [abs(row.dvalue) / row.value_limit for row in verdict.rows]
    return {
        "ok": bool(verdict.ok),
        "reason_classes": classes,
        "n_rows": len(verdict.rows),
        "worst_position_window_utilisation": (max(pos) if pos else None),
        "worst_value_window_utilisation": (max(val) if val else None),
    }


def build_evidence() -> dict:
    fg = _load_fringe_gate()
    freqs = band_axis()
    fsr = C0 / (2.0 * N_INDEX * D_M)
    cell_half = fg.CELL_HALF_WIDTHS_PER_FSR * fsr

    extrema = fg.analytic_slab_extrema(
        EPS_R, D_M, float(freqs[0]), float(freqs[-1]), c0=C0
    )
    fringes = []
    for kind, f_an, v_an in extrema:
        shift = fg.yee_dispersion_shift_hz(
            f_an, N_INDEX, DX_M, DT_S, c0=C0
        )
        w = fg.position_window_hz(
            f_an, n_index=N_INDEX, dx=DX_M, dt=DT_S,
            df_bin_hz=DF_BIN_HZ, c0=C0,
        )
        fringes.append({
            "kind": kind,
            "f_analytic_hz": f_an,
            "value_analytic": v_an,
            "yee_dispersion_shift_hz": shift,
            "position_window_hz": w,
            "position_window_over_cell_half_width": w / cell_half,
            # The window is SAFETY * (df_bin/2 + |yee shift|).  This key is the
            # bracketed part alone -- the budget the two PHYSICAL terms actually
            # derive, before the safety factor.  A measured displacement above
            # it is not explained by bin quantisation or by Yee dispersion, and
            # passes only on SAFETY.  See section 14 of the pre-declaration note.
            "position_derived_budget_hz": w / fg.SAFETY,
        })
    max_w = max(row["position_window_hz"] for row in fringes)

    def compare(measured, **overrides):
        kwargs = dict(
            eps_r=EPS_R, d=D_M, n_index=N_INDEX, dx=DX_M, dt=DT_S,
            df_bin_hz=DF_BIN_HZ, c0=C0, label="evidence",
        )
        kwargs.update(overrides)
        return fg.compare_fringes(freqs, measured, **kwargs)

    measured_ideal = ideal_slab_R(freqs)

    # A 400 MHz uniform fringe shift: comfortably inside every search cell, so
    # the detector FINDS it, and it must still FAIL the position gate. This is
    # the non-entailment demonstration in its measured form.
    shift_hz = 400e6
    f_top = max(row["f_analytic_hz"] for row in fringes)
    scale = f_top / (f_top - shift_hz)

    falsifiers = [
        {
            "id": "A_ideal_analytic_slab",
            "criterion": "A",
            "what": "an exact lossless-slab R(f) must PASS",
            "expect_ok": True,
            "verdict": _verdict_summary(fg, compare(measured_ideal)),
        },
        {
            "id": "B_audit_eps_plus_12_33_percent",
            "criterion": "B",
            "what": ("audit defect: the true slab is eps=4.4933 while rfx built "
                     "eps=4.0; every pre-#812 gate passes"),
            "expect_ok": False,
            "verdict": _verdict_summary(fg, compare(
                measured_ideal, eps_r=AUDIT_EPS_TRUE, n_index=None)),
        },
        {
            "id": "B_audit_thickness_plus_8_percent",
            "criterion": "B",
            "what": "audit defect: the true slab is 8.0% thicker",
            "expect_ok": False,
            "verdict": _verdict_summary(fg, compare(
                measured_ideal, d=D_M * AUDIT_D_SCALE)),
        },
        {
            "id": "B_audit_rmax_minus_22_3_percent",
            "criterion": "B",
            "what": ("audit defect: measured R_max reads 22.3% low with the "
                     "fringe POSITIONS untouched"),
            "expect_ok": False,
            "verdict": _verdict_summary(fg, compare(
                measured_ideal * AUDIT_RMAX_SCALE)),
        },
        {
            "id": "B_one_cell_thickness_error",
            "criterion": "B",
            "what": "one dx of thickness error, the smallest the grid expresses",
            "expect_ok": False,
            "verdict": _verdict_summary(
                fg, compare(measured_ideal, d=D_M + DX_M)),
        },
        {
            "id": "B_structureless_curve_is_a_containment_failure",
            "criterion": "B",
            "what": ("a flat R(f) has no fringe: the detector must FAIL with "
                     "CONTAINMENT, never report a cell boundary as an extremum"),
            "expect_ok": False,
            "verdict": _verdict_summary(
                fg, compare(np.full(freqs.shape, fg.slab_R_max(EPS_R)))),
        },
        {
            "id": "B_shift_inside_the_search_cell_still_fails",
            "criterion": "B",
            "what": ("a 400 MHz uniform fringe shift is inside every search "
                     "cell -- it is FOUND, and must still FAIL on POSITION; "
                     "this is what non_entailment_ratio buys"),
            "expect_ok": False,
            "shift_hz": shift_hz,
            "shift_over_cell_half_width": shift_hz / cell_half,
            "verdict": _verdict_summary(fg, compare(ideal_slab_R(freqs * scale))),
        },
    ]

    return {
        "schema": "cv04_fringe_gate_geometry/1",
        "issue": 812,
        "generated_by":
            "validation/crossval/comparators/emit_cv04_fringe_gate_evidence.py",
        "gates_described":
            "validation/crossval/comparators/fringe_gate.py",
        "config": {
            "eps_r": EPS_R,
            "d_m": D_M,
            "n_index": N_INDEX,
            "dx_m": DX_M,
            "dt_s": DT_S,
            "c0_m_per_s": C0,
            "nfft": NFFT,
            "df_bin_hz": DF_BIN_HZ,
            "band_lo_hz": BAND_LO_HZ,
            "band_hi_hz": BAND_HI_HZ,
            "band_n_bins": int(freqs.size),
            "band_provenance": (
                "the contiguous band the committed spectral mask selects "
                "(04_multilayer_fresnel.py:304); a measured property of the "
                "committed run, not a gate threshold"
            ),
        },
        "detector": {
            "reference_blind": False,
            "anchored_on": (
                "the analytic extremum supplies each extremum's KIND and the "
                "CENTRE of its half-fringe search cell"
            ),
            "measured_quantity": (
                "arg-extremum of the measured R(f) strictly inside that cell, "
                "refined to sub-bin resolution by a 3-point parabolic vertex fit"
            ),
            "search_half_width_hz": cell_half,
            "pin_margin_bins": fg.PIN_MARGIN_BINS,
            "withdrawn_alternative_measured":
                withdrawn_reference_blind_probe(fg, freqs, measured_ideal),
            "withdrawn_alternative": (
                "a reference-blind prominence detector was implemented, "
                "measured and withdrawn: on this band it rejected both true "
                "maxima (their outer shoulders are truncated by the band), so "
                "it failed criterion (A) on correct code -- see "
                "docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md "
                "section 9"
            ),
        },
        "windows": {
            "safety": fg.SAFETY,
            "value_limit": fg.FRINGE_VALUE_LIMIT,
            "fsr_hz": fsr,
            "cell_half_widths_per_fsr": fg.CELL_HALF_WIDTHS_PER_FSR,
            "cell_half_width_hz": cell_half,
            "max_position_window_hz": max_w,
            "non_entailment_ratio": cell_half / max_w,
            "non_entailment_meaning": (
                "search half-width / widest gate window. The verdict cannot be "
                "entailed by the search window while this is >> 1 AND an "
                "extremum reaching the cell edge is a CONTAINMENT failure "
                "rather than a reported boundary; both halves are pinned in "
                "tests/crossval/test_crossval_gate_logic.py"
            ),
            "fringes": fringes,
            "meep_abs_limit": fg.MEEP_ABS_LIMIT,
            "meep_cross_limit": fg.MEEP_CROSS_LIMIT,
        },
        "falsifiers": falsifiers,
    }


def main() -> None:
    evidence = build_evidence()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
