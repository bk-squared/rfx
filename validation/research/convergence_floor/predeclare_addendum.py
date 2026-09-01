"""Issue #786 — ADDENDUM to the frozen pre-declaration (append-only).

Committed BEFORE the discriminators it governs are run. It ADDS windows;
it does NOT widen or replace any window in
``predeclared_windows_786.json``. Every original window still stands and
its letter-verdict is still reported.

Why an addendum is needed (both reasons found during instrument bring-up,
before any judged measurement):

A1. D1's exoneration window was written in ABSOLUTE METRES (< 1e-12 m).
    The grid stores its mesh in float32, so node coordinates built by
    cumsum carry an unavoidable ~eps32 * L = 6e-8 * 27 mm ~ 1.6e-9 m
    error. No correctly-realized geometry can ever meet a 1e-12 m
    window: it is below the representation floor. That is a
    SPECIFICATION ERROR in the original window, not a property of the
    fixture. D1's letter-verdict is reported unchanged (it will read
    INCONCLUSIVE); D1b restates the SAME measurement in the natural
    units of the mechanism -- CELLS -- with a window derived from what
    geometry quantization IS, and D1c adds a scale-consistency test the
    original window did not cover at all (the subpixel-smoothed
    material, which is what the ladder actually solves).

A2. D2's control was declared as "the surviving in-band line" of the
    trace-free box. Bring-up (2 cheap runs, judged nothing) found the
    4.0-6.5 GHz band is EMPTY without the trace -- the trace is what puts
    a resonance there -- and the lowest line the anti-symmetric port
    excites is at 11.79 GHz. The control band is therefore declared
    HERE, before the control ladder runs, and the D4a empty-box twin is
    declared as the PRIMARY smooth-field control (same band, same dt,
    same record length as the W4R rung, and an exact reference).

Run: PYTHONPATH=. python -m validation.research.convergence_floor.predeclare_addendum
"""

from __future__ import annotations

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "results",
                   "predeclared_windows_786_addendum.json")

EPS32 = 2.0 ** -24


def build() -> dict:
    return {
        "issue": 786,
        "predeclared_utc": "2026-08-30",
        "relation_to_base": ("ADDS windows. Does not widen or replace any "
                             "window in predeclared_windows_786.json; every "
                             "original letter-verdict is still reported."),

        "D1b_geometry_quantization_in_cells": {
            "what": ("the SAME realized-vs-declared measurement as D1, "
                     "expressed in cells: delta_cells(s) = "
                     "delta_max(s) / dx(s)"),
            "derivation": "first_principles + arithmetic",
            "why": (
                "Geometry quantization is by construction a >= 0.5-cell "
                "event: a declared feature edge snaps to the next node. "
                "The measurement's own noise floor is float32 mesh "
                "storage: eps32 * L / dx = 5.96e-8 * (27 mm / dx), i.e. "
                "~9e-6 cells at the finest rung (L/dx = 144). A window at "
                "1e-3 cells sits two decades above that floor and 500x "
                "below the smallest quantization event that can exist, so "
                "it separates the two without touching either."),
            "exonerate": "delta_cells(s) < 1e-3 at EVERY rung",
            "attribute": ("delta_cells(s) >= 0.25 at >= 1 rung AND "
                          "|Pearson rho| >= 0.8 between the per-rung "
                          "residual and the realized-vs-declared "
                          "electrical-length delta"),
            "inconclusive": "1e-3 <= delta_cells < 0.25",
            "exonerate_cells": 1.0e-3,
            "attribute_cells": 0.25,
            "float32_mesh_floor_cells": float(EPS32 * 144),
        },

        "D1c_smoothed_material_scale_consistency": {
            "what": (
                "the ladder runs with subpixel_smoothing=True, so what it "
                "actually solves is the SMOOTHED permittivity tensor, not "
                "the raw mask. Read the smoothed eps column at a fixed "
                "in-plane index far from the trace and record, for every "
                "interface, the eps values at cell offsets -3..+3 from the "
                "declared interface plane."),
            "derivation": "arithmetic (exact equality; nothing to tune)",
            "exonerate": (
                "the offset->eps map is IDENTICAL at every rung to 1e-5 "
                "relative -- the realized material is self-similar under "
                "refinement, so every rung solves the same structure and "
                "its smoothing error shrinks with the cell"),
            "attribute": ("any rung's offset->eps map differs from the "
                          "others (a scale-dependent material realization)"),
            "tol_rel": 1.0e-5,
            "disclosure": (
                "bring-up already compared s=1.5 and s=0.75 and found the "
                "maps identical. The window is EXACT EQUALITY, so there is "
                "nothing a prior look could have tuned; the full-ladder "
                "measurement is what is judged."),
        },

        "D2_control_band": {
            "what": ("the trace-free control ladder tracks the lowest line "
                     "the anti-symmetric port excites in the trace-free "
                     "box"),
            "band_hz": [10.0e9, 13.0e9],
            "search_hz": [9.0e9, 14.0e9],
            "derivation": ("bring-up instrument fact: without the trace the "
                           "4.0-6.5 GHz band is empty and the lowest "
                           "excited line is at 11.79 GHz. This is a band "
                           "choice (instrument), not a judgement window; "
                           "declared here before the control ladder runs."),
        },

        "D2B_primary_smooth_control": {
            "what": ("the D4a empty-box twin IS a smooth-field ladder at "
                     "5.54 GHz with the same band, dt and record length as "
                     "the W4R rung, and an EXACT reference -- so its "
                     "convergence order needs no fitted reference at all. "
                     "It is declared the PRIMARY smooth-field control for "
                     "D2; the trace-free box (D2-A) is the secondary one."),
            "p_smooth_rule": ("p_smooth = max(p_smooth from D2-A, "
                              "p_smooth_measured from D4a) -- the strongest "
                              "smooth-field order either control achieves"),
        },
    }


def main():
    out = build()
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
