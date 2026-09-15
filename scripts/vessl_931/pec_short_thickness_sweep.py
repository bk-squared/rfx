"""#931 T6 — the pre-declared separation for the PEC-short |S11| gate.

Two changes landed on `test_pec_short_s11_magnitude` together: the geometry
(the short was 0.93 of ONE cell against an unpinned dx, and §1.5 refuses that,
so it is redrawn as SHORT_CELLS whole cells on the node line) and the operator
(stage C made the waveguide S-matrix lane apply the realized PEC edges instead
of folding pec_mask into a sigma = 1e10 cell fill).

The separation, pre-declared in that module's docstring and in
docs/design_notes/931_migration/T6-RECOMPUTE.md before this ran:

    re-run at SHORT_CELLS = 1, 2, 4.
    If |S11| TRACKS the thickness it is the geometry, and the module re-pins
    itself from the thickness it declares.
    If |S11| does NOT move with thickness it is the operator, and it belongs
    with the chain-battery re-measure (T6-waveguide-chain-battery.md), not
    with a number in this module.

A total reflector's |S11| should not depend on how thick it is: everything
past the leading face is dark. So "tracks thickness" is itself a physical
statement about which mechanism is in play, not just a curve fit.

Run: JAX_PLATFORMS=cpu python scripts/vessl_931/pec_short_thickness_sweep.py
"""
from __future__ import annotations

import sys

import numpy as np

import tests.oracle.test_waveguide_port_validation_battery as B


def main() -> int:
    freqs = np.linspace(5.0e9, 7.0e9, 6)
    rows = []
    for cells in (1, 2, 4):
        B.SHORT_CELLS = cells
        sim = B._build_sim(freqs, pec_short_x=0.085,
                           waveform="modulated_gaussian")
        d = float(sim._build_grid().dx)
        faces = sim._pec_short_faces_m
        s, _, port_idx = B._s_matrix(sim, num_periods=40, normalize=False)
        s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
        rows.append((cells, d, faces, s11))
        print(f"[pec-short-sweep] SHORT_CELLS={cells}  dx={d * 1e3:.4f} mm  "
              f"faces=({faces[0] * 1e3:.4f}, {faces[1] * 1e3:.4f}) mm  "
              f"thickness={(faces[1] - faces[0]) * 1e3:.4f} mm")
        print(f"[pec-short-sweep]   |S11| = "
              f"{np.array2string(s11, precision=5)}")
        print(f"[pec-short-sweep]   min={s11.min():.5f} mean={s11.mean():.5f} "
              f"max={s11.max():.5f}")
        sys.stdout.flush()

    mins = np.array([r[3].min() for r in rows])
    spread = float(mins.max() - mins.min())
    print("\n[pec-short-sweep] SUMMARY  min|S11| by SHORT_CELLS: "
          + ", ".join(f"{r[0]}->{r[3].min():.5f}" for r in rows))
    print(f"[pec-short-sweep] spread across thickness = {spread:.5f}")
    if spread <= 0.005:
        # BLIND SPOT of the original pre-declaration, added 2026-09-07 after
        # the post-fix run: both of its branches assumed a deficit EXISTS and
        # only asked who owns it. A flat sweep AT unity owns nothing — it is
        # the physics the sweep was written to test, and it says the thing
        # that used to move with thickness has been removed. Which is what
        # happened: the plug is drawn to the realized guide walls now
        # (a8d59e86), and the leak along the top broad wall is gone.
        if mins.min() >= 0.99:
            print("[pec-short-sweep] VERDICT: NOTHING TO ATTRIBUTE — |S11| is "
                  f"flat in thickness (spread {spread:.5f}) AND at unity "
                  f"(min {mins.min():.5f} >= 0.99). A total reflector "
                  "reflects everything and does not care how thick it is. "
                  "No re-pin: the module's own gate is green at its "
                  "untouched threshold.")
            return 0
        print("[pec-short-sweep] VERDICT: OPERATOR — |S11| is flat in "
              "thickness, so the deficit is not the redraw; it belongs with "
              "the chain-battery re-measure.")
        return 0

    # It tracks thickness — and that is NOT the benign reading the
    # pre-declaration hoped for. Everything behind a total reflector's
    # leading face is dark, so |S11| CANNOT depend on how many cells of PEC
    # sit behind it. A monotone rise toward 1 with thickness means the
    # realized wall is not opaque: the thicker the stack, the less gets
    # through. So the thickness dependence is a defect signature, not a
    # licence to re-pin at whatever thickness the fixture happens to
    # declare. Say both things.
    print("[pec-short-sweep] VERDICT: |S11| TRACKS THICKNESS "
          f"(spread {spread:.5f} > 0.005). A total reflector's |S11| cannot "
          "depend on its thickness — everything past the leading face is "
          "dark — so this is not 'the redraw moved the number by half a "
          "cell'; it says the realized short is not opaque and leaks less "
          "the thicker it is.")
    if mins.max() < 0.99:
        print("[pec-short-sweep] AND the deficit does not close: even the "
              f"thickest arm reaches only {mins.max():.5f} against the 0.99 "
              "gate. Re-pinning this module at any thickness would pin a "
              "leak. Do NOT re-pin; this is the same class as the chain "
              "battery's pec_short DUT and belongs with that re-measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
