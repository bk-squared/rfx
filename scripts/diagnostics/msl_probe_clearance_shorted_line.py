"""Probe-clearance bias against an EXACT truth: a shorted microstrip (#726).

The notch-filter attempt at this measurement failed its settling witness in
both arms, so nothing could be read from it. This fixture removes every
confound:

* the structure is a uniform microstrip terminated in a PEC short. At the
  short a lossless passive structure reflects everything, so |S11| = 1
  (0 dB) at EVERY frequency, not just at one notch — the truth is exact,
  frequency-independent, and needs no analytic model of the discontinuity;
* a short is a strong reflector, i.e. the standing-wave regime the board's
  divider puts the probes in;
* the only variable between arms is the distance from the port (and its
  probe comb) to the short.

Any deviation from 0 dB is extraction error. If the near arm deviates
materially more than the clean arm, preflight is right that probe
clearance biases |S11| and the Z0 guard's "the V*I-split S11/S21 are
unaffected" is wrong in this regime (issue #726).

RECORDED VERDICT (2026-08-28). Source: issue #726, comment of 2026-08-28 (the
same numbers are in docs/research_notes/20260817_cad_import_crossval_arc.md,
local to the primary checkout). short_x 20 mm, deepest probe to short
16.56 mm (clean) vs 0.16 mm (near), the same 5-probe comb in both arms:

  clean  settling -45.5 dB (bar -40: SETTLED). |S11| over 1.2-5.4 GHz:
         mean -0.04 dB, worst bin -0.23 dB, truth 0.00 dB. The extractor
         itself is accurate.
  near   settling witness -6.1 / -5.7 / -4.7 dB at num_periods 150 / 300 /
         600. Quadrupling the run moves it 1.4 dB the wrong way, so this
         arm cannot be READ at any run length. Ungated |S11| mean / worst
         in those records: -0.135 / -0.692, -0.084 / -0.712,
         -0.232 / -1.864 dB.

  Answer to the contradiction — a THIRD option, not either message:
  * preflight's "physical |S11| -> 1 may read as -5 to -10 dB" did NOT
    reproduce: even in the un-settleable records the worst bin stayed
    under 2 dB from the exact truth at every run length;
  * the guard's "the V*I-split S11/S21 are unaffected" is not reassurance
    either: the same condition makes the settling witness unsatisfiable,
    and settling is a precondition for quoting any DFT-derived S value.
  Observed: with the probes 0.16 mm from a PEC short the residual at the
  probes is independent of run length. The issue comment reads that as
  non-propagating content in the reflector's near field (the #388 class);
  this script does not instrument that mechanism.
  A board run whose port carries the clearance warning but whose settling
  witness reads -131 .. -157 dB is not in this regime; the settling number
  decides readability.
"""
from __future__ import annotations

import argparse
import math
import numpy as np

from rfx import Box, Simulation

C0 = 299792458.0
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX = 80e-6
FREQ_MAX = 6e9


N_PROBE_OFFSET = 10      # cells; same comb in both arms
N_PROBE_SPACING = 2      # cells
N_PROBES = 5


def build(line_len: float, port_x: float, short_x: float):
    margin = 2e-3
    LX = short_x + margin
    clearance = 2 * (2 * H_SUB + 8 * DX)
    LY = W_TRACE + 2 * clearance
    LZ = H_SUB + 1.5e-3
    sim = Simulation(freq_max=FREQ_MAX, domain=(LX, LY, LZ), dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_c = LY / 2
    y_lo, y_hi = y_c - W_TRACE / 2, y_c + W_TRACE / 2
    sim.add(Box((0, y_lo, H_SUB), (short_x, y_hi, H_SUB + DX)), material="pec")
    # PEC short: a wall from the trace down to the ground plane at short_x
    sim.add(Box((short_x - DX, y_lo, 0.0), (short_x, y_hi, H_SUB + DX)),
            material="pec")
    # ground plane under everything
    sim.add(Box((0, 0, -DX), (LX, LY, 0.0)), material="pec")
    # BOTH arms use the SAME probe comb (offset/spacing/count) so the only
    # variable is the port's distance to the short. With the auto-resolved
    # comb the near arm's deepest probe landed PAST the short and outside
    # the domain (2026-08-27) — the comb geometry must be pinned, and its
    # extent checked, before the solve.
    sim.add_msl_port(position=(port_x, y_c, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0,
                     n_probe_offset=N_PROBE_OFFSET,
                     n_probe_spacing=N_PROBE_SPACING, n_probes=N_PROBES)
    deepest = port_x + (N_PROBE_OFFSET + (N_PROBES - 1) * N_PROBE_SPACING) * DX
    if deepest >= short_x:
        raise SystemExit(
            f"design error: deepest probe {deepest * 1e3:.2f} mm is at or "
            f"past the short {short_x * 1e3:.2f} mm — the comb must fit "
            "between the port and the reflector")
    return sim, deepest


def run_arm(label, port_x, short_x, num_periods):
    sim, deepest = build(short_x, port_x, short_x)
    gap_mm = (short_x - deepest) * 1e3
    lam_g4 = C0 / (FREQ_MAX * math.sqrt(2.87)) / 4
    print(f"\n=== ARM {label}: port {port_x*1e3:.2f} mm, deepest probe "
          f"{deepest*1e3:.2f} mm, short {short_x*1e3:.2f} mm -> probe-to-short "
          f"gap {gap_mm:.2f} mm (lambda_g/4 at f_max = {lam_g4*1e3:.2f} mm) ===")
    adv = [str(a) for a in sim.preflight()]
    for a in adv:
        if "probe" in a and ("reflector" in a or "unsatisfiable" in a):
            print(f"  ! {a[:200]}")
    res = sim.compute_msl_s_matrix(n_freqs=60, num_periods=num_periods)
    f = np.asarray(res.freqs, float)
    S = np.abs(np.asarray(res.S[0, 0], np.complex128))
    sett = float(np.max(np.asarray(res.settling_db)))
    band = (f > 0.2 * FREQ_MAX) & (f < 0.9 * FREQ_MAX)
    db = 20 * np.log10(np.maximum(S[band], 1e-12))
    print(f"  settling witness: {sett:.1f} dB "
          f"({'SETTLED' if sett < -40 else 'NOT SETTLED — numbers not read'})")
    print(f"  |S11| over {0.2*FREQ_MAX/1e9:.1f}-{0.9*FREQ_MAX/1e9:.1f} GHz: "
          f"mean {db.mean():+.2f} dB, min {db.min():+.2f}, max {db.max():+.2f} "
          f"(truth: 0.00 dB at every bin)")
    return dict(settled=sett < -40, mean=db.mean(), worst=db.min(), sett=sett)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-periods", type=float, default=300.0)
    ap.add_argument("--short-x-mm", type=float, default=20.0)
    a = ap.parse_args()
    short_x = a.short_x_mm * 1e-3
    print("TRUTH: a shorted lossless line reflects everything -> |S11| = 0 dB "
          "at every frequency. Any deviation is extraction error.")
    clean = run_arm("clean", 2e-3, short_x, a.num_periods)
    if not clean["settled"]:
        # Precondition gate: without a settled control arm the comparison is
        # unreadable, so do not spend the second arm (2026-08-27: two
        # attempts were lost by running both arms first and reading after).
        print("\n=== VERDICT ===\n  NOT READ: control arm un-settled "
              f"({clean['sett']:.1f} dB, bar -40). Needed periods ~= "
              f"{a.num_periods * 40.0 / max(abs(clean['sett']), 1e-6):.0f}; "
              "rerun with --num-periods above that. Arm 2 skipped.")
        return
    near = run_arm("near", short_x - 1.6e-3, short_x, a.num_periods)
    print("\n=== VERDICT (pre-declared) ===")
    if not (clean["settled"] and near["settled"]):
        print("  NOT READ: a settling witness failed "
              f"(clean {clean['sett']:.1f} dB, near {near['sett']:.1f} dB). "
              "Raise --num-periods.")
        return
    d = abs(near["mean"]) - abs(clean["mean"])
    print(f"  |S11| mean error: clean {clean['mean']:+.2f} dB, "
          f"near {near['mean']:+.2f} dB (extra error {d:+.2f} dB)")
    print(f"  worst bin:        clean {clean['worst']:+.2f} dB, "
          f"near {near['worst']:+.2f} dB")
    if d >= 2.0:
        print("  -> PREFLIGHT right: probe clearance biases |S11| itself. The "
              "Z0 guard's 'V*I-split S11/S21 are unaffected' is wrong here.")
    else:
        print("  -> GUARD right within 2 dB: |S11| is insulated from the "
              "probe-clearance condition that wrecks Z0.")


if __name__ == "__main__":
    main()
