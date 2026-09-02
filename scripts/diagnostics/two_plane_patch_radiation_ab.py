"""Two-plane PEC realization vs radiation damping — textbook-patch A/B (#706 follow-up).

QUESTION
--------
Does the opt-in two-plane realization of a one-cell PEC sheet (#706) damage the
radiation damping of a radiating mode?  Motivation: on a larger multilayer
structure, the two-plane realization shows in-band modes with much higher
loaded Q than the reference solver; before attributing that to structure-
specific representation, test the mechanism on the canonical fixture whose
answer is pinned three ways.

FIXTURE (verbatim from tests/locks/test_patch_edgefed_resonance_harminv.py)
---------------------------------------------------------------------
Edge-fed patch, eps_r 3.38, one-cell ground/feed/patch sheets, uniform mesh
DX = H_SUB/4, CPML.  PRE-#702 one-plane ring-down spectrum (N_SUB=4, 120 periods, the
tree this A/B ran on): 8.78/Q31, 9.32/Q44 (patch TM010; then-current
OpenEMS 9.20, design-dimension Balanis 9.21), 11.90/Q18 (feed lambda/2),
13.72/Q38. Since the #702 sheet-node material fix the same fixture's fed
TM010 reads 8.16 GHz at N=120 (issue #782) — re-derive the reference
spectrum before re-running this A/B on today's tree.  Q~44 of the TM010 is radiation-
dominated (lossless dielectric, PEC metal).

PRE-DECLARED READINGS (before the run)
--------------------------------------
Observable: Harminv (f, Q) of the patch TM010, A = one-plane vs B = two_plane
on all three sheets.  Settling witness (-40 dB bar) must pass in BOTH arms or
the arm's numbers are not read at all.
  * |Q_B/Q_A - 1| <= 0.30 and TM010 present in both  -> two-plane does NOT
    suppress radiation damping of a textbook radiating mode; the larger
    structure's excess Q is NOT explained by the realization itself.
  * Q_B/Q_A > 1.3 (mode narrows) or TM010 vanishes    -> two-plane damages
    radiation coupling; the mechanism reproduces at textbook scale.
A few-percent frequency shift between arms is EXPECTED (the far face moves
the electrical top of each sheet by one cell) and is not a verdict input.
"""
from __future__ import annotations

import math
import numpy as np

from rfx import Box, Simulation
from rfx.harminv import harminv
from rfx.sources import GaussianPulse

EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
PORT_MARGIN = 5.0e-3
Z_GND = 4e-3
FEED_LEN = 8.0e-3
DOM_X, DOM_Y, DOM_Z = 29.747e-3, 18.130e-3, 12.787e-3
N_SUB_CELLS = 4
DX = H_SUB / N_SUB_CELLS
NUM_PERIODS = 120.0
SETTLING_BAR_DB = -40.0


def build(two_plane: bool) -> Simulation:
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
                     dx=DX, cpml_layers=8, boundary="cpml")
    z_gnd_hi = Z_GND + DX
    z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + DX
    x_patch0 = PORT_MARGIN + FEED_LEN
    y_c = DOM_Y / 2.0
    kw = dict(two_plane=True) if two_plane else {}
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, Z_GND), (DOM_X, DOM_Y, z_gnd_hi)), material="pec", **kw)
    sim.add(Box((0, 0, z_sub_lo), (DOM_X, DOM_Y, z_sub_hi)), material="ro4003c")
    sim.add(Box((0, y_c - W_MSL / 2, z_tr_lo),
                (x_patch0, y_c + W_MSL / 2, z_tr_hi)), material="pec", **kw)
    sim.add(Box((x_patch0, y_c - W / 2, z_tr_lo),
                (x_patch0 + L, y_c + W / 2, z_tr_hi)), material="pec", **kw)
    sim.add_msl_port(
        position=(PORT_MARGIN, y_c, z_sub_lo),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )
    sim.add_probe(position=(x_patch0 + 0.7 * L, y_c - 0.2 * W,
                            0.5 * (z_sub_lo + z_sub_hi)), component="ez")
    return sim


def run_arm(tag: str, two_plane: bool):
    sim = build(two_plane)
    advisories = [str(a) for a in sim.preflight()]
    print(f"\n[{tag}] preflight advisories ({len(advisories)}) — quoted verbatim:")
    for a in advisories:
        print(f"  ! {a}")
    res = sim.run(num_periods=NUM_PERIODS)
    ts = np.asarray(res.time_series).ravel()
    dt = float(res.dt)
    env = np.abs(ts)
    peak = float(np.max(env))
    tail = float(np.max(env[int(len(env) * 0.95):]))
    end_db = 20.0 * math.log10(max(tail, 1e-300) / max(peak, 1e-300))
    settled = end_db < SETTLING_BAR_DB
    print(f"[{tag}] settling witness: {end_db:.1f} dB of peak "
          f"(bar {SETTLING_BAR_DB}) -> {'SETTLED' if settled else 'NOT SETTLED'}")
    modes = [m for m in harminv(ts[int(len(ts) * 0.3):], dt, 6e9, 14e9)
             if m.Q > 2 and abs(m.amplitude) > 1e-9]
    spectrum = sorted((m.freq / 1e9, m.Q, float(abs(m.amplitude))) for m in modes)
    print(f"[{tag}] ring-down spectrum: "
          f"{[f'{f:.2f}/Q{q:.0f}/a{a:.2g}' for f, q, a in spectrum]}")
    return spectrum, settled


def tm010_of(spectrum):
    band = [(f, q, a) for f, q, a in spectrum if 8.0 <= f <= 10.5]
    return max(band, key=lambda t: t[2]) if band else None


def main():
    spec_a, ok_a = run_arm("A one-plane", two_plane=False)
    spec_b, ok_b = run_arm("B two-plane", two_plane=True)
    print("\n=== VERDICT (pre-declared in module docstring) ===")
    if not (ok_a and ok_b):
        print("  NOT READ: a settling witness failed; no Q number is trusted.")
        return
    a, b = tm010_of(spec_a), tm010_of(spec_b)
    print(f"  TM010 A(one-plane): {a}   [pre-#702 reference 9.32/Q44; fed TM010 reads 8.16 GHz since #702 — issue #782]")
    print(f"  TM010 B(two-plane): {b}")
    if a is None or b is None:
        print("  -> TM010 MISSING in one arm: two-plane suppresses the radiating mode.")
        return
    ratio = b[1] / a[1]
    print(f"  Q ratio B/A = {ratio:.2f}  (gate: <=1.30 exonerates, >1.30 convicts)")
    print("  ->", "two-plane does NOT suppress radiation damping at textbook scale"
          if ratio <= 1.30 else
          "two-plane RAISES the radiating mode's Q — mechanism reproduces")


if __name__ == "__main__":
    main()
