"""Sheet vs volume PEC declaration — textbook-patch radiation A/B (#931).

QUESTION
--------
Does declaring a patch antenna's foils as SHEETS (§1.3 — one node plane, no
cell owned, live normal edge) damage the radiation damping of a radiating
mode, compared with declaring the same foils as one-cell VOLUMES (§1.2 — walls
at both drawn faces, normal edge shorted)?

This is the design note's multilayer-board falsifier at textbook scale. The
board A/B (memory 2026-08-28, VESSL 369367256724) measured the OLD pair —
one-plane vs ``two_plane`` — and found the two-plane arm correlated +0.265
with CST against the one-plane arm's +0.829. Neither of those realizations
exists at 2.0. The surviving choice is sheet vs volume, and it is a
DECLARATION, so the question is no longer "which rule" but "does the honest
declaration of a 35 um foil behave".

HISTORY (this file replaced ``two_plane_patch_radiation_ab.py``)
---------------------------------------------------------------
That script asked whether ``two_plane=True`` suppressed radiation damping.
Its arms were the two pre-2.0 realizations of ONE declaration; both are gone
(a one-cell PEC Box is a volume with two faces at every thickness, with no
flag). Its recorded pre-2.0 readings are kept here as dated history, NOT as a
prediction for this file's arms: PRE-#702 one-plane ring-down on this fixture
(N_SUB=4, 120 periods) read 8.78/Q31, 9.32/Q44 (patch TM010; then-current
openEMS 9.20, design-dimension Balanis 9.21), 11.90/Q18, 13.72/Q38; after the
#702 sheet-node material fix the same fixture's fed TM010 read 8.16 GHz
(issue #782). The #702 resample is itself deleted at 2.0, so neither number
is a reference for the arms below.

FIXTURE (unchanged from the file this replaces)
-----------------------------------------------
Edge-fed patch, eps_r 3.38, RO4003C, uniform mesh DX = H_SUB/4 = 196.75 um,
CPML. The three foils — ground, MSL feed, patch — are 35 um copper in the
physical board, i.e. 0.18 of a cell. Arm S declares them as sheets at the
planes they occupy; arm V declares them as the one-cell Boxes the pre-2.0
scripts drew, which is a 196.75 um plate — 5.6x the physical copper.

PRE-DECLARED READINGS (before the run)
--------------------------------------
Observable: Harminv (f, Q) of the patch TM010. Settling witness (-40 dB bar)
must pass in BOTH arms or the arm's numbers are not read at all.

  * |Q_V/Q_S - 1| <= 0.30 and TM010 present in both -> the declaration does
    not decide radiation damping at textbook scale; the sheet declaration is
    safe for the docs' recommended foil route.
  * Q_S/Q_V > 1.3 (the sheet arm's mode narrows) or TM010 absent in the sheet
    arm -> the sheet declaration damages radiation coupling and the public
    guidance "foil is a sheet" must be qualified before it ships.
  * Q_V/Q_S > 1.3 -> the VOLUME arm narrows, which is the expected direction
    of a 196.75 um plate closing the cavity, and is a caveat on drawing foil
    as a one-cell Box, not on the sheet route.

A frequency shift between the arms is EXPECTED and is not a verdict input:
arm V's plates move the electrical top and bottom of the cavity by the plate
thickness, arm S's do not. That shift is the reason the docs tell you to
declare foil as a sheet, and is reported, not gated.

BUILD-TIME CHECK (no solve): each arm asserts, through
``realized_pec_edge_masks`` / ``realized_wall_planes``, that its realized wall
planes are exactly the declared ones — one plane per foil in arm S, two per
foil in arm V. Run ``--check`` for that alone.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from rfx import Box, Simulation
from rfx.harminv import harminv
from rfx.sources import GaussianPulse

# The ONE spelling of the build-time realization check (#931 §1.7). A second
# hand-rolled scan over an edge mask is the drift the single-owner rule exists
# to stop, so this diagnostic reads the same helper the migrated fixtures do.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests"))
from _realized_geometry import assert_wall_planes, node_index, realized  # noqa: E402

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
T_COPPER = 35e-6            # the physical foil; 0.18 of a cell at this DX
NUM_PERIODS = 120.0
SETTLING_BAR_DB = -40.0

# The three foil footprints, in the plane each one occupies.  Arm S puts a
# zero-thickness Box on that plane; arm V extrudes it one cell upward, which
# is what every pre-2.0 script in this repository drew.
#
# ON-LATTICE (§1.3, "off-lattice interfaces"): the file this replaces put the
# substrate floor at Z_GND + DX = 4.19675 mm, which is 21.3 cells above the
# origin — not a node.  A sheet declared there snaps to the nearest node and
# ends up buried half a cell inside the laminate, which the assembly now warns
# about (`sheet_slot_vacuum` class) instead of absorbing into a tie rule.  The
# contract's answer is to draw the board on the lattice, so the floor moves to
# node 21 (4.13175 mm, a 65 um / one-third-cell move) and the substrate is the
# same four cells.  This is a fixture change and it is the reason none of the
# pre-2.0 numbers in the history section above is a reference for these arms.
Z_GND_PLANE = 21 * DX                  # substrate floor, exactly on a node
Z_TRACE_PLANE = Z_GND_PLANE + H_SUB    # substrate top, N_SUB_CELLS above it
X_PATCH0 = PORT_MARGIN + FEED_LEN
Y_C = DOM_Y / 2.0

_FOILS = (
    ("ground", (0.0, 0.0), (DOM_X, DOM_Y), Z_GND_PLANE),
    ("feed", (0.0, Y_C - W_MSL / 2), (X_PATCH0, Y_C + W_MSL / 2), Z_TRACE_PLANE),
    ("patch", (X_PATCH0, Y_C - W / 2), (X_PATCH0 + L, Y_C + W / 2), Z_TRACE_PLANE),
)


def build(kind: str) -> Simulation:
    """``kind`` is ``"sheet"`` or ``"volume"``."""
    if kind not in ("sheet", "volume"):
        raise ValueError(kind)
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
                     dx=DX, cpml_layers=8, boundary="cpml")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, Z_GND_PLANE), (DOM_X, DOM_Y, Z_TRACE_PLANE)),
            material="ro4003c")
    for _name, (x0, y0), (x1, y1), z in _FOILS:
        if kind == "sheet":
            # A 35 um foil is 0.18 of a cell: it is a sheet, and the
            # declaration says so.  thickness= is read only on the lossy
            # path; this is PEC copper, so it is documentation here.
            sim.add_thin_conductor(Box((x0, y0, z), (x1, y1, z)),
                                   thickness=T_COPPER)
        else:
            sim.add(Box((x0, y0, z), (x1, y1, z + DX)), material="pec")
    sim.add_msl_port(
        position=(PORT_MARGIN, Y_C, Z_GND_PLANE),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )
    sim.add_probe(position=(X_PATCH0 + 0.7 * L, Y_C - 0.2 * W,
                            0.5 * (Z_GND_PLANE + Z_TRACE_PLANE)),
                  component="ez")
    return sim


def assert_realized_planes(kind: str, verbose: bool = True) -> dict:
    """Build-time gate (§1.2/§1.3): realized walls == declared planes.

    Three columns, because a column tells you what stands ABOVE it too and
    the ground's footprint is the whole board:

    * ``ground_only`` — past the patch in x and below it in y, so only the
      ground is overhead: one plane in arm S, two in arm V;
    * ``patch`` and ``feed`` — ground plus the upper foil: two planes in
      arm S, four in arm V.

    That is a stronger statement than "the ground's own plane is right": it
    also says no foil put a wall anywhere it was not declared. No solve —
    ``tests/_realized_geometry.realized`` assembles and realizes.
    """
    sim = build(kind)
    rz = realized(sim)
    k_gnd = node_index(rz.grid, 2, Z_GND_PLANE)
    k_tr = node_index(rz.grid, 2, Z_TRACE_PLANE)
    if kind == "sheet":
        want_gnd, want_upper = [k_gnd], [k_gnd, k_tr]
    else:
        want_gnd = [k_gnd, k_gnd + 1]
        want_upper = [k_gnd, k_gnd + 1, k_tr, k_tr + 1]
    x_p0 = X_PATCH0
    columns = (
        ("ground_only", 0.5 * (x_p0 + L + DOM_X), 0.5 * (Y_C - W / 2),
         want_gnd),
        ("patch", x_p0 + 0.5 * L, Y_C, want_upper),
        ("feed", 0.5 * x_p0, Y_C, want_upper),
    )
    out = {}
    for name, x, y, want in columns:
        i = node_index(rz.grid, 0, x)
        j = node_index(rz.grid, 1, y)
        got = assert_wall_planes(sim, 2, expected_planes=want, ij=(i, j),
                                 what=f"{kind} column '{name}'")
        out[name] = dict(realized_planes=got,
                         realized_z_mm=[_node_position(rz.grid, 2, k) * 1e3
                                        for k in got])
        if verbose:
            print(f"  [{kind}] {name:12s} at (x={x*1e3:6.2f}, y={y*1e3:6.2f}) mm"
                  f" -> realized z walls "
                  f"{[f'{v:.4f}' for v in out[name]['realized_z_mm']]} mm")
    if verbose:
        print(f"  [{kind}] declared: ground {Z_GND_PLANE*1e3:.4f} mm "
              f"(node {k_gnd}), feed and patch {Z_TRACE_PLANE*1e3:.4f} mm "
              f"(node {k_tr})")
    return out


def _node_position(grid, axis: int, k: int) -> float:
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    return float(np.asarray(coords_from_uniform_grid(grid)[axis])[k])


def run_arm(tag: str, kind: str):
    sim = build(kind)
    assert_realized_planes(kind)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="build-time realized-plane gate only; no solve")
    args = ap.parse_args()
    if args.check:
        print("build-time realized-plane gate (#931 §1.2/§1.3), no solve:")
        for kind in ("sheet", "volume"):
            assert_realized_planes(kind)
        print("OK: every foil realizes exactly its declared plane(s).")
        return
    spec_s, ok_s = run_arm("S sheet", "sheet")
    spec_v, ok_v = run_arm("V one-cell volume", "volume")
    print("\n=== VERDICT (pre-declared in module docstring) ===")
    if not (ok_s and ok_v):
        print("  NOT READ: a settling witness failed; no Q number is trusted.")
        return
    s, v = tm010_of(spec_s), tm010_of(spec_v)
    print(f"  TM010 S(sheet, 35 um foil declared as a sheet): {s}")
    print(f"  TM010 V(one-cell volume, a 196.75 um plate):    {v}")
    if s is None:
        print("  -> TM010 MISSING in the SHEET arm: the sheet declaration "
              "suppresses the radiating mode. The public 'foil is a sheet' "
              "guidance must be qualified.")
        return
    if v is None:
        print("  -> TM010 MISSING in the VOLUME arm: a one-cell plate closes "
              "the mode. A caveat on drawing foil as a Box, not on sheets.")
        return
    ratio = v[1] / s[1]
    print(f"  Q ratio V/S = {ratio:.2f}  (band 0.77-1.30 = declaration does "
          f"not decide radiation damping)")
    print(f"  f shift V-S = {v[0] - s[0]:+.3f} GHz  (reported, not gated)")
    if 1.0 / 1.30 <= ratio <= 1.30:
        print("  -> the declaration does not decide radiation damping at "
              "textbook scale.")
    elif ratio > 1.30:
        print("  -> the VOLUME arm narrows: the 196.75 um plate closes the "
              "cavity. Caveat on drawing foil as a one-cell Box.")
    else:
        print("  -> the SHEET arm narrows: the sheet declaration damages "
              "radiation coupling. STOP and qualify the public guidance.")


if __name__ == "__main__":
    main()
