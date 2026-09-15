#!/usr/bin/env python3
"""Where do the walls actually stand, and does the patch still read 9.2 GHz?

THE QUESTION, as it stands from 2.0 (#931). A 17 um foil cannot be a cell on
a mm-scale board mesh. The contract's answer is that it is not one: a foil is
declared a SHEET and realized on one node plane, owning no cell. The question
this ladder now asks is whether that declaration, a one-cell PEC volume and a
two-cell PEC volume put their walls where each says it does, what each leaves
inside the cavity, and what the patch then reads.

STAGE 1 (this file, no solve): for each arm, report the wall planes taken
from ``realized_pec_edge_masks`` — the same function the solve applies — the
cavity cell count, and the series-capacitance measure sum(d/eps) against the
physical h_sub/eps_r.

STAGE 2 (--solve): ring down each arm and read the TM010 against 9.3305.

ARMS
  sheet  ground, feed and patch declared with ``add_thin_conductor`` on the
         substrate's own faces. Each realizes ONE wall plane, at the face.
         No metal cell exists, so the cavity is exactly h_sub of eps_r 3.38.
  vol1   the same foils drawn as ONE-cell PEC Boxes — what every pre-2.0
         script in this repository drew. Under the volume rule each realizes
         walls at BOTH drawn faces, so the cavity is again exactly h_sub;
         what differs is that 196.75 um of the domain is now metal on each
         side, five and a half times the physical copper.
  vol2   the same foils two cells thick. Same cavity again; more metal.

  There is no `tp` arm any more. ``two_plane`` was a toggle on ONE
  declaration and it is gone: `vol1` is what it produced, at every
  thickness, with no flag.

ON-LATTICE (#931 sec 1.3). Z_GND moved from 4.000 mm to 21*DX = 4.13175 mm.
At 4.000 mm the substrate floor is 20.33 cells up — not a node — so a sheet
declared there snaps and lands buried inside the laminate. The contract does
not hide that behind a tie rule, so the board is drawn on the lattice. This
is a fixture change; the numbers recorded below were measured before it.

PRE-DECLARED READING (before the run)
  Stage 1 is descriptive: it fixes what each arm HAS. The contract's own
  prediction is that all three arms show sum(d/eps) at 0.0 % against the
  physical stack, because drawn extent equals realized extent in every one
  of them — that is the claim, and stage 1 is where it is falsified if it
  is false.
  Stage 2: the arm whose TM010 is closest to 9.3305 GHz wins. A settling
  witness above -40 dB voids that arm.

RECORDED VERDICT (2026-08-28) — PRE-2.0, kept as dated history. Source:
docs/agent-memory/rfx-known-issues.md, "Added 2026-08-28 (evening) -- A/B
VERDICT ..." and "... canonical edge-fed patch gate RED since #702" (anchor
corrected 2026-08-29 to the realized 43 x 51 raster; that file is local to
the primary checkout). The arms below are the OLD ones (face1 = one-cell
Box under the one-wall rule, face2 = two-cell Box under it, tp = the
two_plane flag), none of which exists now.

  Stage 1 (no solve; re-run on main b5605391, 2026-09-02, same numbers):
    face1  wall planes 4131.8 / 5115.5 um on the patch column -> the walls
           stand 983.75 um apart (5 cells) where the copper faces are 787.0
           um apart: +25.0 % on sum(d/eps).
    face2  the far face of a 2-cell conductor was NOT zeroed either: an
           N-cell conductor stood N wall planes, one per masked cell at that
           cell's LOWER node plane, and its top face plane never. Same
           983.75 um cavity, with the ground's own upper cell (eps 1.00)
           inside it: sum(d/eps) +84.5 %.
    tp     walls 787.0 um apart, 0.0 %.

  Stage 2 (--solve; TM010 against realized-raster Balanis 9.3305 GHz):
    one-plane 8.162 GHz (-12.52 %), 2-cell metal 7.50 (-19.62 %),
    two_plane 8.22 (-11.90 %). Making the cavity exact did not recover
    9.2-9.3; a 25 % cavity-thickness error moves this resonance ~0.7 %.

  --hsweep (dx fixed, h in whole cells, sheet-cell share of the cavity
  1/4 -> 1/6): measured/Balanis ratios 0.882 / 0.888 / 0.893 -- flat, so
  the sheet's own cell was not the cause of the residual post-#702 bias.
  The isolated-patch refinement ladder took that up (known-issues "Added
  2026-08-30", section 3: mostly the Balanis anchor's own error at this
  h/lambda, ~ -2 pp O(dx) at h/4).

STAGE 1 UNDER THE CONTRACT (measured 2026-09-07 on feat/931-docs, no solve,
the arms above redrawn on the lattice). The contract's prediction held on
all three arms:

  arm    wall planes on the patch column (um)     gap   sum d/eps  vs physical
  sheet  4131.8, 4918.8                         787.0     232.8       -0.0 %
  vol1   4131.8, 4328.5, 5115.5, 5312.2         787.0     232.8       -0.0 %
  vol2   4131.8, 4328.5, 4525.3,
         5312.2, 5509.0, 5705.8                 787.0     232.8       -0.0 %

  Every foil realizes walls at BOTH of its drawn faces, so every arm's
  cavity is the drawn 787.0 um of eps_r 3.38 and no metal cell sits inside
  it. The pre-2.0 table read +25.0 % / +84.5 % / 0.0 % for the same three
  drawings. Stage 2 has NOT been re-run under the contract — its arms are
  not the arms above, so the 2026-08-28 TM010 numbers are not comparable and
  are not carried forward as a prediction.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from rfx import Box, Simulation
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources import GaussianPulse

# The ONE spelling of the build-time realization read (#931 §1.7): a second
# hand-rolled scan over an edge mask is the drift the single-owner rule exists
# to stop, and this file's whole job is to report where the walls stand.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "tests"))
from _realized_geometry import realized  # noqa: E402

EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
PORT_MARGIN = 5.0e-3
FEED_LEN = 8.0e-3
DOM_X, DOM_Y, DOM_Z = 29.747e-3, 18.130e-3, 12.787e-3
N_SUB_CELLS = 4
DX = H_SUB / N_SUB_CELLS
Z_GND = 21 * DX            # ON A NODE (see the docstring); was 4.000 mm
NUM_PERIODS = 120.0
SETTLING_BAR_DB = -40.0
TARGET_GHZ = 9.3305        # Balanis on the h/4 REALIZED raster (43 x 51 cells =
#                            8.46025 x 10.03425 mm), not on the design dimensions.
#                            The retired 9.21 was Balanis on the DESIGN L/W; see
#                            issue #782 for every surface that still quotes it.


def build(kind: str, h_sub: float = H_SUB, dx: float = DX) -> Simulation:
    """``kind`` is ``"sheet"``, ``"vol1"`` or ``"vol2"``."""
    if kind not in ("sheet", "vol1", "vol2"):
        raise ValueError(kind)
    n_metal = 2 if kind == "vol2" else 1
    t_metal = 0.0 if kind == "sheet" else n_metal * dx
    dom_z = DOM_Z + (n_metal - 1) * dx
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, dom_z),
                     dx=dx, cpml_layers=8, boundary="cpml")
    z_gnd_hi = Z_GND + t_metal
    z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + h_sub
    x_patch0 = PORT_MARGIN + FEED_LEN
    y_c = DOM_Y / 2.0
    foils = (
        ((0.0, 0.0), (DOM_X, DOM_Y), Z_GND, z_gnd_hi),
        ((0.0, y_c - W_MSL / 2), (x_patch0, y_c + W_MSL / 2),
         z_sub_hi, z_sub_hi + t_metal),
        ((x_patch0, y_c - W / 2), (x_patch0 + L, y_c + W / 2),
         z_sub_hi, z_sub_hi + t_metal),
    )
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    for (x0, y0), (x1, y1), z_lo, z_hi in foils[:1]:
        _add_foil(sim, kind, x0, y0, x1, y1, z_lo, z_hi)
    sim.add(Box((0, 0, z_sub_lo), (DOM_X, DOM_Y, z_sub_hi)), material="ro4003c")
    for (x0, y0), (x1, y1), z_lo, z_hi in foils[1:]:
        _add_foil(sim, kind, x0, y0, x1, y1, z_lo, z_hi)
    sim.add_msl_port(position=(PORT_MARGIN, y_c, z_sub_lo), width=W_MSL,
                     height=h_sub, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6))
    sim.add_probe(position=(x_patch0 + 0.7 * L, y_c - 0.2 * W,
                            0.5 * (z_sub_lo + z_sub_hi)), component="ez")
    return sim


def _add_foil(sim, kind, x0, y0, x1, y1, z_lo, z_hi):
    if kind == "sheet":
        sim.add_thin_conductor(Box((x0, y0, z_lo), (x1, y1, z_lo)))
    else:
        sim.add(Box((x0, y0, z_lo), (x1, y1, z_hi)), material="pec")


def rasterization(sim, h_sub=H_SUB, dx=DX):
    """Where tangential E is actually zeroed, and what sits between.

    NOT the conductor mask's gap. The mask marks CELLS; the boundary
    condition lands on NODE PLANES, and a sheet has no cell at all — so a
    cell-based measure cannot see a sheet, and for a volume it cannot see
    the far wall, which stands on a plane outside the body's own cells. The
    walls are therefore read from ``realized_pec_edge_masks``, the ONE
    function the solve applies (#931 sec 1.7).
    """
    rz = realized(sim)
    grid = rz.grid
    eps = np.asarray(sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0].eps_r, dtype=float)
    c = coords_from_uniform_grid(grid)
    z = np.asarray(c.z, dtype=float)
    xc, yc = np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float)
    x_patch0 = PORT_MARGIN + FEED_LEN
    i = int(np.argmin(np.abs(xc - (x_patch0 + 0.5 * L))))
    j = int(np.argmin(np.abs(yc - DOM_Y / 2.0)))
    # a z-normal wall zeroes the in-plane components Ex and Ey; ij= asks the
    # contract's own helper for this column, backward neighbours included
    walls = np.asarray(rz.wall_planes(2, ij=(i, j)), dtype=int)
    out = {"walls_k": walls, "walls_um": [float(z[k] * 1e6) for k in walls],
           "physical_cavity_um": float(h_sub * 1e6),
           "physical_sum_d_over_eps_um": float(h_sub / EPS_R * 1e6)}
    if walls.size >= 2:
        # THE CAVITY, not the outermost pair. A volume stands a wall on every
        # plane it spans, so walls[0]..walls[-1] measures across the metal as
        # well; the field region between the two foils is bounded by the
        # HIGHEST wall below the substrate and the LOWEST wall above it.
        k_mid = int(np.argmin(np.abs(z - (0.5 * (z[int(walls[0])]
                                                 + z[int(walls[-1])])))))
        below = walls[walls <= k_mid]
        above = walls[walls > k_mid]
        if below.size == 0 or above.size == 0:
            below, above = walls[:1], walls[-1:]
        a_, b_ = int(below[-1]), int(above[0])
        out["wall_lo_um"] = float(z[a_] * 1e6)
        out["wall_hi_um"] = float(z[b_] * 1e6)
        out["gap_um"] = float((z[b_] - z[a_]) * 1e6)
        out["gap_cells"] = b_ - a_
        out["sum_d_over_eps_um"] = float(np.sum(dx / eps[i, j, a_:b_]) * 1e6)
        out["eps_in_gap"] = (float(eps[i, j, a_:b_].min()),
                             float(eps[i, j, a_:b_].max()))
    return out


def solve(sim, tag):
    from rfx.harminv import harminv
    adv = [str(a) for a in sim.preflight()]
    print(f"[{tag}] preflight ({len(adv)}), quoted verbatim:")
    for a in adv:
        print(f"   ! {a[:200]}")
    res = sim.run(num_periods=NUM_PERIODS)
    ts = np.asarray(res.time_series).ravel()
    env = np.abs(ts)
    end_db = 20.0 * math.log10(max(float(np.max(env[int(len(env) * 0.95):])), 1e-300)
                               / max(float(np.max(env)), 1e-300))
    modes = [m for m in harminv(ts[int(len(ts) * 0.3):], float(res.dt), 6e9, 14e9)
             if m.Q > 2 and abs(m.amplitude) > 1e-9]
    spec = sorted((m.freq / 1e9, m.Q, float(abs(m.amplitude))) for m in modes)
    band = [t for t in spec if 6.5 <= t[0] <= 11.0]
    tm010 = max(band, key=lambda t: t[2]) if band else None
    print(f"[{tag}] settling {end_db:.1f} dB (bar {SETTLING_BAR_DB})")
    print(f"[{tag}] spectrum {[f'{f:.2f}/Q{q:.0f}/a{a:.2g}' for f, q, a in spec]}")
    return dict(tag=tag, settled=end_db < SETTLING_BAR_DB, end_db=end_db,
                spectrum=spec, tm010=tm010)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solve", action="store_true")
    p.add_argument("--kinds", default="sheet,vol1,vol2")
    p.add_argument("--hsweep", action="store_true",
                   help="dx FIXED, h_sub varied in whole cells")
    a = p.parse_args()
    kinds = a.kinds.split(",")

    if a.hsweep:
        # dx is HELD FIXED and h varied in whole cells, so the sheet cell is a
        # varying fraction (1/n) of the cavity. If the model's error is a
        # velocity/permittivity offset it is CONSTANT across the sweep; if it
        # comes from the sheet's own cell it shrinks as 1/n.
        C0 = 299792458.0

        def balanis(Lm, Wm, h, er):
            ee = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / Wm) ** -0.5
            dL = (0.412 * h * (ee + 0.3) * (Wm / h + 0.264)
                  / ((ee - 0.258) * (Wm / h + 0.8)))
            return C0 / (2 * (Lm + 2 * dL) * np.sqrt(ee)) / 1e9

        print("=== h SWEEP (dx fixed at "
              f"{DX*1e6:.2f} um, h in whole cells) ===")
        print(f"{'h cells':>8} {'h (um)':>9} {'measured':>10} {'Balanis':>9} "
              f"{'ratio':>7} {'settled':>8}")
        for n in (3, 4, 5, 6, 8):
            h = n * DX
            sim = build("sheet", h_sub=h)
            r = solve(sim, f"h={n}cells")
            if r["tm010"] is None:
                print(f"{n:8d} {h*1e6:9.1f}   no mode found")
                continue
            f_meas = r["tm010"][0]
            f_bal = balanis(L, W, h, EPS_R)
            print(f"{n:8d} {h*1e6:9.1f} {f_meas:10.3f} {f_bal:9.3f} "
                  f"{f_meas/f_bal:7.3f} {'yes' if r['settled'] else 'NO':>8}")
        print("\nCONSTANT ratio => a velocity/permittivity offset that does not "
              "come from the sheet cell.\nRatio -> 1 as n grows => the sheet "
              "cell is the cause.")
        return 0

    print("=== STAGE 1: rasterization (no solve) ===")
    print(f"physical cavity {H_SUB*1e6:.1f} um of eps_r {EPS_R} -> "
          f"sum(d/eps) {H_SUB/EPS_R*1e6:.1f} um   dx = {DX*1e6:.2f} um\n")
    print(f"{'arm':<7} {'wall planes on the patch column (um)':>44}")
    for kind in kinds:
        r = rasterization(build(kind))
        ws = ", ".join(f"{v:.1f}" for v in r["walls_um"])
        print(f"{kind:<7} {ws:>44}")
    print()
    print(f"{'arm':<7} {'lo..hi wall (um)':>22} {'gap':>8} {'cells':>6} "
          f"{'sum d/eps':>10} {'vs physical':>12} {'eps in gap':>14}")
    for kind in kinds:
        r = rasterization(build(kind))
        if "gap_um" not in r:
            print(f"{kind:<7} {'<fewer than 2 wall planes>':>44}")
            continue
        rel = (r["sum_d_over_eps_um"] / r["physical_sum_d_over_eps_um"] - 1) * 100
        e = r["eps_in_gap"]
        print(f"{kind:<7} {r['wall_lo_um']:10.1f}..{r['wall_hi_um']:<10.1f} "
              f"{r['gap_um']:8.1f} {r['gap_cells']:6d} "
              f"{r['sum_d_over_eps_um']:10.1f} {rel:+11.1f}% "
              f"{e[0]:6.2f}..{e[1]:<6.2f}")

    if not a.solve:
        print("\n(stage 2 skipped; pass --solve)")
        return 0

    print("\n=== STAGE 2: ring-down vs realized-raster Balanis 9.3305 GHz ===")
    rows = [solve(build(k), k) for k in kinds]
    print(f"\n{'arm':<7} {'settled':>8} {'TM010 GHz':>10} {'Q':>7} "
          f"{'vs 9.3305':>10}")
    for r in rows:
        if not r["settled"]:
            print(f"{r['tag']:<7} {'NO':>8}   not read (truncation)")
            continue
        t = r["tm010"]
        if t is None:
            print(f"{r['tag']:<7} {'yes':>8}   no mode in 6.5-11 GHz")
            continue
        print(f"{r['tag']:<7} {'yes':>8} {t[0]:10.3f} {t[1]:7.1f} "
              f"{(t[0]-TARGET_GHZ)/TARGET_GHZ*100:+8.1f}%")
    print("\nPre-declared rule (docstring): the arm whose TM010 is closest "
          "to the realized-raster Balanis target wins; a settling witness "
          "above the bar voids that arm and its numbers are not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
