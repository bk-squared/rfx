#!/usr/bin/env python3
"""Where do the walls actually stand, and does the patch still read 9.2 GHz?

THE QUESTION. A 17 um foil cannot be a cell on a mm-scale board mesh, so
rfx puts an electric wall on ONE node plane per sheet. That leaves the
cavity between two sheets bounded by planes that are not the copper faces.
On this fixture the cost is large and measurable: the walls stand
dx + h_sub apart where the physics has h_sub.

The board campaign has three configurations that disagree, and every
attempt to make the geometry MORE exact has agreed with the reference
LESS. This fixture is the same question with a known answer (realized-raster Balanis 9.3305
GHz, openEMS 9.20), so it is where the question should be settled.

STAGE 1 (this file, no solve): for each realization, report the wall
planes taken from the conductor mask the solve will use, the cavity cell
count, and the series-capacitance measure sum(d/eps) against the physical
h_sub/eps_r. Anything that disagrees with the declared stack shows up here
before a single time step is spent.

STAGE 2 (--solve): ring down each arm and read the TM010 against 9.3305.

REALIZATIONS
  face1  the committed fixture: each sheet is ONE cell, faces on nodes,
         wall on the cell's LOWER node plane only.
  face2  each sheet is TWO cells. A conductor with >= 2 cells of extent
         has no far-face ambiguity -- the standard Yee treatment zeroes
         both bounding planes -- so this is the reference the other rows
         are measured against. The substrate is unchanged, so the cavity
         between the ground's TOP face and the patch's BOTTOM face is
         exactly h_sub.
  tp     the committed fixture with two_plane=True on every sheet.

PRE-DECLARED READING (before the run)
  Stage 1 is descriptive, not a verdict: it fixes what each arm HAS.
  Stage 2: the arm whose TM010 is closest to 9.3305 GHz wins, and `face2`
  is the ruler -- if `face2` itself misses 9.3305 by more than the coarse
  mesh's own bias (the pre-#702 fixture read 9.32, -0.1%), then the
  sheet realization is NOT what moved this fixture and the search moves
  elsewhere. A settling witness above -40 dB voids that arm.

RECORDED VERDICT (2026-08-28). Source: docs/agent-memory/rfx-known-issues.md,
"Added 2026-08-28 (evening) -- A/B VERDICT ..." and "... canonical edge-fed
patch gate RED since #702" (anchor corrected 2026-08-29 to the realized
43 x 51 raster; the agent-memory file is local to the primary checkout).

  Stage 1 (no solve; re-run from this file on main b5605391, 2026-09-02,
  same numbers):
    face1  wall planes 4131.8 / 5115.5 um on the patch column -> the walls
           stand 983.75 um apart (5 cells; the table prints 983.8) where the
           copper faces are 787.0 um apart: +25.0 % on sum(d/eps), every
           cell in the gap at eps 3.38.
    face2  the far face of a 2-cell conductor is NOT zeroed either: an
           N-cell conductor stands N wall planes, one per masked cell at
           that cell's LOWER node plane, and its top face plane never.  The
           2-cell metal therefore gives the same 983.75 um cavity, here with
           the ground's own upper cell (eps 1.00) inside it, sum(d/eps)
           +84.5 %.  The premise above -- that >= 2 cells of extent removes
           the far-face ambiguity -- is FALSIFIED by this stage; face2 is
           not a ruler.
    tp     walls 787.0 um apart, 0.0 %: two_plane is the only realization
           whose cavity equals the copper-face spacing.

  Stage 2 (--solve; TM010 against the realized-raster Balanis 9.3305 GHz):
    one-plane 8.162 GHz (-12.52 %), 2-cell metal 7.50 (-19.62 %),
    two_plane 8.22 (-11.90 %).  Making the cavity exact does not recover
    9.2-9.3; a 25 % cavity-thickness error moves this resonance ~0.7 %.

  --hsweep (dx fixed, h in whole cells, sheet-cell share of the cavity
  1/4 -> 1/6): measured/Balanis ratios 0.882 / 0.888 / 0.893 -- flat, so
  the sheet's own cell is not the cause of the residual post-#702 bias.
  Where that residual comes from was left open here; the isolated-patch
  refinement ladder took it up (known-issues "Added 2026-08-30", section
  3: mostly the Balanis anchor's own error at this h/lambda, ~ -2 pp O(dx)
  at h/4).
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from rfx import Box, Simulation
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
TARGET_GHZ = 9.3305        # Balanis on the h/4 REALIZED raster (43 x 51 cells =
#                            8.46025 x 10.03425 mm), not on the design dimensions.
#                            The retired 9.21 was Balanis on the DESIGN L/W; see
#                            issue #782 for every surface that still quotes it.


def build(kind: str, h_sub: float = H_SUB, dx: float = DX) -> Simulation:
    n_metal = 2 if kind == "face2" else 1
    t_metal = n_metal * dx
    dom_z = DOM_Z + (n_metal - 1) * dx
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, dom_z),
                     dx=dx, cpml_layers=8, boundary="cpml")
    kw = dict(two_plane=True) if kind == "tp" else {}
    z_gnd_hi = Z_GND + t_metal
    z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + h_sub
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + t_metal
    x_patch0 = PORT_MARGIN + FEED_LEN
    y_c = DOM_Y / 2.0
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, Z_GND), (DOM_X, DOM_Y, z_gnd_hi)), material="pec", **kw)
    sim.add(Box((0, 0, z_sub_lo), (DOM_X, DOM_Y, z_sub_hi)), material="ro4003c")
    sim.add(Box((0, y_c - W_MSL / 2, z_tr_lo),
                (x_patch0, y_c + W_MSL / 2, z_tr_hi)), material="pec", **kw)
    sim.add(Box((x_patch0, y_c - W / 2, z_tr_lo),
                (x_patch0 + L, y_c + W / 2, z_tr_hi)), material="pec", **kw)
    sim.add_msl_port(position=(PORT_MARGIN, y_c, z_sub_lo), width=W_MSL,
                     height=h_sub, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6))
    sim.add_probe(position=(x_patch0 + 0.7 * L, y_c - 0.2 * W,
                            0.5 * (z_sub_lo + z_sub_hi)), component="ez")
    return sim


def _two_plane_cells(sim, grid):
    """The two_plane cell mask the runner would build, or None."""
    entries = [e for e in getattr(sim, "_geometry", [])
               if getattr(e, "two_plane", False)]
    if not entries:
        return None
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    c = coords_from_uniform_grid(grid)
    m = None
    for e in entries:
        sub = np.asarray(e.shape.mask_on_coords(c.x, c.y, c.z), dtype=bool)
        m = sub if m is None else (m | sub)
    return m


def rasterization(sim, h_sub=H_SUB, dx=DX):
    """Where tangential E is actually zeroed, and what sits between.

    NOT the conductor mask's gap. The mask marks CELLS; the boundary
    condition lands on NODE PLANES, and for a one-cell sheet the wall is on
    the cell's LOWER plane only -- so the sheet's own cell is INSIDE the
    field region, carrying whatever permittivity the rasterizer left there.
    Measuring the gap between masks hides exactly that cell, which is the
    quantity this whole question is about. So the walls are read from
    ``tangential_edge_masks``, the same function the solve applies.
    """
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    from rfx.boundaries.pec import tangential_edge_masks
    grid = sim._build_grid()
    cond = np.asarray(sim.conductor_mask(grid), dtype=bool)
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, dtype=float)
    c = coords_from_uniform_grid(grid)
    z = np.asarray(c.z, dtype=float)
    xc, yc = np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float)
    x_patch0 = PORT_MARGIN + FEED_LEN
    i = int(np.argmin(np.abs(xc - (x_patch0 + 0.5 * L))))
    j = int(np.argmin(np.abs(yc - DOM_Y / 2.0)))

    mex, mey, _ = tangential_edge_masks(cond, (False, False, False))
    # two_plane adds its far-face planes through a SEPARATE function that
    # apply_pec_mask ORs in; measuring only tangential_edge_masks reports a
    # two_plane arm as if the flag did nothing.
    tpm = _two_plane_cells(sim, grid)
    if tpm is not None and bool(np.asarray(tpm).any()):
        from rfx.boundaries.pec import two_plane_extension_masks
        ex2, ey2, _ = two_plane_extension_masks(cond, np.asarray(tpm),
                                                (False, False, False))
        mex = np.asarray(mex) | np.asarray(ex2)
        mey = np.asarray(mey) | np.asarray(ey2)
    # a z-normal wall zeroes the in-plane components Ex and Ey
    walls = np.flatnonzero(np.asarray(mex)[i, j, :] | np.asarray(mey)[i, j, :])
    out = {"walls_k": walls, "walls_um": [float(z[k] * 1e6) for k in walls],
           "physical_cavity_um": float(h_sub * 1e6),
           "physical_sum_d_over_eps_um": float(h_sub / EPS_R * 1e6)}
    if walls.size >= 2:
        # THE CAVITY, not the outermost pair. A conductor can stand several
        # wall planes (two_plane adds its far face; a thick body stands one
        # per cell); the field region between two sheets is bounded by the
        # HIGHEST wall below the substrate and the LOWEST wall above it.
        # Taking walls[0]..walls[-1] measures across the metal as well and
        # reported a two_plane arm at +109% when its cavity is exact.
        k_mid = int(np.argmin(np.abs(z - (0.5 * (z[int(walls[0])]
                                                 + z[int(walls[-1])])))))
        below = walls[walls <= k_mid]
        above = walls[walls > k_mid]
        if below.size == 0 or above.size == 0:
            below, above = walls[:1], walls[-1:]
        a, b = int(below[-1]), int(above[0])
        out["wall_lo_um"] = float(z[a] * 1e6)
        out["wall_hi_um"] = float(z[b] * 1e6)
        out["gap_um"] = float((z[b] - z[a]) * 1e6)
        out["gap_cells"] = b - a
        out["sum_d_over_eps_um"] = float(np.sum(dx / eps[i, j, a:b]) * 1e6)
        out["eps_in_gap"] = (float(eps[i, j, a:b].min()),
                             float(eps[i, j, a:b].max()))
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
    p.add_argument("--kinds", default="face1,face2,tp")
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
            sim = build("face1", h_sub=h)
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
    print("\nRead against the pre-declared rule in this file's docstring: "
          "face2 is the ruler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
