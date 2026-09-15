"""Non-uniform mesh workflow: resolving a thin substrate without a huge grid.

A 2.4 GHz FR4 patch antenna has a 1.5 mm substrate inside a ~100 mm domain —
uniform cells fine enough for the substrate would waste millions of cells in
air.  This tutorial shows the practical recipe AND the trap that comes with it:

  Rule 1 — REGISTER GEOMETRY TO THE BUILT MESH, NOT THE INTENDED ONE (#325).
     ``smooth_grading`` inserts transition cells that SHIFT the fine band in
     absolute z.  A substrate Box placed at pre-smoothing coordinates silently
     rasterizes onto coarse cells (here: 2 coarse cells instead of 6 fine —
     demonstrated below, with the realized cell layout printed for both the
     broken and the fixed placement).  The fix: derive every layer's z from
     where the built ``dz_profile`` actually put the fine cells, keep a
     uniform-fine BUFFER so the grading transition sits clear of the
     resonator, and assert the realized cell count.  The committed lock for
     this pattern is ``tests/unit/nonuniform/test_patch_uniform_fine_substrate.py``.

  Rule 2 — NU TRADES ACCURACY FOR CELLS.  At matched dx a graded mesh is NOT
     more accurate than uniform — use it to make big problems tractable, not
     to chase digits.

  Rule 3 — FOIL IS A SHEET.  The ground and the patch are 35 um of copper on
     a 1.5 mm board; no mesh this tutorial can afford resolves that, and the
     lattice ownership contract (#931) says so out loud: a foil is declared
     with ``add_thin_conductor`` on a zero-thickness Box and is realized as a
     footprint on ONE node plane, owning no cell.  So the fine band reserves
     cells for the SUBSTRATE only.  Reserving a cell for the metal (what this
     script used to do) hands the solve a vacuum cell inside the cavity: the
     realized board reads 2.0 mm instead of 1.5 mm.  A ``Box`` with
     ``material="pec"`` is the other declaration — a VOLUME, walls on both
     faces with the cells between them shorted — and is the right one for a
     plate, an iris or a post.

  Rule 4 — A PROPERLY RESOLVED PATCH IS MULTI-MODE.  Once the substrate is
     truly 6 cells, harminv shows the patch's real modes: TM01 on the 38 mm
     width and TM10 on the 29.5 mm length at comparable amplitude, plus a
     higher mode.  Picking "the mode closest to the textbook estimate" is
     mode-AMBIGUOUS; identifying the RADIATING mode needs the far field —
     see ``examples/tutorials/patch_antenna_demo.py`` for that workflow.
     This script prints the full mode list and gates on no single frequency.

Runtime: the pre-#931 board measured 1169 s (19 min) at num_periods=120 on a
64-core CPU run alone, settling -53.8 dB.  Under the ownership contract the
foils are sheets and the cavity lost its two vacuum cells, which raised the
modal Q: the same 120 periods measured -30.9 dB, under-settled by the witness
in part [5].  The run is 200 periods for that reason and measures -41.4 dB,
in 840 s of FDTD (VESSL 369367259280, CPU lane).  Parts [1]-[3], the mesh
lesson itself, are grid-only arithmetic and cost nothing to run.

The modes this now resolves, and why they are the patch's and not the mesh's
(VESSL 369367259280):

    1.9037 GHz  Q =  99.9   TM01 on the 38 mm width
    2.4446 GHz  Q =  51.6   TM10 on the 29.5 mm length — the design mode
    3.1847 GHz  Q =  97.9   TM11
    3.7588 GHz  Q = 116.7   TM02

TM10 sits 0.9 % under the Balanis estimate 2.4235 GHz.  The check that the
list is a patch and not four numbers: TM11 predicted from the measured pair as
sqrt(TM01^2 + TM10^2) is 3.100 GHz against 3.1847 measured, and TM02 predicted
as 2 x TM01 is 3.807 GHz against 3.7588 — both inside the coarse-mode spread,
so the ladder closes on itself.  The pre-contract board showed only three
modes (2.1393 / 2.7266 / 3.5067 GHz, VESSL 369367259021); every one of them
was HIGH, because a vacuum cell on each face of the cavity put air in series
with the laminate and lowered eps_eff.

Run:
  python examples/tutorials/nonuniform_patch_demo.py
"""

import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import (
    Box,
    Simulation,
    realized_pec_edge_masks,
    realized_wall_planes,
)
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading
from rfx.harminv import harminv

C0 = 2.998e8
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "nonuniform_patch_demo")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Geometry constants (identical to validation/crossval/05_patch_antenna.py)
# =============================================================================
f_design  = 2.4e9
eps_r     = 4.3
h_sub     = 1.5e-3
W         = 38.0e-3            # patch width  (y)  -> TM01 resonant length
L         = 29.5e-3            # patch length (x)  -> TM10 resonant length
gx        = 60.0e-3
gy        = 55.0e-3
air_above = 25.0e-3
air_below = 12.0e-3
probe_inset = 8.0e-3

dx     = 1.0e-3
n_cpml = 8
n_sub  = 6
dz_sub = h_sub / n_sub          # 0.25 mm
n_buf  = 8                      # uniform-fine buffer cells on EACH side of the
                                # stack: pushes the grading transition ~2 mm
                                # clear of the resonator (the lock test checks
                                # n_buf = 8/12/16 all give >= 2 mm clearance).

n_below = int(math.ceil(air_below / dx))
n_above = int(math.ceil(air_above / dx))

dom_x = gx + 2 * 10e-3
dom_y = gy + 2 * 10e-3

gx_lo = (dom_x - gx) / 2;  gx_hi = gx_lo + gx
gy_lo = (dom_y - gy) / 2;  gy_hi = gy_lo + gy
patch_x_lo = dom_x / 2 - L / 2;  patch_x_hi = dom_x / 2 + L / 2
patch_y_lo = dom_y / 2 - W / 2;  patch_y_hi = dom_y / 2 + W / 2
feed_x = patch_x_lo + probe_inset
feed_y = dom_y / 2

# =============================================================================
# Analytic reference (Balanis, Ch. 14) — PRINTED REFERENCE ONLY, not a gate:
# the transmission-line model is 5-8 % approximate for finite ground planes,
# and (Rule 3) no single-frequency comparison identifies the radiating mode.
# =============================================================================
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / W) ** (-0.5)
delta_L = 0.412 * h_sub * ((eps_eff + 0.3) * (W / h_sub + 0.264)) / \
          ((eps_eff - 0.258) * (W / h_sub + 0.8))
f_an = C0 / (2 * (L + 2 * delta_L) * math.sqrt(eps_eff))

print("=" * 60)
print("Nonuniform-mesh patch demo — register geometry to the BUILT mesh")
print("=" * 60)
print(f"  Patch: W={W*1e3:.1f} mm (TM01), L={L*1e3:.1f} mm (TM10), εr={eps_r}")
print(f"  Balanis TM10 estimate ≈ {f_an/1e9:.4f} GHz "
      f"(approximate reference, not a gate)")


def realized_layout(dz_profile, layers):
    """Print the REALIZED per-layer cell layout for a built dz profile.

    ``layers`` is a list of (name, z_lo, z_hi, intended_cells).  A cell
    belongs to a layer when its centre falls inside [z_lo, z_hi).  Returns
    True when every layer realized its intended cell count.
    """
    edges = np.concatenate([[0.0], np.cumsum(dz_profile)])
    centers = 0.5 * (edges[:-1] + edges[1:])
    print(f"  {'layer':<10} {'intended':>8} {'realized':>8}   "
          f"realized span / cell sizes")
    ok = True
    for name, z_lo, z_hi, intended in layers:
        idx = np.where((centers >= z_lo) & (centers < z_hi))[0]
        sizes = ", ".join(f"{dz_profile[i]*1e3:.3f}" for i in idx) or "-"
        span = (f"[{edges[idx[0]]*1e3:.2f}, {edges[idx[-1]+1]*1e3:.2f}] mm"
                if len(idx) else "(no cells)")
        flag = "" if len(idx) == intended else "   <-- MISREGISTERED"
        if len(idx) != intended:
            ok = False
        print(f"  {name:<10} {intended:>8} {len(idx):>8}   "
              f"{span}  dz = {sizes} mm{flag}")
    return ok


# =============================================================================
# PART 1 — THE TRAP (grid-only, costs nothing): place the stack at the
# INTENDED pre-smoothing z and look at what actually rasterizes.
# =============================================================================
print("\n[1] THE TRAP — geometry at fixed pre-smoothing z:")
raw_dz_naive = np.concatenate([
    np.full(n_below, dx),       # air below the board
    np.full(n_sub, dz_sub),     # substrate
    np.full(n_above, dx),       # air above the patch
])
dz_naive = smooth_grading(raw_dz_naive, max_ratio=1.3)
# The intended coordinates ignore the transition cells smooth_grading inserted:
naive_layers = [
    ("substrate", air_below, air_below + h_sub, n_sub),
]
naive_ok = realized_layout(dz_naive, naive_layers)
print("  => smooth_grading inserted transition cells BELOW the fine band and "
      "shifted it up;\n     a Box at the intended z lands on coarse cells "
      "(issue #325 — preflight's\n     graded-box-rasterization advisory "
      "warns about exactly this class).")
assert not naive_ok, "expected the naive placement to misregister (the lesson)"

# =============================================================================
# PART 2 — THE FIX: buffered fine band + z DERIVED from the built profile.
# (Same pattern as tests/unit/nonuniform/test_patch_uniform_fine_substrate.py::
#  build_uniform_fine_z and scripts/diagnostics/patch_tutorial_rfx.py.)
# =============================================================================
print("\n[2] THE FIX — buffered fine band, z derived from the built mesh:")
raw_dz = np.concatenate([
    np.full(n_below, dx),
    np.full(n_buf + n_sub + n_buf, dz_sub),   # buf + substrate + buf
    np.full(n_above, dx),
])
dz_profile = smooth_grading(raw_dz, max_ratio=1.3)
edges = np.concatenate([[0.0], np.cumsum(dz_profile)])
fi = np.where(np.isclose(dz_profile, dz_sub, rtol=1e-6))[0]
assert len(fi) >= n_sub + 2 * n_buf, \
    f"fine band lost: expected >= {n_sub + 2*n_buf} fine cells, got {len(fi)}"
f0 = int(fi[0]) + n_buf                    # skip the lower buffer cells
z_sub_lo, z_sub_hi = float(edges[f0]), float(edges[f0 + n_sub])
# The ground and the patch are FOIL, so they are SHEETS on the substrate's own
# two node planes — zero thickness, no cell of their own (#931 §1.3).  The
# fine band therefore reserves no cell for them; before the lattice ownership
# contract it reserved one each, and the ground's reserved cell sat inside the
# realized cavity carrying vacuum.
z_gnd, z_patch = z_sub_lo, z_sub_hi

fixed_layers = [
    ("substrate", z_sub_lo, z_sub_hi, n_sub),
]
fixed_ok = realized_layout(dz_profile, fixed_layers)
assert fixed_ok, "mesh-derived registration must realize the intended cells"
assert abs((z_sub_hi - z_sub_lo) - h_sub) < 1e-9

# transition clearance: nearest cell that is neither fine nor coarse
centers = 0.5 * (edges[:-1] + edges[1:])
is_trans = ~(np.isclose(dz_profile, dz_sub, rtol=1e-6) |
             np.isclose(dz_profile, dx, rtol=1e-6))
tc = centers[is_trans]
clearance = (float(min(np.min(np.abs(tc - z_gnd)),
                       np.min(np.abs(tc - z_patch))))
             if len(tc) else float("inf"))
print(f"  grading-transition clearance from the stack: {clearance*1e3:.2f} mm "
      f"(buffer n_buf={n_buf})")

# =============================================================================
# PART 3 — mesh economics (Rule 2): what the graded mesh buys.
# =============================================================================
nx = int(math.ceil(dom_x / dx))
ny = int(math.ceil(dom_y / dx))
nz_nu = len(dz_profile)
dom_z = float(edges[-1])
nz_uniform_equiv = int(math.ceil(dom_z / dz_sub))
ratio = (nx * ny * nz_uniform_equiv) / (nx * ny * nz_nu)
print(f"\n[3] Mesh economics: NU {nx}x{ny}x{nz_nu} vs uniform-at-dz_sub "
      f"{nx}x{ny}x{nz_uniform_equiv}  ->  {ratio:.2f}x fewer cells")

# =============================================================================
# PART 4 — build + preflight (output is part of the result) + run.
# FR4 is modelled lossless (sigma=0) so preflight's infinite-Q advisory will
# fire: legitimate here — we quote mode FREQUENCIES, not absolute Q (real FR4
# tan_delta ~0.02 would cap Q near ~50).
# The second line names the realized PEC sheets and whether each landed on the
# plane it declared (#931 §1.3): preflight collects sheets, so a sheet-declared
# conductor is visible to it.  It is a statement of where the metal is, not a
# finding about this geometry.
# =============================================================================
src_z   = z_sub_lo + dz_sub * 2.5
probe_z = src_z

sim = Simulation(
    freq_max=4e9,
    domain=(dom_x, dom_y, 0),
    dx=dx,
    dz_profile=dz_profile,
    boundary="cpml",
    cpml_layers=n_cpml,
)
sim.add_material("fr4", eps_r=eps_r, sigma=0.0)
# Ground and patch: zero-thickness Boxes through add_thin_conductor, i.e.
# SHEETS on the substrate's two faces.  add_thin_conductor warns that a
# metal's sigma_bulk makes this a lossless PEC sheet and that `thickness` is
# unread — expected, and the reason foil is declared this way instead of as a
# Box: a Box is a VOLUME and would put a shorted cell of metal into a 1.5 mm
# board with a wall on each of its faces.
sim.add_thin_conductor(Box((gx_lo, gy_lo, z_gnd), (gx_hi, gy_hi, z_gnd)))
sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="fr4")
sim.add_thin_conductor(Box((patch_x_lo, patch_y_lo, z_patch),
                           (patch_x_hi, patch_y_hi, z_patch)))
sim.add_source(
    position=(feed_x, feed_y, src_z),
    component="ez",
    waveform=GaussianPulse(f0=f_design, bandwidth=1.2),
    amplitude_kind="current",
)
sim.add_probe(
    position=(dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, probe_z),
    component="ez",
)

# --- Build-time realization check (no solve): declared planes == realized ---
# Read from the contract's own realization (realized_pec_edge_masks then
# realized_wall_planes, #931 §1.7) applied to the arrays the assembly hands
# the stepper, so this cannot drift from what the solve applies.
# `_assemble_materials_nu` is private only because the realized edge set has
# no public accessor yet; the two functions it feeds are public.
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid  # noqa: E402

_grid = sim._build_nonuniform_grid()
_sheets: list = []
_mat, _deb, _lor, _pec_cells = sim._assemble_materials_nu(
    _grid, pec_sheets=_sheets)
_z_nodes = np.asarray(coords_from_nonuniform_grid(_grid).z, dtype=float)
_walls = realized_wall_planes(
    realized_pec_edge_masks(_pec_cells, sheets=_sheets,
                            periodic=sim._periodic_flags()), 2)
k_gnd = int(np.argmin(np.abs(_z_nodes - z_gnd)))
k_patch = int(np.argmin(np.abs(_z_nodes - z_patch)))
print(f"  realized conductor planes along z: {_walls} "
      f"(declared ground {k_gnd} at {z_gnd*1e3:.3f} mm, "
      f"patch {k_patch} at {z_patch*1e3:.3f} mm)")
assert _walls == sorted({k_gnd, k_patch}), \
    f"realized z wall planes {_walls} != declared {sorted({k_gnd, k_patch})}"
assert abs(float(_z_nodes[k_patch] - _z_nodes[k_gnd]) - h_sub) < 1e-9, \
    "realized cavity thickness != declared h_sub"

print("\n[4] Preflight (advisories below are part of the result):")
sim.preflight(strict=False)

# 120 periods settled the PRE-#931 board (-53.8 dB, 2026-09-05). It does not
# settle this one: with the vacuum cells gone from the cavity the modes are
# higher-Q, and the #931 re-solve (VESSL 369367259177) measured -30.9 dB at
# 120 — UNDER-SETTLED by the script's own witness. The binding mode is TM01
# on the 38 mm width (1.9037 GHz, Q = 99.9), the slowest decayer in the set.
#
# 200 periods measures -41.4 dB (VESSL 369367259280). That clears the -40 dB
# bar, but by 1.4 dB, which is less margin than the number below was chosen
# for — worth knowing before anyone shortens this run or quotes a Q off it.
# The two estimates that picked 200 bracketed badly: TM01's free decay says
# -43.0 dB at 200 (1.6 dB pessimistic, close), while extrapolating the
# measured -30.9 dB linearly in time says -51.5 dB (10 dB optimistic, wrong —
# it charges the source ramp to the decay). The MEASURED slope between the two
# runs is 0.131 dB/period (-30.9 at 120, -41.4 at 200); size any future change
# from that, not from either estimate.
#
# Raising the run length is the response to an unsettled run; lowering the bar
# would not be.
n_periods = 200   # -41.4 dB measured; see the docstring runtime note
print(f"\nRunning NU simulation (num_periods={n_periods})...")
t0 = time.time()
result = sim.run(num_periods=n_periods)
print(f"Done in {time.time() - t0:.1f} s")

# =============================================================================
# PART 5 — settling witness BEFORE any frequency is quoted (#332/G1).
# =============================================================================
ts = np.asarray(result.time_series).ravel()
dt_val = float(result.dt)
amp = np.abs(ts)
peak = float(amp[int(len(amp) * 0.15):].max())
end = float(amp[-max(1, len(amp) // 20):].mean())
settle_db = (20 * math.log10(end / peak)
             if peak > 0 and end > 0 else float("-inf"))
print(f"\n[5] Settling witness: end/peak = {settle_db:.1f} dB "
      f"({'OK (<-40)' if settle_db < -40 else 'UNDER-SETTLED (>-40): raise num_periods for gate-grade numbers'})")

# =============================================================================
# PART 6 — harminv: the FULL mode list (Rule 3), no single-mode gate.
# =============================================================================
skip = int(len(ts) * 0.3)
signal = ts[skip:]
modes = [m for m in harminv(signal, dt_val, 1.5e9, 3.6e9)
         if m.Q > 2 and m.amplitude > 1e-8]
modes.sort(key=lambda m: m.freq)

print("\n[6] Harminv modes (all, sorted by frequency):")
for m in modes:
    print(f"  f = {m.freq/1e9:.4f} GHz   Q = {m.Q:5.1f}   amp = {m.amplitude:.2e}")
print("""
  Reading this honestly (Rule 4): with the substrate truly resolved to 6 cells
  the patch shows its REAL modes — the lower one lives on the wider W=38 mm
  dimension (TM01), the middle on L=29.5 mm (TM10, the radiating design mode),
  plus a higher-order mode.  Their amplitudes at a single probe are comparable,
  so "strongest" or "closest to the textbook number" would be an arbitrary,
  geometry-sensitive pick.  Identifying the RADIATING mode takes a far-field
  criterion (broadside beam + radiated power) — that workflow lives in
  examples/tutorials/patch_antenna_demo.py.""")

print(f"  Pass criterion of THIS tutorial: the substrate rasterized to "
      f"{n_sub} fine cells (asserted in [2]) — the mesh lesson, not a "
      f"frequency match.")

# =============================================================================
# PART 7 — plots: ringdown + the built dz profile with the REALIZED substrate.
# =============================================================================
t_axis = np.arange(len(ts)) * dt_val * 1e9   # ns
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_axis, ts, lw=0.7)
ax.axvline(skip * dt_val * 1e9, color="r", ls="--", lw=1.0, label="Harminv start")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez (a.u.)")
ax.set_title("Probe Ez — patch ringdown (NU runner)")
ax.legend()
fig.tight_layout()
plot_ts = os.path.join(OUT_DIR, "probe_ez_timeseries.png")
fig.savefig(plot_ts, dpi=120)
plt.close(fig)
print(f"\nPlot (a): {plot_ts}")

z_centers_mm = centers * 1e3
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.stem(z_centers_mm, dz_profile * 1e3, markerfmt="C0.", basefmt="k-",
         linefmt="C0-")
ax2.axvspan(z_sub_lo * 1e3, z_sub_hi * 1e3, alpha=0.15, color="orange",
            label="FR4 substrate (REALIZED z)")
ax2.set_xlabel("z (mm)")
ax2.set_ylabel("dz (mm)")
ax2.set_title("Non-uniform z mesh — substrate shaded at its realized position")
ax2.legend()
fig2.tight_layout()
plot_mesh = os.path.join(OUT_DIR, "dz_mesh_profile.png")
fig2.savefig(plot_mesh, dpi=120)
plt.close(fig2)
print(f"Plot (b): {plot_mesh}")
