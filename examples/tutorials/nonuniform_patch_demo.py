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
     this pattern is ``tests/test_issue325_uniform_fine_substrate.py``.

  Rule 2 — NU TRADES ACCURACY FOR CELLS.  At matched dx a graded mesh is NOT
     more accurate than uniform — use it to make big problems tractable, not
     to chase digits.

  Rule 3 — A PROPERLY RESOLVED PATCH IS MULTI-MODE.  Once the substrate is
     truly 6 cells, harminv shows the patch's real modes (TM01 on the 38 mm
     width and TM10 on the 29.5 mm length at comparable amplitude, plus a
     higher mode).  Picking "the mode closest to the textbook estimate" is
     mode-AMBIGUOUS; identifying the RADIATING mode needs the far field —
     see ``examples/tutorials/patch_antenna_demo.py`` for that workflow.
     This script prints the full mode list and does not gate on any single
     frequency.

Runtime: ~13 min on CPU (measured 783 s; the num_periods=120 FDTD run
dominates and settles to -40.8 dB.  The mesh lesson itself — parts [1]-[3] —
is grid-only arithmetic and costs nothing).

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

from rfx import Simulation, Box
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
    np.full(n_below, dx),       # air below GP
    np.full(1, dz_sub),         # GP cell
    np.full(n_sub, dz_sub),     # substrate
    np.full(n_above, dx),       # air above patch
])
dz_naive = smooth_grading(raw_dz_naive, max_ratio=1.3)
# The intended coordinates ignore the transition cells smooth_grading inserted:
naive_layers = [
    ("ground",    air_below - dz_sub, air_below,                  1),
    ("substrate", air_below,          air_below + h_sub,          n_sub),
    ("patch",     air_below + h_sub,  air_below + h_sub + dz_sub, 1),
]
naive_ok = realized_layout(dz_naive, naive_layers)
print("  => smooth_grading inserted transition cells BELOW the fine band and "
      "shifted it up;\n     a Box at the intended z lands on coarse cells "
      "(issue #325 — preflight's\n     graded-box-rasterization advisory "
      "warns about exactly this class).")
assert not naive_ok, "expected the naive placement to misregister (the lesson)"

# =============================================================================
# PART 2 — THE FIX: buffered fine band + z DERIVED from the built profile.
# (Same pattern as tests/test_issue325_uniform_fine_substrate.py::
#  build_uniform_fine_z and scripts/diagnostics/patch_tutorial_rfx.py.)
# =============================================================================
print("\n[2] THE FIX — buffered fine band, z derived from the built mesh:")
raw_dz = np.concatenate([
    np.full(n_below, dx),
    np.full(n_buf + 1 + n_sub + 1 + n_buf, dz_sub),  # buf+GP+sub+patch+buf
    np.full(n_above, dx),
])
dz_profile = smooth_grading(raw_dz, max_ratio=1.3)
edges = np.concatenate([[0.0], np.cumsum(dz_profile)])
fi = np.where(np.isclose(dz_profile, dz_sub, rtol=1e-6))[0]
assert len(fi) >= 2 + n_sub + 2 * n_buf, \
    f"fine band lost: expected >= {2 + n_sub + 2*n_buf} fine cells, got {len(fi)}"
f0 = int(fi[0]) + n_buf                    # skip the lower buffer cells
z_gnd_lo,   z_gnd_hi   = float(edges[f0]),     float(edges[f0 + 1])
z_sub_lo,   z_sub_hi   = float(edges[f0 + 1]), float(edges[f0 + 1 + n_sub])
z_patch_lo, z_patch_hi = z_sub_hi,             float(edges[f0 + 2 + n_sub])

fixed_layers = [
    ("ground",    z_gnd_lo,   z_gnd_hi,   1),
    ("substrate", z_sub_lo,   z_sub_hi,   n_sub),
    ("patch",     z_patch_lo, z_patch_hi, 1),
]
fixed_ok = realized_layout(dz_profile, fixed_layers)
assert fixed_ok, "mesh-derived registration must realize the intended cells"
assert abs((z_sub_hi - z_sub_lo) - h_sub) < 1e-9

# transition clearance: nearest cell that is neither fine nor coarse
centers = 0.5 * (edges[:-1] + edges[1:])
is_trans = ~(np.isclose(dz_profile, dz_sub, rtol=1e-6) |
             np.isclose(dz_profile, dx, rtol=1e-6))
tc = centers[is_trans]
clearance = (float(min(np.min(np.abs(tc - z_gnd_lo)),
                       np.min(np.abs(tc - z_patch_hi))))
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
sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="fr4")
sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
            (patch_x_hi, patch_y_hi, z_patch_hi)), material="pec")
sim.add_source(
    position=(feed_x, feed_y, src_z),
    component="ez",
    waveform=GaussianPulse(f0=f_design, bandwidth=1.2),
)
sim.add_probe(
    position=(dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, probe_z),
    component="ez",
)

print("\n[4] Preflight (advisories below are part of the result):")
sim.preflight(strict=False)

n_periods = 120   # honest witness below; ~200 periods reaches ~-50 dB
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
  Reading this honestly (Rule 3): with the substrate truly resolved to 6 cells
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
