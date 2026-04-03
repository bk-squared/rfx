"""Example 7: auto_configure() for a Patch Antenna

Demonstrates the auto_configure() workflow:
  1. Define geometry as (Box, material_name) tuples
  2. Call auto_configure(geometry, freq_range, materials, accuracy="standard")
  3. Inspect config.summary() — dx, non-uniform dz, CPML, n_steps
  4. Build Simulation(**config.to_sim_kwargs()) from the config

2-panel figure:
  Panel 1 — dz_profile bar chart: cell index vs dz (mm)
             showing fine substrate cells vs coarse air cells
  Panel 2 — SimConfig parameter summary as a formatted text box

Save: examples/07_auto_configure.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.auto_config import auto_configure, SimConfig

# ---------------------------------------------------------------------------
# Patch antenna geometry (2.4 GHz on FR4)
# ---------------------------------------------------------------------------

F0 = 2.4e9
C0 = 3e8
EPS_R = 4.4
H_SUB = 1.6e-3   # 1.6 mm substrate

# Analytical patch dimensions
W = C0 / (2 * F0) * np.sqrt(2 / (EPS_R + 1))
EPS_EFF = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 * H_SUB / W) ** (-0.5)
DL = 0.412 * H_SUB * ((EPS_EFF + 0.3) * (W / H_SUB + 0.264) /
                       ((EPS_EFF - 0.258) * (W / H_SUB + 0.8)))
L = C0 / (2 * F0 * np.sqrt(EPS_EFF)) - 2 * DL

MARGIN = 20e-3   # 20 mm air margin around patch

print(f"Patch: L={L*1e3:.1f} mm, W={W*1e3:.1f} mm, h={H_SUB*1e3:.1f} mm")

# Domain starts at (0, 0, 0); substrate is at z=0..h_sub
DOM_X = L + 2 * MARGIN
DOM_Y = W + 2 * MARGIN

MATERIALS = {
    "pec":  {"eps_r": 1.0,   "sigma": 1e10},
    "fr4":  {"eps_r": EPS_R, "sigma": 2 * np.pi * F0 * 8.854e-12 * EPS_R * 0.02},
    "air":  {"eps_r": 1.0,   "sigma": 0.0},
}

GEOMETRY = [
    # Ground plane (zero-thickness — acts as PEC boundary in z)
    (Box((0,          0,          0),           (DOM_X, DOM_Y, H_SUB * 0.01)),  "pec"),
    # FR4 substrate
    (Box((0,          0,          0),           (DOM_X, DOM_Y, H_SUB)),         "fr4"),
    # Patch (PEC, thin layer on top of substrate)
    (Box((MARGIN,     MARGIN,     H_SUB),       (MARGIN + L, MARGIN + W,
                                                  H_SUB + H_SUB * 0.01)),       "pec"),
]

FREQ_RANGE = (1e9, 4e9)   # cover the 2.4 GHz band

# ---------------------------------------------------------------------------
# Auto-configure
# ---------------------------------------------------------------------------

print("\nCalling auto_configure()...")
config = auto_configure(
    geometry=GEOMETRY,
    freq_range=FREQ_RANGE,
    materials=MATERIALS,
    accuracy="standard",
)

summary_text = config.summary()
print(summary_text)

if config.warnings:
    for w in config.warnings:
        print(f"  WARNING: {w}")

# ---------------------------------------------------------------------------
# Build Simulation from config
# ---------------------------------------------------------------------------

sim = Simulation(**config.to_sim_kwargs())
print(f"\nSimulation built: {sim._domain[0]*1e3:.1f} x "
      f"{sim._domain[1]*1e3:.1f} x {sim._domain[2]*1e3:.1f} mm, "
      f"dx={sim._dx*1e3 if sim._dx else 'auto'} mm")

# ---------------------------------------------------------------------------
# Figure: 2-panel
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("auto_configure() — Patch Antenna on FR4", fontsize=13)

# -- Panel 1: dz_profile bar chart -----------------------------------------
if config.dz_profile is not None:
    dz_mm = config.dz_profile * 1e3
    cell_indices = np.arange(len(dz_mm))

    # Colour cells: fine (substrate) cells are darker
    dz_mean = np.mean(dz_mm)
    colors = ["#2171b5" if dz < dz_mean * 0.75 else "#9ecae1"
              for dz in dz_mm]

    ax1.bar(cell_indices, dz_mm, color=colors, edgecolor="none", width=1.0)
    ax1.axhline(config.dx * 1e3, color="orange", ls="--", linewidth=1.5,
                label=f"dx = {config.dx*1e3:.3f} mm (xy)")
    ax1.set_xlabel("Cell index (z-axis)")
    ax1.set_ylabel("dz (mm)")
    ax1.set_title("Non-Uniform dz Profile\n(dark = fine substrate cells)")
    ax1.legend(fontsize=9)

    # Annotate min/max
    ax1.text(0.02, 0.97,
             f"dz_min = {dz_mm.min():.3f} mm\ndz_max = {dz_mm.max():.3f} mm\n"
             f"n_cells = {len(dz_mm)}",
             transform=ax1.transAxes,
             va="top", ha="left", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray"))
else:
    # Uniform grid — show a flat bar chart instead
    n_z = max(1, int(round(config.domain[2] / config.dx)))
    dz_mm_uniform = np.full(n_z, config.dx * 1e3)
    ax1.bar(np.arange(n_z), dz_mm_uniform, color="#9ecae1",
            edgecolor="none", width=1.0)
    ax1.set_xlabel("Cell index (z-axis)")
    ax1.set_ylabel("dz (mm)")
    ax1.set_title("Uniform dz Profile\n(no non-uniform z required)")
    ax1.text(0.02, 0.97,
             f"dz = {config.dx*1e3:.3f} mm\nn_cells = {n_z}",
             transform=ax1.transAxes,
             va="top", ha="left", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray"))

# -- Panel 2: SimConfig text summary ----------------------------------------
ax2.axis("off")

# Build human-readable summary lines
lines = [
    ("accuracy",      config.accuracy),
    ("dx",            f"{config.dx*1e3:.3f} mm  ({config.cells_per_wavelength:.0f} cells/λ_min)"),
    ("domain x",      f"{config.domain[0]*1e3:.1f} mm"),
    ("domain y",      f"{config.domain[1]*1e3:.1f} mm"),
    ("domain z",      f"{config.domain[2]*1e3:.1f} mm"),
    ("cpml_layers",   str(config.cpml_layers)),
    ("n_steps",       f"{config.n_steps:,}"),
    ("sim_time",      f"{config.sim_time_ns:.1f} ns"),
    ("freq_min",      f"{config.freq_range[0]/1e9:.2f} GHz"),
    ("freq_max",      f"{config.freq_range[1]/1e9:.2f} GHz"),
    ("non-uniform z", "yes" if config.uses_nonuniform else "no"),
    ("margin",        f"{config.margin*1e3:.1f} mm"),
]
if config.warnings:
    lines.append(("WARNINGS", f"{len(config.warnings)} (see console)"))

col_width = 0.45
row_height = 0.065
x_key, x_val = 0.05, 0.5
y_start = 0.93

ax2.text(0.5, 0.99, "SimConfig Summary", ha="center", va="top",
         fontsize=12, fontweight="bold", transform=ax2.transAxes)

for i, (key, val) in enumerate(lines):
    y = y_start - i * row_height
    bg = "#f0f4ff" if i % 2 == 0 else "#ffffff"
    ax2.add_patch(plt.Rectangle((0.02, y - row_height * 0.55),
                                  0.96, row_height * 0.9,
                                  transform=ax2.transAxes,
                                  color=bg, zorder=0))
    ax2.text(x_key, y - row_height * 0.1, key + ":",
             transform=ax2.transAxes,
             va="center", ha="left", fontsize=10, color="#333333",
             fontweight="bold")
    ax2.text(x_val, y - row_height * 0.1, val,
             transform=ax2.transAxes,
             va="center", ha="left", fontsize=10, color="#1a1a7a")

ax2.set_title("SimConfig Parameters", fontsize=11)

plt.tight_layout()
out_path = "examples/07_auto_configure.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
