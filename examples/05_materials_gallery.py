"""Example 5: Materials Gallery

Demonstrates different material types in a PEC cavity:
  - Vacuum (eps_r=1.0)
  - FR4 substrate (eps_r=4.4)
  - Lossy dielectric (eps_r=2.5, sigma=0.1)
  - Debye dispersive material (water at 20°C)

Each sub-simulation uses a small PEC cavity with a point source.
The 2×2 figure shows the Ez field slice at the same timestep for
each material, with color indicating field strength.

Save: examples/05_materials_gallery.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse
from rfx.materials.debye import DebyePole

# ---------------------------------------------------------------------------
# Cavity parameters (shared across all sub-simulations)
# ---------------------------------------------------------------------------

FREQ_MAX = 10e9          # 10 GHz upper band
CAVITY_SIZE = 0.02       # 20 mm cube cavity
DX = 5e-4                # 0.5 mm cell size
N_STEPS = 200            # short run — enough to see field pattern
SOURCE_POS = (CAVITY_SIZE * 0.35, CAVITY_SIZE * 0.35, CAVITY_SIZE * 0.50)
PROBE_POS  = (CAVITY_SIZE * 0.65, CAVITY_SIZE * 0.65, CAVITY_SIZE * 0.50)

# ---------------------------------------------------------------------------
# Material definitions
# ---------------------------------------------------------------------------

MATERIALS = [
    {
        "label": "Vacuum\n(eps_r=1.0)",
        "kwargs": {},                         # uses library "vacuum"
        "lib_name": "vacuum",
    },
    {
        "label": "FR4\n(eps_r=4.4)",
        "kwargs": {"eps_r": 4.4, "sigma": 0.025},
        "lib_name": None,
    },
    {
        "label": "Lossy Dielectric\n(eps_r=2.5, σ=0.1 S/m)",
        "kwargs": {"eps_r": 2.5, "sigma": 0.1},
        "lib_name": None,
    },
    {
        "label": "Water 20°C\n(Debye, eps_inf=4.9)",
        "kwargs": {
            "eps_r": 4.9,
            "sigma": 0.0,
            "debye_poles": [DebyePole(delta_eps=74.1, tau=8.3e-12)],
        },
        "lib_name": None,
    },
]


def run_cavity(mat_def):
    """Build and run a small PEC cavity with the given filling material."""
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE),
        boundary="pec",
        dx=DX,
    )

    if mat_def["lib_name"] is not None:
        # Use material from library directly via add()
        sim.add(
            Box((0, 0, 0), (CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE)),
            material=mat_def["lib_name"],
        )
    else:
        sim.add_material("fill", **mat_def["kwargs"])
        sim.add(
            Box((0, 0, 0), (CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE)),
            material="fill",
        )

    sim.add_source(SOURCE_POS, "ez",
                   waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.8))
    sim.add_probe(PROBE_POS, "ez")

    result = sim.run(n_steps=N_STEPS)
    return result


# ---------------------------------------------------------------------------
# Run all four simulations
# ---------------------------------------------------------------------------

print("Running 4 material sub-simulations...")
results = []
for mat in MATERIALS:
    print(f"  {mat['label'].replace(chr(10), ' ')} ...")
    results.append(run_cavity(mat))
print("Done.\n")

# ---------------------------------------------------------------------------
# Build 2×2 visualization grid
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Ez Field Slice — Material Gallery (PEC Cavity, z-mid plane)",
             fontsize=14, y=1.01)

sim_ref = Simulation(
    freq_max=FREQ_MAX,
    domain=(CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE),
    boundary="pec",
    dx=DX,
)
grid = sim_ref._build_grid()
z_mid = grid.nz // 2

for ax, result, mat in zip(axes.ravel(), results, MATERIALS):
    state = result.state
    ez = np.asarray(state.ez)           # (nx, ny, nz)
    slc = ez[:, :, z_mid].T             # (ny, nx) — imshow expects (row, col)

    vmax = float(np.max(np.abs(slc))) or 1.0
    im = ax.imshow(
        slc,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        extent=[0, CAVITY_SIZE * 1e3, 0, CAVITY_SIZE * 1e3],
    )
    fig.colorbar(im, ax=ax, label="Ez (V/m)", fraction=0.046, pad=0.04)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(mat["label"], fontsize=11)

    # Mark source and probe positions
    ax.plot(SOURCE_POS[0] * 1e3, SOURCE_POS[1] * 1e3,
            "g^", markersize=8, label="source")
    ax.plot(PROBE_POS[0] * 1e3, PROBE_POS[1] * 1e3,
            "rs", markersize=8, label="probe")
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out_path = "examples/05_materials_gallery.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
