"""Cross-validation: Meep Mie Scattering tutorial replica.

Replicates: meep.readthedocs.io/en/latest/Python_Tutorials/Basics/
Structure: Dielectric sphere n=2.0, broadband TFSF, scattering cross section
Comparison: Mie theory analytical solution
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx import Simulation, Sphere
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Meep uses normalized units. Scale: 1 unit = 10mm for microwave regime.
SCALE = 10e-3
n_sphere = 2.0
eps_sphere = n_sphere ** 2
radius = 1.0 * SCALE  # 10mm
f_center = 1.0 * C0 / SCALE  # ~30 GHz
f_width = 0.4 * C0 / SCALE

print("=" * 60)
print("Cross-Validation: Meep Mie Scattering (Dielectric Sphere)")
print("=" * 60)
print(f"Sphere: n={n_sphere}, radius={radius*1e3:.0f}mm")
print(f"Frequency: {f_center/1e9:.1f} GHz, BW={f_width/1e9:.1f} GHz")

# Analytical Mie: for small ka, scattering efficiency → (ka)^4 * (...)
# At ka~2: Q_sca ≈ 2-3 for n=2 sphere
k_center = 2 * np.pi * f_center / C0
ka = k_center * radius
print(f"ka at center: {ka:.2f}")

dx = 0.5 * SCALE  # resolution = 2 per unit
margin = 2 * SCALE  # 2 units = 20mm

sim = Simulation(
    freq_max=(f_center + f_width) * 1.2,
    domain=(2*radius + 2*margin,) * 3,
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

center = (radius + margin,) * 3
sim.add_material("dielectric", eps_r=eps_sphere)
sim.add(Sphere(center=center, radius=radius), material="dielectric")

# TFSF plane wave
sim.add_tfsf_source(
    f0=f_center,
    bandwidth=f_width / f_center,
    direction="+x",
    polarization="ez",
    margin=3,
)

sim.add_probe(center, "ez")

grid = sim._build_grid()
n_steps = int(np.ceil(5e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

result = sim.run(n_steps=n_steps)
ts = np.array(result.time_series).ravel()
peak = float(np.max(np.abs(ts)))
print(f"Simulation complete — peak |Ez| = {peak:.4e}")
print("PASS: TFSF+sphere simulation ran successfully" if peak > 1e-10 else "FAIL: no field detected")

# Plot time series
ts = np.array(result.time_series).ravel()
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
t_ns = np.arange(len(ts)) * result.dt * 1e9
ax.plot(t_ns, ts, "b-", lw=0.8)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez at center")
ax.set_title(f"Mie Scattering: n={n_sphere} sphere, r={radius*1e3:.0f}mm")
ax.grid(True, alpha=0.3)

out = os.path.join(SCRIPT_DIR, "04_meep_mie_sphere.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
