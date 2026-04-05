---
title: "Far-Field & Radar Cross Section"
sidebar:
  order: 11
---

rfx computes far-field radiation patterns via near-to-far-field transform (NTFF) and radar cross section (RCS) via TFSF + NTFF integration.

## Radiation Pattern (NTFF)

The NTFF approach records tangential E/H fields on a closed Huygens box during simulation, then computes far-field integrals.

```python
from rfx import Simulation, make_ntff_box, compute_far_field, radiation_pattern
import numpy as np

sim = Simulation(freq_max=5e9, domain=(0.1, 0.1, 0.1), boundary="cpml")
# ... add antenna geometry and source ...

# Define NTFF box enclosing the antenna
ntff_box = make_ntff_box(grid, corner_lo=(0.02, 0.02, 0.02),
                         corner_hi=(0.08, 0.08, 0.08),
                         freqs=np.array([3e9]))

result = sim.run(n_steps=1000, ntff=ntff_box)

# Compute far-field
theta = np.linspace(0, np.pi, 181)
phi = np.linspace(0, 2*np.pi, 360)
ff = compute_far_field(result.ntff_data, ntff_box, grid, theta, phi)

# Radiation pattern in dB
pattern_dB = radiation_pattern(ff)  # (n_freqs, n_theta, n_phi)
```

## Directivity

```python
from rfx import directivity

D = directivity(ff)  # (n_freqs,) in dBi
print(f"Directivity: {D[0]:.1f} dBi")
```

## Radar Cross Section (RCS)

RCS measures the electromagnetic scattering from a target illuminated by a plane wave.

```python
from rfx import compute_rcs
import numpy as np

# Define scatterer (PEC plate, sphere, etc.)
grid = Grid(freq_max=6e9, domain=(0.10, 0.10, 0.10), dx=0.002, cpml_layers=10)
materials = init_materials(grid.shape)
# ... add PEC object to materials ...

result = compute_rcs(
    grid, materials,
    n_steps=1000,
    f0=3e9,
    bandwidth=0.5,
    polarization="ez",
    theta_obs=np.linspace(0, np.pi, 91),
    phi_obs=np.array([0.0, np.pi/2]),
    freqs=np.array([3e9]),
)

print(f"Monostatic RCS: {result.monostatic_rcs[0]:.1f} dBsm")
print(f"RCS range: {result.rcs_dbsm.min():.1f} to {result.rcs_dbsm.max():.1f} dBsm")
```

## Plotting RCS

```python
from rfx import plot_rcs

plot_rcs(result, freq_idx=0, mode="polar")    # Polar plot
plot_rcs(result, freq_idx=0, mode="rectangular")  # dBsm vs angle
```

## Notes

- NTFF box must be **inside** the CPML region and **outside** all sources/scatterers
- For RCS, the TFSF box is placed automatically around the scatterer
- The scattered field (outside TFSF) is what the NTFF box captures
- Currently supports normal-incidence TFSF only (oblique in development)
