---
title: "Far-Field & Radar Cross Section"
sidebar:
  order: 11
---

rfx computes far-field radiation patterns from a near-to-far-field transform
(NTFF), and radar cross section (RCS) by combining TFSF plane-wave illumination
with the same NTFF integration.

## Radiation Pattern (NTFF)

An NTFF box records the tangential E and H fields on a closed Huygens surface
during the time stepping. After the run, those surface fields are converted to
equivalent currents and integrated to the far field — so the pattern comes from
the recorded surface, not from probes sitting in the radiating near field.

```python
from rfx import Simulation, compute_far_field, radiation_pattern
import numpy as np

sim = Simulation(freq_max=5e9, domain=(0.1, 0.1, 0.1), boundary="cpml")
# ... add the radiator geometry and its source (see the Materials & Geometry
#     and Sources & Ports guides) ...

# NTFF box: encloses the radiator but stays inside the simulation region,
# clear of the CPML absorber.
sim.add_ntff_box(
    corner_lo=(0.02, 0.02, 0.02),
    corner_hi=(0.08, 0.08, 0.08),
    freqs=np.array([3e9]),
)

# Inspect coded placement/support advisories and fail on blocking errors.
preflight = sim.preflight()
print(preflight.format())
preflight.raise_for_failure()

result = sim.run(n_steps=1000)

# Far field over a (theta, phi) grid. run() stores the raw NTFF data,
# the box spec, and the grid on the result.
theta = np.linspace(0, np.pi, 181)
phi = np.linspace(0, 2 * np.pi, 360)
ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)

# Normalized pattern in dB (peak = 0 dB), shape (n_freqs, n_theta, n_phi).
pattern_dB = radiation_pattern(ff)
```

`compute_far_field` returns a `FarFieldResult` (fields `E_theta`, `E_phi`,
`theta`, `phi`, `freqs`) that both `radiation_pattern` and `directivity`
consume.

## Directivity

```python
from rfx import directivity

D = directivity(ff)  # (n_freqs,) in dBi
print(f"Directivity: {D[0]:.1f} dBi")
```

`directivity` integrates radiated power over the whole sphere (D = 4π·U_max /
P_rad), so `ff` must be sampled over the full range — `theta` in [0, π] and
`phi` in [0, 2π], as in the block above. A single cut plane yields the wrong
P_rad.

For inverse-design objectives that change total radiated power, use
`maximize_directivity(theta_target, phi_target, log_ratio=True)` so the gradient
follows the full directivity ratio instead of a stopped-power proxy. See the
[Inverse Design](/rfx/guide/inverse-design/) guide.

## Radar Cross Section (RCS)

`compute_rcs` illuminates a target with a TFSF plane wave, captures the
scattered field (the field outside the TFSF box) on an NTFF box, and reports
RCS(θ, φ) = 4π r² |E_scat|² / |E_inc|². It is a functional API: you pass a
`Grid` and a `MaterialArrays` holding the scatterer, not a `Simulation`.

```python
import numpy as np
import jax.numpy as jnp
from rfx import Grid, Box, compute_rcs
from rfx.geometry.csg import rasterize
from rfx.core.yee import MaterialArrays

f0 = 3e9              # illumination centre frequency (lambda = 0.1 m)
dx = 0.01            # ~lambda/10 at 3 GHz; leaves room for scatterer + TFSF + NTFF + CPML
grid = Grid(
    freq_max=f0 * 1.5,
    domain=(0.12, 0.12, 0.12),
    dx=dx,
    cpml_layers=8,
)

# PEC plate: 4 cm square, one cell thick, centred and normal to x.
c = 0.06
plate = Box(
    corner_lo=(c - dx / 2, c - 0.02, c - 0.02),
    corner_hi=(c + dx / 2, c + 0.02, c + 0.02),
)
# rasterize takes (shape, eps_r, sigma) tuples; sigma = 1e7 S/m approximates PEC.
eps_r, sigma = rasterize(grid, [(plate, 1.0, 1e7)])
materials = MaterialArrays(
    eps_r=eps_r, sigma=sigma, mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
)

result = compute_rcs(
    grid, materials,
    n_steps=400,
    f0=f0,
    bandwidth=0.5,            # fractional bandwidth of the Gaussian excitation
    polarization="ez",
    theta_obs=np.linspace(0.01, np.pi - 0.01, 91),
    phi_obs=np.array([0.0, np.pi / 2]),
    freqs=np.array([f0]),
)

# monostatic_rcs is evaluated exactly at the backscatter direction
# (theta=pi/2, phi=pi for the +x incidence), independent of theta_obs/phi_obs.
print(f"Monostatic RCS: {result.monostatic_rcs[0]:.1f} dBsm")
print(f"Bistatic RCS range: {result.rcs_dbsm.min():.1f} to {result.rcs_dbsm.max():.1f} dBsm")
```

`RCSResult` carries `rcs_dbsm` and `rcs_linear` (shape `(n_freqs, n_theta,
n_phi)`), `monostatic_rcs` (`(n_freqs,)` dBsm, always populated — evaluated
exactly at the backscatter direction opposite the incident propagation
vector, so it does not depend on the observation grid), and the
`freqs` / `theta` / `phi` axes.

### Validation scope: monostatic yes, bistatic not yet

Only `monostatic_rcs` (the backscatter bin) is cross-validated against the
exact Mie series — ~0.06 dB at ka ≈ 1 on a PEC sphere. The full
`rcs_dbsm` / `rcs_linear` **bistatic pattern is not validated** at the
auto-placed default NTFF box (`ntff_offset=1`, one cell off the TFSF
boundary, deep in the reactive near field). At off-backscatter angles the
default setup can be several dB to ~20 dB off — a ka ≈ 1 PEC sphere shows a
spurious forward-oblique lobe near 25–55° scattering angle measured ~10 dB
high versus Mie. Enlarging `ntff_offset` alone does **not** close this at
test scale (it can worsen the backscatter bin), because the oblique error is
dominated by the staircased curved surface and forward TFSF-face
contamination, not box distance. Treat off-backscatter cuts as qualitative
until a converged (finer-resolution) far-field setup is used; only the
monostatic number carries the cross-method gate.

## Plotting RCS

```python
from rfx import plot_rcs

plot_rcs(result, freq_idx=0, polar=True)    # polar cut
plot_rcs(result, freq_idx=0, polar=False)   # dBsm vs angle
```

`plot_rcs` selects one frequency (`freq_idx`) and one φ cut (`phi_idx`, default
0). For the other plotting helpers see
[Visualization & Result Analysis](/rfx/guide/visualization-and-analysis/).

## Notes

- Keep NTFF box surfaces in the ordinary simulation region — outside every
  source, scatterer, and radiator, with margin from the CPML absorber.
  `sim.preflight()` warns when a box extends into the absorber.
- Check clearance from **all six NTFF faces** to the nearest relevant geometry
  or active source. Preflight emits `ntff_near_field` below λ/2 at `f_max`, with
  a stronger warning below λ/4. This includes lateral edges of finite PEC
  sheets, not only the sheet's broad face. Move the Huygens surface outward
  rather than assuming a source-to-box distance alone proves far-field
  separation.
- For fixed-period pulsed runs on CPML/UPML, a warning that the run ended above
  -40 dB of peak means the recorded NTFF data still contains a hot transient.
  Increase the run length or use `until_decay=...` before interpreting the
  pattern.
- `compute_rcs` places the TFSF and NTFF boxes automatically from `cpml_layers`,
  `tfsf_margin`, and `ntff_offset` (the NTFF box sits `ntff_offset` cells outside
  the TFSF boundary); you supply only the grid and the scatterer materials.
- Report the incident direction, polarization, observation angles, and frequency
  band alongside any published RCS number.
