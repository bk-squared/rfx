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
from rfx import GaussianPulse, Simulation, compute_far_field, radiation_pattern
import numpy as np

sim = Simulation(
    freq_max=5e9,
    domain=(0.20, 0.20, 0.20),
    boundary="cpml",
    dx=4e-3,
    cpml_layers=8,
)
sim.add_source(
    (0.10, 0.10, 0.10),
    "ez",
    waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
)

# NTFF box: encloses the radiator but stays inside the simulation region,
# clear of the CPML absorber.
sim.add_ntff_box(
    corner_lo=(0.045, 0.045, 0.045),
    corner_hi=(0.155, 0.155, 0.155),
    freqs=np.array([3e9]),
)

# Inspect coded placement/support advisories and fail on blocking errors.
preflight = sim.preflight()
print(preflight.format())
if (preflight.by_code("absorber_overlap") or
        preflight.by_code("ntff_near_field")):
    raise RuntimeError("move the NTFF box before running")
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
    # Two-run complex subtraction is required for the validated bistatic path.
    subtract_incident_reference=True,
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

### Validation scope: raw backscatter and reference-subtracted bistatic RCS

`monostatic_rcs` is always evaluated from the unsubtracted run at the exact
backscatter direction. It agrees with the exact Mie series to about 0.06 dB for
the committed ka ≈ 1 PEC-sphere case.

With the default `subtract_incident_reference=False`, the full
`rcs_dbsm` / `rcs_linear` **bistatic pattern is not validated**. An empty-domain
run produces the same forward-oblique lobe as the target run, showing that the
dominant raw-pattern lobe is target-independent leakage from the discrete TFSF
boundary, not a scatterer staircase effect. Increasing `ntff_offset` alone does
not remove that leakage. Use the default path for the validated monostatic
quantity; treat its off-backscatter bins as qualitative.

For bistatic work, set `subtract_incident_reference=True`. rfx then performs a
second vacuum run with the same TFSF and NTFF setup and subtracts the complex
far fields before forming RCS. This doubles the solve cost. In the committed
ka ≈ 1 PEC-sphere H-plane comparison against exact Mie, subtraction reduces the
largest 15–90° forward-oblique difference from 10.5 dB to 1.2 dB, gives a
full-pattern dB correlation of 0.965 and a mean absolute difference of 0.42 dB,
and leaves the backscatter difference at about 0.06 dB. That evidence covers
the stated sphere, frequency, polarization, angle cut, and discretization; it
does not validate every target or setup. An independent Bempp cube fixture
confirms the raw forward-oblique discrepancy on a second geometry, but does not
validate a reference-subtracted cube pattern.

After subtraction, remaining error can come from curved-surface staircasing,
deep-pattern-null sensitivity, NTFF placement, and CPML reflection. For a new
target, repeat with finer cells and a longer run, increase the domain until the
NTFF surface-to-target distance is adequate for the angles of interest, vary
the NTFF placement and CPML thickness, and compare with an analytic or
independent reference. The plate example above computes a corrected pattern;
the sphere evidence does not by itself validate the plate values.

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

- Keep all six NTFF faces outside sources, scatterers, radiators, and CPML.
  `sim.preflight()` warns when a box enters the absorber. Its
  `ntff_near_field` check measures face-normal clearance to tangentially
  overlapping geometry boxes or point entries created by `add_source()` and
  lumped/wire `add_port()`. It warns
  below λ/2 at `f_max` and more strongly below λ/4. This is a placement
  heuristic, not a nearest-3-D-distance or Huygens-validity test. TFSF
  boundaries and MSL source positions are not included and require manual
  clearance checks. Finite geometry such as PEC sheets is included only where
  its bounding box overlaps the NTFF face tangentially.
- The `ntff_small_ground_plane` advisory fires when an NTFF box is present
  and the largest PEC sheet backed by an `add_source()` / lumped-wire
  `add_port()` entry is under ~1λ across at the highest requested NTFF
  frequency. A sub-wavelength ground plane radiates a pattern shaped by
  ground-plane edge diffraction (broadside dip, off-axis side peaks) — that
  is expected physics, not a solver defect. Such a fixture stays valid for
  resonance and impedance work; for a clean broadside pattern use a ground
  plane of at least ~1.4λ. Deliberately small ground planes are legitimate:
  the advisory is a warning, never a blocker, so interpret the pattern
  accordingly and continue. TFSF-illuminated plates (RCS targets) do not
  trigger it.
- The automatic `ring-down truncated` warning examines a recorded probe time
  series, not the NTFF accumulators. It is emitted only for an
  absorbing-boundary run with a `GaussianPulse` entry from `add_source()`,
  lumped/wire `add_port()`, or `add_msl_port()`, automatic `num_periods`, and a
  probe series. TFSF excitation, non-`GaussianPulse` source entries, and the
  explicit-`n_steps` example above receive no automatic tail verdict. Repeat
  the calculation with a longer duration and compare the pattern, or use
  `until_decay` on the supported uniform or non-uniform CPML/UPML runners;
  neither method replaces observable-specific convergence testing.
- `compute_rcs` places the TFSF and NTFF boxes automatically from `cpml_layers`,
  `tfsf_margin`, and `ntff_offset` (the NTFF box sits `ntff_offset` cells outside
  the TFSF boundary); you supply only the grid and the scatterer materials.
- Report the incident direction, polarization, observation angles, and frequency
  band alongside any published RCS number.
