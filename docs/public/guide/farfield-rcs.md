---
title: "Far-Field & Radar Cross Section"
sidebar:
  order: 11
---

This page shows how to get radiation patterns and directivity from a
near-to-far-field transform (NTFF), and how to compute radar cross section
(RCS) with a plane wave. It also says which RCS numbers you can trust.

## Radiation pattern

An NTFF box is a closed surface around your radiator. During the run rfx
accumulates the tangential E and H on its six faces at the frequencies you
ask for. Afterwards `compute_far_field` turns those into equivalent currents
and integrates them to the far field. The pattern therefore comes from the
recorded surface, not from probes in the near field.

```python
import numpy as np
from rfx import GaussianPulse, Simulation, compute_far_field, radiation_pattern

sim = Simulation(freq_max=5e9, domain=(0.20, 0.20, 0.20), dx=4e-3,
                 boundary="cpml", cpml_layers=8)
sim.add_source((0.10, 0.10, 0.10), "ez",
               waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
               amplitude_kind="current")

# The box encloses the source with about half a wavelength of clearance.
sim.add_ntff_box(corner_lo=(0.045, 0.045, 0.045),
                 corner_hi=(0.155, 0.155, 0.155),
                 freqs=np.array([3e9]))

# A probe away from the source lets rfx check that the fields rang down.
sim.add_probe((0.12, 0.10, 0.10), "ez")

report = sim.preflight()
if report.by_code("absorber_overlap") or report.by_code("ntff_near_field"):
    raise RuntimeError("move the NTFF box before running:\n" + report.format())

result = sim.run(n_steps=1000)

theta = np.linspace(0, np.pi, 91)            # radians
phi = np.linspace(0, 2 * np.pi, 72, endpoint=False)
ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)

pattern_db = radiation_pattern(ff)   # (n_freqs, n_theta, n_phi), peak = 0 dB
```

`compute_far_field` returns a `FarFieldResult` with `E_theta`, `E_phi`,
`theta`, `phi` and `freqs`. Angles are in radians. The mesh here is coarse
for speed; see [Limits](#limits) for what a real pattern needs.

The far field is a Fourier transform of the recorded surface fields, so the
run must last until the fields have died out. rfx judges that from your probe
records: when a run has an NTFF box, it warns if a probe is still ringing at
the end, and warns that the check is missing if you added no probe. Put the
probe where the field is live but not on the source cell.

### Placing the box

- Put every face between the radiator and the absorber, with clearance from
  both. Preflight's `ntff_near_field` check warns when a face is closer than
  λ/2 to a source, port or tangentially overlapping geometry, and more
  strongly below λ/4. `absorber_overlap` fires when the box reaches the
  absorber.
- The near-field check does not see plane-wave (TFSF) boundaries or
  microstrip feed positions. Check those clearances yourself.
- A small ground plane gives an `ntff_small_ground_plane` advisory when it is
  under about 1 λ across. Edge diffraction then shapes the pattern (a
  broadside dip, off-axis peaks). That is real physics, not a solver error;
  use a ground plane of about 1.4 λ or more if you want a clean broadside
  pattern.

## Directivity

```python
from rfx import directivity

D = directivity(ff)   # (n_freqs,) in dBi
print(f"Directivity: {D[0]:.2f} dBi")   # a short dipole is about 1.76 dBi
```

`directivity` integrates the radiated power over the whole sphere
(D = 4π U_max / P_rad). Sample `theta` over [0, π] and `phi` over [0, 2π), as
above. A single cut gives the wrong P_rad.

For optimization, `compute_far_field_jax(..., max_phase_bytes=4e9)` is the
differentiable version. The byte budget splits the theta grid into chunks to
bound memory; the result is the same. For a directivity objective use
`maximize_directivity(theta_target, phi_target, log_ratio=True)`, so the
gradient sees changes in total radiated power. See
[Inverse Design](/rfx/guide/inverse-design/).

## Radar cross section

`compute_rcs` illuminates a target with a plane wave travelling along +x,
records the scattered field on an NTFF box, and returns
RCS(θ, φ) = 4π r² |E_scat|² / |E_inc|². It is a functional API: you pass a
`Grid` and `MaterialArrays` holding the target, not a `Simulation`.

```python
import jax.numpy as jnp
from rfx import Box, Grid, compute_rcs
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import rasterize

f0 = 3e9                     # lambda = 0.1 m
dx = 0.01                    # lambda / 10: coarse, for speed
grid = Grid(freq_max=f0 * 1.5, domain=(0.12, 0.12, 0.12), dx=dx, cpml_layers=8)

# A 4 cm square plate, one cell thick, normal to x. compute_rcs takes material
# arrays, so the plate is a high-conductivity fill (1e7 S/m), not a declared
# PEC conductor.
c = 0.06
plate = Box((c - dx / 2, c - 0.02, c - 0.02), (c + dx / 2, c + 0.02, c + 0.02))
eps_r, sigma = rasterize(grid, [(plate, 1.0, 1e7)])
materials = MaterialArrays(eps_r=eps_r, sigma=sigma,
                           mu_r=jnp.ones(grid.shape, dtype=jnp.float32))

rcs = compute_rcs(
    grid, materials, n_steps=400,
    f0=f0, bandwidth=0.5, polarization="ez",
    theta_obs=np.linspace(0.01, np.pi - 0.01, 91),
    phi_obs=np.array([0.0, np.pi / 2]),
    freqs=np.array([f0]),
    subtract_incident_reference=True,   # needed for the bistatic pattern
)

print(f"Monostatic RCS: {rcs.monostatic_rcs[0]:.1f} dBsm")
print(f"Bistatic range: {rcs.rcs_dbsm.min():.1f} to {rcs.rcs_dbsm.max():.1f} dBsm")
```

`RCSResult` holds `rcs_dbsm` and `rcs_linear` with shape
`(n_freqs, n_theta, n_phi)`, the `freqs` / `theta` / `phi` axes, and
`monostatic_rcs` (dBsm, one value per frequency). The monostatic value is
evaluated exactly at the backscatter direction (θ = π/2, φ = π for +x
incidence), so it does not depend on the observation grid you pass.

### Which RCS numbers to trust

**Monostatic (backscatter).** `monostatic_rcs` is the quantity checked
against the exact Mie series for a PEC sphere at ka ≈ 1 (λ/40 cells). On that
committed case it reads 0.73 dB above Mie. It is always taken from the plain
run, whatever you set for `subtract_incident_reference`.

**Bistatic.** With the default `subtract_incident_reference=False`, the
**bistatic pattern is not validated**: off-backscatter angles can be several
dB to about 20 dB wrong, typically as a spurious forward-oblique lobe. The
reason is geometric. The plane wave is injected between two x planes, so the
region carrying the incident field is a slab that is infinite in y and z. The
side faces of any closed NTFF box cut through that slab and record the
incident field at full strength. Moving the box with a larger `ntff_offset`
does not help, because every closed box has side faces in the slab. An empty
run with no target shows the same lobe.

Set `subtract_incident_reference=True` for bistatic work. rfx then repeats
the run without the target and subtracts the two far fields (as complex
values) before forming the RCS, which removes the incident-field
contribution. It doubles the cost. On the Mie sphere it removes most of the
forward-oblique lobe; that comparison covers one sphere, frequency,
polarization and cut, not every target.

After subtraction, the remaining error comes from staircasing of curved
surfaces, sensitivity near deep pattern nulls, box placement and absorber
reflection. For a new target: refine the mesh, lengthen the run, vary
`ntff_offset` and `cpml_layers`, and compare with an analytic or independent
result.

`RCSResult` does not record whether subtraction was used, so keep your call
settings with any saved result.

### Box placement and oblique incidence

`compute_rcs` places the plane-wave boundary `tfsf_margin` cells in from the
absorber and the NTFF box `ntff_offset` cells outside the plane-wave
boundary. The box must enclose the whole injected region, and rfx refuses to
run otherwise, naming the faces that don't clear it.

- **Normal incidence** (`theta_inc=0`, the default): the defaults
  (`tfsf_margin=3`, `ntff_offset=1`) are valid.
- **Oblique incidence** (`theta_inc` in degrees, tilted in the x-y plane):
  this uses an open-domain plane wave for `polarization="ez"` on a uniform
  grid only, with z-invariant targets. Its y faces are inset from the
  absorber, not from the injection planes, so the box encloses the injected
  region only for `1 <= ntff_offset <= tfsf_margin - 2`. Raise `tfsf_margin`
  if you need a larger offset. At oblique incidence the direction of the
  specular peak is checked; treat the absolute RCS as accurate to about
  ±2 dB. The oblique model is 2.5-D (periodic in z), so the returned RCS
  belongs to a strip of height 2 `dx`; scale by (L_z / (2 dx))² for a strip
  of height L_z.

If you build an `NTFFBox` yourself around a plane-wave source, it too must
surround the injected region; rfx refuses a box that doesn't.

## Plotting

```python
from rfx import plot_radiation_pattern, plot_rcs

plot_radiation_pattern(ff, freq_idx=0, phi_idx=0)   # E-plane cut at phi = 0
plot_rcs(rcs, freq_idx=0, polar=True)                # polar cut, phi_idx = 0
plot_rcs(rcs, freq_idx=0, polar=False)               # dBsm vs angle
```

Both helpers return a matplotlib figure for one frequency and one φ cut. See
[Visualization & Result Analysis](/rfx/guide/visualization-and-analysis/)
for the other helpers.

## Limits

- **Pattern changes when you run longer** → the record was cut. The
  ring-down check reads your probes, not the NTFF surface, so it is a guide,
  not a proof. Repeat with a longer run (or `until_decay=`) and compare the
  patterns.
- **Directivity looks wrong** → sample the full sphere; a single cut gives
  the wrong total power.
- **Box too close to the radiator or touching the absorber** → preflight
  warns; move the faces out. It does not check plane-wave boundaries or
  microstrip feeds for you.
- **Bistatic RCS with a strong forward-oblique lobe** → you left
  `subtract_incident_reference=False`; turn it on. A larger `ntff_offset`
  does not fix it.
- **RCS of a new target** → only the monostatic sphere case is checked
  against Mie. Run your own convergence study and compare with a reference.
- **Oblique incidence refuses your setup** → it needs `polarization="ez"`, a
  uniform grid, a z-invariant target and `ntff_offset <= tfsf_margin - 2`.
- **Reporting an RCS** → state the incident direction, polarization,
  observation angles and frequency band with it.

The [support matrix](https://github.com/bk-squared/rfx/blob/main/docs/guides/support_matrix.md)
lists the current status of each far-field and RCS path.
