---
title: "Visualization & Result Analysis"
sidebar:
  order: 15
---

rfx ships matplotlib-based helpers for common RF post-processing: S-parameters,
field slices, radiation patterns, RCS, and probe time series. Each helper takes
plain arrays (`result.s_params`, `result.freqs`, `result.time_series`) or the
result objects the analysis APIs return (`FarFieldResult`, `RCSResult`). For the
most common plots, `Result` also exposes convenience methods —
`result.plot_s_params()`, `result.plot_smith()`, and `result.plot_time_series()` —
that forward to the functions below.

## Built-in Visualization

### S-Parameter Plots

Setup used by the snippets below:

```python
import numpy as np
from rfx import GaussianPulse, Simulation

sim = Simulation(
    freq_max=15e9,
    domain=(0.06, 0.06, 0.06),
    dx=1.5e-3,
    boundary="cpml",
    cpml_layers=6,
)
for x_port in (0.028, 0.032):
    sim.add_port(
        (x_port, 0.030, 0.030),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=8e9, bandwidth=0.8),
    )
sim.add_probe((0.030, 0.036, 0.030), "ez")
sim.add_ntff_box(
    corner_lo=(0.012, 0.012, 0.012),
    corner_hi=(0.048, 0.048, 0.048),
    freqs=np.array([12e9]),
)

result = sim.run(n_steps=800, compute_s_params=True)
```

Use the S-matrix and frequency grid stored on the result:

```python
from rfx import plot_s_params

fig = plot_s_params(result.s_params, result.freqs, db=True)
```

### Field Distribution

Pass the final field state together with the grid metadata:

```python
from rfx import plot_field_slice

grid = result.grid
fig = plot_field_slice(
    result.state,
    grid,
    component="ez",
    axis="z",
    index=grid.nz // 2,
    title="Ez at the z-midplane",
)
```

### Radiation Pattern

First compute the far field, then plot it:

```python
import numpy as np
from rfx import compute_far_field, plot_radiation_pattern

theta = np.linspace(0.0, np.pi, 181)
phi = np.array([0.0])
ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)
fig = plot_radiation_pattern(ff, freq_idx=0)
```

`theta` and `phi` are in **radians**. The example above uses a full
elevation sweep with a single azimuth cut.

### RCS

`plot_rcs()` takes the `RCSResult` returned by `compute_rcs()`:

```python
import jax.numpy as jnp
from rfx import Box, Grid, compute_rcs, plot_rcs
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import rasterize

# TFSF illumination cannot share a Simulation with ports, so the RCSResult
# comes from its own scattering run: a PEC plate, one cell thick.
f0 = 3e9
dx = 0.01
grid = Grid(freq_max=f0 * 1.5, domain=(0.12, 0.12, 0.12), dx=dx, cpml_layers=8)
c = 0.06
plate = Box(
    corner_lo=(c - dx / 2, c - 0.02, c - 0.02),
    corner_hi=(c + dx / 2, c + 0.02, c + 0.02),
)
eps_r, sigma = rasterize(grid, [(plate, 1.0, 1e7)])
materials = MaterialArrays(
    eps_r=eps_r, sigma=sigma, mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
)
rcs_result = compute_rcs(
    grid, materials,
    n_steps=400,
    f0=f0,
    bandwidth=0.5,
    polarization="ez",
    theta_obs=np.linspace(0.01, np.pi - 0.01, 91),
    phi_obs=np.array([0.0, np.pi / 2]),
    freqs=np.array([f0]),
)

fig = plot_rcs(rcs_result, freq_idx=0, polar=True)
```

For how to produce `rcs_result` from a scattering run, see
[Far-Field & RCS](/rfx/guide/farfield-rcs/).

### Time-Domain Signal

Plot probe time series with the timestep used to record them:

```python
from rfx import plot_time_series

fig = plot_time_series(result.time_series, result.dt, labels=["Probe 1"])
```

## Programmatic Analysis

rfx results are NumPy or JAX arrays, so you can use standard Python analysis
libraries directly.

### Frequency-Domain Analysis

```python
import numpy as np
from scipy.signal import find_peaks

# Custom FFT analysis from time-domain data
ts = np.array(result.time_series[:, 0])  # first probe
spectrum = np.fft.rfft(ts)
freqs = np.fft.rfftfreq(len(ts), d=result.dt)

# Find resonant frequencies (peaks)
peaks, _ = find_peaks(np.abs(spectrum), height=np.max(np.abs(spectrum)) * 0.1)
print(f"Resonances at: {freqs[peaks] / 1e9} GHz")
```

### S-Parameter Post-Processing

```python
import numpy as np

freqs = result.freqs
s11 = result.s_params[0, 0, :]

# Smith chart (impedance)
z_in = 50 * (1 + s11) / (1 - s11)

# Group delay
phase = np.unwrap(np.angle(result.s_params[1, 0, :]))
group_delay = -np.gradient(phase) / np.gradient(2 * np.pi * freqs)

# Return loss
return_loss_db = -20 * np.log10(np.abs(s11))

# VSWR
vswr = (1 + np.abs(s11)) / (1 - np.abs(s11))
```

### Field Energy and Power

```python
import jax.numpy as jnp

EPS_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6

grid = result.grid
state = result.state

# Electric energy density
u_e = 0.5 * EPS_0 * (state.ex**2 + state.ey**2 + state.ez**2)
# Magnetic energy density
u_h = 0.5 * MU_0 * (state.hx**2 + state.hy**2 + state.hz**2)
# Approximate total stored energy on a uniform cubic grid
stored_energy = float(jnp.sum(u_e + u_h) * grid.dx**3)
```

This is an estimate: it uses the vacuum permittivity `EPS_0` everywhere (no
per-cell `eps_r` weighting, so it under-counts energy inside dielectrics) and
treats the staggered Yee field components as if they were co-located. The final
line also assumes a **uniform cubic grid** — for non-uniform spacing, integrate
with the actual cell-volume weights instead of `grid.dx**3`.

### Export for External Tools

```python
from rfx import (
    read_touchstone_full,
    save_snapshots,
    save_state,
    write_touchstone,
)

# Legacy-compatible Touchstone for ADS/CST/HFSS. Shape is (n_ports, n_ports, n_freqs).
write_touchstone("device.s2p", result.s_params, result.freqs, z0=50.0)

# Four lumped ports supply the row-wise 4-port result exported below.
four_port_sim = Simulation(
    freq_max=10e9,
    domain=(0.06, 0.06, 0.06),
    dx=2e-3,
    boundary="cpml",
    cpml_layers=6,
)
for port_position in (
    (0.026, 0.030, 0.030),
    (0.034, 0.030, 0.030),
    (0.030, 0.026, 0.030),
    (0.030, 0.034, 0.030),
):
    four_port_sim.add_port(
        port_position, "ez", impedance=50.0,
        waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
    )
four_port_result = four_port_sim.run(n_steps=600, compute_s_params=True)

# Metadata-rich Touchstone 2.0 export for a standard row-wise 4-port result
write_touchstone(
    "device_v2.s4p",
    four_port_result.s_params,
    four_port_result.freqs,
    version="2.0",
    layout="standard",
    port_z0=[50.0, 50.0, 50.0, 50.0],
    information={"Project": "demo", "Tool": "rfx"},
)
network = read_touchstone_full("device_v2.s4p")

# HDF5 for the final field state
save_state("fields.h5", result.state, grid=result.grid)

# HDF5 for saved snapshots, when present
if result.snapshots is not None:
    save_snapshots("snapshots.h5", result.snapshots, grid=result.grid, dt=result.dt)
```

## External Analysis Workflows

Hand summaries to notebooks or reports, but retain the raw arrays and plots
with the summary. A useful summary records the frequency grid, S-parameters,
resonant peaks, bandwidth, return loss, and the exact pass/fail rule used to
judge the result — state the metric rather than only that a design "looks good".

Alongside exported plots or Touchstone files, keep a small machine-readable
manifest with the command, git SHA, support status, and metric used to produce
the figure, so the artifact can be reproduced.
