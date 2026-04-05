---
title: "Visualization & Result Analysis"
sidebar:
  order: 15
---

rfx ships matplotlib-based helpers for the most common RF post-processing tasks.
The helpers expect the same objects that the simulation and analysis APIs return,
so prefer passing raw arrays or result fields directly.

## Built-in Visualization

### S-Parameter Plots

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
from rfx import plot_rcs

fig = plot_rcs(rcs_result, freq_idx=0, polar=True)
```

If you want the full computation flow, see `docs/public/guide/farfield-rcs.md`.

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
from rfx.core.yee import EPS_0, MU_0

grid = result.grid
state = result.state

# Electric energy density
u_e = 0.5 * EPS_0 * (state.ex**2 + state.ey**2 + state.ez**2)
# Magnetic energy density
u_h = 0.5 * MU_0 * (state.hx**2 + state.hy**2 + state.hz**2)
# Total stored energy for a uniform cubic grid
stored_energy = float(jnp.sum(u_e + u_h) * grid.dx**3)
```

The final line assumes a **uniform cubic grid**. For non-uniform spacing,
integrate with the actual cell-volume weights instead of `grid.dx**3`.

### Export for External Tools

```python
from rfx import save_state, save_snapshots, write_touchstone

# Touchstone for ADS/CST/HFSS
write_touchstone("device.s2p", result.s_params, result.freqs, z0=50.0)

# HDF5 for the final field state
save_state("fields.h5", result.state, grid=result.grid)

# HDF5 for saved snapshots, when present
if result.snapshots is not None:
    save_snapshots("snapshots.h5", result.snapshots, grid=result.grid, dt=result.dt)
```

## External Analysis Workflows

You can hand summaries to notebooks, scripts, or LLMs, but keep the raw arrays
and plots as the source of truth. A good summary includes the frequency grid,
S-parameters, resonant peaks, bandwidth, return loss, and the pass/fail rule you
used to judge the result.

When you write public notes or reports, describe the exact metric you used
instead of saying only that a design "looks good".
