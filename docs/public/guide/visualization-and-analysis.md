---
title: "Visualization & Result Analysis"
sidebar:
  order: 15
---

This page shows how to plot rfx results, record field snapshots with correct
positions and times, post-process results with NumPy, and export them to
other tools.

rfx's plotting helpers use matplotlib and return a `Figure`. They take plain
arrays (`result.s_params`, `result.freqs`, `result.time_series`) or the result
objects of the analysis functions (`FarFieldResult`, `RCSResult`).

## A result to work with

The examples below share one small run: two lumped ports, a probe and an NTFF
box. The mesh is coarse so the page runs quickly.

```python
import numpy as np
from rfx import GaussianPulse, Simulation

sim = Simulation(freq_max=15e9, domain=(0.06, 0.06, 0.06), dx=1.5e-3,
                 boundary="cpml", cpml_layers=6)
for x_port in (0.028, 0.032):
    sim.add_port((x_port, 0.030, 0.030), "ez", impedance=50.0,
                 waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
sim.add_probe((0.030, 0.036, 0.030), "ez")
sim.add_ntff_box(corner_lo=(0.012, 0.012, 0.012),
                 corner_hi=(0.048, 0.048, 0.048),
                 freqs=np.array([12e9]))

result = sim.run(n_steps=800, compute_s_params=True)
```

## Plots

### S-parameters and Smith chart

```python
from rfx import plot_s_params, plot_smith

fig = plot_s_params(result.s_params, result.freqs, db=True)   # every S_ij in dB
fig = plot_smith(np.asarray(result.s_params[0, 0]), np.asarray(result.freqs),
                 z0=50.0)
```

`result.plot_s_params()`, `result.plot_smith()` and
`result.plot_time_series()` are shortcuts for the same functions.

### Probe time series

```python
from rfx import plot_time_series

fig = plot_time_series(result.time_series, result.dt, labels=["probe"])
```

### Field slice

`plot_field_slice` plots one component of the final field state on a plane.
`index` is a grid index along `axis`, counting the absorber cells:

```python
from rfx import plot_field_slice

grid = result.grid
k_mid = grid.position_to_index((0.030, 0.030, 0.030))[2]
fig = plot_field_slice(result.state, grid, component="ez", axis="z",
                       index=k_mid, title="Ez at z = 30 mm")
```

`result.state` holds the fields after the last step only. To see how the
field evolves, record snapshots (below).

### Radiation pattern and RCS

```python
from rfx import compute_far_field, plot_radiation_pattern

theta = np.linspace(0.0, np.pi, 181)      # radians
phi = np.array([0.0])
ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)
fig = plot_radiation_pattern(ff, freq_idx=0)
```

`plot_rcs(rcs_result, freq_idx=0, polar=True)` plots the `RCSResult` from
`compute_rcs`. Both are covered on [Far-Field & RCS](/rfx/guide/farfield-rcs/).

## Field snapshots

Pass a `SnapshotSpec` to `run(snapshot=...)` to record fields during the run.
A slice through one plane keeps memory small:

```python
from rfx import SnapshotSpec

snap_sim = Simulation(freq_max=10e9, domain=(0.04, 0.04, 0.02), dx=1e-3,
                      boundary="cpml", cpml_layers=6)
snap_sim.add_source((0.020, 0.020, 0.010), "ez",
                    waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
                    amplitude_kind="current")

# slice_index counts grid cells from the outer edge of the absorber, which
# lies outside the domain: z = 0 is index cpml_layers.
k = 6 + round(0.010 / 1e-3)
spec = SnapshotSpec(interval=10, components=("ez",), slice_axis=2, slice_index=k)
snap_run = snap_sim.run(n_steps=300, snapshot=spec)

frames = np.asarray(snap_run.snapshots["ez"])     # (n_frames, nx, ny)
axes = snap_run.snapshot_axes["ez"]
print(axes.dims)                                  # ('frame', 'x', 'y')
print(axes.times_s[:3])                           # seconds, one per frame
assert k == snap_run.grid.position_to_index((0.020, 0.020, 0.010))[2]
x_m, y_m = axes.coords["x"], axes.coords["y"]     # sample positions (m)
```

Use `axes.times_s` and `axes.coords` for your plot axes rather than working
them out yourself:

- Frames are `interval` steps apart. With the default `interval=10`, frame
  `k` is the field after step `10 * (k + 1)`, not after step `k`. A run of N
  steps records `N // interval` frames.
- H components are recorded half a time step earlier than E, and each
  component sits at its own staggered position. `times_s` and `coords`
  account for both.
- `slice_index` and the coordinate arrays include the absorber cells.

`run(until_decay=...)` records snapshots too; the frame count then depends on
where the run stopped. To see the frame layout before running, call
`rfx.snapshot_axes(grid, spec, n_steps)`.

## Analysis with NumPy

Results are NumPy or JAX arrays, so the usual tools work directly.

### Spectrum of a probe

```python
ts = np.asarray(result.time_series[:, 0])
spectrum = np.fft.rfft(ts)
f_fft = np.fft.rfftfreq(len(ts), d=result.dt)
print(f"spectral peak at {f_fft[np.argmax(np.abs(spectrum[1:])) + 1] / 1e9:.2f} GHz")
```

A plain FFT of a record that has not decayed smears the spectrum. For
resonant frequencies and Q, use `result.find_resonances()`; see
[Probes and S-Parameters](/rfx/guide/probes-sparams/#resonances-harminv).

### S-parameter quantities

```python
freqs = np.asarray(result.freqs)
s11 = np.asarray(result.s_params[0, 0])
s21 = np.asarray(result.s_params[1, 0])

z_in = 50 * (1 + s11) / (1 - s11)                          # input impedance (ohm)
return_loss_db = -20 * np.log10(np.abs(s11))
vswr = (1 + np.abs(s11)) / (1 - np.abs(s11))
phase = np.unwrap(np.angle(s21))
group_delay = -np.gradient(phase) / np.gradient(2 * np.pi * freqs)   # seconds
```

### Stored energy (rough estimate)

```python
EPS_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
st = result.state

u_e = 0.5 * EPS_0 * (st.ex**2 + st.ey**2 + st.ez**2)
u_h = 0.5 * MU_0 * (st.hx**2 + st.hy**2 + st.hz**2)
stored_energy = float(np.sum(u_e + u_h) * grid.dx**3)
```

This uses vacuum permittivity everywhere (it undercounts energy in
dielectrics), treats the staggered components as co-located, and assumes a
uniform cubic grid. On a graded mesh, weight each cell by its own volume.

## Export

```python
import os
import tempfile
from rfx import load_snapshots, save_snapshots, save_state, write_touchstone

out = tempfile.mkdtemp()

# Touchstone for circuit tools. Shape is (n_ports, n_ports, n_freqs).
write_touchstone(os.path.join(out, "device.s2p"), result.s_params, result.freqs,
                 z0=50.0)

# HDF5: the final field state.
save_state(os.path.join(out, "fields.h5"), result.state, grid=result.grid)

# HDF5: snapshots, with their positions and frame times.
save_snapshots(os.path.join(out, "snapshots.h5"), snap_run.snapshots,
               grid=snap_run.grid, dt=snap_run.dt, axes=snap_run.snapshot_axes)
loaded, meta = load_snapshots(os.path.join(out, "snapshots.h5"))
times = meta["axes"]["ez"].times_s
```

Always pass `axes=result.snapshot_axes` to `save_snapshots`. With only `dt=`,
a reader would naturally assume one frame per time step, which is wrong
unless `interval=1`. `load_snapshots` returns the saved axes as
`meta["axes"]`. Touchstone 2.0 options and reading files back are on
[Probes and S-Parameters](/rfx/guide/probes-sparams/#touchstone-export).

When you hand a figure or file to someone else, keep the script, the rfx
version and the settings (mesh, run length, calculator options) with it, so
the result can be reproduced.

## Limits

- **Frame times off by a factor of `interval`** → read `times_s` from
  `result.snapshot_axes`; don't assume frame k is step k.
- **Snapshots of full 3-D fields run out of memory** → record a slice
  (`slice_axis`, `slice_index`), fewer components or a larger `interval`.
- **FFT peaks don't match `find_resonances`** → the record hasn't decayed;
  run longer or use Harminv.
- **Energy estimate looks low in a dielectric** → the estimate ignores
  `eps_r`; weight by the per-cell permittivity for a better number.
