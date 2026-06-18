---
title: "Waveguide Ports"
sidebar:
  order: 10
---

rfx supports rectangular waveguide ports with analytical TE/TM mode profiles.
S-parameter claims are intentionally bounded: the documented full-matrix path is
`compute_waveguide_s_matrix(...)` under the rectangular-guide evidence envelope
in `docs/guides/sparameter_support_matrix.md`.

Current evidence level: Recommended for the documented WR-style rectangular-guide
cases. The main gates are in `tests/test_waveguide_port_validation_battery.py`
(empty-guide max `|S11| < 0.02`, passivity `< 1.02`, PEC-short
`0.99 <= min(|S11|)` and `max(|S11|) < 1.03`) and
`examples/crossval/11_waveguide_port_wr90.py` (analytic Airy/reference-plane
gates, with external references reported as available or skipped).

Waveguide ports do **not** use `run(compute_s_params=True)` for full
multi-port matrices. Use `compute_waveguide_s_matrix(...)` for the S-matrix;
`run(...)` exposes only single-port `result.waveguide_sparams` diagnostics.

## Single Port

```python
sim = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02),
                 boundary="cpml", cpml_layers=10, dx=0.002)

sim.add_waveguide_port(
    0.01,                    # x-position of port plane (meters)
    mode=(1, 0),             # TE10 dominant mode
    mode_type="TE",
    freqs=jnp.linspace(4.5e9, 8e9, 50),
    f0=6e9,                  # Center frequency for excitation pulse
    name="input",
)

result = sim.run(n_steps=500, compute_s_params=False)
# Access calibrated S-params
sp = result.waveguide_sparams["input"]
print(f"|S11| mean: {np.mean(np.abs(sp.s11)):.3f}")
print(f"|S21| mean: {np.mean(np.abs(sp.s21)):.3f}")
```

## Two-Port S-Matrix

For transmission measurements, use two ports with opposite directions:

```python
sim.add_waveguide_port(0.01, direction="+x", name="left",
                       mode=(1, 0), freqs=freqs, f0=6e9)
sim.add_waveguide_port(0.09, direction="-x", name="right",
                       mode=(1, 0), freqs=freqs, f0=6e9)

result = sim.compute_waveguide_s_matrix(num_periods=30)
S = result.s_params  # (2, 2, n_freqs) complex

s11 = S[0, 0, :]  # Reflection at port 1
s21 = S[1, 0, :]  # Transmission port 1 → port 2
s12 = S[0, 1, :]  # Transmission port 2 → port 1 (reciprocal: S12 ≈ S21)
```

## Two-Run Normalization

For the documented empty-guide envelope, two-run normalization cancels Yee-grid
dispersion and should keep `|S21|` near unity:

```python
result = sim.compute_waveguide_s_matrix(num_periods=30, normalize=True)
```

This runs a reference simulation (empty waveguide) to cancel Yee-grid numerical
dispersion. Do not extrapolate this to arbitrary branches/T-junctions without a
per-port reference-geometry validation artifact.

## Multi-Axis Ports

Ports can be placed on any axis-normal boundary:

```python
# Y-normal ports for a y-directed waveguide
sim.add_waveguide_port(0.01, direction="+y", name="bottom")
sim.add_waveguide_port(0.09, direction="-y", name="top")
```

## Disjoint Aperture Ports (N-port)

Multiple ports on the same boundary for parallel-guide or branch networks:

```python
sim.add_waveguide_port(0.01, y_range=(0.0, 0.04), z_range=(0.0, 0.02),
                       direction="+x", name="left_lo")
sim.add_waveguide_port(0.01, y_range=(0.06, 0.10), z_range=(0.0, 0.02),
                       direction="+x", name="left_hi")
```

## Calibration Options

```python
# Report S-params at the snapped measurement planes (default)
sim.add_waveguide_port(0.01, calibration_preset="measured")

# Report S11 at source plane, S21 at probe plane
sim.add_waveguide_port(0.01, calibration_preset="source_to_probe")

# Explicit reporting planes with de-embedding
sim.add_waveguide_port(0.01, reference_plane=0.012, probe_plane=0.034)
```

For reverse-mode AD or memory-heavy waveguide runs on the uniform Yee path,
`compute_waveguide_s_matrix(checkpoint_segments=K)` reuses the segmented
checkpointing machinery from the core runner. `K` must divide the timestep
count exactly; non-uniform waveguide extraction rejects this knob rather than
silently falling back to the linear-memory scan.
