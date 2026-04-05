---
title: "Waveguide Ports"
sidebar:
  order: 10
---

rfx supports rectangular waveguide ports with analytical TE/TM mode profiles for accurate S-parameter extraction.

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

For high-accuracy S21 (normalized |S21| = 1.0000 for empty guide):

```python
result = sim.compute_waveguide_s_matrix(num_periods=30, normalize=True)
```

This runs a reference simulation (empty waveguide) to cancel Yee-grid numerical dispersion.

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
