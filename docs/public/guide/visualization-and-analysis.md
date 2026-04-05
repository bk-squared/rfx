---
title: "Visualization & Result Analysis"
sidebar:
  order: 15
---

## Built-in Visualization

rfx provides matplotlib-based plotting functions for common RF analysis tasks.

### S-Parameter Plots

```python
from rfx import plot_s_params

result = sim.run(n_steps=500, compute_s_params=True)
plot_s_params(result)  # |S11|, |S21| magnitude and phase vs frequency
```

### Field Distribution

```python
from rfx import plot_field_slice

# Snapshot at a specific timestep
plot_field_slice(result.state.ez, axis="z", index=grid.nz//2,
                 title="Ez at z-midplane")
```

### Radiation Pattern

```python
from rfx import plot_radiation_pattern

ff = compute_far_field(ntff_data, ntff_box, grid, theta, phi)
plot_radiation_pattern(ff, freq_idx=0)  # Polar plot
```

### RCS

```python
from rfx import plot_rcs

rcs_result = compute_rcs(grid, materials, ...)
plot_rcs(rcs_result, freq_idx=0, mode="polar")
```

### Time-Domain Signal

```python
from rfx import plot_time_series

plot_time_series(result)  # Ez vs time at probe locations
```

## Programmatic Analysis

rfx results are NumPy/JAX arrays — use any Python analysis tools directly.

### Frequency-Domain Analysis

```python
import numpy as np

# Custom FFT analysis from time-domain data
ts = np.array(result.time_series[:, 0])  # First probe
spectrum = np.fft.rfft(ts)
freqs = np.fft.rfftfreq(len(ts), d=grid.dt)

# Find resonant frequencies (peaks)
from scipy.signal import find_peaks
peaks, _ = find_peaks(np.abs(spectrum), height=np.max(np.abs(spectrum)) * 0.1)
print(f"Resonances at: {freqs[peaks] / 1e9} GHz")
```

### S-Parameter Post-Processing

```python
# Smith chart (impedance)
s11 = result.s_params[0, 0, :]
z_in = 50 * (1 + s11) / (1 - s11)  # Input impedance

# Group delay
phase = np.unwrap(np.angle(result.s_params[1, 0, :]))
group_delay = -np.gradient(phase) / np.gradient(2 * np.pi * result.s_param_freqs)

# Return loss
return_loss_dB = -20 * np.log10(np.abs(s11))

# VSWR
vswr = (1 + np.abs(s11)) / (1 - np.abs(s11))
```

### Field Energy and Power

```python
import jax.numpy as jnp
from rfx.core.yee import EPS_0, MU_0

state = result.state
# Electric energy density
u_e = 0.5 * EPS_0 * (state.ex**2 + state.ey**2 + state.ez**2)
# Magnetic energy density
u_h = 0.5 * MU_0 * (state.hx**2 + state.hy**2 + state.hz**2)
# Total stored energy
total_energy = float(jnp.sum(u_e + u_h) * grid.dx**3)
```

### Export for External Tools

```python
# Touchstone for ADS/CST/HFSS
from rfx import write_touchstone
write_touchstone("device.s2p", freqs, s_matrix, z0=50.0)

# CSV for spreadsheet analysis
np.savetxt("s_params.csv",
           np.column_stack([freqs/1e9, np.abs(s11), np.abs(s21)]),
           header="freq_GHz,|S11|,|S21|", delimiter=",")

# HDF5 for large datasets
from rfx import save_state, save_snapshots
save_state("fields.h5", result.state)
```

---

# AI-Assisted Design & Analysis

rfx is designed to work synergistically with AI models. The JAX foundation enables both gradient-based optimization and seamless integration with machine learning pipelines.

## 1. LLM as Design Assistant

Use Claude, GPT, or other LLMs to generate rfx simulation scripts from natural language specifications.

### Workflow

```
User: "Design a 5 GHz bandpass filter using coupled
       microstrip lines on FR4 substrate"
  ↓
LLM generates rfx script:
  - Simulation setup (domain, materials, ports)
  - Geometry (coupled lines with gap)
  - Optimization objective (S21 > -3dB in passband)
  ↓
rfx runs simulation + optimization
  ↓
LLM interprets results:
  - "S11 < -15dB from 4.8-5.2 GHz (good matching)"
  - "S21 ripple of 0.8dB (consider increasing coupled length)"
  - Suggests parameter adjustments
```

### Prompt Template for LLM

```
You are an RF engineer using rfx (JAX FDTD simulator).
Given this specification:
- Frequency: {f_center} GHz, bandwidth: {bw} GHz
- Substrate: {material}, thickness: {h} mm
- Constraints: {size_limit}, {layer_count}

Generate a complete rfx Python script that:
1. Sets up the simulation domain
2. Defines the geometry
3. Adds appropriate ports
4. Runs optimization to meet the spec
5. Plots and saves results

Use rfx.Simulation builder API. Available materials: {MATERIAL_LIBRARY}.
Available geometry: Box, Sphere, Cylinder with boolean CSG.
Available objectives: minimize_s11, maximize_s21, maximize_bandwidth.
```

### Result Interpretation with LLM

```python
# After simulation, format results for LLM analysis
import json

analysis_context = {
    "freq_ghz": list(freqs / 1e9),
    "s11_db": list(20 * np.log10(np.abs(s11))),
    "s21_db": list(20 * np.log10(np.abs(s21))),
    "bandwidth_10dB": float(bw_10dB / 1e9),
    "min_return_loss": float(np.min(-20 * np.log10(np.abs(s11)))),
    "insertion_loss": float(np.mean(-20 * np.log10(np.abs(s21[passband])))),
}

prompt = f"""Analyze these S-parameter results for a 5 GHz bandpass filter:
{json.dumps(analysis_context, indent=2)}

Evaluate:
1. Does it meet the -10dB return loss requirement?
2. What is the 3dB bandwidth?
3. What design changes would improve performance?
"""
```

## 2. ML Surrogate Models

Train neural networks on rfx simulation data for fast design space exploration.

### Data Generation

```python
import jax
import jax.numpy as jnp
import numpy as np

def generate_training_data(n_samples=1000):
    """Generate (geometry_params, S_params) pairs."""
    data = []
    for i in range(n_samples):
        # Random geometry parameters
        width = np.random.uniform(0.005, 0.020)
        length = np.random.uniform(0.010, 0.050)
        eps_r = np.random.uniform(2.0, 10.0)

        # Run rfx simulation
        sim = Simulation(freq_max=10e9, domain=(...), boundary="cpml")
        sim.add(Box(..., width, length), material_eps=eps_r)
        sim.add_port(...)
        result = sim.run(n_steps=300, compute_s_params=True)

        data.append({
            "params": [width, length, eps_r],
            "s11": np.array(result.s_params[0, 0, :]),
            "s21": np.array(result.s_params[1, 0, :]),
        })
    return data
```

### Surrogate Training (JAX/Flax)

```python
import flax.linen as nn

class SurrogateModel(nn.Module):
    """Predicts S-parameters from geometry parameters."""
    n_freqs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        # Output: real + imag parts of S11 and S21
        return nn.Dense(self.n_freqs * 4)(x)

# Train on rfx data, then use for fast sweep:
# 1000x faster than full FDTD per evaluation
# Use for initial design space exploration
# Fine-tune promising designs with full rfx simulation
```

### Hybrid Optimization

```python
# Phase 1: Surrogate-based coarse search (seconds)
best_params = surrogate_optimize(model, objective, bounds, n_iter=10000)

# Phase 2: rfx fine-tuning with jax.grad (minutes)
sim = setup_simulation(best_params)
result = optimize(sim, region, objective, n_iters=20, lr=0.01)
```

## 3. Gradient-Based Neural Topology Optimization

rfx's differentiability enables end-to-end training of neural networks that output optimized geometries.

```python
class DesignNetwork(nn.Module):
    """Neural network that outputs eps_r distribution."""
    grid_shape: tuple

    @nn.compact
    def __call__(self, spec):
        # spec = [f_center, bandwidth, target_s11]
        x = nn.Dense(256)(spec)
        x = nn.relu(x)
        x = nn.Dense(np.prod(self.grid_shape))(x)
        x = nn.sigmoid(x)  # [0, 1]
        eps_r = 1.0 + 5.0 * x.reshape(self.grid_shape)  # [1, 6]
        return eps_r

# End-to-end differentiable:
# spec → DesignNetwork → eps_r → rfx FDTD → S-params → loss
# jax.grad flows through everything including the FDTD
```

## 4. Automated Design Reports

Generate comprehensive reports by combining rfx results with LLM analysis.

```python
def generate_design_report(result, spec, output_path="report.md"):
    """Auto-generate a design report with plots and AI analysis."""
    import matplotlib.pyplot as plt

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # ... S-params, Smith chart, field distribution, convergence ...
    fig.savefig("plots.png")

    # Format for LLM
    report_data = extract_metrics(result, spec)

    # LLM generates analysis text
    analysis = llm_analyze(report_data)  # Your LLM API call

    # Assemble report
    report = f"""# RF Design Report
## Specification
{spec}

## Results
![Plots](plots.png)

## Analysis
{analysis}

## Recommendations
{llm_recommend(report_data)}
"""
    with open(output_path, "w") as f:
        f.write(report)
```

## Recommended AI Workflow

```
1. SPECIFY    → Describe design in natural language
2. GENERATE   → LLM creates rfx simulation script
3. SIMULATE   → rfx runs FDTD on GPU
4. OPTIMIZE   → jax.grad refines design (or surrogate + fine-tune)
5. ANALYZE    → LLM interprets S-params, suggests improvements
6. ITERATE    → Repeat 2-5 until spec is met
7. REPORT     → Auto-generate documentation with plots + analysis
```

This workflow leverages:
- **LLM** for specification → code translation and result interpretation
- **rfx** for physics-accurate simulation with autodiff
- **ML surrogate** for fast design space exploration
- **GPU** for accelerating both FDTD and neural network training
