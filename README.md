# rfx

```text
██████╗ ███████╗██╗  ██╗
██╔══██╗██╔════╝╚██╗██╔╝
██████╔╝█████╗   ╚███╔╝
██╔══██╗██╔══╝   ██╔██╗
██║  ██║██║     ██╔╝ ██╗
╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
```

**Differentiable 3D FDTD electromagnetic simulator for RF and microwave engineering — powered by JAX.**

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/BK3536/rfx/actions/workflows/test.yml/badge.svg)](https://github.com/BK3536/rfx/actions)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## Highlights

| | |
|---|---|
| **GPU-accelerated** | ~3,000 Mcells/s on RTX 4090 via `jax.lax.scan` JIT |
| **Differentiable** | `jax.grad` through full time-stepping for inverse design |
| **Cross-validated** | 0.000--0.007% agreement vs Meep and OpenEMS |
| **Non-uniform mesh** | Graded z-profiles for thin substrates without global refinement |
| **Auto-configuration** | `auto_configure()` derives dx, domain, CPML, timesteps from geometry |
| **260+ tests** | PEC cavities, dielectric materials, S-params, far-field, optimization |

## Installation

```bash
# From source (recommended)
git clone https://github.com/BK3536/rfx.git
cd rfx
pip install -e ".[dev]"
```

> **GPU:** Complete the [JAX GPU setup](https://jax.readthedocs.io/en/latest/installation.html) for your platform before running large simulations.

## Quick Start

```python
from rfx import Simulation, Box, GaussianPulse
import numpy as np

# PEC cavity with dielectric slab — simulate up to 10 GHz
sim = Simulation(freq_max=10e9, domain=(0.048, 0.032, 0.032), boundary="pec")
sim.add_material("slab", eps_r=2.2)
sim.add(Box((0.016, 0, 0), (0.032, 0.032, 0.032)), material="slab")

# Lumped port + probe
sim.add_port((0.006, 0.016, 0.016), "ez", impedance=50.0,
             waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
sim.add_probe((0.042, 0.016, 0.016), "ez")

result = sim.run(num_periods=30)

# S11
s11_dB = 20 * np.log10(np.abs(result.s_params[0, 0, :]) + 1e-12)
```

## Patch Antenna with Non-Uniform Mesh

```python
from rfx import Simulation, Box, GaussianPulse, auto_configure
import numpy as np

# Auto-configure from geometry and frequency range
geometry = [
    (Box((0, 0, 0), (0.06, 0.06, 0.0016)), "fr4"),
    (Box((0, 0, 0), (0.06, 0.06, 0)), "pec"),
    (Box((0.015, 0.011, 0.0016), (0.0444, 0.049, 0.0016)), "pec"),
]
materials = {"fr4": {"eps_r": 4.4, "sigma": 0.025}, "pec": {"eps_r": 1.0, "sigma": 1e10}}
config = auto_configure(geometry, (1e9, 4e9), materials=materials, accuracy="standard")
print(config.summary())  # Shows dx, non-uniform dz, CPML, n_steps

# Build simulation from auto-config
sim = Simulation(**config.to_sim_kwargs())
sim.add_material("fr4", eps_r=4.4, sigma=0.025)
for shape, mat in geometry:
    sim.add(shape, material=mat)
sim.add_source((0.025, 0.03, 0.0008), "ez", waveform=GaussianPulse(f0=2.4e9))
sim.add_probe((0.025, 0.03, 0.0008), "ez")

result = sim.run(n_steps=config.n_steps)
modes = result.find_resonances(freq_range=(1.5e9, 3.5e9))
for m in modes:
    print(f"f={m.freq/1e9:.4f} GHz, Q={m.Q:.0f}")
```

## Inverse Design

```python
from rfx import Simulation, Box, GaussianPulse
from rfx.optimize import DesignRegion, optimize
from rfx.optimize_objectives import minimize_s11

sim = Simulation(freq_max=10e9, domain=(0.048, 0.032, 0.032), boundary="pec")
sim.add_port((0.006, 0.016, 0.016), "ez", impedance=50.0,
             waveform=GaussianPulse(f0=5e9, bandwidth=0.8))

region = DesignRegion(
    corner_lo=(0.016, 0.008, 0.008),
    corner_hi=(0.032, 0.024, 0.024),
    eps_range=(1.0, 4.4),
)

result = optimize(sim, region, minimize_s11, n_iters=40, lr=0.05)
print(f"Final loss: {result.loss_history[-1]:.4f}")
```

## Key Features

### Simulation Engine
- 3D/2D Yee solver with CFS-CPML absorbing boundaries
- True PEC mask (component-specific tangential zeroing)
- Non-uniform z mesh for thin substrates (auto-detected)
- Periodic boundaries, TFSF plane-wave source

### Sources & Ports
- GaussianPulse, ModulatedGaussian, CW, custom waveforms
- Lumped ports and multi-cell wire ports (conductor-to-conductor)
- Waveguide ports with modal decomposition
- Polarized sources (Jones vector: circular, LHCP, slant45)

### Materials
- Dispersive: Debye, Lorentz/Drude
- Magnetic (`mu_r` validated)
- Thin conductors with subcell correction
- Subpixel smoothing (Farjadpour/Kottke)
- Built-in library: PEC, FR4, Rogers 4003C, copper, alumina, PTFE, water

### Analysis
- S-parameters: lumped, wire (JIT-DFT), waveguide (N-port matrix)
- Harminv resonance extraction (Matrix Pencil Method)
- NTFF far-field, radiation patterns, RCS
- Far-field polarization (axial ratio, tilt, sense)
- Touchstone I/O, HDF5 checkpoints, VTK export

### Optimization
- Design regions with `jax.grad` through full simulation
- Adam optimizer, gradient checkpointing
- Objective library: minimize_s11, maximize_s21, target_impedance, maximize_bandwidth

## Cross-Validation

Three-way validation against Meep and OpenEMS:

| Geometry | rfx vs Meep | rfx vs OpenEMS | rfx vs Analytical |
|----------|-------------|----------------|-------------------|
| PEC cavity TM110 | 0.010% | **0.000%** | 0.013% |
| FR4 cavity (eps=4.4) | 0.006% | — | 0.013% |
| Rogers (eps=3.55) | 0.007% | — | 0.026% |
| PTFE (eps=2.2) | 0.004% | — | 0.004% |
| Alumina (eps=9.8) | 0.005% | — | 0.038% |

## GPU Performance (RTX 4090)

| Grid | Steps | Time | Throughput |
|------|-------|------|------------|
| 23^3 | 200 | 0.087s | 28 Mcells/s |
| 33^3 | 300 | 0.086s | 125 Mcells/s |
| 43^3 | 400 | 0.103s | 310 Mcells/s |
| 63^3 | 500 | 0.095s | 1,310 Mcells/s |

Gradient (reverse-mode AD): ~0.31s for all grid sizes.

## Documentation

Full documentation: **[remilab.ai/rfx](https://remilab.ai/rfx/)**

- [User Guide](https://remilab.ai/rfx/guide/) — Installation, API, materials, sources, probes
- [AI Agent Guide](https://remilab.ai/rfx/agent/) — Auto-configuration, prompt templates, design workflows

## Citation

```bibtex
@software{kim_rfx_2026,
  author       = {Byungkwan Kim},
  title        = {rfx: JAX-based differentiable 3D FDTD simulator},
  institution  = {REMI Lab, Chungnam National University},
  year         = {2026},
  url          = {https://github.com/BK3536/rfx}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://remilab.cnu.ac.kr) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.

AI-assisted development: Claude (Anthropic), Codex (OpenAI).
