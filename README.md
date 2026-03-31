# `rfx`

```text
██████╗ ███████╗██╗  ██╗
██╔══██╗██╔════╝╚██╗██╔╝
██████╔╝█████╗   ╚███╔╝
██╔══██╗██╔══╝   ██╔██╗
██║  ██║██║     ██╔╝ ██╗
╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
```

JAX-based differentiable 3D FDTD electromagnetic simulator for RF and microwave engineering.

[![PyPI](https://img.shields.io/badge/PyPI-coming_soon-lightgrey)](#installation)
[![Tests](https://img.shields.io/badge/tests-placeholder-lightgrey)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

## What is rfx?

`rfx` is a JAX-native finite-difference time-domain (FDTD) simulator built for differentiable electromagnetic modeling in RF and microwave engineering. It supports end-to-end automatic differentiation, so gradients can flow through the entire simulation with `jax.grad` for optimization and inverse design workflows. The project targets practical RF and microwave regimes up to X-band (10 GHz), with a focus on simulation workflows that remain programmable, composable, and accelerator-friendly. It is positioned to be competitive with tools such as Meep and OpenEMS while adding native autodiff as a first-class capability.

## Key Features

- **3D and 2D FDTD** with CFS-CPML absorbing boundaries
- **Differentiable** simulation with `jax.grad` and `jax.checkpoint` for reverse-mode AD
- **Sources:** Gaussian pulse, CW, custom waveforms, TFSF plane wave, lumped ports, and waveguide ports
- **Materials:** dispersive (Debye, Lorentz/Drude), magnetic (`mu_r`), thin conductors, and subpixel smoothing
- **S-parameters:** N-port extraction, waveguide modal decomposition, and two-run normalization
- **Far-field:** NTFF, radiation patterns, and radar cross section (RCS) computation
- **Inverse design:** Adam optimizer with design regions
- **I/O:** Touchstone (`.sNp`) files and HDF5 checkpoints
- **GPU accelerated** via JAX
- **187 tests**, cross-validated against Meep and OpenEMS

## Installation

Install the base package from PyPI:

```bash
pip install rfx
```

For GPU-enabled workflows:

```bash
pip install rfx[gpu]
```

> **Note:** For CUDA or other accelerator backends, complete the appropriate
> [JAX GPU setup](https://jax.readthedocs.io/en/latest/installation.html)
> for your platform before running large simulations.

## Quick Start

The example below creates a PEC cavity with a dielectric slab, excites it with a lumped port, runs the simulation, extracts S11, and plots the result.

```python
import matplotlib.pyplot as plt
from rfx import Simulation

sim = (
    Simulation((48, 32, 32), spacing=1e-3, dt="auto")
    .background(eps_r=1.0)
    .pec_box((0, 0, 0), (48, 32, 32))
    .box((18, 8, 8), (30, 24, 24), eps_r=2.2)
    .lumped_port("P1", start=(6, 16, 16), stop=(10, 16, 16), impedance=50.0)
    .gaussian_pulse("P1", f0=5e9, fwidth=2e9)
    .run(steps=1200)
)

freq, s11 = sim.s_parameters(port="P1", ref_impedance=50.0)["S11"]
plt.plot(freq / 1e9, 20 * s11.log10_mag())
plt.xlabel("Frequency [GHz]")
plt.ylabel("|S11| [dB]")
plt.show()
```

## Inverse Design Example

This sketch shows a simple design-region optimization that adjusts `eps_r` between two ports to minimize reflection.

```python
import matplotlib.pyplot as plt
from rfx import Simulation, optim

sim = Simulation((80, 24, 24), spacing=1e-3).waveguide_port("in").waveguide_port("out")
design = sim.design_region((28, 6, 6), (52, 18, 18), init_eps_r=2.0)
opt = optim.Adam(learning_rate=5e-2)
history = sim.optimize(design, optimizer=opt, steps=80, objective=lambda out: out.s11_power())

plt.plot(history["loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
```

## Documentation

Documentation is currently centered around in-code docstrings and runnable examples in the `examples/` directory. Start with the builder API examples and then inspect individual object and method docstrings for parameter details and simulation options.

## Citation

If you use `rfx` in academic work, please cite:

```bibtex
@software{kim_rfx,
  author       = {Byungkwan Kim},
  title        = {rfx: JAX-based differentiable 3D FDTD simulator},
  institution  = {REMI Lab, Chungnam National University},
  year         = {2026},
  url          = {https://github.com/BK3536/rfx},
  note         = {BibTeX placeholder -- update after formal release}
}
```

## License

`rfx` is released under the [MIT License](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://github.com/BK3536) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.
