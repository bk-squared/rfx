---
title: "rfx Documentation"
sidebar:
  order: -1
---

`rfx` is a JAX-based differentiable FDTD simulator for RF and microwave
engineering. This index tracks the **v1.3.1** release surface and highlights
selected post-release updates already merged on `main`.

## Recent highlights

- **Published RF validation**: 5-case error table against Balanis, Pozar, and analytical references.
- **Distributed execution**: single-host **multi-GPU** FDTD via `jax.pmap`.
- **Modern optimization workflows**: time-domain proxy objectives and NTFF-aware far-field objectives.
- **Research-grade examples**: `examples/50_advanced/` plus GPU validation scripts.

## How this guide is organized

The sidebar already groups pages by job-to-be-done. The table below gives the
same structure in one place so you can jump directly to the right section.

### Getting Started

| Guide | Description |
|---|---|
| [Installation](/rfx/guide/installation/) | Python/JAX install, GPU notes, dev setup |
| [Quick Start](/rfx/guide/quickstart/) | First simulation with the current stable high-level API |
| [Your First Patch Antenna](/rfx/guide/first-patch/) | First end-to-end success path for a practical RF structure |

### Modeling & Setup

| Guide | Description |
|---|---|
| [Simulation API](/rfx/guide/api-reference/) | `Simulation`, `Result`, materials, sources, probes, ports, and NTFF helpers |
| [Materials & Geometry](/rfx/guide/materials-geometry/) | Material library, Debye/Lorentz models, CSG shapes, and PCB stackup basics |
| [Sources & Ports](/rfx/guide/sources-ports/) | Soft sources, lumped/wire ports, waveguide ports, and Floquet workflows |
| [Probes & S-Parameters](/rfx/guide/probes-sparams/) | DFT probes, S-matrix extraction, Harminv, de-embedding, and exports |
| [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/) | `dz_profile`, `auto_configure()`, thin-substrate workflow |
| [Waveguide Ports](/rfx/guide/waveguide-ports/) | Modal waveguide excitation and calibrated multi-mode S-matrix extraction |
| [Floquet Ports](/rfx/guide/floquet-ports/) | Bloch-periodic unit-cell workflows for phased and periodic structures |

### Analysis & Validation

| Guide | Description |
|---|---|
| [Validation](/rfx/guide/validation/) | Published RF benchmarks, cross-solver checks, and workflow-validation guidance |
| [Convergence Study](/rfx/guide/tutorial-convergence/) | Mesh-refinement analysis and accuracy verification workflow |
| [Far-Field & RCS](/rfx/guide/farfield-rcs/) | NTFF radiation patterns and scattering workflows |
| [Antenna Metrics](/rfx/guide/antenna-metrics/) | Gain, efficiency, beamwidth, bandwidth, and front-to-back ratio |
| [Visualization & Analysis](/rfx/guide/visualization-and-analysis/) | Plots, exports, post-processing, and result interpretation |
| [Solver Comparison](/rfx/guide/comparison/) | Feature and workflow comparison vs. Meep and OpenEMS |

### Design & Optimization

| Guide | Description |
|---|---|
| [Inverse Design](/rfx/guide/inverse-design/) | Gradient-based optimization, proxy objectives, and NTFF-aware advanced workflows |
| [Topology Optimization](/rfx/guide/topology-optimisation/) | Density-based inverse design with filtering and projection |
| [Parametric Sweeps](/rfx/guide/parametric-sweeps/) | Sequential sweeps and `jax.vmap` batch evaluation |
| [Material Fitting](/rfx/guide/material-fitting/) | CSV import, Debye/Lorentz fitting, and differentiable fitting workflows |
| [Patch Antenna Design](/rfx/guide/tutorial-patch-antenna/) | Full rectangular patch design workflow |
| [Microstrip Filter Design](/rfx/guide/tutorial-microstrip-filter/) | Coupled-line filter workflow with two-port analysis |

### Advanced & Research Methods

| Guide | Description |
|---|---|
| [Advanced Features](/rfx/guide/advanced/) | Multi-GPU, material fitting, mixed precision, nonlinear materials, and research examples |
| [Conformal PEC](/rfx/guide/conformal-pec/) | Dey-Mittra method for curved PEC conductors |
| [SBP-SAT Subgridding](/rfx/guide/subgridding/) | Local mesh refinement with JIT performance |
| [Gradient Behavior](/rfx/guide/gradient-behavior/) | Where gradients are strong, weak, or noisy |
| [Geometry & Limitations](/rfx/guide/geometry-and-limitations/) | Supported workflows, strengths, and current trade-offs |

## Start Here

If you're new to `rfx`, start here:

1. [Installation](/rfx/guide/installation/)
2. [Quick Start](/rfx/guide/quickstart/)
3. [Simulation API](/rfx/guide/api-reference/)
4. [Sources & Ports](/rfx/guide/sources-ports/)
5. [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/)
6. [Validation](/rfx/guide/validation/)
7. [Advanced Features](/rfx/guide/advanced/) once you want distributed runs, proxy objectives, or research-style workflows

## Project & maintainer guides

| Guide | Description |
|---|---|
| [Migration Guide](/rfx/guide/migration/) | Mapping Meep/OpenEMS workflows into rfx |
| [Changelog](/rfx/guide/changelog/) | Release notes and current-main capability updates |
| [Contributing](/rfx/guide/contributing/) | Maintainer workflow, testing, linting, and coding conventions |

## API Reference

Generate API documentation locally:

```bash
pip install pdoc
pdoc rfx -o docs/api
```

Then open `docs/api/index.html` in your browser.

## Quick Links

- GitHub: [bk-squared/rfx](https://github.com/bk-squared/rfx)
- Public docs snapshot: [remilab.ai/rfx](https://remilab.ai/rfx/)
- Top-level API exports: `rfx.__init__`
- Package metadata: `pyproject.toml`
- Advanced example bundle: `examples/50_advanced/`
