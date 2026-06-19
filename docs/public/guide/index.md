---
title: "rfx Documentation"
sidebar:
  order: -1
---

`rfx` is a JAX-based differentiable FDTD simulator for RF and microwave engineering.

## How to read these docs

| Surface | What to expect |
|---|---|
| **Recommended default** | uniform Cartesian Yee RF workflows: cavity, waveguide, patch-style resonance, probes, Harminv, selected S-parameter workflows, and benchmarked far-field workflows |
| **Bounded port-family envelopes** | lumped/wire, microstrip-line, rectangular waveguide, and coaxial-line workflows each use their matching calculator and stated limits |
| **Curated public API** | user-facing builders, observables, and reporting helpers; lower-level symbols are supporting reference |
| **Other repository code** | may exist in the repository, but is not a documented public workflow unless a guide and support entry cover it |

Start with the recommended default lane unless a public guide explicitly routes you to a bounded envelope.

## Getting Started

| Guide | Description |
|---|---|
| [Installation](/rfx/guide/installation/) | Python/JAX install, GPU notes, dev setup |
| [Quick Start](/rfx/guide/quickstart/) | First simulation with the current high-level API |
| [Your First Patch Antenna](/rfx/guide/first-patch/) | First end-to-end resonance workflow |

## Modeling & Setup

| Guide | Description |
|---|---|
| [Simulation API](/rfx/guide/api-reference/) | `Simulation`, `Result`, materials, sources, probes, ports, and NTFF helpers |
| [Materials & Geometry](/rfx/guide/materials-geometry/) | Material library, Debye/Lorentz models, CSG shapes, and PCB stackup basics |
| [Sources & Ports](/rfx/guide/sources-ports/) | Soft sources, lumped/wire ports, microstrip-line ports, waveguide ports, and bounded coaxial-line reflection |
| [Probes & S-Parameters](/rfx/guide/probes-sparams/) | DFT probes, S-matrix helpers, Harminv, de-embedding, and exports |
| [Memory Reduction](/rfx/guide/memory-reduction/) | How to reduce FDTD/AD memory without crossing validation boundaries |
| [Waveguide Ports](/rfx/guide/waveguide-ports/) | Modal waveguide excitation and S-matrix extraction |

## Analysis & Validation

| Guide | Description |
|---|---|
| [Validation](/rfx/guide/validation/) | Public validation overview and support status |
| [Benchmark Table](/rfx/guide/benchmarks/) | Current A/B cross-validation cases and reproduce commands |
| [Convergence Study](/rfx/guide/tutorial-convergence/) | Mesh-refinement workflow |
| [Far-Field & RCS](/rfx/guide/farfield-rcs/) | NTFF radiation patterns and scattering workflows |
| [Visualization & Analysis](/rfx/guide/visualization-and-analysis/) | Plots, exports, post-processing, and result interpretation |

## Design & Optimization

| Guide | Description |
|---|---|
| [Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/) | Gradient concepts for microwave engineers, mapped from Meep adjoint terminology to rfx autodiff workflows |
| [Inverse Design](/rfx/guide/inverse-design/) | Gradient-based optimization with proxy objectives and validation caveats |
| [Gradient Behavior](/rfx/guide/gradient-behavior/) | Where gradients are strong, weak, or noisy |
| [Parametric Sweeps](/rfx/guide/parametric-sweeps/) | Sequential sweeps and `jax.vmap` batch evaluation |
| [Patch Antenna Design](/rfx/guide/tutorial-patch-antenna/) | Practical rectangular patch workflow |

## Secondary hubs

- [Examples](/rfx/examples/) — recommended public runnable paths
- [Validation](/rfx/validation/) — support and validation overview
- [API](/rfx/api/) — curated public API contract

## Quick Links

- GitHub: [bk-squared/rfx](https://github.com/bk-squared/rfx)
- Public docs snapshot: [remilab.ai/rfx](https://remilab.ai/rfx/)
- Top-level API exports: `rfx.__init__`
- Package metadata: `pyproject.toml`
