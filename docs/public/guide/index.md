---
title: "rfx Documentation"
sidebar:
  order: -1
---

`rfx` is a JAX-based differentiable FDTD simulator for RF and microwave engineering.

## How to read these docs

| Lane | What to expect |
|---|---|
| **Recommended default** | uniform Cartesian Yee RF workflows: cavity, waveguide, patch-style resonance, probes, Harminv, selected S-parameter workflows, and benchmarked far-field workflows |
| **Shadow** | non-uniform mesh thin-substrate workflows |
| **Port-family evidence envelopes** | rectangular waveguide S-matrices and coaxial transmission-line reflection where the support matrix states an envelope |
| **Experimental / under active validation** | distributed execution, Floquet/Bloch, SBP-SAT subgridding, generalized planar ports, and inverse-design extensions |

Start with the recommended default lane unless you specifically need an advanced feature and can stay inside its support envelope.

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
| [Sources & Ports](/rfx/guide/sources-ports/) | Soft sources, lumped/wire ports, waveguide ports, coaxial-line reflection, and experimental port surfaces |
| [Probes & S-Parameters](/rfx/guide/probes-sparams/) | DFT probes, S-matrix helpers, Harminv, de-embedding, and exports |
| [Memory Reduction](/rfx/guide/memory-reduction/) | How to reduce FDTD/AD memory without crossing validation boundaries |
| [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/) | Shadow thin-substrate mesh workflows |
| [Waveguide Ports](/rfx/guide/waveguide-ports/) | Modal waveguide excitation and S-matrix extraction |
| [Floquet Ports](/rfx/guide/floquet-ports/) | Experimental Bloch-periodic unit-cell workflows |

## Analysis & Validation

| Guide | Description |
|---|---|
| [Validation](/rfx/guide/validation/) | Public validation overview and support status |
| [Convergence Study](/rfx/guide/tutorial-convergence/) | Mesh-refinement workflow |
| [Far-Field & RCS](/rfx/guide/farfield-rcs/) | NTFF radiation patterns and scattering workflows |
| [Antenna Metrics](/rfx/guide/antenna-metrics/) | Gain, efficiency, beamwidth, bandwidth, and front-to-back ratio |
| [Visualization & Analysis](/rfx/guide/visualization-and-analysis/) | Plots, exports, post-processing, and result interpretation |
| [Solver Comparison](/rfx/guide/comparison/) | Feature and workflow comparison vs. Meep and OpenEMS |

## Design & Optimization

| Guide | Description |
|---|---|
| [Inverse Design](/rfx/guide/inverse-design/) | Gradient-based optimization and advanced objectives |
| [Topology Optimization](/rfx/guide/topology-optimisation/) | Density-based inverse design with filtering and projection |
| [Parametric Sweeps](/rfx/guide/parametric-sweeps/) | Sequential sweeps and `jax.vmap` batch evaluation |
| [Material Fitting](/rfx/guide/material-fitting/) | CSV import, Debye/Lorentz fitting, and differentiable fitting workflows |
| [Patch Antenna Design](/rfx/guide/tutorial-patch-antenna/) | Practical rectangular patch workflow |
| [Microstrip Filter Design](/rfx/guide/tutorial-microstrip-filter/) | Experimental coupled-line filter workflow |

## Advanced / Experimental

| Guide | Description |
|---|---|
| [Advanced Features](/rfx/guide/advanced/) | Distributed runs, material fitting, mixed precision, nonlinear materials, and advanced workflows |
| [Conformal PEC](/rfx/guide/conformal-pec/) | Dey-Mittra method for curved PEC conductors |
| [SBP-SAT Subgridding](/rfx/guide/subgridding/) | Experimental local mesh refinement |
| [Gradient Behavior](/rfx/guide/gradient-behavior/) | Where gradients are strong, weak, or noisy |
| [Geometry & Limitations](/rfx/guide/geometry-and-limitations/) | Supported workflows, strengths, and current trade-offs |

## Secondary hubs

- [Examples](/rfx/examples/) — recommended public runnable paths
- [Validation](/rfx/validation/) — support and validation overview
- [API](/rfx/api/) — curated public API contract
- [Generated API](/rfx/api/generated/) — subordinate generated symbol reference
- [AI Agent Guide](/rfx/agent/overview/) — safe agent prompts, auto-configuration, and review workflow

## Quick Links

- GitHub: [bk-squared/rfx](https://github.com/bk-squared/rfx)
- Public docs snapshot: [remilab.ai/rfx](https://remilab.ai/rfx/)
- Top-level API exports: `rfx.__init__`
- Package metadata: `pyproject.toml`
