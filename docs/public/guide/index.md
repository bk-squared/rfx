---
title: "rfx Documentation"
sidebar:
  order: -1
---

`rfx` is a JAX-based differentiable FDTD simulator for RF and microwave
engineering. The current public docs track the **v1.3** release surface while
also highlighting important post-release updates already merged on `main`.

## Recent highlights

- **Published RF validation**: 5-case error table against Balanis, Pozar, and analytical references.
- **Distributed execution**: single-host **multi-GPU** FDTD via `jax.pmap`.
- **Modern optimisation workflows**: time-domain proxy objectives and NTFF-aware far-field objectives.
- **Research-grade examples**: `examples/50_advanced/` plus GPU validation scripts.

## Core Guides

| Guide | Description |
|---|---|
| [Installation](/rfx/guide/installation/) | Python/JAX install, GPU notes, dev setup |
| [Quick Start](/rfx/guide/quickstart/) | First simulation with the current stable high-level API |
| [Simulation API](/rfx/guide/api-reference/) | `Simulation`, `Result`, materials, sources, probes, ports, and NTFF helpers |
| [Sources & Ports](/rfx/guide/sources-ports/) | Soft sources, polarized sources, lumped/wire/waveguide ports, Floquet workflows |
| [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/) | `dz_profile`, `auto_configure()`, thin-substrate workflow |
| [Waveguide Ports](/rfx/guide/waveguide-ports/) | Modal waveguide excitation and calibrated multi-mode S-matrix extraction |
| [Inverse Design](/rfx/guide/inverse-design/) | Gradient-based optimisation, proxy objectives, and NTFF-aware advanced workflows |
| [Far-Field & RCS](/rfx/guide/farfield-rcs/) | NTFF radiation patterns and scattering workflows |
| [Validation](/rfx/guide/validation/) | Published RF benchmarks, cross-solver checks, and workflow-validation guidance |
| [Advanced Features](/rfx/guide/advanced/) | Multi-GPU, material fitting, mixed precision, nonlinear materials, research examples |
| [Geometry & Limitations](/rfx/guide/geometry-and-limitations/) | Primitives, supported workflows, and current trade-offs |
| [Visualization & Analysis](/rfx/guide/visualization-and-analysis/) | Plots, exports, post-processing, interpretation |

## Start Here

If you are new to `rfx`, the shortest path is usually:

1. [Installation](/rfx/guide/installation/)
2. [Quick Start](/rfx/guide/quickstart/)
3. [Simulation API](/rfx/guide/api-reference/)
4. [Sources & Ports](/rfx/guide/sources-ports/)
5. [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/)
6. [Validation](/rfx/guide/validation/)
7. [Advanced Features](/rfx/guide/advanced/) once you want distributed runs, proxy objectives, or research-style workflows

## Project / Maintainer Guides

| Guide | Description |
|---|---|
| [Migration Guide](/rfx/guide/migration/) | Mapping Meep/OpenEMS workflows into rfx |
| [Changelog](/rfx/guide/changelog/) | Release notes and current-main capability updates |
| [Contributing](/rfx/guide/contributing/) | Developer / maintainer workflow, testing, linting, and coding conventions |

## API Reference

Generate API documentation locally:

```bash
pip install pdoc
pdoc rfx -o docs/api
```

Then open `docs/api/index.html` in your browser.

## Quick Links

- GitHub: [BK3536/rfx](https://github.com/BK3536/rfx)
- Public docs snapshot: [remilab.ai/rfx](https://remilab.ai/rfx/)
- Top-level API exports: `rfx.__init__`
- Package metadata: `pyproject.toml`
- Advanced example bundle: `examples/50_advanced/`
