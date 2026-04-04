# rfx Documentation

`rfx` is a JAX-based differentiable FDTD simulator for RF and microwave
engineering. The guide set below focuses on the current v1.0 surface:

- **non-uniform z meshing** for thin substrates,
- **lumped/wire/waveguide ports**,
- **lumped RLC elements**,
- **via / curved-patch geometry helpers**,
- **field animation export**,
- and **benchmark/validation workflows**.

## Core Guides

| Guide | Description |
|---|---|
| [Installation](installation.md) | Python/JAX install, GPU notes, dev setup |
| [Quick Start](quickstart.md) | First simulation with the current high-level API |
| [Simulation API](simulation_api.md) | `Simulation`, `Result`, materials, sources, probes, ports |
| [Sources & Ports](sources_ports.md) | Soft sources, polarized sources, lumped/wire/waveguide ports |
| [Non-Uniform Mesh](nonuniform_mesh.md) | `dz_profile`, `auto_configure()`, thin-substrate workflow |
| [Waveguide Ports](waveguide_ports.md) | Modal waveguide excitation and calibrated S-matrix extraction |
| [Inverse Design](inverse_design.md) | Gradient-based optimization with `jax.grad` |
| [Far-Field & RCS](farfield_rcs.md) | NTFF radiation patterns and RCS workflows |
| [Validation](validation.md) | What is strongly benchmarked vs. what is still evolving |
| [Advanced Features](advanced.md) | Dispersive media, lumped RLC, via/curved geometry, animation |
| [Geometry & Limitations](geometry_and_limitations.md) | Primitives, CSG, current boundaries/limitations |
| [Visualization & Analysis](visualization_and_analysis.md) | Plots, exports, post-processing, interpretation |

## Start Here

If you are new to `rfx`, the shortest path is usually:

1. [Installation](installation.md)
2. [Quick Start](quickstart.md)
3. [Simulation API](simulation_api.md)
4. [Sources & Ports](sources_ports.md)
5. [Non-Uniform Mesh](nonuniform_mesh.md)
6. [Validation](validation.md)

## Tutorials

Step-by-step design workflows that walk through complete problems from
specification to verified result. Each tutorial is self-contained and
aimed at graduate students in RF/microwave engineering.

| Tutorial | Description |
|---|---|
| [2.4 GHz Patch Antenna](tutorial-patch-antenna.md) | Analytical design, non-uniform mesh, resonance extraction, convergence check, optional topology optimisation |
| [Coupled-Line Bandpass Filter](tutorial-microstrip-filter.md) | Even/odd mode theory, coupled microstrip geometry, S-parameter analysis, parametric gap sweep |
| [Mesh Convergence & Verification](tutorial-convergence.md) | Why convergence matters, `convergence_study()`, Richardson extrapolation, log-log error plots, common pitfalls |

## Project / Maintainer Guides

| Guide | Description |
|---|---|
| [Migration Guide](migration.md) | Mapping Meep/OpenEMS workflows into rfx |
| [Comparison](comparison.md) | Fair comparison with Meep, OpenEMS, FDTDX, and commercial tools |
| [Changelog](changelog.md) | Release notes and major capability changes |
| [Contributing](contributing.md) | Developer / maintainer workflow, testing, and coding conventions |

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
