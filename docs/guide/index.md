# rfx Documentation

**rfx** is a JAX-based differentiable 3D FDTD electromagnetic simulator for RF and microwave engineering.

## Guides

| Guide | Description |
|-------|-------------|
| [Installation](installation.md) | Install rfx, set up GPU, development mode |
| [Quick Start](quickstart.md) | Your first simulation in 15 minutes |
| [Simulation API](simulation_api.md) | Simulation builder, materials, sources, probes |
| [Waveguide Ports](waveguide_ports.md) | S-matrix extraction, multi-port, calibration |
| [Inverse Design](inverse_design.md) | Gradient-based optimization with jax.grad |
| [Far-Field & RCS](farfield_rcs.md) | Radiation patterns, radar cross section |
| [Advanced Features](advanced.md) | Dispersive materials, CFS-CPML, subpixel smoothing |
| [Geometry & Limitations](geometry_and_limitations.md) | Primitives, CSG, what rfx can/can't do, tool comparison |
| [Visualization & AI Analysis](visualization_and_analysis.md) | Plots, post-processing, LLM-assisted design, ML surrogates |
| [Migration Guide](migration.md) | Coming from Meep or OpenEMS? Start here |
| [Changelog](changelog.md) | Version history and release notes |
| [Contributing](contributing.md) | Dev setup, testing, PR workflow |

## Quick Links

- **GitHub**: [BK3536/rfx](https://github.com/BK3536/rfx)
- **License**: MIT
- **API Reference**: All public classes and functions have comprehensive docstrings. Use `help(rfx.Simulation)` in Python.
