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

> Start with the uniform Cartesian Yee solver. Feature support and its limits
> live in the [support matrix](docs/guides/support_matrix.md); per-port-family
> S-parameter limits live in the
> [S-parameter support matrix](docs/guides/sparameter_support_matrix.md).

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## At a Glance

| | |
|---|---|
| **GPU-accelerated** | 200³ grid on an RTX 4090: **7,266 Mcells/s** with PEC walls, **2,087 Mcells/s** with CPML absorbers — an open-boundary simulation pays the absorber, so quote the second for antenna/scattering work. Measured by marginal-cost differencing (`scripts/diagnostics/gpu_throughput_bench.py`); see the benchmark guide for other cards. |
| **Differentiable** | `jax.grad` through the time-domain solver for sensitivity and inverse design |
| **RF workflow tools** | materials, sources, probes, ports, S-parameters, Harminv, far-field / RCS |
| **Per-family S-parameters** | lumped/wire, microstrip, rectangular waveguide, and coaxial paths use distinct calculators |
| **Preflight guards** | `sim.preflight()` surfaces setup errors and support-boundary issues before a run |
| **Cross-validated** | public cases mapped to Meep / OpenEMS / Palace / analytic references, each with a reproduce command |

## Installation

```bash
pip install rfx-fdtd                  # CPU
pip install "jax[cuda12]" rfx-fdtd    # GPU (JAX + CUDA)
```

Development install:

```bash
git clone https://github.com/bk-squared/rfx.git
cd rfx && pip install -e ".[all]"
```

## Quick Start

```python
from rfx import Box, GaussianPulse, Simulation

sim = Simulation(
    freq_max=5e9,
    domain=(0.14, 0.06, 0.05),
    dx=2e-3,
    boundary="cpml",
    cpml_layers=8,
)
sim.add_material("slab", eps_r=2.2, sigma=0.01)
sim.add(Box((0.07, 0.018, 0.018), (0.09, 0.042, 0.032)), material="slab")
sim.add_source(
    (0.03, 0.03, 0.025),
    "ez",
    waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
)
sim.add_probe((0.11, 0.03, 0.025), "ez")

preflight = sim.preflight()
print(preflight.format())
preflight.raise_for_failure()

result = sim.run(n_steps=1200)
print(result.time_series.shape)
```

For a real antenna workflow — including the mesh, time-window, and reference
checks required before reporting RF results — follow the
[First Patch tutorial](docs/public/guide/first-patch.mdx).

## Interfaces

Beyond the Python API:

- **Dashboard** — `pip install "rfx-fdtd[dashboard]" && rfx-dashboard`: browser GUI for building, running, and inspecting a simulation.
- **Experiment CLI** — `rfx experiment run <spec.json>`: versioned CPU runs from a strict JSON spec, with `submit`/`status`/`cancel`.
- **Studio + MCP** — `pip install "rfx-fdtd[studio]" && rfx studio`: local app with append-only experiment revisions, approval-gated MCP actions, and an optional LLM Design Copilot.

All three share the same `ExperimentSpec` format. Details, safety model, and
remote deployment: [Studio, CLI, and MCP Experiments](docs/public/guide/studio-experiments.mdx).

## Differentiable Design

JAX-traced objectives for inverse design — sensitivity calculations through the
discrete solver, validated per port family. Runnable examples with
finite-difference cross-checks live in
[`examples/inverse_design/`](examples/inverse_design/); background in the
[Autodiff and Adjoint guide](docs/public/guide/autodiff-adjoint.mdx).

## Validation

Every public cross-validation case is mapped to a named analytic or external
reference with a reproduce command and acceptance gates. Start with
[Cross-Validation and Accuracy](docs/public/guide/validation.mdx) for the
support limits, then [Benchmarks](docs/public/guide/benchmarks.mdx) for the
per-case numbers. The CPU-feasible subset runs locally:

```bash
PYTHONPATH=. python scripts/run_crossval_cpu.py
```

Exit codes: `0` all gates passed, `1` a gate failed, `2` a required external
reference was unavailable (inconclusive, not silently green).

## Documentation

Full documentation: **[remilab.ai/rfx](https://remilab.ai/rfx/)**

- Start here: [public landing page](docs/public/index.mdx) · [validation hub](docs/public/validation/index.mdx) · [examples hub](docs/public/examples/index.mdx)
- Tutorials: [patch antenna](docs/public/guide/tutorial-patch-antenna.mdx) · [convergence study](docs/public/guide/tutorial-convergence.mdx) · ordered learning path in [`examples/tutorials/`](examples/README.md)
- Guides: [migration from Meep/OpenEMS](docs/public/guide/migration.md) · [changelog](docs/public/guide/changelog.mdx) · [contributing](docs/public/guide/contributing.md)
- **AI coding agents**: purpose-built docs in [`docs/agent/`](docs/agent/overview.mdx) — operating rules, [repo map](docs/agent/repo-map.mdx), [port selection](docs/agent/port-selection.mdx), and task recipes.

## Citation

```bibtex
@software{kim_rfx_2026,
  author       = {Byungkwan Kim},
  title        = {rfx: JAX-based differentiable 3D FDTD simulator for RF engineering},
  institution  = {REMI Lab, Chungnam National University},
  year         = {2026},
  url          = {https://github.com/bk-squared/rfx}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://remilab.cnu.ac.kr) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.
