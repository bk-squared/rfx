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

**v1.6.6** — JAX-based RF/FDTD workflows, GPU-oriented execution, practical examples, structured setup guards, and port-family validation envelopes.

> **Project status:** `rfx` is an actively validated RF/FDTD simulator. Use the uniform Cartesian Yee RF lane first; additional workflows should be used only inside their documented evidence envelopes.

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## At a Glance

The recommended starting point is the **uniform Cartesian Yee RF/FDTD lane**. Public claims are scoped to documented workflows and bounded evidence envelopes rather than every importable symbol.

| | |
|---|---|
| **GPU-accelerated** | 7,309 Mcells/s on RTX 4090, 5,249 on A6000 via `jax.lax.scan` JIT |
| **Differentiable workflows** | `jax.grad` through selected JAX-traced time-domain objectives; final RF claims still require the relevant validation envelope |
| **RF workflow tools** | materials, sources, probes, ports, S-parameter helpers, Harminv, far-field utilities |
| **Waveguide modal ports** | analytical TE/TM eigenmodes for documented rectangular-guide S-matrix envelopes |
| **Port-family S-parameter routing** | lumped/wire, microstrip-line, rectangular waveguide, and coaxial-line workflows use different calculators and evidence envelopes |
| **Documented benchmark evidence** | public validation cases mapped to analytic or external references with stated envelopes |
| **Regression checks** | CI and local checks support development without replacing feature-specific validation |
| **Documentation scope** | public guides cover maintained workflows and explicitly bounded support envelopes |

## Current Highlights

- **Public workflow scope**: user guides lead with uniform Yee RF workflows, documented waveguide/MSL/lumped/wire port envelopes, bounded coaxial line reflection, and conservative differentiable design loops.
- **Structured preflight and runtime guards**: `preflight()` and `preflight_sparameters()` return coded `PreflightReport` issues while remaining list/string compatible; `run()`, `forward()`, S-matrix calculators, sweeps, and optimizers surface NaN/passivity/setup problems earlier.
- **Waveguide S-matrix evidence**: rectangular waveguide S-matrices have the most mature documented port-family evidence envelope; use the support matrix for exact limits rather than broadening the claim.
- **Coaxial line reflection**: `compute_coaxial_line_reflection(...)` is the bounded one-port coaxial transmission-line reflection path. It is not a general coaxial network solver.
- **Curated star-import surface**: `rfx.__all__` exposes the supported public API; per-step kernels and bookkeeping helpers remain available for compatibility but are outside the curated API surface.
- **Maintained validation lanes**: a full-suite PR gate, API-reference drift checks, a maintained GPU suite harness, weekly external-crossval CI, and on-demand source-built-OpenEMS validation use the honest exit-code convention (reference-missing is a visible SKIP, never a silent pass).

## Installation

```bash
pip install rfx-fdtd
```

GPU support (JAX + CUDA):

```bash
pip install "jax[cuda12]" rfx-fdtd
```

For development:

```bash
git clone https://github.com/bk-squared/rfx.git
cd rfx && pip install -e ".[all]"
```

## Quick Start

```python
from rfx import Simulation, Box, GaussianPulse

# 2.4 GHz patch antenna on FR4
sim = Simulation(freq_max=4e9, domain=(0.08, 0.06, 0.025), boundary="cpml")
sim.add(Box((0.0, 0.0, 0.0), (0.08, 0.06, 0.0016)), material="fr4")
sim.add(Box((0.02, 0.01, 0.0016), (0.049, 0.049, 0.0016)), material="pec")
sim.add(Box((0.0, 0.0, 0.0), (0.08, 0.06, 0.0)), material="pec")
sim.add_source((0.029, 0.03, 0.0008), "ez",
               waveform=GaussianPulse(f0=2.4e9, bandwidth=0.8))
sim.add_probe((0.029, 0.03, 0.0008), "ez")

result = sim.run(n_steps=8000)
modes = result.find_resonances(freq_range=(1.5e9, 3.5e9))
print(f"Resonance: {modes[0].freq/1e9:.4f} GHz  Q={modes[0].Q:.0f}")
```

> This compact snippet is for API orientation, not a resolved reference case.
> It uses a coarse mesh and may print non-fatal preflight advisories. For a
> resolved rectangular patch workflow with geometry, mesh, material, and
> validation notes, follow the [First Patch tutorial](docs/public/guide/first-patch.mdx).

## Differentiable Design Workflows

rfx supports selected JAX-traced objectives for inverse design. Treat these as
**sensitivity calculations through the implemented discrete solver**, not as
standalone RF validation. For S-parameter-driven work, use the calculator that
matches the port family and keep the final claim inside that family's evidence
envelope.

The runnable gradient examples include:

- [`examples/inverse_design/differentiable_s11_design.py`](examples/inverse_design/differentiable_s11_design.py) — differentiable S11 objective with a finite-difference cross-check.
- [`examples/inverse_design/multilayer_ar_coating.py`](examples/inverse_design/multilayer_ar_coating.py) — conservative differentiable material-profile demo.

For the conceptual background, see the public [Autodiff and Adjoint Background](docs/public/guide/autodiff-adjoint.mdx) guide.

## Support Contracts

Support contracts keep public claims aligned with evidence:

- [`docs/guides/support_matrix.md`](docs/guides/support_matrix.md) and [`docs/guides/support_matrix.json`](docs/guides/support_matrix.json) — feature support status and public-scope rules.
- [`docs/guides/sparameter_support_matrix.md`](docs/guides/sparameter_support_matrix.md) and [`docs/guides/sparameter_support_matrix.json`](docs/guides/sparameter_support_matrix.json) — port-family calculators, result schemas, and evidence envelopes.

The public API surface is pinned by a symbol/signature inventory gate and the curated public docs.

## GPU Performance (Measured)

| GPU | VRAM | Peak Mcells/s (200^3) |
|-----|------|-----------------------|
| RTX 4090 | 24 GB | **7,309** |
| RTX 3090 | 24 GB | **5,847** |
| RTX A6000 | 48 GB | **5,249** |

| Grid | RTX 4090 | RTX 3090 | A6000 |
|------|----------|----------|-------|
| 50^3 (125K) | 751 | 455 | 483 |
| 100^3 (1M) | 5,332 | 2,502 | 2,410 |
| 150^3 (3.4M) | 6,258 | 4,797 | 4,158 |
| 200^3 (8M) | 7,309 | 5,847 | 5,249 |

Gradient (reverse-mode AD): ~3-4x forward pass with `jax.checkpoint`.

## Accuracy Validation

Representative public validation cases are tied to named analytic or textbook references:

| Structure | Reference | Error |
|-----------|-----------|-------|
| Patch antenna resonance | Balanis Ch 14 | 1.97% |
| WR-90 TE10 cutoff | Analytical | 0.60% |
| Dielectric cavity TM110 | Analytical | 0.016% |
| Microstrip Z0 | Hammerstad-Jensen | 0.47% |
| Coupled-line filter | Pozar Ch 8 | 22.5% (formula limitation) |

The full cross-validation suite (Meep / OpenEMS / Palace / analytic references, one reproduce command per case) is summarized in the public validation docs. The CPU-feasible subset runs locally via `python scripts/run_crossval_cpu.py` using the repo-wide exit-code convention (0 = full pass with external reference, 1 = self-check failure, 2 = reference unavailable — visibly skipped, never silently green).

For practical public examples, start with `examples/crossval/05_patch_antenna.py` for the patch workflow and `examples/crossval/11_waveguide_port_wr90.py` for rectangular waveguide ports. Use scripts outside the recommended example set only as local diagnostics unless a public guide and support matrix entry state otherwise.

## Key Features

### Core Simulator
- 3D/2D Yee FDTD with CFS-CPML (kappa_max=1.0 default — measured sweep showed kappa_max>1 degrades guided-mode absorption)
- Mixed precision (float16 fields, 2x memory reduction)
- Auto-configuration from geometry + frequency range
- Structured preflight reports for setup, support-boundary, and S-parameter routing checks

### Sources & Ports
- GaussianPulse, ModulatedGaussian, CW, custom waveforms
- Lumped/wire feed ports and lumped RLC (series/parallel ADE); calibrated S-parameter workflows depend on the selected port family
- Specialized microstrip-line ports through `compute_msl_s_matrix(...)`
- Rectangular waveguide modal ports through `compute_waveguide_s_matrix(...)`
- Coaxial transmission-line reflection through the bounded `compute_coaxial_line_reflection(...)` envelope

### Materials
- Debye/Lorentz/Drude dispersive, Kerr nonlinear (chi3)
- Subpixel smoothing, thin conductor correction
- Library: pec, fr4, rogers4003c, copper, alumina, water_20c, ...

### Analysis & Optimization
- S-parameter tooling with per-family calculators: lumped/wire, MSL, waveguide, and coaxial-line workflows use different APIs and evidence envelopes
- Harminv resonance extraction (MPM)
- Documented far-field, RCS, radiation-pattern, and polarization workflows
- Differentiable proxy-objective optimization for selected workflows
- Parametric sweep + jax.vmap batch evaluation
- Smith chart, de-embedding, Touchstone I/O (.s2p/.s4p/.snp)
- Auto convergence study with Richardson extrapolation

### Geometry & Workflow
- Box, Sphere, Cylinder (CSG), Via, CurvedPatch
- PCB stackup builder (2-layer, 4-layer presets)
- Field animation (GIF/MP4)
- Artifact/report exporters for reproducible review bundles

## Documentation

Full documentation: **[remilab.ai/rfx](https://remilab.ai/rfx/)**

Repo-level support and reference-lane contract artifacts:

- `docs/guides/support_matrix.md`
- `docs/guides/reference_lane_contract.md`

Canonical public-doc sources in this repo:

- `docs/public/index.mdx` — public `/rfx/` landing page
- `docs/public/guide/` — public guide pages
- `docs/public/examples/` — runnable example hubs
- `docs/public/validation/` — quantitative evidence and lane-label hubs
- `docs/public/api/` — curated public API pages
- `docs/guides/public_docs_architecture.md` — ownership, sync, and deploy rules
- `docs/guides/public_docs_maintenance.md` — release/docs-sync checklist, outdated-doc policy, and support-matrix cadence

### Public docs maintenance workflow

1. Edit the source pages in `docs/public/index.mdx`, `docs/public/guide/`, `docs/public/examples/`, `docs/public/validation/`, or `docs/public/api/`.
2. Validate the source tree:
   ```bash
   python scripts/check_public_docs_manifest.py
   ```
3. Export the updated snapshot to gitops:
   ```bash
   python scripts/export_public_docs_to_gitops.py
   ```
4. Keep the source repo CI in sync with `.github/workflows/public-docs-source.yml`.

Source-side CI for this flow lives in:

- `.github/workflows/public-docs-source.yml`

Gitops-side snapshot/build CI lives in the deploy repo:

- `remilab-sites-gitops/.github/workflows/rfx-public-docs-sync.yml`

### Start here
- [Public landing page](docs/public/index.mdx)
- [Validation hub](docs/public/validation/index.mdx) — support overview and lane labels
- [Examples hub](docs/public/examples/index.mdx) — current runnable entry points

### Tutorials
- [Patch Antenna Design](docs/public/guide/tutorial-patch-antenna.mdx) — practical patch workflow from local resonance run to external cross-check
- [Convergence Study](docs/public/guide/tutorial-convergence.mdx) — mesh independence methodology

### Guides
- [Migration from Meep/OpenEMS](docs/public/guide/migration.md)
- [Changelog](docs/public/guide/changelog.mdx)
- [Contributing](docs/public/guide/contributing.md)

## Citation

```bibtex
@software{kim_rfx_2026,
  author       = {Byungkwan Kim},
  title        = {rfx: JAX-based differentiable 3D FDTD simulator for RF engineering},
  institution  = {REMI Lab, Chungnam National University},
  year         = {2026},
  version      = {1.6.6},
  url          = {https://github.com/bk-squared/rfx}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://remilab.cnu.ac.kr) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.
