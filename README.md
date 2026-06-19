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

**v1.6.5 package** — JAX-based RF/FDTD workflows, GPU-oriented execution, practical examples, structured setup guards, and port-family validation envelopes.

> **Project status (June 2026):** `rfx` is an actively validated RF/FDTD simulator. Use the uniform Cartesian Yee RF lane first; additional workflows should be used only inside their documented evidence envelopes.

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## At a Glance

The recommended starting point is the **uniform Cartesian Yee RF/FDTD lane**. Public claims are scoped to documented workflows and bounded evidence envelopes rather than every importable symbol.

| | |
|---|---|
| **GPU-accelerated** | 7,309 Mcells/s on RTX 4090, 5,249 on A6000 via `jax.lax.scan` JIT |
| **Differentiable** | `jax.grad` through time-domain workflows for inverse design |
| **RF workflow tools** | materials, sources, probes, ports, S-parameter helpers, Harminv, far-field utilities |
| **Waveguide modal ports** | analytical TE/TM eigenmodes for documented rectangular-guide S-matrix envelopes |
| **Port-family S-parameter routing** | lumped/wire, microstrip-line, rectangular waveguide, and coaxial-line workflows use different calculators and evidence envelopes |
| **Published benchmark evidence** | 5-case benchmark against Balanis/Pozar (patch 1.97%, cavity 0.016%) |
| **Regression checks** | CI and local checks support development without replacing feature-specific validation |
| **Documentation scope** | public guides cover maintained workflows and explicitly bounded support envelopes |

## Current main highlights

- **Public workflow scope (June 2026)**: user guides lead with uniform Yee RF workflows, documented waveguide/MSL/lumped/wire port envelopes, bounded coaxial line reflection, and conservative differentiable design loops.
- **Structured preflight and runtime guards (current `main`)**: `preflight()` and `preflight_sparameters()` return coded `PreflightReport` issues while remaining list/string compatible; `run()`, `forward()`, S-matrix calculators, sweeps, and optimizers surface NaN/passivity/setup problems earlier.
- **Waveguide S-matrix evidence (current `main`)**: rectangular waveguide S-matrices have the strongest current port-family evidence envelope; use the support matrix for exact limits rather than broadening the claim.
- **Coaxial line reflection (current `main`)**: `compute_coaxial_line_reflection(...)` is the bounded one-port coaxial transmission-line reflection path. It is not a general coaxial network solver.
- **Curated star-import surface (current `main`)**: `rfx.__all__` exposes the supported public API; per-step kernels and bookkeeping helpers remain available for compatibility but are outside the curated API surface.
- **Maintained validation lanes (current `main`)**: a full-suite PR gate, API-reference drift checks, a maintained GPU suite harness, weekly external-crossval CI, and on-demand source-built-OpenEMS validation use the honest exit-code convention (reference-missing is a visible SKIP, never a silent pass).

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

> This compact snippet favours brevity over a production-clean mesh, so `run()`
> prints a few non-fatal preflight advisories (thin PEC sheets; substrate/feed
> near the CPML absorber) and the reported resonance is only approximate (this
> coarse mesh lands ~10–15% high). For a
> properly resolved patch — PEC ground, non-uniform z-mesh, lossy FR4, and a
> cross-checked resonance — follow the
> [First Patch tutorial](docs/public/guide/first-patch.mdx).

## Differentiable S-Parameters

`jax.grad` flows end-to-end through the **public S-parameter API** — the
gradient of a measured S-parameter with respect to material design variables
in one call, no adjoint plumbing:

```python
import jax

def objective(eps_plug):
    eps = base_eps.at[plug_region].set(eps_plug)   # design variable
    res = sim.compute_waveguide_s_matrix(normalize=False, eps_override=eps)
    s11 = res.s_params[0, 0, target_freq_idx]
    return jnp.abs(s11) ** 2

grad_fn = jax.grad(objective)        # that's the whole adjoint setup
g = grad_fn(2.0)
```

The runnable version with a finite-difference cross-check is
[`examples/inverse_design/differentiable_s11_design.py`](examples/inverse_design/differentiable_s11_design.py)
(measured: FD↔AD relative error 4.5e-4, ~15 s on CPU). The CI-gated
correctness lock is `tests/test_sparam_ad_end_to_end.py`. Memory-bounded
reverse mode for long runs is available via `checkpoint_segments`.

## Support Contracts

Machine-readable and maintainer-facing support contracts keep public claims aligned with evidence:

- [`docs/guides/support_matrix.md`](docs/guides/support_matrix.md) and [`docs/guides/support_matrix.json`](docs/guides/support_matrix.json) — feature support status and public-scope rules.
- [`docs/guides/sparameter_support_matrix.md`](docs/guides/sparameter_support_matrix.md) and [`docs/guides/sparameter_support_matrix.json`](docs/guides/sparameter_support_matrix.json) — port-family calculators, result schemas, and evidence envelopes.
- [`docs/guides/public_surface_scope_inventory_20260617.md`](docs/guides/public_surface_scope_inventory_20260617.md) — maintainer-only list of public-repo code surfaces that stay outside user docs until support evidence and public workflow docs exist.

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

Benchmarked against Balanis "Antenna Theory" and Pozar "Microwave Engineering":

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
- Material fitting: CSV import, Debye/Lorentz pole fitting
- Differentiable material fitting (jax.grad through FDTD)
- Library: pec, fr4, rogers4003c, copper, alumina, water_20c, ...

### Analysis & Optimization
- S-parameter tooling with per-family calculators: lumped/wire, MSL, waveguide, and coaxial-line workflows use different APIs and evidence envelopes
- Harminv resonance extraction (MPM)
- Far-field, RCS, radiation patterns, polarization
- Antenna metrics: gain, efficiency, HPBW, F/B ratio, bandwidth
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

### For AI coding agents
Working from a fresh clone with an LLM agent? Start with the purpose-built agent docs:
- [Agent overview & operating rules](docs/agent/overview.mdx) — start here; includes a safe prompt skeleton
- [Repo map & feature discovery](docs/agent/repo-map.mdx) — how the code is organized + the grep map for finding existing primitives before writing new code
- [Port selection](docs/agent/port-selection.mdx) — pick the right port family / S-parameter calculator
- Task recipes: [waveguide S-params](docs/agent/recipe-waveguide-sparams.mdx) · [R(f)/T(f) measurement](docs/agent/recipe-rt-measurement.mdx) · [resonance extraction](docs/agent/recipe-resonance-extraction.mdx) · [differentiable design loop](docs/agent/recipe-design-loop.mdx) · [parameter sweeps](docs/agent/recipe-parameter-sweeps.mdx) · [failed-gate triage](docs/agent/recipe-failed-gate-triage.mdx)

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
  version      = {1.6.5},
  url          = {https://github.com/bk-squared/rfx}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://remilab.cnu.ac.kr) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.
