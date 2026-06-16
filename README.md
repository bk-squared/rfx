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

**v1.6.4 package** — JAX-based RF/FDTD workflows, GPU-oriented execution, practical examples, structured setup guards, and port-family validation envelopes.

> **Project status (June 2026):** `rfx` remains an actively validated research/product simulator. Use the uniform Cartesian Yee RF lane first; advanced lanes are promoted only inside explicitly documented evidence envelopes rather than as blanket simulator guarantees.

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## At a Glance

The recommended starting point is the **uniform Cartesian Yee RF/FDTD lane**. Non-uniform mesh, distributed execution, Floquet/Bloch, SBP-SAT subgridding, generalized planar ports, and inverse-design extensions remain lane-scoped; coaxial line reflection and rectangular-waveguide S-matrices now have stronger port-family evidence envelopes documented below.

| | |
|---|---|
| **GPU-accelerated** | 7,309 Mcells/s on RTX 4090, 5,249 on A6000 via `jax.lax.scan` JIT |
| **Differentiable** | `jax.grad` through full time-stepping for inverse design |
| **Topology optimization** | Density-based with filtering, projection, and beta continuation |
| **Conformal PEC** | Dey-Mittra weights — experimental; known NaN at fine mesh (dx <= 2 mm); 2nd-order convergence on curved conductors not validated (staircase is the supported PEC floor) |
| **Multi-GPU** | Single-host multi-GPU distributed FDTD with 1D slab decomposition (experimental lane) |
| **Waveguide modal ports** | Analytical TE/TM eigenmodes for documented rectangular-guide S-matrix envelopes |
| **Coaxial line reflection** | One-port coaxial transmission-line reflection envelope via `compute_coaxial_line_reflection(...)` |
| **Floquet ports** | Phased-array unit-cell analysis with Bloch periodic BC (experimental lane; analytic/external promotion still pending) |
| **Non-uniform x/y/z profiles** | Thin-substrate shadow lane (graded `dz`, per-cell `dx/dy`); forward fields validated — NU **S-matrix extraction of non-trivial devices is an open issue** (strict-xfail tracked) and rejects the differentiable overrides rather than silently dropping them |
| **Published benchmark evidence** | 5-case benchmark against Balanis/Pozar (patch 1.97%, cavity 0.016%) |
| **Regression checks** | CI and local checks support development without replacing feature-specific validation |

## Current main highlights

- **Structured preflight and runtime guards (current `main`)**: `preflight()` and `preflight_sparameters()` return coded `PreflightReport` issues while remaining list/string compatible; `run()`, `forward()`, S-matrix calculators, sweeps, and optimizers now surface NaN/passivity/setup problems earlier.
- **Coaxial line reflection envelope (current `main`)**: `compute_coaxial_line_reflection(...)` is the validated coaxial transmission-line reflection path, with analytic broad-E5 and independent MEEP broad-E4 short/open evidence; the older single-plane `compute_coaxial_s_matrix(...)` path is deprecated/experimental.
- **Waveguide S-matrix memory control (current `main`)**: `compute_waveguide_s_matrix(checkpoint_segments=...)` threads segmented checkpointing through uniform waveguide extractors for AD-heavy runs with bit-identical forward results in regression tests.
- **Periodic × CPML correctness (v1.6.3)**: `set_periodic_axes("xy")` + `boundary="cpml"` only allocates CPML on non-periodic faces. Required for normal-incidence absorber / FSS / RIS setups.
- **Current practical anchors**: `examples/crossval/05_patch_antenna.py` for the patch cross-check and `examples/nonuniform_patch_demo.py` for a quick repo-local thin-substrate demo.
- **Curated star-import surface (current `main`)**: `rfx.__all__` exposes the supported public API (176 names); per-step kernels and internal bookkeeping stay importable for back-compat but are no longer part of the star-import surface.
- **Maintained validation lanes (current `main`)**: a full-suite PR gate (4-shard fast suite + API-reference drift gate + agent-docs hygiene), a maintained GPU suite harness (`scripts/vessl_gpu_suite.yaml`), a weekly external-crossval CI lane with real Meep, and an on-demand source-built-OpenEMS lane (run per release) — all using the honest exit-code convention (reference-missing is a visible SKIP, never a silent pass).
- **Public docs separate lanes**: start with the uniform Yee RF lane; treat advanced features as experimental unless a guide or support matrix gives a narrower claims-bearing envelope.

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

## Differentiable S-Parameters

`jax.grad` flows end-to-end through the **public S-parameter API** — the
gradient of a measured S-parameter with respect to material design variables
in one call, no adjoint plumbing:

```python
import jax

def objective(eps_plug):
    eps = base_eps.at[plug_region].set(eps_plug)   # design variable
    res = sim.compute_waveguide_s_matrix(normalize=True, eps_override=eps)
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

## For AI Agents

rfx ships agent-facing operating docs designed for autonomous use from a
fresh clone:

- [Agent overview & operating rules](docs/agent/overview.mdx) — safe-lane
  defaults and a prompt skeleton
- [Repo map & feature discovery](docs/agent/repo-map.mdx) — grep the API
  surface before writing low-level code
- [Port & source selection](docs/agent/port-selection.mdx) — device class →
  primitive → validated compute path
- Task recipes: [waveguide S-parameters](docs/agent/recipe-waveguide-sparams.mdx),
  [R/T measurement](docs/agent/recipe-rt-measurement.mdx),
  [resonance extraction](docs/agent/recipe-resonance-extraction.mdx),
  [parameter sweeps](docs/agent/recipe-parameter-sweeps.mdx),
  [end-to-end design loop](docs/agent/recipe-design-loop.mdx)
- [Failed-gate triage](docs/agent/recipe-failed-gate-triage.mdx) — decision
  tree for failed gates whose terminal nodes are this project's
  implemented-and-falsified dead ends (do not re-attempt them)
- Machine-readable support contracts:
  [`docs/guides/support_matrix.json`](docs/guides/support_matrix.json),
  [`docs/guides/sparameter_support_matrix.json`](docs/guides/sparameter_support_matrix.json)
  — the S-parameter matrix carries a per-family `ad_traceable` column, locked
  by a contract test that fails when a new compute entry point ships without
  an autodiff classification
- The public API surface is pinned: a symbol/signature inventory gate
  regenerates the pdoc reference in CI and fails on undocumented API drift
- Every `run()`/`forward()`/`optimize()` auto-runs a structured preflight
  (machine-readable codes); S-matrix extractors auto-run a passivity guard.

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

The full cross-validation suite (Meep / OpenEMS / Palace / analytic
references, one reproduce command per case) is tabulated on the
[benchmarks page](docs/public/guide/benchmarks.mdx); the CPU-feasible subset
runs locally via `python scripts/run_crossval_cpu.py` using the repo-wide
exit-code convention (0 = full pass with external reference, 1 = self-check
failure, 2 = reference unavailable — visibly skipped, never silently green).

For practical public examples, start with `examples/crossval/05_patch_antenna.py`
for the patch workflow and `examples/crossval/11_waveguide_port_wr90.py` for
rectangular waveguide ports. Treat non-uniform workflows as shadow unless the
relevant guide says otherwise; treat distributed, Floquet/Bloch, subgridding,
generalized planar ports, and advanced inverse-design workflows as experimental
unless the relevant guide says otherwise. Use the coaxial line reflection
method only inside its documented coax transmission-line envelope. Known open
accuracy items are tracked honestly as GitHub issues rather than hidden
(see the issue tracker; the failed-gate triage recipe routes them).

## Key Features

### Core Simulator
- 3D/2D Yee FDTD with CFS-CPML (kappa_max=1.0 default — measured sweep showed kappa_max>1 degrades guided-mode absorption)
- Conformal PEC via Dey-Mittra (experimental — known NaN at fine mesh; staircased PEC is the supported floor)
- Non-uniform z-mesh for thin substrates (shadow qualification lane)
- Multi-GPU via jax.pmap (experimental distributed lane)
- Mixed precision (float16 fields, 2x memory reduction)
- Auto-configuration from geometry + frequency range

### Sources & Ports
- GaussianPulse, ModulatedGaussian, CW, custom waveforms
- Lumped/wire feed ports and lumped RLC (series/parallel ADE); calibrated
  S-parameter workflows depend on the selected port family
- Rectangular waveguide modal ports (analytical TE/TM eigenmodes, documented
  rectangular-guide workflow)
- Coaxial transmission-line reflection through the documented
  `compute_coaxial_line_reflection(...)` envelope
- Floquet ports with Bloch periodic BC (experimental lane)
- Oblique TFSF (2D TMz + TEz auxiliary grids)

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
- Topology optimization (density filter + projection + beta schedule)
- Parametric sweep + jax.vmap batch evaluation
- Smith chart, de-embedding, Touchstone I/O (.s2p/.s4p/.snp)
- Auto convergence study with Richardson extrapolation

### Geometry & Workflow
- Box, Sphere, Cylinder (CSG), Via, CurvedPatch
- PCB stackup builder (2-layer, 4-layer presets)
- RIS unit cell workflow
- Pre-AMR error indicator + neural surrogate export
- Field animation (GIF/MP4)
- Streamlit web dashboard

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
- `docs/agent/` — public AI-agent pages
- `docs/guides/public_docs_architecture.md` — ownership, sync, and deploy rules
- `docs/guides/public_docs_maintenance.md` — release/docs-sync checklist, stale-doc policy, and support-matrix cadence

### Public docs maintenance workflow

1. Edit the source pages in `docs/public/index.mdx`, `docs/public/guide/`, or `docs/agent/`.
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
- [Non-Uniform Mesh guide](docs/public/guide/nonuniform-mesh.mdx) — practical thin-substrate shadow-lane workflows

### Tutorials
- [Patch Antenna Design](docs/public/guide/tutorial-patch-antenna.mdx) — practical patch workflow from local resonance run to external cross-check
- [Microstrip Filter](docs/public/guide/tutorial-microstrip-filter.mdx) — Coupled-line BPF from Pozar
- [Convergence Study](docs/public/guide/tutorial-convergence.mdx) — Mesh independence methodology

### Guides
- [Migration from Meep/OpenEMS](docs/public/guide/comparison.mdx)
- [Changelog](docs/public/guide/changelog.mdx)
- [Contributing](docs/public/guide/contributing.md)

## Citation

```bibtex
@software{kim_rfx_2026,
  author       = {Byungkwan Kim},
  title        = {rfx: JAX-based differentiable 3D FDTD simulator for RF engineering},
  institution  = {REMI Lab, Chungnam National University},
  year         = {2026},
  version      = {1.6.4},
  url          = {https://github.com/bk-squared/rfx}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://remilab.cnu.ac.kr) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.
