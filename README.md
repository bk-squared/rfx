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

**v1.5.0** — 500+ tests, GPU-benchmarked (7,309 Mcells/s on RTX 4090), published RF benchmarks, and expanded preflight/crossval tooling.

> **Project status (April 2026):** `rfx` is still in active validation and early conceptualization. Treat the current `main` branch as an initial-stage release rather than a finalized, fully qualified simulator; the support surface, validation evidence, and higher-level workflows will continue to be tightened and expanded in upcoming iterations.

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/test.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## At a Glance

Current correctness-bearing support is centered on the **uniform Cartesian Yee
reference lane**. Non-uniform graded-z, distributed execution, SBP-SAT
subgridding, and Floquet/Bloch workflows should be treated according to
`docs/guides/support_matrix.md`, not as blanket guarantees.

| | |
|---|---|
| **GPU-accelerated** | 7,309 Mcells/s on RTX 4090, 5,249 on A6000 via `jax.lax.scan` JIT |
| **Differentiable** | `jax.grad` through full time-stepping for inverse design |
| **Topology optimization** | Density-based with filtering, projection, and beta continuation |
| **Conformal PEC** | Dey-Mittra method for 2nd-order accuracy on curved conductors |
| **Multi-GPU** | Single-host multi-GPU distributed FDTD with 1D slab decomposition (experimental lane) |
| **SBP-SAT subgridding** | JIT-compiled local mesh refinement (experimental in 3D) |
| **Multi-mode ports** | Analytical TE/TM eigenmodes for waveguide S-matrix |
| **Floquet ports** | Phased-array unit-cell analysis with Bloch periodic BC (experimental lane) |
| **Accuracy validated** | 5-case benchmark against Balanis/Pozar (patch 1.97%, cavity 0.016%) |
| **500+ tests** | CI with ruff lint + pytest, 0 xfails |

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

Cross-validation vs Meep/OpenEMS: 0.000-0.007% on cavities and waveguides.

## Key Features

### Core Simulator
- 3D/2D Yee FDTD with CFS-CPML (kappa=5.0, < -40 dB reflectivity)
- Conformal PEC via Dey-Mittra (2nd-order on curved surfaces)
- SBP-SAT subgridding with JIT (experimental in 3D, stable in 1D/2D)
- Non-uniform z-mesh for thin substrates (shadow qualification lane)
- Multi-GPU via jax.pmap (experimental distributed lane)
- Mixed precision (float16 fields, 2x memory reduction)
- Auto-configuration from geometry + frequency range

### Sources & Ports
- GaussianPulse, ModulatedGaussian, CW, custom waveforms
- Lumped/wire ports, lumped RLC (series/parallel ADE)
- Multi-mode waveguide ports (analytical TE/TM eigenmodes)
- Floquet ports with Bloch periodic BC (experimental lane)
- Oblique TFSF (2D TMz + TEz auxiliary grids)

### Materials
- Debye/Lorentz/Drude dispersive, Kerr nonlinear (chi3)
- Subpixel smoothing, thin conductor correction
- Material fitting: CSV import, Debye/Lorentz pole fitting
- Differentiable material fitting (jax.grad through FDTD)
- Library: pec, fr4, rogers4003c, copper, alumina, water_20c, ...

### Analysis & Optimization
- S-parameters (lumped, wire, waveguide N-port)
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
- `docs/agent/` — public AI-agent pages
- `docs/guides/public_docs_architecture.md` — ownership, sync, and deploy rules

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

### Tutorials
- [Patch Antenna Design](docs/public/guide/tutorial-patch-antenna.mdx) — Complete 2.4 GHz design flow
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
  version      = {1.5.0},
  url          = {https://github.com/bk-squared/rfx}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://remilab.cnu.ac.kr) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.

### AI-Assisted Development

This project was developed with AI coding assistants orchestrated via [oh-my-claudecode](https://github.com/yeachan-heo/oh-my-claudecode):

- **Claude** (Anthropic) — Architecture design, physics validation, cross-validation, code review, documentation
- **Codex** (OpenAI) — Feature implementation, test generation, review, debugging
