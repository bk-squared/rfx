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

**v1.6.6** — GPU-scaled `lax.scan` time-stepping, per-port-family S-parameter extractors, and structured preflight guards that surface setup errors before a run.

> **Project status:** active RF/FDTD simulator. Start with the uniform Cartesian Yee lane; other workflows are supported only within the evidence envelopes documented in the [support matrix](docs/guides/support_matrix.md).

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## At a Glance

Start with the **uniform Cartesian Yee RF/FDTD lane**; the table below scopes each capability to its supported claim.

| | |
|---|---|
| **GPU-accelerated** | 7,309 Mcells/s on RTX 4090, 5,249 on A6000 via `jax.lax.scan` JIT |
| **Differentiable** | `jax.grad` through the JAX-traced time-domain solver for sensitivity and inverse design |
| **RF workflow tools** | materials, sources, probes, ports, S-parameter helpers, Harminv, far-field / RCS |
| **Waveguide modal ports** | analytic TE/TM eigenmodes for rectangular-guide S-matrices |
| **Per-family S-parameters** | lumped/wire, microstrip-line, rectangular waveguide, and coaxial-line paths use distinct calculators |
| **Cross-validated** | public cases mapped to Meep / OpenEMS / Palace / analytic references, each with a reproduce command |

## Current Highlights

- **Documented workflows**: uniform Yee RF simulation; rectangular-waveguide, MSL, lumped, and wire ports; one-port coaxial-line reflection; and conservative differentiable design loops.
- **Structured preflight and runtime guards**: `preflight()` and `preflight_sparameters()` return a coded `PreflightReport` (a `list` subclass, so existing list/string checks still work); `run()`, `forward()`, the S-matrix calculators, sweeps, and optimizers surface NaN / passivity / setup problems early.
- **Waveguide S-matrices** are the most mature port family; consult the [S-parameter support matrix](docs/guides/sparameter_support_matrix.md) for exact limits rather than broadening the claim.
- **Coaxial-line reflection**: `compute_coaxial_line_reflection(...)` is the one-port coaxial transmission-line reflection path — not a general coaxial network solver.
- **Curated star-import surface**: `rfx.__all__` is the supported public API; per-step kernels and bookkeeping helpers stay importable for back-compatibility but sit outside it.
- **Maintained validation lanes**: a full-suite PR gate, API-reference drift checks, a GPU-suite harness, weekly external-crossval CI, and on-demand source-built-OpenEMS validation, all under the honest exit-code convention (a missing reference is a visible SKIP, never a silent pass).

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

### Interactive dashboard (GUI)

rfx ships an experimental Streamlit dashboard — a browser GUI for assembling a
simulation, running it, and inspecting results (S-parameter plots, Smith chart,
field slices, Touchstone export) without writing Python. Install the optional
dependency and launch:

```bash
pip install "rfx-fdtd[dashboard]"
rfx-dashboard
```

### Versioned CPU experiments

The first AI-native execution lane accepts a strict JSON patch-antenna spec,
compiles it to native rfx configuration/Python, runs preflight and simulation
in a CPU-only subprocess, and stores S11 in an immutable content-addressed
artifact:

```bash
rfx experiment validate tests/fixtures/experiments/patch_antenna_cpu_v1.json
rfx experiment run tests/fixtures/experiments/patch_antenna_cpu_v1.json
```

The bundled fixture is a fast structural smoke test, not a calibrated RF
reference. Durable asynchronous runs also support `submit`, `status`, and
`cancel`; their default local state directory is `.rfx-experiments/`.

### AI-native rfx Studio

Install the Studio extra and launch the packaged local application:

```bash
pip install "rfx-fdtd[studio]"
rfx studio --workspace .rfx-studio
```

Studio keeps immutable experiment revisions, generated Python, geometry,
preflight, durable CPU runs, S-parameter/Smith analysis, comparisons, bounded
studies, and replay bundles in one workflow. The **Setup** view organizes the
canonical model as domain/mesh, material assignments, ports, boundaries,
observations, and the CPU solve contract. Successful runs add an **RF evidence
summary** with sampled S11/VSWR/bandwidth, sweep coverage, provenance, and an
explicit list of engineering evidence that was not captured. Its MCP endpoint
uses the same application service; every mutation or costly action requires an
exact-argument human approval and is append-only audited. The default bind is loopback-only.
Local MCP hosts can launch the same contract over stdio with
`rfx-agent-mcp --workspace .rfx-studio`.

The built-in Design Copilot turns a natural-language RF intent into a bounded
`ExperimentSpec` JSON Patch, compiles it without saving, and shows the semantic
diff, deterministic Python, preflight, and CPU estimate before approval. Set an
OpenAI API key to enable LLM proposals; without one, Studio visibly uses its
narrow deterministic offline planner:

```bash
export OPENAI_API_KEY=...              # never stored in the workspace
export RFX_OPENAI_MODEL=gpt-5.5        # optional; choose an entitled model
rfx studio --workspace .rfx-studio
```

Provider responses use `store=false`. The model cannot execute Python, shell,
or solver tools from this surface, and a proposal never creates a revision or
starts a run. Generated Python remains read-only because the canonical editable
source is the spec; direct users edit the synchronized ExperimentSpec JSON.
Authenticated lab-server deployment, migration/backup/restore, and TLS/Origin
requirements are documented in
[`docs/guides/studio_operations.md`](docs/guides/studio_operations.md); MCP client
setup is in [`docs/agent/mcp-studio.mdx`](docs/agent/mcp-studio.mdx).

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
> may print non-fatal preflight advisories and the reported resonance is only
> approximate (coarse mesh lands ~10–15% high). For a properly resolved patch
> workflow, follow the [First Patch tutorial](docs/public/guide/first-patch.mdx).

## Differentiable Design Workflows

rfx supports selected JAX-traced objectives for inverse design. Treat these as
**sensitivity calculations through the implemented discrete solver**, not as
standalone RF validation. `forward(...)` objectives can be wrapped in an outer
`jax.jit`, `optimize(...)` exposes default-off multi-start / best-iterate /
step-clamp safeguards, and uniform single-device `forward(...)` can differentiate
registered lumped R/L/C values through `rlc_values_override`. For
S-parameter-driven work, use the calculator that matches the port family and
keep the final claim inside that family's evidence envelope.

The runnable gradient examples include:

- [`examples/inverse_design/differentiable_s11_design.py`](examples/inverse_design/differentiable_s11_design.py) — differentiable S11 objective with a finite-difference cross-check (measured: FD↔AD relative error 4.5e-4, ~15 s on CPU); CI-gated by `tests/test_sparam_ad_end_to_end.py`.
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

Each A/B cross-validation case is mapped to a named analytic or external reference. The values below are the **accept envelopes** enforced by each script's own gate code — not single-point marketing errors. Several cases (e.g. `11_waveguide_port_wr90`) are diagnostic reporters whose authoritative gates live in `tests/`.

| Case (`examples/crossval/`) | Upstream reference | Accept envelope (from the script's gate) |
|---|---|---|
| `05_patch_antenna` | OpenEMS patch tutorial + Balanis TL analytic | rfx vs OpenEMS Harminv < 20%; vs Balanis TL < 10%; self-consistency < 5%; \|S11\| ≤ 1 |
| `04_multilayer_fresnel` | Analytic transfer-matrix / Fresnel (Taflove Ch 5) | `T`, `R` mean error and `R+T` energy deviation each < 0.05 |
| `02_ring_resonator` | Meep "Basics" tutorial + Harminv | mean mode-frequency error < 5%; ≥ 2 modes matched |
| `09_half_symmetric_waveguide` | Analytic rectangular cavity TE₁₀₁ (Pozar Ch 6) | `f` within 10% of analytic; PMC-half vs full PEC < 5% |
| `11_waveguide_port_wr90` | Analytic Airy + Meep/Palace JSON when present | empty-guide max\|S11\| < 0.02, min\|S21\| > 0.97; PEC-short mean \|S11\| ∈ [0.97, 1.03] |
| `06b_msl_notch_filter_uniform` | Analytic quarter-wave stub notch | notch-frequency error < 15%; depth < −10 dB; Z₀ median ∈ (40, 65) Ω |

For the per-case metric, reproduce command, and last-touched commit, see the [benchmark table](docs/public/guide/benchmarks.mdx); for lane guidance and support boundaries, read [Cross-Validation and Accuracy](docs/public/guide/validation.mdx) first.

The CPU-feasible subset runs locally via `PYTHONPATH=. python scripts/run_crossval_cpu.py`, using the repo-wide exit-code convention: `0` = full pass including the external cross-check, `1` = a self-check / numeric gate failed, `2` = the external reference (e.g. Meep / OpenEMS) was unavailable, so the case is visibly skipped — never silently green.

For a first read, start with `examples/crossval/05_patch_antenna.py` (patch workflow) and `examples/crossval/11_waveguide_port_wr90.py` (rectangular waveguide ports). Treat scripts outside this validated set as local diagnostics unless a public guide and support-matrix entry list them.

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
- Material fitting: `fit_debye` / `fit_lorentz` (Debye/Lorentz pole fitting from data); `differentiable_material_fit` (jax.grad-traced fitting through FDTD)
- Library: pec, fr4, rogers4003c, copper, alumina, water_20c, ...

### Analysis & Optimization
- Per-family S-parameter extraction (lumped/wire, MSL, waveguide, coaxial-line — see Sources & Ports above)
- Harminv resonance extraction (MPM)
- Documented far-field, RCS, radiation-pattern, and polarization workflows
- Antenna metrics: `antenna_gain`, `antenna_efficiency`, `half_power_beamwidth`, `front_to_back_ratio`, `antenna_bandwidth` (rfx/antenna.py)
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
  version      = {1.6.6},
  url          = {https://github.com/bk-squared/rfx}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://remilab.cnu.ac.kr) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.
