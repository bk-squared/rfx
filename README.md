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

> **Project status:** active RF/FDTD simulator. Start with the uniform Cartesian
> Yee solver. For other features, follow the geometry, mesh, frequency, and
> calculator limits in the [support matrix](docs/guides/support_matrix.md).

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)
[![Docs](https://img.shields.io/badge/docs-remilab.ai%2Frfx-blue)](https://remilab.ai/rfx/)

## At a Glance

Start with the **uniform Cartesian Yee RF/FDTD solver**. The table below
summarizes the documented capabilities and their limits.

| | |
|---|---|
| **GPU-accelerated** | 7,309 Mcells/s on RTX 4090, 5,249 on A6000 via `jax.lax.scan` JIT |
| **Differentiable** | `jax.grad` through the JAX-traced time-domain solver for sensitivity and inverse design |
| **RF workflow tools** | materials, sources, probes, ports, S-parameter helpers, Harminv, far-field / RCS |
| **Waveguide modal ports** | analytic TE/TM eigenmodes for rectangular-guide S-matrices |
| **Per-family S-parameters** | lumped/wire, microstrip-line, rectangular waveguide, and coaxial-line paths use distinct calculators |
| **Cross-validated** | public cases mapped to Meep / OpenEMS / Palace / analytic references, each with a reproduce command |

## Supported Workflows and Runtime Checks

- **Documented solvers and ports**: uniform Yee RF simulation;
  rectangular-waveguide, microstrip-line (MSL), lumped, and wire ports;
  one-port coaxial-line reflection; and conservative differentiable design
  loops.
- **Preflight and result checks**: `preflight()` and
  `preflight_sparameters()` return a coded `PreflightReport` (a `list`
  subclass, so existing list/string checks still work). Normal, non-traced runs
  can warn about non-finite output, no recorded field energy, an incompletely
  settled probe signal, S-parameter passivity violations, and low-signal MSL
  wave extraction. These checks do not alter returned arrays and do not replace
  convergence or cross-solver validation.
- **Versioned experiments**: Studio, the experiment CLI, and MCP use the same
  `ExperimentSpec` format. CPU runs execute in subprocesses and can be
  inspected, bundled, and replayed; succeeded S-parameter runs can also be
  compared. Studio and MCP store append-only revisions. The interfaces share
  records only when they use the
  same `--workspace` path.
- **Waveguide S-matrices** are the most mature port family; consult the [S-parameter support matrix](docs/guides/sparameter_support_matrix.md) and do not apply its results outside the documented modes, mesh, frequency, geometry, and reference-plane limits.
- **Coaxial-line reflection**: `compute_coaxial_line_reflection(...)` requires
  float32 precision, a nonperiodic 3D second-order uniform Yee grid, CPML on
  all six boundary faces with positive thickness on both z faces,
  `cpml_axes="z"`, and exactly one `face="top"` coaxial port. The method builds
  its own line, source, monitors, and termination; it rejects separately
  registered simulation objects rather than ignoring them. See the
  [S-parameter support matrix](docs/guides/sparameter_support_matrix.md#coaxial-port)
  for the accepted arguments and result limits.
- **Public imports**: `rfx.__all__` defines the supported star-import API.
  Per-step kernels and bookkeeping helpers remain importable for compatibility
  but are not part of that API.
- **Validation checks**: PR tests, API-reference drift checks, GPU tests,
  weekly external comparisons, and on-demand source-built OpenEMS checks report
  a missing reference as `SKIP`, not as a pass.

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

From a source checkout, the experiment CLI reads a strict, versioned JSON spec,
generates the rfx configuration and Python, runs preflight and simulation in a CPU-only
subprocess, and saves the run inputs and analysis artifacts:

```bash
rfx experiment validate tests/fixtures/experiments/patch_antenna_cpu_v1.json
rfx experiment run tests/fixtures/experiments/patch_antenna_cpu_v1.json
```

The repository fixture is not installed with the wheel. It runs a short setup
check, not a calibrated RF calculation. Background runs also support `submit`,
`status`, and `cancel`; their
default local state directory is `.rfx-experiments/`.

### rfx Studio

Install the Studio extra and launch the packaged local application:

```bash
pip install "rfx-fdtd[studio]"
rfx studio --workspace .rfx-studio
```

Studio stores append-only experiment revisions, generated Python, geometry,
preflight reports, CPU runs, comparisons, approvals, and replay bundles. The
current UI directly creates and plots the bundled patch-antenna workflow. Other
experiment types are available through the CLI, API, or MCP; studies are
available through the API or MCP. The Studio guide states which results the UI
can display. Before an MCP action
changes data or starts costly work, Studio shows the exact arguments and
requires approval, then records the decision in an append-only log. The default
bind is loopback-only. Local MCP clients can use the same operations over stdio
with `rfx-agent-mcp --workspace .rfx-studio`.

The built-in Design Copilot turns a natural-language RF request into an
`ExperimentSpec` JSON Patch restricted by the spec schema. Before an experiment
and revision are created in the configured workspace, the proposal panel shows
the field changes, a pass/block summary from an in-memory solver preflight, and
the CPU estimate. After revision 1 exists, the main editor shows generated
Python and geometry; unsaved edits use a preview-only compiler check, while
saving or explicit validation runs the full solver preflight. Set an OpenAI API
key to enable LLM proposals; without one, Studio visibly uses a deterministic
rule-based planner:

```bash
export OPENAI_API_KEY=...              # never stored in the workspace
# Optional: set RFX_OPENAI_MODEL to a model ID available to your account.
rfx studio --workspace .rfx-studio
```

Provider responses use `store=false`. The model cannot execute Python, shell,
or solver tools through this feature, and a proposal never creates a revision
or starts a run. Generated Python remains read-only; edit the synchronized
`ExperimentSpec` JSON instead.
The [Studio, CLI, and MCP Experiments](docs/public/guide/studio-experiments.mdx)
guide covers the UI, CLI, MCP approval flow, workspace backup/restore, and
authenticated remote deployment.

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

This example records a dielectric pulse response; it does not claim a resonant
frequency or Q. Resolve any preflight warnings introduced by model changes. For
a patch antenna, follow the [First Patch
tutorial](docs/public/guide/first-patch.mdx) to build a model that passes
preflight, then perform the mesh, time-window, and reference checks it requires
before reporting RF results.

## Differentiable Design Workflows

rfx supports selected JAX-traced objectives for inverse design. Treat these as
**sensitivity calculations through the implemented discrete solver**, not as
standalone RF validation. `forward(...)` objectives can be wrapped in an outer
`jax.jit`, `optimize(...)` exposes default-off multi-start / best-iterate /
step-clamp safeguards, and uniform single-device `forward(...)` can differentiate
registered lumped R/L/C values through `rlc_values_override`. For
S-parameter-driven work, use the calculator that matches the port family and
apply that port family's stated validation limits to the final result.

The runnable gradient examples include:

- [`examples/inverse_design/differentiable_s11_design.py`](examples/inverse_design/differentiable_s11_design.py) — differentiable S11 objective with a finite-difference cross-check (measured: FD↔AD relative error 4.5e-4, ~15 s on CPU); CI-gated by `tests/test_sparam_ad_end_to_end.py`.
- [`examples/inverse_design/multilayer_ar_coating.py`](examples/inverse_design/multilayer_ar_coating.py) — conservative differentiable material-profile demo.

For the conceptual background, see the public [Autodiff and Adjoint Background](docs/public/guide/autodiff-adjoint.mdx) guide.

## Support and Validation Files

These files keep public statements aligned with the available evidence:

- [`docs/guides/support_matrix.md`](docs/guides/support_matrix.md) and [`docs/guides/support_matrix.json`](docs/guides/support_matrix.json) — feature support status and public-scope rules.
- [`docs/guides/sparameter_support_matrix.md`](docs/guides/sparameter_support_matrix.md) and [`docs/guides/sparameter_support_matrix.json`](docs/guides/sparameter_support_matrix.json) — port-family calculators, result schemas, and validated limits.

The supported public API is checked against a symbol/signature inventory and
the curated public docs.

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

Each cross-validation case is mapped to a named analytic or external
reference. The values below are **acceptance thresholds**, not measured-error
summaries. Some scripts print diagnostic tables; for example, the regression
checks for `11_waveguide_port_wr90` are defined in `tests/`.

| Case (`validation/crossval/`) | Upstream reference | Acceptance thresholds from the script |
|---|---|---|
| `05_patch_antenna` | OpenEMS patch tutorial + Balanis TL analytic | rfx vs OpenEMS Harminv < 20%; vs Balanis TL < 10%; self-consistency < 5%; numerical passivity advisory \|S11\| < 1.05 |
| `04_multilayer_fresnel` | Analytic transfer-matrix / Fresnel (Taflove Ch 5) | `T`, `R` mean error and `R+T` energy deviation each < 0.05 |
| `02_ring_resonator` | Meep "Basics" tutorial + Harminv | mean mode-frequency error < 5%; ≥ 2 modes matched |
| `09_half_symmetric_waveguide` | Analytic rectangular cavity TE₁₀₁ (Pozar Ch 6) | `f` within 10% of analytic; PMC-half vs full PEC < 5% |
| `11_waveguide_port_wr90` | Analytic matched guide, PEC short, and Airy slab; optional Meep/OpenEMS/Palace tables | script gates band-mean linear-magnitude differences: empty S11 < 0.02 and S21 < 0.03; PEC-short magnitude < 0.05 plus band-mean round-trip phase < 15°; slab S11/S21 < 0.10/0.07 with the documented phase and complex-S gates |
| `06b_msl_notch_filter_uniform` | Analytic quarter-wave stub notch | notch-frequency error < 15%; depth < −10 dB; Z₀ median ∈ (40, 65) Ω |

For each metric and reproduce command, see [Benchmarks](docs/public/guide/benchmarks.mdx). Read [Cross-Validation and Accuracy](docs/public/guide/validation.mdx) first for the applicable support limits.

The CPU-feasible subset runs locally via `PYTHONPATH=. python scripts/run_crossval_cpu.py`, using the repo-wide exit-code convention: `0` = every configured gate passed, including references marked `required_for_script_pass`; `1` = a self-check or numeric gate failed; `2` = a required external reference such as Meep or OpenEMS was unavailable, so the comparison is inconclusive rather than silently green. Optional reference files do not determine the exit code unless the manifest marks them as required.

To learn rfx, start with the ordered tutorials in `examples/tutorials/` (see `examples/README.md` for the learning path). The cross-validation scripts referenced above are measurement fixtures we maintain to verify accuracy — read them for evidence, not as lessons. Treat scripts outside the tutorial and validated sets as local diagnostics unless a public guide and support-matrix entry list them.

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
- Coaxial transmission-line reflection through
  `compute_coaxial_line_reflection(...)` within the exact setup and argument
  limits in the
  [S-parameter support matrix](docs/guides/sparameter_support_matrix.md#coaxial-port)

### Materials
- Debye/Lorentz/Drude dispersive, Kerr nonlinear (chi3)
- Subpixel smoothing, thin conductor correction
- Material fitting: `fit_debye` / `fit_lorentz` (Debye/Lorentz pole fitting from data); `differentiable_material_fit` (jax.grad-traced fitting through FDTD)
- Library: pec, fr4, rogers4003c, copper, alumina, water_20c, ...

### Analysis & Optimization
- Per-family S-parameter extraction (lumped/wire, MSL, waveguide, coaxial-line — see Sources & Ports above)
- Harminv resonance extraction (MPM)
- Documented far-field, RCS, radiation-pattern, and linear-polarization workflows
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

Repository support files:

- `docs/guides/support_matrix.md`
- `docs/guides/reference_lane_contract.md`

Public-documentation sources in this repo:

- `docs/public/index.mdx` — public `/rfx/` landing page
- `docs/public/guide/` — public guide pages
- `docs/public/examples/` — runnable example hubs
- `docs/public/validation/` — quantitative evidence and support-status pages
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
- [Studio, CLI, and MCP Experiments](docs/public/guide/studio-experiments.mdx) — local UI, versioned CPU runs, replay, and MCP approvals
- [Validation hub](docs/public/validation/index.mdx) — support overview and current limits
- [Examples hub](docs/public/examples/index.mdx) — current runnable entry points

### For AI coding agents
Working from a fresh clone with an LLM agent? Start with the purpose-built agent docs:
- [Agent overview & operating rules](docs/agent/overview.mdx) — start here; includes a safe prompt skeleton
- [Repo map & feature discovery](docs/agent/repo-map.mdx) — how the code is organized + the grep map for finding existing primitives before writing new code
- [Port selection](docs/agent/port-selection.mdx) — pick the right port family / S-parameter calculator
- Task recipes: [waveguide S-params](docs/agent/recipe-waveguide-sparams.mdx) · [R(f)/T(f) measurement](docs/agent/recipe-rt-measurement.mdx) · [resonance extraction](docs/agent/recipe-resonance-extraction.mdx) · [differentiable design loop](docs/agent/recipe-design-loop.mdx) · [parameter sweeps](docs/agent/recipe-parameter-sweeps.mdx) · [failed-gate triage](docs/agent/recipe-failed-gate-triage.mdx)

### Tutorials
- [Patch Antenna Design](docs/public/guide/tutorial-patch-antenna.mdx) — patch setup and validation workflow, including the separate external cross-check
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
