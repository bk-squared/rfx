---
title: "rfx v1.0.x Remaining TODO"
date: 2026-04-04
type: plan
project: rfx
project_code: rfx
tags: [roadmap, v1.0, v1.1, release]
status: preliminary
references: []
figures: []
---

## v1.0.x Patch — Resolved

- [x] `run_until_decay` RLC + wire port support → `d31f364`
- [x] Fresnel test tolerance investigation → probe artifact confirmed, 30% justified → `80efb2e`
- [x] JIT scan body wire port DFT sampling order → moved before source injection → `c15f186`
- [x] `Simulation.auto()` empty geometry guard → 4B cells → 176K → `e39e148`
- [x] Oblique TFSF `ey` polarization → sign bug fixed + oblique ey guard → `8461a8c`
- [x] Oblique TFSF `-x` direction test → `ec09883`
- [x] SBP-SAT penalty coefficient configurable (tau) → `af70400`
- [x] True series RLC current tracking → capacitor charge ADE → `e2e445a`

## Remaining — Partially Resolved or Discovered During Patches

### Oblique TFSF ey + oblique angle (NEW)
- **Status**: NotImplementedError guard added (T5)
- **Issue**: 2D TMz aux grid only handles ez oblique. ey oblique needs TEz aux grid (xz-plane)
- **Fix**: Implement TEz 2D auxiliary grid or generalized 2D aux grid supporting both polarizations
- **Priority**: 중 — blocks full oblique polarization support

### SBP-SAT alpha cap (DISCOVERED in T7)
- **Status**: tau parameter plumbed through, but `min(alpha, 0.5)` cap makes all tau values identical at standard CFL
- **Issue**: `cb_vac * 2 * tau / dx` always exceeds 0.5 cap → tau has no effect
- **Fix**: Revisit Cheng et al. 2025 derivation, consider relaxing cap or scaling differently
- **Priority**: 낮음 — stability is fine, just no tuning knob yet

### Fresnel probe accuracy (UNDERSTOOD in T2)
- **Status**: Root cause documented — single-point phase sampling artifact
- **Issue**: 30% tolerance needed because scattered/incident probes at different x-positions have oblique phase mismatch
- **Fix**: DFT plane probe with oblique phase de-rotation (k-vector aware normalization)
- **Priority**: 낮음 — physics is correct (2.3% at best alignment), only measurement is imprecise

### PyPI publish
- **Status**: Resolved — published on PyPI as `rfx-fdtd` 1.0.0
- **Install**: `pip install rfx-fdtd`
- **Import**: `import rfx`
- **Follow-up**: Keep docs explicit about package name vs import name

## v1.1 (Mid-term)

- [ ] Multi-mode waveguide ports (MPB eigenmode solver integration)
- [ ] Dispersive materials on non-uniform mesh (currently ValueError)
- [ ] SBP-SAT JIT overhead benchmark (uniform vs subgridded characterization)
- [ ] Cross-validation 9-case quantification (coupled filter, via etc. currently qualitative)
- [ ] Sphinx/pdoc API reference auto-generation
- [ ] GPU multi-device via jax.pmap domain decomposition
- [ ] TEz 2D auxiliary grid for ey oblique TFSF (from T5 finding)
- [ ] DFT plane Fresnel probe with phase de-rotation (from T2 finding)
- [ ] SBP-SAT alpha cap investigation (from T7 finding)

## v2+ (Long-term)

- [ ] General-purpose FDTD beyond RF/microwave (photonics, THz) — aggressive, scope carefully
- [ ] Cloud/distributed execution
- [ ] Web-based simulation interface
- [ ] MPB eigenmode solver native integration
