# SBP-SAT z-slab Phase 1 crosswalk

Status: handoff support doc  
Date: 2026-04-16  
Branch: `plan/sbp-sat-zslab-ralplan`

## Purpose

> **Current planning note (2026-04-23):** this crosswalk is historical input.
> Use `docs/guides/sbp_sat_zslab_phase1_full_spec.md` for the current
> implementation-aware roadmap and next-phase execution plan.

This document translates the approved Phase 1 plan into a **paper-to-code / brownfield-to-target crosswalk** for future implementation.

It exists to keep three things aligned:

1. the shared canonical rewrite spec
2. the current `rfx` brownfield code
3. the future **z-slab-only, single-stepper, norm-compatible** implementation lane

## Non-negotiable invariants

Future implementation must preserve all three:

1. **z-slab only**
   - full-span x/y
   - local z-range refinement only
   - interface faces only: `z_lo`, `z_hi`

2. **single canonical stepper**
   - one authoritative Phase 1 runtime path
   - no overlay / injection fallback
   - no parallel legacy “almost-canonical” path

3. **norm-compatible face ops**
   - explicit face norms
   - prolongation / restriction defined from those norms
   - no naked `mean()` / `repeat()` as the source of truth

## Crosswalk table

| Concept | Current brownfield location | Current state | Phase 1 target | Action |
|---|---|---|---|---|
| External API entrypoint | `rfx/api.py::Simulation.add_refinement(...)` | Already z-oriented, but still allows broader interpretation and PML-overlap warning behavior | Keep this entrypoint, but narrow semantics to z-slab-only and hard-fail unsupported configs | **Preserve + narrow** |
| Runtime orchestration path | `rfx/runners/subgridded.py`, `rfx/subgridding/jit_runner.py`, `rfx/subgridding/runner.py` | Multiple paths / broader-box assumptions still visible | One canonical Phase 1 path calling one canonical z-slab stepper | **Converge** |
| 3D solver core | `rfx/subgridding/sbp_sat_3d.py` | 6-face logic, mean/repeat transfer shortcuts, broader experimental surface | New Phase 1 z-slab solver module with z-face-only coupling | **Replace** |
| 1D SBP-SAT reference lane | `rfx/subgridding/sbp_sat_1d.py` | Useful for energy/interface intuition, but different temporal/interface formulation | Use only as a conceptual reference; do not force 1D structure onto 3D Phase 1 | **Selective reuse** |
| Face extraction/scatter | currently implicit via raw array slices inside `sbp_sat_3d.py` | Slice-based coupling with no explicit face-DOF abstraction | Dedicated extractor/scatter helpers for z-face tangential `E/H` traces | **Introduce** |
| Transfer operators | `_downsample_2d`, `_upsample_2d` in `sbp_sat_3d.py` | Mean/repeat based shortcuts | Face-operator builders rooted in explicit norms | **Replace** |
| SAT penalty helper | embedded alpha logic in `sbp_sat_3d.py`; historic alpha drift also reflected in tests | Improved from older cap bug, but still too embedded in the legacy file | One explicit coefficient helper used by implementation + tests | **Centralize** |
| Energy accounting | `compute_energy_3d` in `sbp_sat_3d.py` | Correct intent: do not double-count overlap | Preserve overlap-exclusion principle in the canonical z-slab lane | **Reuse principle, re-home implementation** |
| Misuse policy for PML/CPML | `rfx/api.py::add_refinement(...)` | warning only | hard-fail | **Tighten** |
| Test strategy | `tests/test_sbp_sat_3d.py`, `tests/test_sbp_sat_alpha.py`, `tests/test_sbp_sat_jit.py` | smoke/JIT/regression heavy | operator + misuse + smoke + benchmark ladder | **Replace/expand** |

## Proposed target file structure

The shared canonical spec strongly suggests separating face operators, solver core, and energy accounting.

Recommended target structure:

```text
rfx/subgridding/
  face_ops.py
  sbp_sat_zslab.py
  energy.py
```

Recommended semantics:

- `face_ops.py`
  - explicit z-face norm builders
  - prolongation / restriction builders
  - norm-compatibility checks

- `sbp_sat_zslab.py`
  - z-slab config/state
  - tangential face extract/scatter helpers
  - SAT-H / SAT-E application
  - one canonical step function

- `energy.py`
  - overlap-safe total energy accounting for the z-slab lane

This does **not** force final filenames, but anything equivalent must still preserve the three invariants above.

## Symbol-level mapping

### API layer

| Current | Phase 1 interpretation | Notes |
|---|---|---|
| `Simulation.add_refinement(z_range, ratio, xy_margin, tau)` | keep method name; treat as z-slab-only Phase 1 entrypoint | `xy_margin` must stop implying general box support |

### Brownfield 3D solver helpers

| Current helper / pattern | Problem | Phase 1 replacement |
|---|---|---|
| `_downsample_2d(...)` | mean-based restriction as source of truth | restriction derived from face norms |
| `_upsample_2d(...)` | repeat-based prolongation as source of truth | prolongation derived from face norms |
| `_shared_node_coupling_3d(...)` | 6-face `E` coupling | z-face-only `E` SAT coupling helper |
| `_shared_node_coupling_h_3d(...)` | 6-face `H` coupling | z-face-only `H` SAT coupling helper |
| raw `c_slice` / `f_slice` trace coupling | no explicit trace-DOF abstraction | `extract_tangential_*_face`, `scatter_tangential_*_face` |
| `step_subgrid_3d(...)` | broader-than-Phase-1 stepper | one canonical `step_subgrid_zslab(...)` |

### Tests

| Current test surface | Keep? | Phase 1 intent |
|---|---|---|
| `tests/test_sbp_sat_3d.py` | partially | migrate smoke intent, but narrow scope and rename to Phase 1 semantics |
| `tests/test_sbp_sat_alpha.py` | partially | keep source-of-truth alpha/tau intent, remove stale legacy assumptions |
| `tests/test_sbp_sat_jit.py` | likely reshape | JIT/runtime coverage must not reintroduce forbidden CPML/general-box assumptions |

## Brownfield classification

### Reuse directly

- overlap-safe energy-accounting principle
- global-`dt` framing
- existing z-oriented public API entrypoint concept

### Reuse only as concept, not code

- 1D interface-energy reasoning from `sbp_sat_1d.py`
- existing alpha/tau regression intent
- runtime scan/JIT ideas

### Replace outright

- 6-face coupling logic
- mean/repeat transfer shortcuts
- warning-based PML-overlap behavior
- tests that encode forbidden Phase 1 behavior as supported

## Notation crosswalk skeleton

This section is intentionally simple and should be expanded before implementation starts.

| Math / spec term | Intended code concept | Expected owner |
|---|---|---|
| `H_c_face`, `H_f_face` | face-norm builders | `face_ops.py` |
| `P` | cell-centered linear coarse-to-fine face prolongation | `face_ops.py` |
| `R` | fine-to-coarse face restriction derived from norms (`P^T / ratio`) | `face_ops.py` |
| tangential `E` trace on `z_lo`, `z_hi` | extracted `(Ex, Ey)` z-face DOFs | `sbp_sat_zslab.py` |
| tangential `H` trace on `z_lo`, `z_hi` | extracted `(Hx, Hy)` z-face DOFs | `sbp_sat_zslab.py` |
| SAT penalty coefficients | one coefficient helper | `sbp_sat_zslab.py` or tightly paired helper |
| total energy | overlap-safe energy diagnostic | `energy.py` |

## Implementation-order guardrail

Future execution should follow this order:

1. face-op definitions
2. norm-compatibility tests
3. trace extract/scatter helpers
4. SAT-H coupling
5. SAT-E coupling
6. single canonical z-slab stepper
7. energy accounting
8. misuse tests
9. smoke test
10. benchmark tests
11. legacy-path fencing/removal

## Explicit do-not-do list

Do **not**:

- widen back to arbitrary 6-face support during Phase 1
- leave a hidden overlay/injection fallback in runtime wiring
- let `mean()` / `repeat()` remain the mathematical source of truth
- preserve CPML overlap as warning-only behavior
- let JIT/runtime tests silently keep broader support claims alive
