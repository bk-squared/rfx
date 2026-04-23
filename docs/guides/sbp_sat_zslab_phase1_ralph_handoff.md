# SBP-SAT z-slab Phase 1 — ralph handoff

Status: execution handoff  
Date: 2026-04-16  
Branch: `plan/sbp-sat-zslab-ralplan`

## Purpose

> **Current planning note (2026-04-23):** use
> `docs/guides/sbp_sat_zslab_phase1_full_spec.md` and
> `.omx/plans/ralplan-sbp-sat-next-phase.md` before executing. This file is a
> historical Phase-1 handoff.

This document is the **execution handoff** for a future `$ralph` implementation pass.

Use it together with:

- `docs/guides/sbp_sat_zslab_phase1_plan.md`
- `docs/guides/sbp_sat_zslab_phase1_crosswalk.md`
- `.omx/plans/prd-sbp-sat-zslab-phase1.md`
- `.omx/plans/test-spec-sbp-sat-zslab-phase1.md`

## Start here

Work only in:

- `/root/workspace/byungkwan-workspace/.worktrees/rfx-sbp-sat-zslab-ralplan`

Do **not** implement this lane in the busy main checkout.

## Mission

Implement the approved **Phase 1 z-slab-only SBP-SAT lane** while preserving:

1. **z-slab only**
2. **single canonical stepper**
3. **norm-compatible face ops**

If any proposed shortcut violates one of those, reject the shortcut.

## Hard constraints

### Scope constraints

- full-span x/y only
- local z-slab refinement only
- interface faces `z_lo`, `z_hi` only
- no arbitrary 6-face support
- no temporal sub-stepping
- no CPML + subgrid coexistence

### Runtime constraints

- one canonical Phase 1 stepper only
- no overlay / overwrite / injection fallback
- no hidden second implementation path left as “temporary”

### Numerical constraints

- explicit face norms
- norm-compatible prolongation / restriction
- tangential `E/H` SAT coupling on z-faces
- overlap-safe energy accounting

## First implementation decisions already made

These are **not open questions anymore**:

1. **Workspace strategy**
   - use the isolated worktree already created

2. **Public-facing entrypoint**
   - preserve `Simulation.add_refinement(...)`
   - narrow it to Phase 1 z-slab semantics
   - hard-fail unsupported arguments/configurations

3. **Validation posture**
   - benchmark-gated, not smoke-only

## Ordered execution plan

### Step 1 — notation crosswalk

Before changing solver code:

- expand the notation crosswalk from `docs/guides/sbp_sat_zslab_phase1_crosswalk.md`
- make the mapping between:
  - face norms
  - `P/R`
  - tangential z-face traces
  - SAT-H / SAT-E corrections
  - energy accounting
  explicit enough that later code and tests can share one vocabulary

### Step 2 — face ops

Introduce the canonical face-operator layer:

- explicit z-face norms
- prolongation builder
- restriction builder derived from norms
- norm-compatibility checks

Do this **before** writing the new stepper.

### Step 3 — trace helpers

Add explicit helpers for:

- tangential `E` extraction/scatter on `z_lo`, `z_hi`
- tangential `H` extraction/scatter on `z_lo`, `z_hi`

Do not let stepper internals manipulate trace slices ad hoc.

### Step 4 — canonical SAT coupling

Implement:

- SAT-H coupling on z-faces
- SAT-E coupling on z-faces

Coupling must use the canonical face-op layer, not mean/repeat shortcuts.

### Step 5 — single canonical stepper

Implement or converge to one Phase 1 stepper:

- one authoritative runtime path
- no parallel legacy fallback
- preserve global `dt`

### Step 6 — misuse policy

Tighten the API and runtime checks so unsupported configs hard-fail:

- CPML overlap
- non-z-slab requests
- partial x/y refinement

### Step 7 — verification ladder

Ship tests in this order:

1. operator/property
2. misuse
3. smoke
4. benchmark
5. legacy-path fencing/removal

## Expected write scope

Likely files:

- `rfx/subgridding/sbp_sat_3d.py` or replacement z-slab module
- `rfx/subgridding/jit_runner.py`
- `rfx/subgridding/runner.py`
- `rfx/runners/subgridded.py`
- `rfx/api.py`
- `tests/test_sbp_sat_3d.py`
- `tests/test_sbp_sat_alpha.py`
- `tests/test_sbp_sat_jit.py`
- new face-op / energy files if introduced

Support docs:

- `docs/guides/sbp_sat_zslab_phase1_crosswalk.md`
- `docs/guides/sbp_sat_zslab_phase1_plan.md`

## Acceptance gates

Do not claim completion until all are satisfied:

### Operator/property

- norm-compatibility test passes
- face helper invariants pass
- alpha/tau source-of-truth test passes

### Misuse

- CPML overlap hard-fails
- partial x/y refinement hard-fails
- non-z-slab box requests hard-fail

### Smoke

- `max(E_t)/E_0 <= 1.02`
- `E_1000/E_0 <= 1.00`

### Benchmark

- reflection error `<= 0.05`
- transmission amplitude error `<= 5%`
- transmission phase error `<= 5°`

### Migration

- legacy overlay/E-only path removed, unreachable, or explicitly fenced off
- only one canonical Phase 1 stepper remains authoritative

## Suggested staffing if ralph needs help

Primary owner:

- `executor`

Useful support lanes:

- `architect` for notation/API boundary checks
- `test-engineer` for operator + benchmark suite design
- `verifier` for final evidence review
- `writer` for support-boundary/docs cleanup after code evidence exists

## Stop / escalate conditions

Stop and reassess if:

- implementation pressure starts re-expanding scope to general 3D box support
- a shortcut depends on warning-only CPML overlap behavior
- runtime wiring requires keeping two canonical steppers alive
- benchmark fixtures cannot be defined cleanly enough to enforce the numeric thresholds

## Completion packet

When the implementation is ready, final reporting should include:

- changed files
- what was simplified or removed
- verification evidence by layer
- remaining risks
- whether any legacy surface is still present but intentionally unsupported
