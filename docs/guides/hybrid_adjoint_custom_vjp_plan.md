# Hybrid adjoint-AD via `jax.custom_vjp` — phased plan

Status: plan  
Date: 2026-04-16  
Branch: `plan/hybrid-adjoint-custom-vjp`

## Canonical authority

Primary technical authority:

- `docs/research_notes/2026-04-16_hybrid_adjoint_ad_handover.md`

This tracked doc is a branch-visible planning artifact derived from that handover.  
Execution details should remain consistent with the handover unless explicitly revised.

## Workspace

Planning lane:

- worktree: `/root/workspace/byungkwan-workspace/.worktrees/rfx-hybrid-adjoint-ralplan`
- branch: `plan/hybrid-adjoint-custom-vjp`

This work should stay isolated from the busy main checkout.

## Overall staged goals

### Phase 1A — canonical seam extraction
- identify the real step/replay seam in `simulation.py`
- document carry ownership and replay requirements

### Phase 1B — correctness POC
- build a Strategy A `custom_vjp` proof of concept on that seam
- keep scope uniform / lossless / PEC-only
- prove gradient agreement vs pure AD

### Phase 2 — seam hardening
- stabilize seam boundaries and replay contract for later subsystem growth

### Phase 3 — subsystem expansion
- add CPML, Debye, Lorentz, then NTFF only if still needed

### Phase 4 — optimization integration
- integrate the hybrid path selectively into `optimize.py` / `topology.py`
- define fallback policy instead of making it a blanket default

## Phase 1 goal

Phase 1 is **not** “implement hybrid adjoint for rfx.”  
Phase 1 is:

> prove that hybrid adjoint-AD is correct on the real `simulation.py` seam using a narrow Strategy A POC.

### Phase 1 success criteria

- the canonical seam is explicitly identified
- replay contract is documented
- tiny PEC fixture is fixed and reproducible
- pure AD vs hybrid gradient relative error `<= 1e-4`
- unsupported physics paths are explicitly denied or routed back to pure AD
- memory-evidence plan exists on the real carry

### Phase 1 non-goals

- CPML adjoint
- Debye/Lorentz adjoint
- NTFF adjoint
- Strategy B
- full optimize/topology integration
- production API stabilization

## Core tradeoff

The POC must stay narrow enough to be correct, but not so narrow that it avoids the real seam that later phases must use.
