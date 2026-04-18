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
- expose a public inspection surface for support/fallback and replay inventory so later phases stop depending on private prep helpers
- expose a public context-builder for the supported seam so replay users can target a stable contract instead of private prep state
- expose a public context-consumption helper so supported replay users can execute the stable seam without importing lower-level module helpers
- ensure the public replay context carries the supported baseline material state needed to re-execute lossless dielectric cases faithfully
- expose one public prepared bundle surface so later phases can consume inspection + executable context from a single stable contract
- keep the older inspect/build/execute APIs layered on top of that bundle rather than letting the contract fragment again
- expose direct execution from the prepared bundle so later phases can stay on a single stable preparation contract end-to-end
- make the prepared bundle itself more self-contained so later phases can depend on one stable object rather than unpacking separate report/context fields everywhere
- keep pushing execution behavior onto that self-contained bundle so later phases can consume one stable object end-to-end
- keep pushing execution behavior down into the replay context itself so later phases have one canonical execution path under every higher-level helper
- keep pushing result-shaping behavior down as well so later phases avoid reintroducing wrapper duplication above the canonical seam objects
- keep collapsing even tiny result-wrapper construction into seam-owned helpers so higher layers stay as thin delegates
- keep pushing unsupported-path messaging down too, so later phases share one canonical reason surface instead of rebuilding error text in higher layers
- keep pushing seam metadata down as well, so later phases can consume one canonical prepared object without unpacking report internals again
- push bundle construction down too, so later phases keep one canonical seam-owned assembly path instead of rebuilding public objects in higher layers
- keep consolidating object creation onto the seam-owned types themselves (not just free functions) so later phases have clearer ownership and fewer parallel construction paths
- keep consolidating bundle creation onto the seam-owned types across both supported and unsupported inspected states so the API layer sheds more branching over time
- keep completing the helper families symmetrically (including the inspected-runner prepare alias) so later phases do not grow lopsided helper surfaces again
- keep routing public convenience methods back through the seam-owned input surface so the API layer converges toward one canonical Phase 1 entry object
- keep pushing prepared-runner-to-seam translation down as well, so later phases do not drift back into API-owned reconstruction helpers
- keep pushing prepared-runner-to-report translation down too, so later phases have one seam-owned path even for inspection rebuilding
- keep consolidating construction on the canonical seam-owned types themselves (not only helper functions) so later phases have fewer parallel creation paths
- keep letting the canonical seam-owned types own report creation from lower seam objects too, so the remaining helper functions can shrink toward thin aliases
- expose the seam-owned input spec explicitly so later phases can extend preparation from one canonical input object rather than re-threading loose arguments again
- keep consolidating input construction variants on the seam-owned type itself, so supported and unsupported preparation share one canonical input-construction surface
- keep making that input object self-contained so later phases can operate from one canonical seam-owned entry surface before even constructing prepared bundles
- push direct execution onto that input object where safe, so the API layer becomes a thinner delegation shell over the seam-owned contract
- where helpful, expose thin public wrappers over that input-owned execution path rather than reintroducing separate execution logic higher up
- complete the remaining forward helper aliases symmetrically as the contract stabilizes, so callers do not split between input/context/prepared execution surfaces
- extend that same symmetry to the prepared-runner and inspected-runner execution paths too, so every major seam state now has a canonical forward helper instead of forcing callers back through manual context plumbing
- do the same for inspection access, so the public Phase 1 surface family can converge around the seam-owned input object instead of parallel API-specific paths
- round out prepare/context access on that same input-centered public surface family so callers can stay on one canonical Phase 1 entry object end-to-end
- expose the corresponding seam-owned helper aliases at the package root as the contract stabilizes, so callers do not split between input-surface usage and deep-module imports
- expose the seam-owned inventory type at the package root too, so callers can type against the replay inventory contract without deep-module imports
- expose the seam-owned field-carry type at the package root too, so callers can type against the replay carry contract carried by built contexts without deep-module imports
- expose the remaining low-level seam-owned execution primitives at the package root too, so callers can stay on the public contract surface even when they need raw time-series execution or the custom_vjp factory itself
- expose the canonical support-reason helper at the package root too, so callers can reuse the unsupported-path gating contract without deep-module imports
- complete the public Simulation execution wrappers down to the lower runner-state surfaces too, so callers do not have to drop out of the high-level API just to execute seam-owned prepared/inspected runner snapshots
- round out the same high-level API symmetry for inspect/prepare/context access on those runner-state surfaces too, so all major seam states can be consumed from one public API family
- complete the same high-level API symmetry for input-building on those runner-state surfaces too, so every major Phase 1 seam state can enter the canonical input object from one public API family
- where safe, let the input object expose the supported replay context directly too, so later phases can move more supported-path access off the bundle layer
- continue collapsing repeat inspection/preparation access onto cached seam-owned views so later phases do not re-trigger parallel construction paths unnecessarily
- keep standardizing unsupported object construction in the seam module as well, so later phases do not drift back into ad hoc error-object assembly in higher layers
- keep pulling the remaining unsupported special cases (like the non-uniform bundle/report path) behind seam-owned helpers for consistency
- where possible, collapse those helpers into seam-owned type constructors/classmethods too, so the canonical object types own their own unsupported construction paths
- extend that same consistency to the input surface too, so even unsupported preparation can stay on canonical seam-owned entry objects
- keep the top-level and runner-state public API families symmetric across input/inspect/prepare/context/forward so later phases do not fork surface-specific behavior again
- keep the top-level and runner-state omitted-`n_steps` behavior aligned across supported and unsupported paths so later subsystem expansion does not silently fork timestep-resolution semantics
- keep contract regressions broad enough to catch export drift, default/signature drift, omitted-n_steps drift, and unsupported reason-text drift before subsystem expansion begins
- keep API and seam-owned wrapper/prep signatures explicitly annotated so later subsystem expansion can reuse a clearer typed contract instead of rediscovering wrapper intent from implementation details

### Phase 3 — subsystem expansion
- extend beyond the now-supported CPML / Debye / Lorentz time-series seam only where evidence still justifies it
- prioritize remaining unsupported physics such as NTFF / waveguide-port accumulation and other non-time-series surfaces only if still needed

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

### Original Phase 1 POC non-goals

- CPML adjoint
- Debye/Lorentz adjoint
- NTFF adjoint
- Strategy B
- full optimize/topology integration
- production API stabilization

These were the **initial POC limits**. The current Phase 2 seam-hardening work has already moved beyond that original narrow forward surface.

## Current execution snapshot

- the seam regression floor is currently `pytest -q tests/test_hybrid_adjoint_phase1.py` → `215 passed`
- repeated supported-path setup in the regression file has been consolidated behind smaller reusable helpers
- the current supported replay seam now covers:
  - uniform grids
  - PEC and CPML boundaries
  - lossless, Debye-dispersive, and Lorentz-dispersive materials with zero conductivity
  - point-source / point-probe `time_series` objectives
- the current explicit unsupported / rejected surface still includes:
  - non-uniform grids
  - periodic / Floquet-style paths
  - mixed Debye+Lorentz dispersion
  - Drude-shaped Lorentz poles
  - NTFF / waveguide-port accumulation
  - lumped or wire-port source paths

## Core tradeoff

The POC must stay narrow enough to be correct, but not so narrow that it avoids the real seam that later phases must use.
