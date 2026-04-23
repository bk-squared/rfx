# Hybrid Adjoint Macro Roadmap

Status: active macro roadmap
Date: 2026-04-23
Worktree: `/root/workspace/byungkwan-workspace/.worktrees/rfx-hybrid-adjoint-ralplan`

## Purpose

This document replaces the older micro-phase mindset (`4A`, `4B`, `4C`, `4D`, ...) with a
small number of macro phases that are large enough to justify their coordination overhead.

Use this roadmap when:
- deciding what the next major roadmap phase should be
- opening a new `$ralplan` for a major phase
- explaining why a task belongs to the current phase vs a future one
- checking which execution mode (`ralph`, `team`, `autoresearch`, `autopilot`) best matches the work

## Current baseline

### Committed major milestones
- seam extraction / hardening complete
- Phase 3A: CPML complete
- Phase 3B: Debye complete
- Phase 3C: Lorentz complete
- Phase 3D: NTFF complete
- Phase 4A: selective `optimize.py` hybrid routing complete
- Phase 4B: one-port proxy optimize blocker-support complete
- Phase 4C / Phase I closure: zero-sigma dielectric topology carve-out complete

### Current worktree state
- Phase II Practical Workflow Coverage is complete locally and ready to commit:
  - CPML zero-sigma dielectric source/probe topology hybrid support
  - one-excited + optional one-passive lumped-port proxy optimize hybrid support
  - cookbook/support-contract alignment
- next major roadmap target is **Phase III — Strategy B / Scale-up**

## Operating principles

1. **A macro phase exists only when user value or algorithmic boundaries change materially.**
2. **Inside a macro phase, use workstreams or milestones, not new phase numbers.**
3. **Planning should be broad; verification should remain granular.**
4. **Do not open a new macro phase for a single fixture, helper, or narrow support fence.**
5. **Prefer seam-owned support contracts over per-surface drift.**

## Macro phases

---

## Phase I — Usable Hybrid Optimization Surface

### Goal
Turn the hybrid adjoint from a correctness-focused experimental seam into a usable, explicitly bounded
optimization surface across the main inverse-design entry points.

### Includes
- stabilization and commit of the current Phase 4C zero-sigma topology carve-out
- support/fallback/reporting alignment across:
  - `Simulation` seam helpers
  - `optimize()`
  - `topology_optimize()`
- docs and cookbook updates so the supported subset matches the real code
- completion notes explaining what is now genuinely usable

### Does **not** include
- broad topology support
- broad port-family support beyond already landed one-port subset
- Strategy B / time-reversal / checkpoint redesign
- making hybrid the default everywhere

### Success criteria
- `optimize()` and `topology_optimize()` both have explicit hybrid subsets with clear fallback semantics
- docs match the real support boundary
- support/reason surfaces are stable enough to cite directly in future planning
- the worktree is clean and the phase can be summarized without caveats about hidden behavior drift

### Best execution mode
- **Default:** `ralph`
- **Why:** same file family, same support-contract discipline, same verification owner

---

## Phase II — Practical Workflow Coverage

### Goal
Expand hybrid support from the currently usable optimization subset into the **practical workflows users are most likely to try next**.

### Typical workstreams inside this phase
- broader topology coverage beyond the narrow zero-sigma dielectric subset
- broader port-family coverage beyond the one-port proxy carve-out
- seam/report/fallback consolidation where support logic still feels fragmented
- representative docs/examples for those newly supported practical workflows

### Does **not** include
- Strategy B memory work
- full default enablement
- “support everything” ambitions

### Success criteria
- common optimize/topology workflows are supported without re-entering seam surgery for every small use case
- support inspection/reporting feels unified, not fragmented by entry point
- docs/examples reflect actual supported workflows, not theoretical subsets

### Best execution mode
- **Planning:** `$ralplan`
- **Execution:** `ralph` by default, `team` only if the phase naturally splits into low-conflict lanes

### Guidance
If this phase starts fracturing into many tiny blocker-specific “mini-phases”, stop and regroup around the dominant workflow family instead.

---

## Phase III — Strategy B / Scale-up

### Goal
Move from correctness-first Strategy A replay to a memory-scaled hybrid adjoint that can support materially larger workloads.

### Includes
- Strategy B / time-reversal / checkpointed reconstruction design and implementation
- benchmark harnesses for realistic optimize/topology workloads
- correctness checks against Strategy A or pure AD where applicable
- memory/runtime evidence strong enough to justify the added complexity

### Does **not** include
- broad workflow-surface expansion unless required by the benchmark plan
- release/productization cleanup work

### Success criteria
- realistic workloads show meaningful scale or memory benefit
- the algorithmic tradeoff is measurable and documented
- correctness is not weakened to obtain scale

### Best execution mode
- **First:** `autoresearch` for benchmark/evidence framing if the exact scale target is unclear
- **Then:** `$ralplan --deliberate`
- **Execution:** `team`

### Guidance
If this phase cannot show compelling benchmark value, do not let it sprawl into an architecture hobby project.

---

## Phase IV — Productization / Default Policy

### Goal
Turn the hybrid adjoint from an advanced explicit capability into a product-quality, policy-driven feature surface.

### Includes
- explicit selection policy across `pure_ad`, `hybrid`, `auto`, and (if landed) Strategy B paths
- default-policy decisions for optimize/topology surfaces
- support-matrix cleanup and release-quality docs/examples
- migration guidance and boundary communication
- any release-facing enablement logic that depends on the earlier phases being complete

### Does **not** include
- new physics support families
- new replay algorithms

### Success criteria
- users can understand when and why a given mode is selected
- docs, examples, and actual runtime behavior align
- the default-policy story is defendable with verification and benchmark evidence

### Best execution mode
- **Planning:** `$ralplan`
- **Execution:** `team` or `ralph` depending on how much is policy/docs vs code
- **Limited use of `autopilot`:** only for low-risk housekeeping/docs chores, not for core policy design

---

# Mode-selection guidance

## `ralph`
Use as the **default execution mode** for core hybrid-adjoint development.

Best for:
- seam-boundary work
- replay correctness work
- feature slices where one owner should carry the implementation through verification

## `team`
Use when a macro phase has **truly separable lanes**.

Good examples:
- implementation lane
- test lane
- docs lane
- benchmark/verification lane

Avoid when:
- multiple lanes need to edit the same seam files heavily

## `autoresearch`
Use **before** execution when the next major step depends on evidence more than code.

Best for:
- Strategy B prioritization
- benchmark design
- evaluating whether a blocker family or algorithmic shift matters more next

## `autopilot`
Do **not** use as the default for core hybrid-adjoint algorithm/support work.

Allowed only for:
- low-risk docs chores
- examples cleanup
- release-notes / housekeeping style work after the hard support decisions are already made

## Recommended default by phase
- **Phase I:** `ralph`
- **Phase II:** `$ralplan` -> `ralph` or `team`
- **Phase III:** `autoresearch` -> `$ralplan --deliberate` -> `team`
- **Phase IV:** `$ralplan` -> `team` or `ralph`; `autopilot` only as a helper for low-risk cleanup

# Invocation guidance

When you need to open a major planning loop, cite this roadmap explicitly.

Examples:

```text
$ralplan from .omx/plans/hybrid-adjoint-macro-roadmap.md focus: Phase II Practical Workflow Coverage
```

```text
$ralplan from .omx/plans/hybrid-adjoint-macro-roadmap.md focus: Phase III Strategy B / Scale-up
```

```text
$ralph execute current Phase I closure from .omx/plans/hybrid-adjoint-macro-roadmap.md
```

# Anti-fragmentation rule

Do **not** create a new macro phase unless at least one of the following is true:
- replay algorithm changes materially
- user-facing policy changes materially
- the dominant workflow family changes materially
- benchmark/scale goals change materially

If the change is only:
- a new negative test family
- a helper extraction
- a small support fence adjustment
- a fixture migration
- a docs sync

then it is a **milestone inside the current macro phase**, not a new phase.

# Near-term recommendation

If starting from the current repo state:
1. commit the completed Phase II Practical Workflow Coverage worktree closure
2. begin **Phase III — Strategy B / Scale-up** with benchmark/evidence framing before implementation
3. use `autoresearch` first if the scale target is unclear, then `$ralplan --deliberate` and `team` execution when the Strategy B plan is evidence-backed
4. avoid reopening micro-phase numbering unless a genuinely new algorithm/policy boundary appears
