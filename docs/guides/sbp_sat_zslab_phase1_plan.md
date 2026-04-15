# SBP-SAT z-slab Phase 1 plan

Status: plan  
Date: 2026-04-16  
Branch: `plan/sbp-sat-zslab-ralplan`

## Canonical authority

The primary design authority for this lane is the shared rewrite document:

- https://chatgpt.com/share/69dfae8d-b3cc-83ab-b286-cba725271684

This note is the repo-tracked local planning record derived from that spec.  
The corresponding local ralplan workflow artifacts live in:

- `.omx/plans/prd-sbp-sat-zslab-phase1.md`
- `.omx/plans/test-spec-sbp-sat-zslab-phase1.md`

Tracked companion docs:

- `docs/guides/sbp_sat_zslab_phase1_crosswalk.md`
- `docs/guides/sbp_sat_zslab_phase1_ralph_handoff.md`

## Workspace decision

Because other agents are already working locally, **do not use the busy main checkout for this lane**.

Use:

- separate branch
- separate git worktree

Current isolated worktree:

- `/root/workspace/byungkwan-workspace/.worktrees/rfx-sbp-sat-zslab-ralplan`

Important clarification:

- **No additional extra folder is needed beyond this isolated worktree.**
- The requirement is **checkout isolation**, not repeated folder creation.

## Scope

Phase 1 is explicitly:

- **z-slab only**
- full-span x/y
- local z-slab refinement only
- interface faces `z_lo`, `z_hi` only
- same global `dt` on coarse/fine grids
- tangential `E/H` SAT coupling
- explicit face norms
- norm-compatible transfer operators
- no overlay/injection
- CPML overlap hard-fail
- one canonical stepper only

Working shorthand for this lane:

- z-slab only
- single canonical stepper
- norm-compatible face ops

## Non-goals

Do not treat these as Phase 1 deliverables:

- arbitrary 6-face 3D box refinement
- partial x/y refinement
- temporal sub-stepping
- CPML + subgrid coexistence
- parallel legacy + canonical runtime paths
- public claims of general 3D SBP-SAT support

## Brownfield findings

Current repo state that motivates the rewrite lane:

- `rfx/subgridding/sbp_sat_3d.py` still carries 6-face logic and mean/repeat transfer shortcuts.
- `rfx/runners/subgridded.py` already effectively forces full-span x/y.
- `rfx/api.py` warns on PML overlap rather than hard-failing.
- `rfx/subgridding/jit_runner.py` and `tests/test_sbp_sat_jit.py` assume a broader runtime surface than this narrowed Phase 1 scope.
- `tests/test_sbp_sat_3d.py` and `tests/test_sbp_sat_alpha.py` are mostly smoke/regression oriented, not benchmark-gated.

## External primary references worth consulting

Must-have:

1. https://doi.org/10.1109/TAP.2022.3230553
2. https://doi.org/10.1109/TAP.2025.10836194
3. https://doi.org/10.1006/jcph.1998.6114
4. https://doi.org/10.1007/s10915-018-0723-9

Nice-to-have:

- https://doi.org/10.1016/j.jcp.2023.112510
- https://doi.org/10.1002/1098-2760%2820001205%2927%3A5%3C334%3AAID-MOP14%3E3.0.CO%3B2-A

## Pre-implementation gates

Before implementation starts:

1. keep the isolated worktree as the only active authoring surface for this lane
2. finish PRD + test spec review
3. add a paper-to-code notation crosswalk
4. preserve `Simulation.add_refinement(...)`, but narrow it to z-slab-only semantics and hard-fail unsupported args/configurations
5. lock benchmark-oriented verification criteria, not smoke tests alone

## Immediate next steps

1. finalize consensus review of the PRD/test spec
2. map reusable / replaceable / forbidden parts of current subgridding code
3. hand off to implementation only after the verification contract is stable
