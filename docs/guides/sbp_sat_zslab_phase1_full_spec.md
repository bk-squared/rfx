# SBP-SAT subgridding full specification plan

Status: planning specification
Date: 2026-04-23
Branch: `plan/sbp-sat-zslab-ralplan`
Worktree: `/root/workspace/byungkwan-workspace/.worktrees/rfx-sbp-sat-zslab-ralplan`

## 0. Purpose and reading contract

This document is the next canonical planning surface for applying SBP-SAT
subgridding to `rfx`. It supersedes the older Phase-1-only handoff framing in
the sense that the branch is no longer pre-implementation: the worktree already
contains z-slab face operators, SAT helpers, runtime wiring, and tests.

This is still a planning document, not a release claim. Public documentation
must continue to describe SBP-SAT subgridding as experimental until the
main-branch drift, boundary model, benchmark definitions, and unsupported
runtime surfaces are reconciled.

### Current evidence

- `Simulation.add_refinement(...)` currently accepts `z_range`, `ratio`,
  `xy_margin`, and `tau`, while rejecting `xy_margin` and CPML/UPML in the
  Phase-1 path (`rfx/api.py:533-592`).
- Additional subgridding preflight rejects NTFF, DFT planes, waveguide ports,
  Floquet ports, TFSF, and lumped RLC (`rfx/api.py:2933-2961`).
- Face operators are implemented in `rfx/subgridding/face_ops.py:1-179`.
- The current 3D SBP-SAT core contains z-slab config/state, canonical `dt`,
  face trace helpers, SAT helpers, compatibility wrappers, the stepper, and
  overlap-safe energy accounting (`rfx/subgridding/sbp_sat_3d.py:1-556`).
- The runtime path builds a full-span x/y fine slab from `add_refinement(...)`
  and dispatches to `run_subgridded_jit(...)`
  (`rfx/runners/subgridded.py:13-225`).
- JIT execution calls `step_subgrid_3d(...)` inside one `jax.lax.scan`
  (`rfx/subgridding/jit_runner.py:25-148`).
- Current tests cover API guards, face operators, z-slab smoke, alpha/tau,
  JIT, and proxy cross-validation (`tests/test_sbp_sat_api_guards.py:1-57`,
  `tests/test_sbp_sat_face_ops.py:1-128`,
  `tests/test_subgrid_crossval.py:1-144`).

## 1. RALPLAN-DR summary

### Principles

1. **Truthful support surface first**: only documented, verified behavior may
   become public support guidance.
2. **Mathematical source of truth**: face norms, transfer operators, SAT
   equations, and energy accounting must be specified before expansion.
3. **One canonical runtime path**: no hidden overlay, injection, or legacy
   near-equivalent path may remain authoritative.
4. **Drift-aware delivery**: plans must account for `origin/main` boundary,
   preflight, crossval, and docs changes before merge or public claims.
5. **Benchmark credibility over smoke success**: stability and proxy tests are
   necessary but do not prove full reflection/transmission correctness.

### Decision drivers

1. The branch is heavily drifted from `origin/main`; the planning baseline must
   include rebase/integration work before publication.
2. Existing code already implements a partial Phase-1 z-slab path; the spec
   must describe actual state and gaps, not only future intent.
3. The target is full SBP-SAT subgridding for `rfx`, so the roadmap must extend
   beyond the first PEC z-slab milestone without over-claiming it.

### Viable options

| Option | Summary | Pros | Cons | Verdict |
|---|---|---|---|---|
| A. Spec-first, implementation-aware roadmap | Write a full internal spec and execution roadmap before more solver work | Aligns code, docs, tests, and drift; safest ralplan seed | Delays additional implementation | **Chosen** |
| B. Continue patching Phase 1 directly | Fix current blockers immediately and document later | Fast short-term progress | Risks encoding unstable contracts; BoundarySpec drift remains hidden | Rejected for next step |
| C. Rebase first, spec later | Resolve `origin/main` drift before writing detailed spec | Avoids documenting stale APIs | Rebase may obscure current SBP-SAT decisions; no stable review artifact | Defer until spec captures expected merge policy |
| D. Public docs first | Replace placeholder public guide now | Visible progress | Premature support claim; benchmark and boundary gaps remain | Rejected |

## 2. Table of contents and section-by-section writing plan

Each section below is a concrete writing unit for the final full specification.
The acceptance criteria are intentionally testable so the document can feed a
future `$ralplan` / `$ralph` execution pass.

### 2.1 Status, authority, and document lifecycle

**Purpose:** establish what this spec governs and how it relates to older
planning artifacts.

**Must include:**

- Canonical artifact path: this file.
- Related older artifacts:
  - `.omx/plans/prd-sbp-sat-zslab-phase1.md`
  - `.omx/plans/test-spec-sbp-sat-zslab-phase1.md`
  - `docs/guides/sbp_sat_zslab_phase1_plan.md`
  - `docs/guides/sbp_sat_zslab_phase1_crosswalk.md`
  - `docs/guides/sbp_sat_zslab_phase1_ralph_handoff.md`
- Lifecycle states:
  1. `planning specification`
  2. `implementation baseline approved`
  3. `post-main-drift reconciled`
  4. `public-doc eligible`
- Explicit rule: older documents are historical inputs, but this document is
  the current roadmap once reviewed.

**Acceptance criteria:**

- A future agent can identify the canonical spec without asking the user.
- The doc states whether it is safe to implement, rebase, or publish from the
  current state.

### 2.2 Current worktree baseline

**Purpose:** make the current implementation state auditable before planning
new work.

**Must include:**

- Git state:
  - current branch: `plan/sbp-sat-zslab-ralplan`
  - current worktree path
  - HEAD at review time: `cb68d31d3bb38ff96040084596cf8849dfa936be`
  - `origin/main` at review time: `4da9415eec7da7feddcc705acf1f93a1154b1bbd`
  - branch tracking divergence:
    `plan/sbp-sat-zslab-ralplan...origin/plan/sbp-sat-zslab-ralplan`
    is 417 commits ahead / 380 behind
  - known drift at review time: 472 commits ahead / 417 behind `origin/main`
  - merge-base: `ae7194d` from 2026-03-28
- Modified files currently in the SBP-SAT lane:
  - `docs/guides/sbp_sat_zslab_phase1_full_spec.md` (this new canonical
    planning spec; currently untracked until staged)
  - `docs/guides/sbp_sat_zslab_phase1_crosswalk.md`
  - `rfx/runners/subgridded.py`
  - `rfx/subgridding/face_ops.py`
  - `rfx/subgridding/sbp_sat_3d.py`
  - `tests/test_sbp_sat_3d.py`
  - `tests/test_sbp_sat_alpha.py`
  - `tests/test_sbp_sat_face_ops.py`
  - `tests/test_sbp_sat_jit.py`
  - `tests/test_subgrid_crossval.py`
- Additional relevant committed branch files:
  - `tests/test_sbp_sat_api_guards.py`
  - `rfx/subgridding/jit_runner.py`
  - `rfx/api.py`

**Acceptance criteria:**

- The baseline section can be refreshed by rerunning `git status`,
  `git rev-list --left-right --count origin/main...HEAD`, and targeted tests.
- The section separates uncommitted local edits from broader branch drift.

### 2.3 Supported surface matrix

**Purpose:** prevent public/API claims from exceeding verified behavior.

**Current supported intent:**

| Surface | Status | Required wording |
|---|---|---|
| Geometry | Phase-1 candidate | full-span x/y, local z slab only |
| Interfaces | Phase-1 candidate | z-faces only: `z_lo`, `z_hi` |
| Boundaries | Candidate all-PEC only | reject CPML/UPML now; after `origin/main`, reject PMC/periodic/mixed specs too |
| Sources | Candidate soft point sources only | source/probe positions must lie inside refined z slab |
| Probes | Candidate point probes only | probe positions must lie inside refined z slab |
| Impedance ports | **Unsupported in Milestone 1** | hard-fail nonzero impedance point ports and wire/extent ports; repair/support moves to a later port-support milestone |
| NTFF/DFT/waveguide/Floquet/TFSF/RLC/coaxial ports | Unsupported | fail fast |
| Arbitrary 3D box refinement | Unsupported | later milestone only |
| CPML + subgrid coexistence | Unsupported | later research milestone only |

**Evidence:**

- `add_refinement(...)` rejects `xy_margin` and CPML/UPML
  (`rfx/api.py:567-576`).
- Preflight rejects several non-Phase-1 features (`rfx/api.py:2933-2961`).
- Runtime currently hard-checks scalar `sim._boundary == "pec"`
  (`rfx/runners/subgridded.py:39-42`).
- The runner still contains partial impedance-port logic that is not currently
  safe (`rfx/runners/subgridded.py:164-211`).
- Milestone 1 therefore chooses rejection, not repair, as the executable
  ralplan policy for impedance ports.

**Acceptance criteria:**

- Every supported feature has at least one positive test.
- Every unsupported feature has a hard-fail test or a tracked test gap.
- Public docs must not describe unsupported features as available.

### 2.4 `origin/main` drift reconciliation

**Purpose:** define how this branch will adapt to the current remote mainline
before merge or public docs.

**Known drift themes:**

1. `BoundarySpec`, per-face boundary types, PMC, CPML/UPML, periodic, and
   per-face padding behavior landed after this branch diverged.
2. Crossval scripts and docs were reorganized.
3. Public docs/support matrices were cleaned up and still mark subgridding as
   experimental / not claims-bearing.

**Required plan content:**

- Add a rebase/integration milestone before public documentation.
- Specify all-PEC-only as the conservative post-merge Phase-1 boundary rule.
- After rebasing onto the `BoundarySpec` mainline, the exact acceptance
  predicate is: all six face tokens are exactly `pec`,
  `periodic_axes() == ""`, `pmc_faces() == set()`, no absorber token exists,
  and no per-face CPML/UPML layer or padding is active. A legacy scalar
  `_boundary == "pec"` check is insufficient because `origin/main` can derive
  scalar `"pec"` from explicit non-absorber specs that still contain PMC.
- Add tests that reject:
  - `BoundarySpec` with any absorber face,
  - `BoundarySpec` with PMC,
  - periodic axes,
  - mixed PEC/PMC/absorber specs,
  - per-face CPML thickness with subgrid.
- Keep support matrix status as experimental until true benchmarks pass.

**Acceptance criteria:**

- A post-rebase test suite proves the SBP-SAT lane does not accidentally accept
  mainline boundary configurations it cannot handle.
- The spec states whether rebase must happen before or after code fixes. This
  plan chooses: **write spec first, fix local blockers second, rebase third,
  public docs last**.

### 2.5 Mathematical contract: norms and transfer operators

**Purpose:** remove ambiguity from the current `P/R` notation.

**Required notation:**

- `r`: refinement ratio.
- Coarse z-face shape: `(n_i, n_j)`.
- Fine z-face shape: `(r n_i, r n_j)`.
- `H_c_face`: diagonal coarse face norm, uniform entry `dx_c^2`.
- `H_f_face`: diagonal fine face norm, uniform entry `dx_f^2`.
- `P_i`, `P_j`: 1D cell-centered linear prolongation matrices.
- `R_i = P_i^T / r`, `R_j = P_j^T / r`.
- `P_face = P_i ⊗ P_j`.
- `R_face = H_c_face^{-1} P_face^T H_f_face`.
- For uniform face cells, `R_face = P_face^T / r^2`.

**Evidence:**

- Current implementation builds `prolong_i`, `prolong_j`,
  `restrict_i = prolong_i.T / ratio`, and `restrict_j = prolong_j.T / ratio`
  (`rfx/subgridding/face_ops.py:83-99`).
- Full 2D application is separable:
  `ops.prolong_i @ coarse_face @ ops.prolong_j.T` and
  `ops.restrict_i @ fine_face @ ops.restrict_j.T`
  (`rfx/subgridding/face_ops.py:122-142`).
- Existing comments that say `R = P^T / ratio` are accurate only for the 1D
  axis operator, not the full 2D face operator.

**Acceptance criteria:**

- The spec distinguishes axis operators from full-face operators.
- `tests/test_sbp_sat_face_ops.py` includes positive tests for interpolation
  weights and norm compatibility (`tests/test_sbp_sat_face_ops.py:20-62`).
- Future code comments must use the same notation.

### 2.6 Mathematical contract: SAT coupling

**Purpose:** define what the current simplified SAT coupling means and what
must be proven before broader claims.

**Current implementation model:**

- `alpha_c = tau / (r + 1)`.
- `alpha_f = tau * r / (r + 1)`.
- Coarse correction uses `restrict(fine_face) - coarse_face`.
- Fine correction uses `prolong(coarse_face) - fine_face`.
- The same pairwise relaxation is currently used for tangential `E` and `H` on
  both z faces.

**Evidence:**

- Penalty coefficients are in `sat_penalty_coefficients(...)`
  (`rfx/subgridding/sbp_sat_3d.py:264-269`).
- Pairwise mismatch relaxation is in `_apply_sat_pair(...)`
  (`rfx/subgridding/sbp_sat_3d.py:348-360`).
- H SAT coupling is applied by `apply_sat_h_zfaces(...)`
  (`rfx/subgridding/sbp_sat_3d.py:383-401`).
- E SAT coupling is applied by `apply_sat_e_zfaces(...)`
  (`rfx/subgridding/sbp_sat_3d.py:404-422`).

**Required spec expansion before support claims:**

- z-face normal sign convention for `z_lo` and `z_hi`.
- Whether SAT-E and SAT-H should differ by sign, material scaling, or timing.
- Exact relationship between simplified array traces and Yee staggering.
- Stability argument for the chosen `tau` range.
- What evidence would invalidate this simplified model.

**Required expansion for the full roadmap beyond z-slab Phase 1:**

- Six-face normal convention and tangential component orientation for
  `x_lo/x_hi`, `y_lo/y_hi`, and `z_lo/z_hi`.
- Edge and corner interaction policy when two or three refined interfaces meet.
- Material-scaled SAT penalties for non-vacuum, magnetic, lossy, Debye, and
  Lorentz materials.
- Discrete energy estimate that covers all accepted faces, edges, and
  materials.
- CFL/sub-stepping contract for equal-`dt`, reduced-`dt`, and future
  sub-stepped variants.
- Explicit limits for nonuniform, dispersive, anisotropic, and nonlinear
  materials until each has its own validation.

**Acceptance criteria:**

- A future implementation change cannot alter SAT sign/scaling without a spec
  diff and test update.
- The doc identifies the current coupling as a candidate implementation, not a
  fully validated final derivation for arbitrary SBP-SAT subgridding.

### 2.7 Indexing and data layout

**Purpose:** make trace extraction/scatter and overlap policy deterministic.

**Required content:**

- Coarse `z_lo`: `k = fk_lo`.
- Coarse `z_hi`: `k = fk_hi - 1`.
- Fine `z_lo`: `k = 0`.
- Fine `z_hi`: `k = nz_f - 1`.
- Tangential E trace: `(Ex, Ey)`.
- Tangential H trace: `(Hx, Hy)`.
- Coarse interior overlap policy:
  - coarse fields remain authoritative outside the refined slab and on
    interface traces;
  - strict coarse interior `fk_lo + 1 : fk_hi - 1` is zeroed to prevent hidden
    redundant dynamics.

**Evidence:**

- Face slice helpers are in `rfx/subgridding/sbp_sat_3d.py:272-279`.
- E/H extraction and scatter helpers are in
  `rfx/subgridding/sbp_sat_3d.py:282-345`.
- Coarse interior zeroing is in `rfx/subgridding/sbp_sat_3d.py:363-380`.

**Acceptance criteria:**

- Trace shape tests cover coarse and fine z-faces.
- Scatter roundtrip tests prove non-face entries are preserved.
- The spec records the Yee-staggering caveat for later high-fidelity SBP-SAT
  derivation work.

### 2.8 Runtime architecture

**Purpose:** specify the canonical runtime path and legacy compatibility policy.

**Current runtime path:**

1. `Simulation.run(...)` dispatches to `_run_subgridded(...)`.
2. `_run_subgridded(...)` imports and calls `run_subgridded_path(...)`
   (`rfx/api.py:2163-2170`).
3. `run_subgridded_path(...)` builds the full-span x/y fine z slab and
   `SubgridConfig3D` (`rfx/runners/subgridded.py:13-81`).
4. Fine-grid materials are rasterized over the z slab
   (`rfx/runners/subgridded.py:93-122`).
5. Sources/probes are mapped to fine-grid indices
   (`rfx/runners/subgridded.py:124-219`).
6. `run_subgridded_jit(...)` runs `step_subgrid_3d(...)` in `jax.lax.scan`
   (`rfx/subgridding/jit_runner.py:108-122`).

**Compatibility shims:**

- `_shared_node_coupling_h_3d(...)` and `_shared_node_coupling_3d(...)` now
  delegate to z-face-only SAT helpers (`rfx/subgridding/sbp_sat_3d.py:425-440`).
- The spec must decide whether to keep these as private compatibility shims,
  rename them, or delete them after import users are removed.

**Required improvements:**

- Add config-level validation so direct `SubgridConfig3D` construction cannot
  bypass full-span x/y and z-slab invariants.
- Implement `validate_subgrid_config_3d(config)` rather than replacing
  `SubgridConfig3D` immediately. Call it from:
  - `init_subgrid_3d(...)` after config construction;
  - `run_subgridded_jit(...)` before state allocation;
  - `step_subgrid_3d(...)` or its closest non-JIT wrapper if per-step
    validation is too expensive under `jax.lax.scan`.
- Reject impedance ports in Milestone 1:
  - nonzero point-port impedance must fail before JIT;
  - wire/extent ports must fail before JIT;
  - support/repair is deferred to a later port-support milestone.
- Remove stale CPML normalization branches from the PEC-only path once the
  source/port policy is final.

**Acceptance criteria:**

- One runtime path is authoritative.
- Direct JIT calls cannot accept unsupported geometry.
- Legacy wrapper names do not imply general 6-face support.

### 2.9 API and preflight contract

**Purpose:** define where invalid user configurations fail.

**Construction-time failures:**

- `ratio <= 1`
- `xy_margin is not None`
- scalar boundary in `("cpml", "upml")`
- invalid `z_range`

**Preflight/run-time failures:**

- feature attached after refinement but before `run(...)`:
  - NTFF
  - DFT plane probes
  - waveguide ports
  - Floquet ports
  - TFSF
  - lumped RLC
  - coaxial ports
  - source/probe outside the refined z slab
  - impedance ports: hard-fail in Milestone 1; support is deferred

**Post-main failures to add:**

- non-all-PEC `BoundarySpec`
- PMC faces
- periodic axes
- mixed per-face boundary specifications
- CPML/UPML per-face thickness or padding with subgrid

**Acceptance criteria:**

- Each hard-fail rule has a named test.
- Errors explain both the rejected configuration and the supported alternative.
- Unsupported configurations fail before long JIT compilation where possible.

**Named hard-fail test matrix:**

| Rule | Failure phase | Expected test | Status |
|---|---|---|---|
| CPML/UPML scalar boundary | construction | `test_subgrid_touching_cpml_fails` | existing |
| `xy_margin` / partial x/y request | construction | `test_partial_xy_refinement_fails` | existing |
| source/probe outside z slab | run mapping/preflight | `test_source_outside_zslab_fails` | existing |
| NTFF with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[ntff]` | existing pattern |
| DFT plane with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[dft_plane]` | existing pattern |
| waveguide port with subgrid | run preflight | `test_subgrid_rejects_waveguide_port` | new explicit test |
| Floquet port with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[floquet]` | existing pattern |
| TFSF with subgrid | run preflight | `test_subgrid_rejects_tfsf_source` | new explicit test |
| lumped RLC with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[lumped_rlc]` | existing pattern |
| coaxial port with subgrid | run preflight | `test_subgrid_rejects_coaxial_port` | new |
| nonzero impedance point port | run preflight before JIT | `test_subgrid_rejects_impedance_point_port` | new |
| impedance wire/extent port | run preflight before JIT | `test_subgrid_rejects_impedance_wire_port` | new |
| direct partial-x/y `SubgridConfig3D` into JIT | JIT entry validation | `test_run_subgridded_jit_rejects_partial_xy_config` | new |
| non-all-PEC `BoundarySpec` | post-rebase construction/preflight | `test_subgrid_rejects_boundaryspec_absorber_faces` | new after rebase |
| PMC face in `BoundarySpec` | post-rebase construction/preflight | `test_subgrid_rejects_boundaryspec_pmc_faces` | new after rebase |
| periodic axes | post-rebase construction/preflight | `test_subgrid_rejects_periodic_axes` | new after rebase |
| mixed per-face boundary specs | post-rebase construction/preflight | `test_subgrid_rejects_mixed_boundaryspec` | new after rebase |
| per-face CPML thickness/padding | post-rebase construction/preflight | `test_subgrid_rejects_per_face_cpml_padding` | new after rebase |

### 2.10 Verification ladder

**Purpose:** define what must pass at each maturity level.

| Layer | Goal | Existing / required tests |
|---|---|---|
| Operator/property | lock face algebra | `tests/test_sbp_sat_face_ops.py` |
| API/misuse | hard-fail unsupported configs | `tests/test_sbp_sat_api_guards.py` plus new port/boundary tests |
| Smoke/stability | prove bounded simple behavior | `tests/test_sbp_sat_3d.py`, `tests/test_sbp_sat_alpha.py` |
| JIT/runtime | prove one canonical scan path | `tests/test_sbp_sat_jit.py` |
| Proxy benchmark | compare probe DFT against uniform-fine | `tests/test_subgrid_crossval.py` |
| True benchmark | compute R/T or S-parameters | new tests required |
| Drift regression | prove `origin/main` integration | new post-rebase tests required |

**Current evidence:**

- Focused suite passed during review:
  `pytest -q tests/test_sbp_sat_api_guards.py tests/test_sbp_sat_3d.py tests/test_sbp_sat_alpha.py tests/test_sbp_sat_face_ops.py tests/test_sbp_sat_jit.py tests/test_subgrid_crossval.py`
  -> `37 passed`; warning count is non-gating and was observed as 7-8 across
  parent/independent review runs.

**Acceptance criteria:**

- Phase-1 implementation-complete requires all non-slow tests plus the proxy
  crossval suite.
- Public-doc eligible requires post-rebase tests plus true benchmark decisions.
- Full SBP-SAT roadmap milestones require separate verification gates.

### 2.11 Benchmark definitions

**Purpose:** prevent proxy comparisons from being mistaken for physical
reflection/transmission validation.

**Current proxy benchmark definition:**

- Run a uniform-fine reference and a subgridded coarse/fine case.
- Sample one time-series probe.
- Compute one-frequency DFT amplitude and phase.
- Require amplitude error `<= 5%` and phase error `<= 5 degrees`.

**Evidence:**

- DFT helper and error calculation are in `tests/test_subgrid_crossval.py:21-50`.
- Reflection-side and transmission-side proxy tests are in
  `tests/test_subgrid_crossval.py:89-144`.
- Transmission probe placement was moved to reduce downstream PEC-cavity
  contamination (`tests/test_subgrid_crossval.py:121-124`).

**Required future true benchmark content:**

- Define incident/reflected/transmitted separation.
- Define time windowing and frequency band.
- Define acceptable amplitude, phase, and energy-balance tolerances.
- Decide whether to use S-parameter extraction or dedicated plane-wave
  monitors.
- Document when slow/GPU markers are required.

**Acceptance criteria:**

- Proxy benchmarks are labeled as proxy benchmarks.
- True R/T benchmark work is its own deliverable, not silently implied by the
  current tests.

### 2.12 Milestones and deliverables

**Purpose:** place Phase 1 inside the full SBP-SAT subgridding roadmap.

#### Milestone 0 — Planning/spec lock

**Deliverables:**

- This full spec.
- Reviewer/critic sign-off.
- Updated pointers from older Phase-1 docs.
- Decision log for ports, BoundarySpec policy, and benchmark claim level.

**Exit gate:**

- No unresolved spec contradiction around support surface, P/R notation, or
  drift ordering.

#### Milestone 1 — Local Phase-1 z-slab hardening

**Deliverables:**

- Config-level Phase-1 validator.
- Port policy implemented:
  - reject nonzero impedance point ports;
  - reject impedance wire/extent ports;
  - reject coaxial ports until separately specified;
  - defer support/repair to a later port-support milestone.
- `validate_subgrid_config_3d(config)` implemented and called at the JIT seam.
- Documentation comments fixed for axis vs face `P/R`.
- Compatibility wrapper policy applied.
- Focused tests passing.

**Exit gate:**

- Current worktree can pass the focused SBP-SAT suite without known broken
  reachable API paths.

#### Milestone 2 — Mainline drift integration

**Deliverables:**

- Rebase or merge against current `origin/main`.
- BoundarySpec-aware preflight.
- All-PEC-only enforcement in the merged API model.
- Updated crossval/docs paths after mainline reorganizations.

**Exit gate:**

- Post-rebase SBP-SAT tests pass.
- BoundarySpec/PMC/periodic/CPML rejection tests pass.

#### Milestone 3 — Benchmark credibility

**Deliverables:**

- Proxy benchmark retained and clearly named.
- True reflection/transmission benchmark spec.
- Either implemented true R/T tests or an explicit deferred research issue.
- Tolerance rationale documented.

**Exit gate:**

- The support matrix states exactly what benchmark evidence exists.

#### Milestone 4 — Public documentation eligibility

**Deliverables:**

- Public-doc quarantine/audit before any positive support claim. Audit at least:
  - `README.md`
  - `docs/public/guide/subgridding.mdx`
  - `docs/public/index.mdx`
  - `docs/public/guide/index.md`
  - `docs/guides/support_matrix.md`
  - `docs/public/api/support-boundaries.mdx`
- Public guide update that remains scoped and experimental.
- Support-boundary update.
- Migration/API notes.
- Example usage limited to supported all-PEC z-slab cases.

**Exit gate:**

- Public docs do not claim arbitrary 3D box, CPML/open boundary, PMC, or port
  support.
- Existing over-claim wording such as "provably stable coupling" is either
  removed or scoped to cited, tested behavior.

#### Milestone 5 — All-PEC arbitrary 6-face box refinement

**Deliverables:**

- General face-operator contract for `x`, `y`, and `z` oriented interfaces.
- Face-normal and tangential-component table for all six faces.
- Edge/corner interaction policy for boxes where refined interfaces meet.
- All-PEC box-refinement implementation plan.
- Uniform-fine benchmarks that exercise oblique waves across non-z faces.

**Exit gate:**

- No all-PEC box-refinement implementation begins until the six-face math,
  edge/corner policy, and benchmarks are specified.

#### Milestone 6 — Boundary coexistence

**Deliverables:**

- `BoundarySpec` coexistence RFC for PMC, periodic, CPML, and UPML.
- Per-face CPML layer/padding interaction contract.
- Open-boundary benchmark definitions distinct from PEC-cavity proxies.
- Hard-fail matrix for every boundary combination still unsupported.

**Exit gate:**

- No non-PEC boundary support is advertised until explicit BoundarySpec tests
  and open-boundary benchmarks pass.

#### Milestone 7 — Ports and observables inside refined regions

**Deliverables:**

- Port-support RFC for impedance point ports, wire/extent ports, coaxial ports,
  waveguide ports, and Floquet ports.
- Source normalization contract for fine-grid material and loss values.
- Probe/DFT/NTFF placement rules across coarse/fine regions.

**Exit gate:**

- Each supported port/observable has positive tests and at least one
  unsupported-placement failure test.

#### Milestone 8 — Materials, dispersion, and time integration

**Deliverables:**

- Material-scaled SAT penalty policy for lossy, magnetic, Debye, Lorentz,
  anisotropic, and nonlinear materials.
- CFL/sub-stepping decision record.
- Discrete energy estimate for accepted material/time-integration cases.
- Benchmark ladder that separates material error from interface error.

**Exit gate:**

- Any material or sub-stepping expansion has a spec, implementation plan, and
  benchmark gate before support matrix promotion.

#### Milestone 9 — Public support promotion

**Deliverables:**

- Support matrix promotion proposal.
- Public docs with exact supported surfaces and examples.
- Release notes and migration caveats.
- Final verifier report tying docs claims to tests and benchmarks.

**Exit gate:**

- Public claims are traceable to passing tests, benchmark evidence, and
  documented unsupported cases.

### 2.13 Ralplan execution handoff

**Purpose:** turn the spec into a staged execution plan without asking the
executor to rediscover the same decisions.

**Recommended `$ralph` path:**

```text
$ralph use docs/guides/sbp_sat_zslab_phase1_full_spec.md as the canonical
SBP-SAT subgridding roadmap. Execute Milestone 1 only: local Phase-1 z-slab
hardening. Do not rebase or update public docs yet.
```

**Recommended `$team` path for later milestones:**

```text
$team "Use docs/guides/sbp_sat_zslab_phase1_full_spec.md. Split work into:
1) BoundarySpec drift integration, 2) config/runtime validator and port policy,
3) benchmark definition and tests, 4) scoped docs/support matrix. Verify through
the Team Verification Path before shutdown."
```

**Available agent roster:**

| Role | Use |
|---|---|
| `architect` | boundary model, SAT/math contract, drift strategy |
| `executor` | validator/runtime/API implementation |
| `test-engineer` | misuse, JIT, benchmark, and drift tests |
| `critic` | plan/design review and contradiction detection |
| `verifier` | completion evidence and test adequacy |
| `writer` | public/internal docs after evidence exists |
| `researcher` | bounded paper/reference lookup only when equations are unclear |

**Team verification path:**

1. Executor reports changed files and exact unsupported feature policy.
2. Test engineer reports focused tests and any slow/GPU test gaps.
3. Architect confirms no support-surface widening.
4. Critic checks that docs, tests, and code agree.
5. Verifier reruns or audits the evidence before final handoff.

### 2.14 Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| BoundarySpec drift accepts unsupported mixed boundaries | Incorrect physics or silent support claims | all-PEC-only rejection tests after rebase |
| Impedance ports remain broken but reachable | User-facing runtime crash | reject in Milestone 1; repair/support only in later port milestone |
| P/R notation remains ambiguous | Future math/code drift | axis-vs-face notation in spec and code comments |
| Proxy benchmarks overclaimed as true R/T | False validation confidence | label proxy tests and create true benchmark milestone |
| Compatibility wrappers imply 6-face support | Maintainer confusion | rename, delete, or document as private shims |
| Rebase is delayed too long | Spec references stale APIs | keep drift section current and re-verify before public docs |
| Direct `SubgridConfig3D` / direct JIT calls bypass API preflight | Unsupported partial-x/y or invalid z slabs execute as if supported | add `validate_subgrid_config_3d(config)` and direct-JIT rejection tests |
| Simplified SAT pairwise relaxation is physically wrong for Yee staggering, signs, or material scaling | Smoke/proxy tests pass while true SBP-SAT physics remains invalid | require explicit SAT equation section, true R/T benchmark milestone, and invalidation criteria before public support |

### 2.15 ADR

**Decision:** Use this implementation-aware full spec as the next canonical
planning artifact, then execute local Phase-1 hardening before mainline drift
integration and public documentation.

**Drivers:**

- The branch already contains partial implementation.
- `origin/main` has boundary/preflight/docs changes that can alter the final
  API integration plan.
- Current tests pass for the narrow path but do not cover all reachable
  runtime/API surfaces.

**Alternatives considered:**

- Continue coding first: rejected because the support contract is still
  ambiguous.
- Rebase first: deferred because current decisions need to be captured before
  conflict resolution.
- Public docs first: rejected because the benchmark and boundary claims are not
  ready.

**Why chosen:**

- A spec-first consolidation gives `$ralplan`, `$ralph`, and `$team` a stable
  shared artifact.
- It prevents Phase-1 local hardening from being confused with full SBP-SAT
  support.

**Consequences:**

- Public documentation waits.
- Some current code may be intentionally rejected/fenced rather than repaired
  immediately, especially impedance ports.
- Rebase work becomes an explicit milestone, not hidden cleanup.

**Follow-ups:**

- Review this file with architect/critic lanes.
- Implement Milestone 1 via `$ralph`.
- Rebase/integrate only after local blockers are fixed or explicitly fenced.

## 3. Self-review checklist

- [x] States current implementation evidence with file/line references.
- [x] Separates Phase-1 candidate behavior from full SBP-SAT roadmap.
- [x] Includes concrete section plans, deliverables, and acceptance criteria.
- [x] Captures `origin/main` drift as a first-class planning input.
- [x] Records port, BoundarySpec, and benchmark gaps.
- [x] Defines `$ralph` and `$team` handoff paths.
- [x] Architect/critic review completed.
- [x] Review feedback applied.

## 4. Review feedback applied

### Critic review

Verdict before revision: `REVISE`.

Applied changes:

- Converted impedance-port handling from an open choice to a Milestone-1
  hard-fail decision.
- Added named hard-fail tests for unsupported runtime/API surfaces.
- Added the new spec file itself to the worktree baseline.
- Specified `validate_subgrid_config_3d(config)` as the config-validation
  implementation path.
- Added risks for direct-JIT bypass and simplified SAT invalidation.

### Architect review

Status before revision: `BLOCK`.

Applied changes:

- Expanded the post-Phase-1 roadmap into sequenced Milestones 5-9 rather than a
  single broad RFC bucket.
- Added an exact post-main `BoundarySpec` all-PEC acceptance predicate.
- Added branch tracking divergence and untracked-spec status to the baseline.
- Added coaxial, waveguide, TFSF, impedance, wire/extent port, and direct-JIT
  bypass gaps to the hard-fail/test matrix.
- Added future full-roadmap math contracts for all six faces, edge/corner
  interactions, material scaling, energy estimates, CFL/sub-stepping, and
  material limits.
- Added public-doc quarantine/audit deliverables for README, public guide,
  site index, guide index, support matrix, and support-boundaries docs.
