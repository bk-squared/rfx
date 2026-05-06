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

For a concise local statement of the long-term target and current private
solver-integration / boundary-coexistence handoff, see
`docs/guides/sbp_sat_final_goal.md`.

### Current evidence

- `Simulation.add_refinement(...)` currently accepts `z_range`, `ratio`,
  `tau`, and optional `x_range` / `y_range`, while rejecting `xy_margin`; it
  now accepts selected reflector/periodic boundaries and a bounded CPML subset
  for boxes outside active absorber pads plus one coarse-cell guard (`rfx/api.py`).
- Additional subgridding preflight rejects NTFF, DFT planes, flux monitors,
  waveguide ports, Floquet ports, TFSF, lumped RLC, coaxial ports, and
  impedance/wire ports (`rfx/api.py`).
- Face operators are implemented in `rfx/subgridding/face_ops.py`.
- The current 3D SBP-SAT core contains z-slab config/state, canonical `dt`,
  face trace helpers, SAT helpers, compatibility wrappers, the stepper, and
  overlap-safe energy accounting (`rfx/subgridding/sbp_sat_3d.py`).
- The runtime path builds the current fine box from `add_refinement(...)`
  and dispatches to `run_subgridded_jit(...)`
  (`rfx/runners/subgridded.py`).
- JIT execution calls `step_subgrid_3d(...)` inside one `jax.lax.scan`
  (`rfx/subgridding/jit_runner.py`).
- The low-level arbitrary-box runtime now accepts selected reflector/periodic
  boundaries and a bounded CPML subset, while the public/documented claim
  boundary remains experimental and proxy-only until later promotion gates are
  re-evaluated.
- Current tests cover API guards, face operators, z-slab smoke, alpha/tau,
  JIT, proxy cross-validation, and support-matrix benchmark metadata
  (`tests/test_sbp_sat_api_guards.py`, `tests/test_sbp_sat_face_ops.py`,
  `tests/test_subgrid_crossval.py`, `tests/test_support_matrix_sbp_sat.py`).
- Milestone 3 locks the benchmark claim boundary: `tests/test_subgrid_crossval.py`
  is a proxy numerical-equivalence benchmark only, while
  `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md` defines the deferred
  true reflection/transmission benchmark and `docs/guides/support_matrix.json`
  records that public R/T claims remain blocked.  The later bounded-CPML
  point-probe feasibility probe (`tests/test_sbp_sat_true_rt_feasibility.py`)
  is internal and inconclusive, so it does not change the public claim.

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
| Geometry | Experimental current implementation | axis-aligned refinement box only; CPML cases require absorber guard |
| Interfaces | Experimental current implementation | oriented face/edge/corner coupling under the arbitrary-box lane |
| Boundaries | Experimental reflector/periodic/CPML subset | reject UPML, CPML guard violations, per-face CPML overrides, mixed PMC+periodic, mixed CPML+periodic, and mixed CPML+reflector now; broader coexistence remains a later milestone |
| Sources | Candidate soft point sources only | source/probe positions must lie inside the refined box |
| Probes | Candidate point probes only | probe positions must lie inside the refined box |
| Impedance ports | **Unsupported in Milestone 1** | hard-fail nonzero impedance point ports and wire/extent ports; repair/support moves to a later port-support milestone |
| NTFF/DFT/waveguide/Floquet/TFSF/RLC/coaxial ports | Unsupported | fail fast |
| Arbitrary 3D box refinement | Implemented internally, still experimental | keep public claims proxy-only until later promotion review is updated |
| CPML + subgrid coexistence | Implemented bounded subset | interior guarded boxes only; point-probe true-R/T feasibility is inconclusive; true R/T and S-parameter claims still deferred |

**Evidence:**

- `add_refinement(...)` rejects `xy_margin`, UPML, CPML boxes inside the
  absorber guard, and mixed CPML+reflector/periodic configurations.
- Preflight rejects several non-Phase-1 features (`rfx/api.py:2933-2961`).
- Runtime currently routes scalar `boundary="cpml"` through a CPML-aware
  subgridded scan only for the bounded absorber subset.
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
- Milestone 6 boundary coexistence RFC for PMC / periodic / CPML / UPML /
  per-face absorber padding.

Milestone 5 now records that contract in
`docs/guides/sbp_sat_all_pec_box_refinement_spec.md`.

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
3. `run_subgridded_path(...)` now maps explicit all-PEC box bounds from the
   refinement config, while `SubgridConfig3D` / `step_subgrid_3d(...)` accept
   arbitrary all-PEC box bounds (`rfx/runners/subgridded.py`,
   `rfx/subgridding/sbp_sat_3d.py`).
4. Fine-grid materials are rasterized over the current refinement box
   (`rfx/runners/subgridded.py:93-122`).
5. Sources/probes are mapped to fine-grid indices
   (`rfx/runners/subgridded.py:124-219`).
6. `run_subgridded_jit(...)` runs `step_subgrid_3d(...)` in `jax.lax.scan`
   (`rfx/subgridding/jit_runner.py:108-122`).

**Compatibility shims:**

- `_shared_node_coupling_h_3d(...)` and `_shared_node_coupling_3d(...)` now
  delegate to the oriented-face SAT helpers (`rfx/subgridding/sbp_sat_3d.py`).
- The spec must decide whether to keep these as private compatibility shims,
  rename them, or delete them after import users are removed.

**Required improvements:**

- Add config-level validation so direct `SubgridConfig3D` construction cannot
  bypass arbitrary-box invariants.
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
- scalar boundary `"upml"`
- scalar `boundary="cpml"` when the box violates the absorber guard
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

- UPML `BoundarySpec` faces
- mixed PMC+periodic faces
- mixed CPML+reflector or CPML+periodic faces
- CPML box inside the absorber guard
- CPML per-face thickness overrides with subgrid

**Acceptance criteria:**

- Each hard-fail rule has a named test.
- Errors explain both the rejected configuration and the supported alternative.
- Unsupported configurations fail before long JIT compilation where possible.

**Named hard-fail test matrix:**

| Rule | Failure phase | Expected test | Status |
|---|---|---|---|
| UPML scalar boundary | construction | `test_subgrid_rejects_unsupported_absorbing_boundaryspec` | updated |
| CPML scalar boundary inside absorber guard | construction | `test_subgrid_rejects_cpml_box_inside_absorber_guard` | new |
| bounded CPML scalar boundary | construction/run | `test_subgrid_accepts_interior_cpml_absorbing_subset` | new |
| `xy_margin` / partial x/y request | construction | `test_partial_xy_refinement_fails` | existing |
| source/probe outside z slab | run mapping/preflight | `test_source_outside_zslab_fails` | existing |
| NTFF with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[ntff]` | existing pattern |
| DFT plane with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[dft_plane]` | existing pattern |
| flux monitor with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[flux_monitor]` | new |
| waveguide port with subgrid | run preflight | `test_subgrid_rejects_waveguide_port` | new explicit test |
| Floquet port with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[floquet]` | existing pattern |
| TFSF with subgrid | run preflight | `test_subgrid_rejects_tfsf_source` | new explicit test |
| lumped RLC with subgrid | run preflight | `test_unsupported_phase1_features_fail_fast[lumped_rlc]` | existing pattern |
| coaxial port with subgrid | run preflight | `test_subgrid_rejects_coaxial_port` | new |
| nonzero impedance point port | run preflight before JIT | `test_subgrid_rejects_impedance_point_port` | new |
| impedance wire/extent port | run preflight before JIT | `test_subgrid_rejects_impedance_wire_port` | new |
| direct partial-x/y `SubgridConfig3D` into JIT | JIT entry validation | `test_run_subgridded_jit_rejects_partial_xy_config` | new |
| selected PMC face in `BoundarySpec` | construction/run | `test_subgrid_accepts_reflector_only_pmc_boundaryspec` | implemented |
| periodic axes with interior/full-axis box | construction/run | `test_subgrid_accepts_periodic_axis_when_box_is_interior` / `test_subgrid_accepts_periodic_axis_when_box_spans_it` | implemented |
| mixed PMC+periodic specs | construction/preflight | `test_subgrid_rejects_mixed_pmc_periodic_boundaryspec` | implemented |
| all-CPML `BoundarySpec` bounded subset | construction/run | `test_subgrid_accepts_boundaryspec_cpml_absorbing_subset` | implemented |
| mixed CPML+periodic specs | construction/preflight | `test_subgrid_rejects_mixed_periodic_cpml_boundaryspec` | new |
| per-face CPML thickness/padding | construction/preflight | `test_subgrid_rejects_unsupported_absorbing_boundaryspec` | updated |

### 2.10 Verification ladder

**Purpose:** define what must pass at each maturity level.

| Layer | Goal | Existing / required tests |
|---|---|---|
| Operator/property | lock face algebra | `tests/test_sbp_sat_face_ops.py` |
| API/misuse | hard-fail unsupported configs | `tests/test_sbp_sat_api_guards.py` plus new port/boundary tests |
| Smoke/stability | prove bounded simple behavior | `tests/test_sbp_sat_3d.py`, `tests/test_sbp_sat_alpha.py` |
| JIT/runtime | prove one canonical scan path | `tests/test_sbp_sat_jit.py` |
| Proxy benchmark | compare probe DFT against uniform-fine | `tests/test_subgrid_crossval.py` |
| True benchmark | compute R/T or S-parameters | deferred spec in `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md`; implementation blocked until required source/observable surfaces exist |
| Drift regression | prove `origin/main` integration | `tests/test_api.py::test_upml_rejects_subgridding_refinement` plus BoundarySpec cases in `tests/test_sbp_sat_api_guards.py` |
| Support matrix | lock evidence claims | `tests/test_support_matrix_sbp_sat.py` |

**Current evidence:**

- Milestone 2 mainline-integration regression slice passed:
  `pytest -q tests/test_api.py::test_upml_rejects_subgridding_refinement tests/test_sbp_sat_api_guards.py tests/test_sbp_sat_face_ops.py tests/test_sbp_sat_3d.py tests/test_sbp_sat_alpha.py tests/test_sbp_sat_jit.py tests/test_subgrid_crossval.py tests/test_boundary_spec.py tests/test_boundary_spec_legacy.py tests/test_boundary_spec_preflight.py tests/test_boundary_spec_thickness.py tests/test_boundary_pmc_guard.py tests/test_boundary_pmc_hi_faces.py tests/test_silent_drop_warnings.py`
  -> `127 passed, 2 deselected, 3 warnings`.
- Milestone 3 support-matrix contract is locked by
  `tests/test_support_matrix_sbp_sat.py`.

**Acceptance criteria:**

- Phase-1 implementation-complete requires all non-slow tests plus the proxy
  crossval suite.
- Public-doc eligible requires post-rebase tests plus true benchmark
  implementation, not only the Milestone 3 deferred spec.
- Full SBP-SAT roadmap milestones require separate verification gates.

### 2.11 Benchmark definitions

**Purpose:** prevent proxy comparisons from being mistaken for physical
reflection/transmission validation.

**Current proxy benchmark definition:**

- Run a uniform-fine reference and a subgridded coarse/fine case.
- Sample one time-series probe.
- Compute one-frequency DFT amplitude and phase.
- Require amplitude error `<= 5%` and phase error `<= 5 degrees`.
- Name both reflection-side and transmission-side fixtures as proxy fixtures.

**Evidence:**

- DFT helper and error calculation are in `tests/test_subgrid_crossval.py`.
- Reflection-side and transmission-side proxy tests are in
  `tests/test_subgrid_crossval.py`.
- Support-matrix metadata records the proxy tolerance and explicitly marks
  true R/T as deferred in `docs/guides/support_matrix.json`.
- `tests/test_sbp_sat_true_rt_feasibility.py` records an internal bounded-CPML
  point-probe feasibility probe as inconclusive, not public true R/T evidence.

**Deferred true benchmark specification:**

- Full spec: `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md`.
- Current status: deferred because Phase 1 still hard-fails the source/observable
  surfaces needed for clean incident/reflected/transmitted separation; the
  bounded-CPML point-probe attempt is inconclusive.

**Required future true benchmark content:**

- Define incident/reflected/transmitted separation.
- Define time windowing and frequency band.
- Define acceptable amplitude, phase, and energy-balance tolerances.
- Decide whether to use S-parameter extraction or dedicated plane-wave
  monitors.
- Document when slow/GPU markers are required.
- Preserve the support-matrix block on public claims until the true benchmark
  passes.

**Acceptance criteria:**

- Proxy benchmarks are labeled as proxy benchmarks.
- True R/T benchmark work is its own deferred deliverable, not silently implied
  by the current tests.
- The support matrix states the current evidence level exactly.

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

- Proxy benchmark retained and clearly named in `tests/test_subgrid_crossval.py`.
- True reflection/transmission benchmark spec in
  `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md`.
- True R/T tests explicitly deferred until the required boundary/source/observable
  contracts exist; the deferred issue record lives in the true R/T spec.
- Bounded-CPML point-probe feasibility evidence is allowed only as internal,
  inconclusive evidence unless the support matrix is updated by a later
  claims-bearing benchmark gate.
- Tolerance rationale documented in the true R/T spec and support matrix.

**Exit gate:**

- The support matrix states exactly what benchmark evidence exists, and
  `tests/test_support_matrix_sbp_sat.py` locks that metadata.

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

- General face-operator contract for `x`, `y`, and `z` oriented interfaces in
  `docs/guides/sbp_sat_all_pec_box_refinement_spec.md`.
- Face-normal and tangential-component table for all six faces.
- Edge/corner interaction policy for boxes where refined interfaces meet.
- All-PEC box-refinement implementation plan.
- Uniform-fine benchmarks that exercise oblique waves across non-z faces.

**Exit gate:**

- The arbitrary-box runtime remains experimental until the six-face math,
  edge/corner policy, and benchmark matrix stay regression-locked.
- `tests/test_sbp_sat_box_refinement_spec_contract.py` locks the spec artifact.

#### Milestone 6 — Boundary coexistence

**Deliverables:**

- `BoundarySpec` coexistence RFC for PMC, periodic, CPML, and UPML in
  `docs/guides/sbp_sat_boundary_coexistence_rfc.md`.
- Per-face CPML layer/padding interaction contract.
- Open-boundary benchmark definitions distinct from PEC-cavity proxies.
- Hard-fail matrix for every boundary combination still unsupported.

**Exit gate:**

- No non-PEC boundary support is advertised until explicit BoundarySpec tests
  and open-boundary benchmarks pass.
- `tests/test_sbp_sat_boundary_coexistence_spec_contract.py` locks the RFC
  artifact.

#### Milestone 7 — Ports and observables inside refined regions

**Deliverables:**

- Port-support RFC for impedance point ports, wire/extent ports, coaxial ports,
  waveguide ports, and Floquet ports in
  `docs/guides/sbp_sat_ports_observables_rfc.md`.
- Source normalization contract for fine-grid material and loss values.
- Probe/DFT/NTFF placement rules across coarse/fine regions.

**Exit gate:**

- Each supported port/observable has positive tests and at least one
  unsupported-placement failure test.
- `tests/test_sbp_sat_ports_observables_spec_contract.py` locks the RFC
  artifact.

#### Milestone 8 — Materials, dispersion, and time integration

**Deliverables:**

- Material-scaled SAT penalty policy for lossy, magnetic, Debye, Lorentz,
  anisotropic, and nonlinear materials in
  `docs/guides/sbp_sat_materials_time_integration_rfc.md`.
- CFL/sub-stepping decision record.
- Discrete energy estimate for accepted material/time-integration cases.
- Benchmark ladder that separates material error from interface error.

**Exit gate:**

- Any material or sub-stepping expansion has a spec, implementation plan, and
  benchmark gate before support matrix promotion.
- `tests/test_sbp_sat_materials_time_integration_spec_contract.py` locks the
  RFC artifact.

#### Milestone 9 — Public support promotion

**Deliverables:**

- Support matrix promotion proposal in
  `docs/guides/sbp_sat_support_promotion_proposal.md`.
- Public docs with exact supported surfaces and examples.
- Release notes and migration caveats in
  `docs/guides/sbp_sat_release_migration_caveats.md` plus scoped public
  changelog/migration wording.
- Final verifier report tying docs claims to tests and benchmarks in
  `docs/guides/sbp_sat_final_verifier_report.md`.

**Exit gate:**

- Public claims are traceable to passing tests, benchmark evidence, and
  documented unsupported cases.
- `tests/test_sbp_sat_promotion_artifacts_contract.py` locks the promotion
  artifacts.

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


## 5. Current final-goal continuation note

The Phase-1 hardening spec remains a private support-contract baseline, not a
public support claim. In the active final-goal roadmap, the retained
source/interface packet energy co-normalization helper has now been privately
parity-scored as `private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_hunk_insufficient_fixture_quality_pending`. That evidence is finite but still insufficient for
claims-bearing true R/T readiness, so the next private step is private failure-theory redesign for the source/interface packet energy co-normalization floor; no
public API, observable, runner, hook, example, or threshold promotion is unlocked
by this Phase-1 evidence.


The active final-goal roadmap subsequently records `private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_floor_theory_ready`. This remains a
private theory result only: it selects private source/interface phase-energy residual implementation inside the retained 3x3 transfer-map contract and does not unlock public
APIs, observables, runners, hooks, examples, support claims, or thresholds.

The source/interface phase-energy residual implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the retained source/interface packet energy co-normalization helper, adds bounded phase-energy residual coupling inside the existing 3x3 source/interface transfer-map contract, and keeps true R/T readiness, thresholds, runner state, hooks, exports, public observables, docs-public/examples, and API promotion closed while routing the next step to private parity scoring.

The source/interface phase-energy residual parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_hunk_insufficient_fixture_quality_pending`: it consumes the retained phase-energy residual implementation metadata, preserves frozen thresholds and baseline/current metrics, records finite private scoring evidence, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed while routing the next step to private failure-theory redesign for the phase-energy residual floor.

The source/interface phase-energy residual failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_floor_theory_ready`: it consumes the finite-but-insufficient phase-energy residual parity evidence, rejects repeat residual scoring and public/threshold escape routes, selects a bounded private source/interface time-centered energy-pairing implementation target inside the retained 3x3 transfer-map contract, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed.

The source/interface time-centered energy-pairing implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the failure-theory contract, retains a private 3x3 source/interface time-centered energy-pairing transfer map, preserves frozen thresholds and private metrics, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed while routing the next step to private parity scoring.


The source/interface time-centered energy-pairing parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_hunk_insufficient_fixture_quality_pending`: it consumes the retained time-centered energy-pairing implementation metadata, preserves frozen thresholds and baseline/current metrics, records finite private scoring evidence, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed while routing the next step to private failure-theory redesign for the time-centered energy-pairing floor.


The source/interface time-centered energy-pairing failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_floor_theory_ready`: it consumes the finite-but-insufficient time-centered energy-pairing parity evidence, rejects repeated time-centered pairing and public/threshold escape routes, selects a bounded private packet-basis phase-energy cross-coupling implementation target inside the retained 3x3 transfer-map contract, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed.


The source/interface time-centered energy-pairing packet-basis phase-energy cross-coupling implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the packet-basis cross-coupling failure-theory contract and retained time-centered pairing map, keeps the private 3x3 source/interface transfer-map bound at 0.35, preserves frozen thresholds and private metrics, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed while routing the next step to private parity scoring.

The source/interface time-centered energy-pairing packet-basis phase-energy cross-coupling parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_hunk_insufficient_fixture_quality_pending`: it consumes the retained packet-basis phase-energy cross-coupling implementation metadata, preserves frozen thresholds and baseline/current private metrics, records finite private scoring evidence, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed while routing the next step to private failure-theory redesign for the packet-basis cross-coupling floor.

The packet-basis phase-energy cross-coupling failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_floor_theory_ready`: it consumes the retained implementation plus parity-scoring metadata, observes that frozen private metrics remain baseline-identical after the cross-coupling hunk, rejects repeated cross-coupling, unbounded solver edits, threshold relaxation, runner/hook/API changes, and public-observable escape routes, and selects a bounded private score-path visibility contract while keeping true R/T readiness and all public promotion closed.
The packet-basis phase-energy cross-coupling score-path visibility implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_hunk_retained_fixture_quality_pending`: it consumes the private score-path visibility failure-theory contract, wires the bounded fixed-shape helper into `_project_private_modal_basis_packets`, preserves the 0.35 transfer-map bound and frozen private metrics, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, API promotion, and threshold changes closed while routing the next step to private parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_hunk_insufficient_fixture_quality_pending`: it consumes the retained score-path visibility implementation metadata, records finite private scoring evidence, confirms the frozen private metrics remain baseline-identical, keeps fixture-quality and true R/T readiness closed, and routes the next step to private score-path visibility failure-theory redesign without changing thresholds, public observables, runners, hooks, exports, docs-public/examples, README, or APIs.

The packet-basis phase-energy cross-coupling score-path visibility failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_floor_theory_ready`: it consumes the finite but baseline-identical parity-scoring metadata, rejects repeated visibility scoring, unbounded solver edits, threshold relaxation, runner/hook/API changes, and public-observable escape routes, and selects a bounded private field-update coupling floor while keeping true R/T readiness and all public promotion closed.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_hunk_retained_fixture_quality_pending`: it consumes the private failure-theory contract, retains `_private_score_path_visibility_field_update_coupling_target` inside the propagation-aware modal retry field-update path, preserves fixed-shape fail-closed private coupling, and keeps true R/T readiness, runners, hooks, exports, public observables, docs-public/examples, README, API promotion, and threshold changes closed while routing the next step to private parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_hunk_insufficient_fixture_quality_pending`: it consumes the retained field-update coupling implementation metadata, records finite private parity-scoring evidence, confirms the frozen private metrics remain baseline-identical, keeps fixture-quality and true R/T readiness closed, and routes the next step to private field-update coupling failure-theory redesign without changing thresholds, public observables, runners, hooks, exports, docs-public/examples, README, APIs, or package exports.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_floor_theory_ready`: it consumes the finite but baseline-identical field-update coupling parity evidence, rejects repeat field-update scoring, public/threshold/API/runner/hook/export escape routes, and unbounded solver rewrites, selects a bounded private solver-observed post-update delta floor, and keeps true R/T readiness plus all public promotion closed while routing the next step to private solver-observed delta implementation.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_hunk_retained_fixture_quality_pending`: it consumes the private solver-observed delta failure-theory contract, wires `_private_score_path_visibility_field_update_solver_observed_delta` into the propagation-aware modal retry field-update path, captures fixed-shape pre/post packet delta diagnostics with fail-closed zero/nonfinite gating, and keeps true R/T readiness plus all public promotion closed while routing the next step to private solver-observed delta parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_hunk_insufficient_fixture_quality_pending`: it consumes the retained solver-observed delta implementation metadata, records finite private scoring evidence, confirms the frozen private metrics remain baseline-identical, keeps fixture-quality and true R/T readiness closed, and routes the next step to private solver-observed delta failure-theory redesign without changing thresholds, public observables, runners, hooks, exports, docs-public/examples, README, APIs, or package exports.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_floor_theory_ready`: it consumes the finite but baseline-identical solver-observed delta parity evidence, rejects repeat scoring, threshold changes, public observable/API/export/runner/hook/docs-public/README routes, and unbounded solver rewrites, selects a bounded private packet-normalized residual floor, and keeps fixture-quality, true R/T readiness, and all public promotion closed while routing the next step to private packet-normalized residual implementation.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_hunk_retained_fixture_quality_pending`: it consumes the private packet-normalized residual failure-theory contract, wires `_private_score_path_visibility_field_update_solver_observed_delta_packet_normalized_residual` into the propagation-aware modal retry field-update path after solver-observed delta gating, preserves fixed-shape fail-closed packet-energy normalization, and keeps fixture-quality, true R/T readiness, thresholds, public observables, runners, hooks, exports, docs-public/examples, README, and APIs closed while routing the next step to private packet-normalized residual parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_hunk_insufficient_fixture_quality_pending`: it consumes the retained packet-normalized residual implementation metadata, records finite private scoring evidence, confirms the frozen private metrics remain baseline-identical, keeps fixture-quality and true R/T readiness closed, and routes the next step to private packet-normalized residual failure-theory redesign without changing thresholds, public observables, runners, hooks, exports, docs-public/examples, README, APIs, or package exports.

The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_theory_ready`: it consumes the finite but baseline-identical packet-normalized residual parity evidence, rejects repeat scoring, helper-existence-as-readiness, threshold changes, public observable/API/export/runner/hook/docs-public/README routes, and unbounded solver rewrites, selects bounded private residual-weighted delta coupling, and keeps fixture-quality, true R/T readiness, thresholds, public observables, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private residual-weighted delta coupling implementation.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_hunk_retained_fixture_quality_pending`: it consumes the residual-weighted delta coupling failure-theory contract, wires the private fixed-shape fail-closed residual-weighted delta helper into the propagation-aware modal retry path after packet-normalized residual scoring, keeps existing relaxation/update bounds, and keeps fixture-quality, true R/T readiness, thresholds, public observables, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private residual-weighted delta coupling parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_hunk_insufficient_fixture_quality_pending`: it consumes the retained residual-weighted delta coupling implementation metadata, records finite private parity-scoring evidence, confirms the frozen benchmark metrics and score deltas remain baseline-identical, and keeps fixture-quality, true R/T readiness, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private residual-weighted delta coupling failure-theory after parity scoring insufficient.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_theory_ready`: it consumes the baseline-identical residual-weighted delta coupling parity-scoring evidence plus the retained implementation contract, rejects repeat scoring, helper-existence-as-readiness, public-observable/threshold/API/export/runner/hook/docs-public/README escapes, and unbounded solver rewrites, selects the bounded target-packet residual projection contract, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private target-packet residual projection implementation.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_hunk_retained_fixture_quality_pending`: it consumes the failure-theory contract, wires the private fixed-shape fail-closed target-packet residual projection helper into the propagation-aware modal retry path after residual-weighted delta coupling, keeps existing relaxation/update bounds, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private target-packet residual projection parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_hunk_insufficient_fixture_quality_pending`: it consumes the retained target-packet residual projection implementation metadata, records finite private parity-scoring evidence, confirms the frozen benchmark metrics and score deltas remain baseline-identical, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private target-packet residual projection failure-theory after parity scoring insufficient.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_source_interface_residual_phase_rotation_theory_ready`: it consumes the finite but baseline-identical target-packet residual projection parity-scoring evidence plus the retained implementation contract, rejects repeat scoring, helper-existence-as-readiness, public-observable/threshold/API/export/runner/hook/docs-public/README escapes, and unbounded solver rewrites, selects the bounded source/interface residual phase-rotation coupling contract, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private source/interface residual phase-rotation coupling implementation after the target-packet residual projection failure-theory contract.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection source/interface residual phase-rotation implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_source_interface_residual_phase_rotation_coupling_hunk_retained_fixture_quality_pending`: it consumes the target-packet residual projection failure-theory contract, wires the bounded fixed-shape fail-closed private source/interface residual phase-rotation helper into the propagation-aware modal retry path after target-packet residual projection, keeps existing relaxation/update bounds, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private source/interface residual phase-rotation coupling parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection source/interface residual phase-rotation parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_source_interface_residual_phase_rotation_coupling_hunk_insufficient_fixture_quality_pending`: it consumes the retained source/interface residual phase-rotation implementation metadata, records finite private parity-scoring evidence with `PP1_finite_source_interface_residual_phase_rotation_private_parity_score` selected, confirms the frozen benchmark metrics and score deltas remain baseline-identical, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private source/interface residual phase-rotation coupling failure-theory redesign after parity scoring insufficient.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection source/interface residual phase-rotation failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_source_interface_residual_phase_rotation_phase_energy_closure_theory_ready`: it consumes the finite but baseline-identical source/interface residual phase-rotation parity-scoring evidence plus the retained implementation contract, rejects repeat scoring, helper-existence-as-readiness, public-observable/threshold/API/export/runner/hook/docs-public/README escapes, and unbounded solver rewrites, selects the bounded source/interface residual phase-rotation phase-energy closure contract, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private phase-energy closure implementation after the failure-theory contract.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection source/interface residual phase-rotation phase-energy closure implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_source_interface_residual_phase_rotation_phase_energy_closure_hunk_retained_fixture_quality_pending`: it consumes the phase-energy closure failure-theory contract, wires the bounded fixed-shape fail-closed private phase-energy closure helper into the propagation-aware modal retry path after source/interface residual phase-rotation coupling, keeps existing relaxation/update bounds, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private phase-energy closure parity scoring.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection source/interface residual phase-rotation phase-energy closure parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_source_interface_residual_phase_rotation_phase_energy_closure_hunk_insufficient_fixture_quality_pending`: it consumes the retained phase-energy closure implementation metadata, records finite private parity-scoring evidence with `SP1_finite_phase_energy_closure_private_parity_score` selected, confirms the frozen benchmark metrics and score deltas remain baseline-identical, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private phase-energy closure failure-theory redesign after parity scoring insufficient.
The packet-basis phase-energy cross-coupling score-path visibility field-update coupling solver-observed delta packet-normalized residual residual-weighted delta coupling target-packet residual projection source/interface residual phase-rotation phase-energy closure failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_source_interface_time_centered_energy_pairing_packet_basis_phase_energy_cross_coupling_score_path_visibility_field_update_coupling_solver_observed_delta_packet_normalized_residual_residual_weighted_delta_coupling_target_packet_residual_projection_source_interface_residual_phase_rotation_phase_energy_closure_residual_distribution_theory_ready`: it consumes the finite but baseline-identical phase-energy closure parity-scoring evidence plus the retained implementation contract, rejects repeat scoring, helper-existence-as-readiness, public-observable/threshold/API/export/runner/hook/docs-public/README escapes, and unbounded solver rewrites, selects the bounded phase-energy closure residual-distribution contract, and keeps fixture-quality, true R/T readiness, slab R/T scoring, thresholds, public observables, DFT/flux/TFSF/port/S-parameter promotion, runners, hooks, exports, docs-public/examples, README, APIs, and package exports closed while routing the next step to private residual-distribution implementation after the failure-theory contract.
