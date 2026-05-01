# SBP-SAT final goal note

Status: internal roadmap note, not a public support claim.

## One-sentence goal

Make SBP-SAT subgridding a production-quality local mesh refinement feature for
`rfx`, where coarse/fine interface coupling is energy-stable, benchmarked with
claims-bearing physical observables, and promoted publicly only for the exact
configurations proven by tests and documentation.

## Intended end state

### 1. Stable coarse/fine coupling

The coarse/fine interface must not be merely visually plausible or proxy-close.
It must satisfy low-level energy-transfer contracts:

- matched coarse/fine projected traces are a no-op;
- zero-work cases do not inject artificial energy;
- SAT update magnitudes remain within declared coupling/update bounds;
- all six face orientations use a consistent tangential component/sign table;
- manufactured energy-ledger residuals pass the unchanged private gate;
- CPML and non-CPML paths share the same interface contract without hooks.

This was the highest-priority private blocker after the solver-integration dry
run: the private solver-integration attempt reached S1 preacceptance, but the
manufactured ledger still failed with `0.16561960778570511 > 0.02`.  The private
operator-projected energy-transfer redesign now records a diagnostic contract
ready state: a ratio-weighted scalar plus skew E/H operator work form closes the
private manufactured ledger below the unchanged `0.02` threshold without
residual-derived coefficients.  That closes the private algebra gate only; it
does not retain a solver hunk or promote any public observable.

### 2. General 3-D refinement box

The long-term feature is not limited to a z-slab fixture.  The intended runtime
surface is an axis-aligned 3-D refinement box with deterministic face, edge, and
corner ownership.  The all-PEC arbitrary-box work is implementation evidence,
but it remains inside the experimental/support-gated envelope until the energy,
boundary, and benchmark gates are satisfied.

### 3. Boundary coexistence

SBP-SAT refinement must coexist with the supported boundary classes only after
explicit coexistence rules and tests pass.  The current roadmap distinguishes:

- all-PEC baseline behavior;
- selected PMC / periodic subsets;
- bounded CPML interior-box coexistence;
- blocked or future classes such as UPML, per-face CPML thickness overrides,
  mixed absorber/reflector classes, and mixed absorber families.

Boundary support must remain tied to the canonical `BoundarySpec`; no parallel
shadow boundary model should become authoritative.

### 4. Claims-bearing true R/T benchmark

Public promotion requires a real physical benchmark, not only point-probe or
proxy numerical equivalence.  The final benchmark surface should provide:

- incident/reflected/transmitted separation;
- calibrated `R(f)` and `T(f)` magnitude and phase;
- energy-balance residuals for appropriate fixtures;
- documented time-window and usable passband selection;
- convergence evidence across at least two resolutions;
- flux, DFT, S-parameter, or equivalent observable contracts that are valid
  under SBP-SAT refinement.

Until these exist and pass, true R/T remains deferred/inconclusive and public
support promotion remains blocked.

### 5. Public promotion discipline

The final public claim should be narrow and evidence-backed: "SBP-SAT
subgridding is supported for these tested configurations, with these boundaries,
observables, and benchmarks."  The feature should not be promoted merely
because a private diagnostic, proxy cross-validation, or one fixture happens to
pass.

Promotion requires synchronized updates to:

- support matrix metadata and tests;
- user-facing support documentation;
- API/runtime hard-fail guards for unsupported combinations;
- benchmark evidence and tolerances;
- regression coverage for solver, boundary, observable, and public-surface
  invariants.

## Current immediate objective

The immediate lane is now the next private solver-integration plan for the
ledger-passing operator-projected energy-transfer contract.  That future lane
must decide whether the private skew E/H work-form contract can be wired into
`sbp_sat_3d.py` as a bounded solver hunk while preserving CPML/non-CPML
symmetry, update/coupling bounds, public-surface closure, and the unchanged
`0.02` manufactured-ledger threshold.

No `sbp_sat_3d.py` production hunk, public true R/T, DFT/flux, TFSF, port,
S-parameter, API, runner, result, hook, env/config, default tau, or public
observable promotion should be admitted until the private manufactured
energy-ledger gate and a separate solver-integration plan both pass without
threshold laundering.

## Related local documents

- `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md` — true R/T benchmark
  requirements and current deferred status.
- `docs/guides/sbp_sat_all_pec_box_refinement_spec.md` — arbitrary all-PEC
  3-D refinement-box contract.
- `docs/guides/sbp_sat_boundary_coexistence_rfc.md` — boundary coexistence
  invariants and blocked/future classes.
- `docs/guides/support_matrix.md` and `docs/guides/support_matrix.json` —
  support status and promotion gate metadata.
- `.omx/plans/ralplan-sbp-sat-private-operator-projected-face-sat-energy-transfer-redesign-after-diagnostic-only-solver-integration-gate-failed.md`
  — historical private energy-transfer redesign plan whose execution produced
  the current ledger-passing private contract.
