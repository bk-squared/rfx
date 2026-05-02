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
residual-derived coefficients.  The follow-up bounded solver-integration lane
now retains that contract through one private same-call solver-local face helper
in `sbp_sat_3d.py`, with all-six-face `t1/t2` slot-map, normal-sign
orientation, CPML/non-CPML parity, and edge/corner guards covered.  The retained
solver-local call disables scalar projection after the existing SAT stages to
avoid double-coupling.  That remains private fixture-quality-pending evidence
only; it does not promote any public observable.  The subsequent private
boundary coexistence validation confirms direct step-path helper execution under
representative all-PEC, selected PMC, periodic, and bounded-CPML accepted
boundary arguments while keeping unsupported `BoundarySpec` classes hard-fail,
but unchanged fixture-quality replay is still blocked by transverse-uniformity
and vacuum-parity errors.

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

The immediate lane is now private plane-wave transverse phase-coherence
architecture redesign after the operator/mortar implementation lane failed
closed. The private solver-local
skew E/H helper is retained as a bounded
production-context hunk, the accepted boundary subset has direct
helper-execution evidence, and the F0-F4 private
fixture-quality repair ladder
preserves the failing boundary-expanded baseline. The previous analytic source
lanes isolated the old sheet-source phase-front failure, recorded
`private_uniform_plane_wave_source_self_oracle_ready` for a private prototype
uniform plane-wave source self-oracle, and then recorded
`private_uniform_plane_wave_reference_contract_ready` for the private
same-contract uniform reference. The follow-up subgrid-vacuum lane recorded
`private_plane_wave_subgrid_vacuum_fixture_blocked_no_public_promotion`; the
fixture-path wiring lane recorded
`private_plane_wave_fixture_path_wiring_blocked_no_public_promotion`; the
request/spec adapter design lane recorded
`private_runner_plane_wave_adapter_design_ready`; and the adapter implementation
lane recorded `private_plane_wave_adapter_implemented_parity_pending`. The
private subgrid-vacuum parity scoring lane now records
`private_subgrid_vacuum_plane_wave_parity_failed_no_public_promotion`: the
private adapter scores reproducibly, but unchanged transverse phase-spread,
transverse magnitude-CV, vacuum magnitude-error, and vacuum phase-error gates
fail. The follow-up blocker repair/design lane now records
`private_plane_wave_interface_floor_repair_design_required`: the failed parity
packet remains the baseline, source/measurement relabeling is rejected, and the
next executable lane explicitly widened production scope for a private
interface-floor implementation. That implementation lane now records
`no_private_plane_wave_interface_floor_repair`: no bounded local characteristic
or operator-projection-only hunk is accepted as material improvement over the
failed parity packet. The follow-up architecture/root-cause redesign lane now
records `private_plane_wave_interface_energy_form_root_cause_identified`:
source/staggering, projection-only, and fixture-geometry explanations are not
selected, so the next executable unit is an interface energy-form architecture
repair design before any new implementation hunk. That design lane now records
`private_plane_wave_interface_energy_form_implementation_contract_ready`: a
future implementation must preserve the failed baseline, improve the dominant
parity blocker, retain paired-metric guards, run CPML/non-CPML and slow private
metadata checks, and keep public surfaces closed. The follow-up implementation
lane now records `no_private_plane_wave_energy_form_implementation`: the frozen
parity packet remains the baseline, no bounded F1/F2 local solver hunk is
retained as material improvement, and true R/T readiness is still locked. The
follow-up failure theory/design lane now records
`private_plane_wave_energy_form_redesign_contract_ready`: the failed local
implementation attempt is explained as missing an operator-owned,
time-centered interface energy-state contract, and the next bounded unit is a
private operator/mortar time-centered energy-form implementation plan. That
implementation lane now records
`no_private_plane_wave_operator_mortar_energy_form_implementation`: the current
solver staging does not expose a bounded owner for that time-centered interface
energy state. The goal gate therefore remains claims-closed until a transverse
phase-coherence architecture redesign identifies a production-safe ownership
model, a later implementation improves the parity metrics, and a later true R/T
readiness lane proves the hunk improves the real solver path without breaking
CPML/non-CPML symmetry, boundary guards, update/coupling bounds, or the
unchanged `0.02` manufactured-ledger threshold.

No public true R/T, DFT/flux, TFSF, port, S-parameter, API, runner, result,
hook, env/config, default tau, or public observable promotion should be admitted
until boundary coexistence, fixture-quality, true-R/T readiness, and promotion
gates pass without threshold laundering.

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
  the ledger-passing private contract.
- `.omx/plans/ralplan-sbp-sat-private-bounded-solver-integration-after-energy-transfer-ledger-closure.md`
  — private bounded solver-integration plan whose execution retained the
  operator-projected solver-local hunk while keeping public surfaces closed.
