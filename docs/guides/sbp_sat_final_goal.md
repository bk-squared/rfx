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

The immediate lane is now private plane-wave observable-proxy modal retry
implementation after the face-packet state hunk was retained. The private
owner state shape now propagates through CPML and non-CPML subgrid steps plus
JIT-runner initialization, and same-step E/H owner scan capture plus
owner-backed joint phase/CV scoring are retained. The physical phase/CV
correction lane failed closed because the retained scorer is face-scalar
diagnostic state; the follow-up architecture lane now records
`private_plane_wave_face_local_modal_correction_contract_ready` by mapping phase
and magnitude owner references to private face-local tangential modal
contracts with paired phase/CV, vacuum-regression, and CPML/non-CPML symmetry
guards. The follow-up implementation lane now records
`no_private_plane_wave_face_local_modal_correction_implementation`: no bounded
field-update hunk is retained because paired phase/CV and vacuum-regression
gates were not honestly improved. The follow-up failure-theory lane now records
`private_plane_wave_face_local_modal_failure_theory_contract_ready`: same-step
diagnostic owner-reference feedback and characteristic-vs-observable
modal-basis mismatch explain the failed hunk, so the next retry requires a
lagged owner reference and observable-aligned modal basis under paired gates.
The implementation retry lane now records
`no_private_plane_wave_face_local_modal_retry_implementation`: lagged owner
timing alone is insufficient, and the claims-bearing transverse plane-DFT
observable cannot be imported into solver-local field updates without a private
observable-proxy modal basis. The architecture lane now records
`private_plane_wave_observable_proxy_modal_retry_contract_ready`: a solver-local
transverse face energy/phase proxy can stand in for the benchmark plane-DFT
distribution without importing the DFT observable into field updates, while
lagged owner state plus paired phase/CV and vacuum guards remain required before
implementation. The implementation lane now records
`no_private_plane_wave_observable_proxy_modal_retry_implementation`: the current
solver owner state stores scalar phase/magnitude references per active face,
while a solver-local modal proxy requires packed face-local distributions with
offsets and masks before any field-update hunk can be retained. The next design
lane must define that packed face-packet state shape without public API, hook, or
benchmark-observable coupling. That design lane now records
`private_plane_wave_proxy_face_packet_state_contract_ready`: packed face-local
proxy buffers, `FACE_ORIENTATIONS`-derived index metadata, and CPML/non-CPML
initialization symmetry can be specified without public API, hooks, benchmark
DFT, or public observable promotion. The implementation lane now records
`private_plane_wave_proxy_face_packet_capture_hunk_retained_fixture_quality_pending`:
the private owner state carries fixed-shape packed face-local proxy references,
offsets, masks, orientation metadata, and CPML/non-CPML/JIT initialization
symmetry. The next implementation lane must retry the private observable-proxy
modal correction using that packed state without public observable promotion.
True R/T readiness is still pending. The private solver-local
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
energy state. The follow-up transverse phase-coherence architecture lane now
records `private_plane_wave_phase_coherence_staging_contract_ready`: a single
private interface-state owner must preserve transverse phase spread and
magnitude CV jointly. The follow-up implementation lane now records
`no_private_plane_wave_phase_coherence_staging_implementation`: K1 would require
a solver-wide interface-state owner that changes solver state shape and scan
staging, while K2 is only a joint scoring guard and cannot repair the solver
path without that owner. The follow-up architecture lane now records
`private_plane_wave_solver_wide_interface_state_owner_contract_ready`: one
private owner and scan/staging state-shape contract are defined for the next
bounded solver hunk. The follow-up implementation lane now records
`no_private_plane_wave_solver_wide_interface_state_owner_implementation`: owner
state shape and same-step scan wiring require a private solver-state propagation
boundary contract for JAX pytree shape and runner/JIT initialization before any
bounded hunk can be retained. The follow-up boundary design lane now records
`private_plane_wave_solver_state_owner_propagation_contract_ready`: state pytree
shape plus runner/JIT initialization boundaries are defined before retrying the
bounded owner implementation. The follow-up implementation lane now records
`private_plane_wave_runner_jit_owner_propagation_hunk_retained_fixture_quality_pending`:
the private owner state shape is propagated through CPML/non-CPML steps and
JIT-runner initialization. The follow-up owner scan-wiring/joint-scoring lane
now records
`private_plane_wave_owner_joint_parity_scoring_hunk_retained_fixture_quality_pending`:
same-step E/H owner scan capture and owner-backed joint phase/CV scoring are
retained. The follow-up physical correction lane now records
`no_private_plane_wave_owner_backed_physical_phase_cv_correction`: no field-
update correction is retained before a face-local modal correction architecture
can prove paired phase/CV and vacuum-regression gates. The architecture lane
has now proven the private contract and the implementation lane has failed
closed without retaining a solver hunk. The failure-theory lane has now
defined the retry contract, and the retry implementation lane has failed closed
without retaining a lagged/observable-aligned hunk. The observable-proxy
architecture lane has now defined the private contract, and the observable-proxy
implementation lane has failed closed on missing packed face-packet state. The
face-packet state-shape design lane has now defined the private contract, and
the face-packet implementation lane has retained the packed state/capture hunk.
The goal gate therefore remains claims-closed until a later modal retry and
readiness lane proves the hunk improves the real solver path without breaking
CPML/non-CPML symmetry, boundary guards, update/coupling bounds, or the unchanged
`0.02` manufactured-ledger threshold.

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
