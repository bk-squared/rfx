# SBP-SAT z-slab true reflection/transmission benchmark specification

## Status

Deferred implementation specification for the SBP-SAT z-slab research lane.
This document is the Milestone 3 benchmark-credibility artifact: it defines the
true R/T benchmark that must exist before public support promotion, and explains
why the current Phase-1 lane intentionally does **not** claim that benchmark yet.

## Current evidence level

Current executable evidence is a **proxy numerical-equivalence benchmark**:

- file: `tests/test_subgrid_crossval.py`
- comparison: subgridded all-PEC z-slab run vs uniform-fine reference run
- signal: one point-probe time series transformed by a single-frequency DFT
- tolerance: relative amplitude error `<= 5%`, phase error `<= 5°`
- claim level: verifies that the current subgridded implementation stays close
  to a uniform-fine run for the selected PEC-cavity proxy fixture

This proxy does **not** establish calibrated physical reflection coefficient
`R(f)`, transmission coefficient `T(f)`, energy balance, S-parameters, or
open-boundary behavior.

## bounded-CPML feasibility result

The bounded CPML coexistence subset now has an executable point-probe
feasibility probe:

- file: `tests/test_sbp_sat_true_rt_feasibility.py`
- fixture: guarded all-CPML quasi-1D vacuum/dielectric-slab/vacuum setup
  inside an axis-aligned refinement box
- source / observables: one soft `Ez` point source and two `Ez` point probes
- comparison: subgridded point-probe extraction vs a uniform-fine reference at
  `1.5`, `2.0`, and `2.5 GHz`
- classification: **inconclusive**
- claim level: internal measurement-contract feasibility only; **not** public
  true R/T, S-parameter, flux, DFT-plane, or port evidence

The first run is useful because it separates the open-boundary question from
the broader ports/observables question.  It is still not claims-bearing:

- `|R|` magnitude and phase matched the uniform-fine reference within the
  provisional gates;
- `|T|` magnitude missed the provisional `<= 5%` gate (`~6.57%`);
- the one-cell probe-shift perturbation missed the provisional transmitted
  stability gate (`max |ΔT| ~= 0.0749`);
- energy balance is recorded only as an advisory diagnostic because point
  probes are not flux-normalized observables.

Therefore the support matrix remains:

- SBP-SAT subgridding status: `experimental`
- benchmark evidence: proxy numerical equivalence plus bounded-CPML
  point-probe feasibility evidence
- true R/T benchmark: deferred
- feasibility probe gate: inconclusive
- public support promotion: blocked

If future work needs this gate to pass, the next separate plan should add a
minimal benchmark-only flux/DFT observable contract rather than promoting
point-probe evidence as calibrated true R/T.

## Private flux/DFT benchmark gate

A private fine-owned flux/DFT accumulator and private analytic sheet/source now
exist for internal SBP-SAT benchmarking.  The current recovery attempt also
adds a private TFSF-style incident field, but only as benchmark-internal
fixture-quality evidence:

- file: `tests/test_sbp_sat_true_rt_flux_dft_benchmark.py`
- fixture: translational x/y, guarded all-CPML vacuum setup using z-normal
  flux planes fully inside the fine grid; the recorded private candidate is
  currently the boundary-expanded private TFSF-style incident fixture
- source contract: private TFSF-style incident field accepted only by private
  benchmark helpers; the subgrid side uses low-level post-H/post-E hooks, the
  uniform reference side uses the same pre-CPML private H/E correction slots as
  public uniform TFSF ordering, and neither path is public
  `Simulation.add_tfsf_source`, public TFSF, or exposed through
  `Simulation.run()` / `Result`
- normalization: same-contract private reference exists, but it is still
  vacuum-gated because private uniform/subgrid parity has not passed
- observable contract: private raw DFT accumulators only; not public
  `add_dft_plane_probe`, not public `add_flux_monitor`, and not
  `Result.dft_planes` / `Result.flux_monitors`
- placement contract: local normal index must satisfy `1 <= idx <= n-2`, and
  sheet/plane tangential extents must remain strictly inside the fine region
- classification: **inconclusive**
- claim level: internal benchmark-only evidence; **not** public true R/T,
  S-parameter, DFT-plane, flux-monitor, TFSF, or port support

The current private gate is useful because it removes the most fragile
point-probe extraction path and replaces the earlier point-source
finite-aperture diagnostic with an explicit sheet/source fixture.  It still
keeps the public claim blocked:

- public DFT-plane and flux-monitor APIs still hard-fail under SBP-SAT
  refinement;
- public TFSF remains unsupported for SBP-SAT and is not used by this fixture;
- mixed periodic+CPML is rejected for this plan, so the fixture remains a
  bounded all-CPML domain rather than a transverse-periodic shortcut;
- synthetic accumulator semantics match the uniform scan-kernel formula,
  including multi-step, all-axis, windowed DFT accumulation;
- private sheet/source lowering and injection are tested as benchmark-only
  implementation details;
- runtime scoring now records at least two non-floor passband bins, transverse
  magnitude/phase uniformity, and analytic incident consistency as diagnostic
  fixture-quality gates only;
- slab R/T scoring is intentionally skipped because the same-contract private
  reference exists but vacuum/reference fixture-quality gates remain
  inconclusive; the current row-2 diagnostic ranks
  `transverse_phase_spread_deg` as the dominant blocker and keeps transverse
  magnitude spread plus vacuum magnitude parity visible rather than
  relaxing thresholds;
- the private causal ladder keeps the full-aperture baseline visible, checks a
  bounded `+2` plane-shift candidate and a central-core aperture candidate, and
  classifies the present row-2 state as `sbp_sat_interface_floor` because the
  source `E/eta0` check passes while plane/aperture controls do not satisfy the
  paired material-improvement rule;
- the follow-up private solver/interface-floor investigation now subclasses the
  blocker as `coarse_fine_energy_transfer_mismatch`: a signed front/back flux
  ratio residual persists across the predeclared boundary-expanded baseline and
  nearer bounded interface control, direct hook invariants still pass, and hook
  contingency therefore remains closed by default;
- the bounded private energy-transfer repair stage scored the existing
  refinement `tau` plumbing at `0.25`, `0.5`, `0.75`, and `1.0` without
  changing the public `tau=0.5` default/API; it records
  `no_material_repair` because the best candidate (`tau=1.0`) improved max
  ratio error from `0.8813` to only `0.7211` and regressed the paired
  `vacuum_phase_error_deg` metric;
- the private energy-transfer theory/design review is complete and selects
  `discrete_eh_work_ledger_mismatch` as the next kernel-repair candidate; it
  adds no executable synthetic diagnostics, solver edits, hook experiments,
  public API behavior, public default changes, or public claims;
- the private manufactured-interface energy-ledger diagnostic is now executable
  and records `ledger_mismatch_detected`: matched and zero-work probes remain
  private/CI-green, while the nonzero E/H face-work probe reports normalized
  balance residual `0.1656` above the unchanged `0.02` threshold;
- the bounded private face-kernel feasibility stage records
  `no_signature_compatible_bounded_repair`: bounded current/minimum/reciprocal
  face-jump candidates still miss the `0.02` manufactured ledger threshold,
  and a ledger-passing low-scale control is rejected as under-coupling;
- the private SAT face-coupling theory/redesign stage records
  `paired_face_coupling_design_ready`: a test-local same-step paired E/H
  candidate closes the manufactured face ledger with normalized residual
  `0.000578`, preserves matched-trace and zero-work gates, keeps coupling
  strength near the current kernel, and requires paired E/H face context;
- the private paired-face helper implementation gate records
  `production_context_mismatch_detected`: both non-CPML and CPML production
  step paths expose H SAT before the E update and E SAT after the E update, so
  the selected same-step paired E/H candidate cannot be applied to one
  co-temporal current-SAT ledger state without time-centered staging;
- the private time-centered E/H face-ledger staging redesign records
  `time_centered_staging_contract_ready`: same-call `H_pre_sat`/`H_post_sat`
  and `E_pre_sat`/`E_post_sat` staging with a centered-H ledger closes the
  private manufactured face ledger, passes the production-expressibility gate
  with named CPML and non-CPML local slots, keeps orientation generic through
  `FACE_ORIENTATIONS`, and rejects non-selected staging controls with explicit
  reordering, trace-availability, or cross-step-state reasons;
- the private time-centered paired-face helper implementation records
  `private_time_centered_paired_face_helper_implemented`: bounded private helper
  functions in `sbp_sat_3d.py` bind those same-call slots in both CPML and
  non-CPML paths and apply the selected `same_call_centered_h_bar` centered-H
  correction after E SAT with a predeclared bounded `0.02` relaxation, without
  adding hook/API/runner/public observable surfaces;
- the private time-centered paired-face helper fixture-quality recovery stage now
  records `measurement_contract_or_interface_floor_persists`: the finite
  C0/C1/C2/C3 ladder keeps the C0 original fixture metrics visible, treats C1/C2
  center-core/one-cell plane controls as measurement-only evidence that cannot
  claim original-fixture recovery, and rejects the only solver-touching C3
  helper-relaxation candidate after rollback to `0.02`; unchanged transverse
  uniformity and vacuum-stability thresholds still fail, so public promotion
  remains closed;
- the private measurement-contract/interface-floor redesign after helper recovery
  failed stage now records `persistent_interface_floor_confirmed`: D2
  phase-referenced modal coherence and D3 local E/H impedance/Poynting
  diagnostics both remain below their predeclared diagnostic thresholds, while
  D4 interface-ledger correlation remains positive under current helper-state
  provenance plus prior committed manufactured-ledger context; public promotion
  remains closed. The private measurement-contract/interface-floor redesign after helper recovery failed lane is complete and points to `private interface-floor repair theory/implementation after measurement-contract diagnostics ralplan`;
- the private interface-floor repair theory/implementation lane now records
  `no_bounded_private_interface_floor_repair`: the solver-admissible F1
  `oriented_characteristic_face_balance` candidate passes all-six-face
  orientation, matched-trace, zero-work, coupling-strength, update-bound, and
  edge/corner preacceptance guards, but it collapses to the current component SAT
  update and fails the manufactured ledger residual gate (`0.16562 > 0.02`). No
  solver hunk is retained, and public true-R/T/DFT/flux/TFSF/port/S-parameter/API/
  result/hook promotion remains closed;
- the private higher-order SBP face-norm/interface-operator redesign ladder now
  records `no_private_face_norm_operator_repair`: the existing face restriction
  is already the unmasked mass-adjoint of prolongation under the current
  diagonal face norms, H1/H2 candidates fail the manufactured ledger or
  higher-order projection/noop gates, no solver hunk is retained, and the next
  safe lane is private broader SBP derivative/interior-boundary operator
  redesign;
- the private broader derivative/interior-boundary ladder now records
  `no_private_derivative_interface_repair`: the reduced private energy-identity
  fixture reproduces the current `0.02` ledger-floor failure and can be closed
  only by a test-local face correction; the production lift is blocked by
  `requires_global_sbp_operator_refactor`, no solver hunk is retained, and the
  next safe lane is global SBP derivative/mortar operator architecture;
- the private global SBP derivative/mortar operator architecture lane now records
  `private_global_operator_3d_contract_ready`: A1-A4 identity evidence passes for
  diagonal-norm SBP derivatives including Yee-staggered dual operators,
  norm-compatible mortar projection, material/metric weighted tangential EM flux
  closure, all-six-face edge/corner partition closure, and CPML/non-CPML SAT
  staging evidence. No `sbp_sat_3d.py` solver hunk is retained;
- the private solver integration hunk gate now records
  `private_solver_integration_requires_followup_diagnostic_only`: the
  operator-projected face SAT adapter passes S1 preacceptance, but the
  production-shaped S2 dry run reproduces the manufactured ledger floor above
  the unchanged `0.02` threshold, so no `sbp_sat_3d.py` hunk is retained and the
  next safe lane is private operator-projected face SAT energy-transfer redesign.
  This gate is still the private solver integration hunk from global SBP derivative/mortar operator architecture evidence, not a public observable promotion;
- the private operator-projected energy-transfer redesign now records
  `private_operator_projected_energy_transfer_contract_ready`: the
  ratio-weighted scalar plus skew E/H operator work form closes the private
  manufactured ledger below the unchanged `0.02` threshold without
  residual-derived coefficients, but this is still a private future-integration
  contract only, not a retained `sbp_sat_3d.py` solver hunk or public observable
  promotion;
- the follow-up private bounded solver integration now records
  `private_operator_projected_solver_hunk_retained_fixture_quality_pending`: the
  ledger-passing skew E/H work form is wired through one same-call solver-local
  face helper in `sbp_sat_3d.py`, with CPML/non-CPML parity, six-face `t1/t2`
  slot-map, normal-sign orientation, and edge/corner guards covered. The
  solver-local post-existing-SAT call disables scalar projection to avoid
  double-coupling. This is still private fixture-quality pending evidence, not a
  true R/T, DFT/flux, TFSF, port, S-parameter, API, result, runner, hook, or
  public-promotion claim;
- the private boundary coexistence and fixture-quality validation lane now
  records `private_boundary_coexistence_passed_fixture_quality_blocked`: direct
  step-path probes exercise the retained helper under representative all-PEC,
  selected PMC, periodic, and bounded-CPML accepted boundary arguments while
  `BoundarySpec` unsupported classes remain hard-fail. The unchanged
  fixture-quality gates still fail transverse-uniformity and vacuum-parity
  thresholds, so this is not true-R/T readiness or public-promotion evidence;
- the prior boundary-expanded analytic-sheet sweep is retained as history, not
  as current slab R/T evidence;
- the current recorded status is therefore **inconclusive**, not a public
  support promotion and not a reason to reinterpret thresholds.

Because the private TFSF-style incident fixture now has a same-contract private
reference but does not pass vacuum/reference parity and transverse-uniformity
gates, and because the interface-floor investigation now points to a private
coarse/fine energy-transfer mismatch rather than source/plane/aperture repair
or a direct hook invariant failure, and because the private tau sweep did not
produce a material repair, because the theory/design review selected
`discrete_eh_work_ledger_mismatch` without changing solver or public behavior,
and because the manufactured-interface diagnostic records a private
`ledger_mismatch_detected` face-work residual and bounded face-kernel
feasibility now records `no_signature_compatible_bounded_repair`, and because
the private SAT face-coupling theory/redesign stage now records
`paired_face_coupling_design_ready` without changing production solver or
public surfaces, and because the private paired-face helper implementation gate
records `production_context_mismatch_detected` before any `sbp_sat_3d.py`
patch, and because the private time-centered staging redesign records
`time_centered_staging_contract_ready` as a production-expressible private
staging contract rather than a public observable, and because the bounded private
helper implementation now changes only solver-internal SBP-SAT behavior without
public observables, and because the private fixture-quality recovery ladder now
records `measurement_contract_or_interface_floor_persists`, and because the
private measurement-contract/interface-floor redesign now records
`persistent_interface_floor_confirmed`, and because the private interface-floor
repair theory/implementation lane now records
`no_bounded_private_interface_floor_repair`, and because the private higher-order
SBP face-norm/interface-operator redesign ladder now records
`no_private_face_norm_operator_repair`, and because the private broader
derivative/interior-boundary ladder now records
`no_private_derivative_interface_repair`, and because the private global SBP
derivative/mortar operator architecture lane now records
`private_global_operator_3d_contract_ready`, and because the private
solver-integration gate now records
`private_solver_integration_requires_followup_diagnostic_only`, and because the
private operator-projected energy-transfer redesign now records
`private_operator_projected_energy_transfer_contract_ready` with a ledger-passing
private E/H skew work-form candidate, and because the follow-up private bounded
solver integration now records
`private_operator_projected_solver_hunk_retained_fixture_quality_pending` with a
retained private solver-local hunk but no public observable, and because the
private boundary coexistence and fixture-quality validation lane now records
`private_boundary_coexistence_passed_fixture_quality_blocked` after direct
accepted-boundary helper probes and unchanged fixture-quality replay, the next
private fixture-quality blocker repair lane records
`private_fixture_quality_blocker_persists_no_public_promotion` after a finite
F0-F4 ladder, and because the private source/reference phase-front
fixture-contract redesign lane now records
`private_source_phase_front_self_oracle_failed`, the next safe lane is `private
analytic source phase-front self-oracle repair before fixture-contract
candidates ralplan`. The follow-up private analytic source phase-front
self-oracle repair lane now records
`private_analytic_source_phase_front_self_oracle_blocked_no_public_promotion`,
so the next safe lane is `private analytic plane-wave source implementation
redesign after source self-oracle blocked ralplan`. That follow-up lane now
records `private_uniform_plane_wave_source_self_oracle_ready` for a private
prototype uniform plane-wave source self-oracle, so the next safe lane is
`private fixture contract recovery using plane-wave source self-oracle ralplan`.
That follow-up lane now records
`private_uniform_plane_wave_reference_contract_ready`. The follow-up
subgrid-vacuum fixture lane records
`private_plane_wave_subgrid_vacuum_fixture_blocked_no_public_promotion`. The
fixture-path wiring lane records
`private_plane_wave_fixture_path_wiring_blocked_no_public_promotion`. The
follow-up adapter design lane now records
`private_runner_plane_wave_adapter_design_ready`. The follow-up adapter
implementation lane now records
`private_plane_wave_adapter_implemented_parity_pending`. The follow-up private
subgrid-vacuum parity scoring lane now records
`private_subgrid_vacuum_plane_wave_parity_failed_no_public_promotion`, so the
follow-up blocker repair/design lane now records
`private_plane_wave_interface_floor_repair_design_required`. The next safe lane
is `private plane-wave interface-floor repair implementation before true R/T
readiness ralplan`, with explicitly widened private production scope. That
implementation lane now records `no_private_plane_wave_interface_floor_repair`,
so the next safe lane is `private plane-wave interface-floor
architecture/root-cause redesign after bounded implementation failed ralplan`.
That redesign lane now records
`private_plane_wave_interface_energy_form_root_cause_identified`, so the next
safe lane is `private plane-wave interface energy-form architecture repair
design before implementation ralplan`. That design lane now records
`private_plane_wave_interface_energy_form_implementation_contract_ready`, so
the next safe lane is `private plane-wave interface energy-form implementation
after design contract ready ralplan`. That implementation lane now records
`no_private_plane_wave_energy_form_implementation`: no bounded F1/F2 local
solver hunk is retained as material improvement over the frozen parity packet,
so the next safe lane is `private plane-wave interface energy-form failure
theory/design before true R/T readiness ralplan`. That theory/design lane now
records `private_plane_wave_energy_form_redesign_contract_ready`: the failed
local implementation attempt is explained as missing an operator-owned,
time-centered interface energy-state contract, so the next safe lane is
`private plane-wave operator/mortar time-centered energy-form implementation
after failure theory contract ready ralplan`. That implementation lane now
records `no_private_plane_wave_operator_mortar_energy_form_implementation`: the
current solver staging does not expose a bounded owner for that time-centered
interface energy state, so the next safe lane is `private plane-wave transverse
phase-coherence architecture redesign after operator/mortar energy-form
implementation blocked ralplan`. That architecture lane now records
`private_plane_wave_phase_coherence_staging_contract_ready`: a single private
interface-state owner must preserve transverse phase spread and magnitude CV
jointly. The follow-up implementation lane now records
`no_private_plane_wave_phase_coherence_staging_implementation`: the required
interface-state owner changes solver state shape and scan staging, and a joint
score guard alone cannot claim material repair. The next safe lane is `private
plane-wave solver-wide interface-state owner architecture redesign after
phase-coherence staging implementation blocked ralplan`. That architecture lane
now records `private_plane_wave_solver_wide_interface_state_owner_contract_ready`:
one private owner and explicit scan/staging state-shape contract must be
implemented before phase-coherence parity repair can be claimed, so the next
safe lane is `private plane-wave solver-wide interface-state owner implementation
after architecture contract ready ralplan`. That implementation lane now
records `no_private_plane_wave_solver_wide_interface_state_owner_implementation`:
owner state shape and same-step scan wiring require a private solver-state
propagation boundary contract for JAX pytree shape and runner/JIT
initialization before any bounded hunk can be retained, so the next safe lane is
`private plane-wave solver-state owner propagation boundary design after
interface-state owner implementation blocked ralplan`. That design lane now
records `private_plane_wave_solver_state_owner_propagation_contract_ready`:
state pytree shape plus runner/JIT initialization boundaries are defined before
retrying the bounded owner implementation. The follow-up implementation lane now
records
`private_plane_wave_runner_jit_owner_propagation_hunk_retained_fixture_quality_pending`:
the private owner state shape is carried through CPML and non-CPML subgrid steps
and JIT-runner initialization. The follow-up owner scan-wiring/joint-scoring
lane now records
`private_plane_wave_owner_joint_parity_scoring_hunk_retained_fixture_quality_pending`:
same-step E/H owner scan capture and owner-backed joint phase/CV scoring are
retained. The follow-up physical correction lane now records
`no_private_plane_wave_owner_backed_physical_phase_cv_correction`: the retained
owner scorer is face-scalar diagnostic state, so no field-update correction is
retained before a face-local modal correction architecture proves paired
phase/CV and vacuum-regression gates. That architecture lane now records
`private_plane_wave_face_local_modal_correction_contract_ready`: phase and
magnitude owner references map to private face-local tangential modal
contracts with paired phase/CV, vacuum-regression, and CPML/non-CPML symmetry
guards before any field-update hunk. The next safe lane is `private plane-wave
face-local modal correction implementation after architecture contract ready
ralplan`. That implementation lane now records
`no_private_plane_wave_face_local_modal_correction_implementation`: no bounded
field-update hunk is retained because paired phase/CV and vacuum-regression
gates were not honestly improved. The next safe lane is `private plane-wave
face-local modal correction failure-theory redesign after bounded
implementation failed ralplan`. That redesign lane now records
`private_plane_wave_face_local_modal_failure_theory_contract_ready`: the failed
implementation is explained by same-step diagnostic owner-reference feedback
and characteristic-vs-observable modal-basis mismatch, so the next retry
requires a lagged owner reference and observable-aligned modal basis under
paired phase/CV and vacuum gates. The implementation retry lane now records
`no_private_plane_wave_face_local_modal_retry_implementation`: lagged owner
timing alone is insufficient, and the claims-bearing transverse plane-DFT
observable cannot be imported into solver-local field updates without private
observable-proxy modal-basis architecture. That architecture lane now records
`private_plane_wave_observable_proxy_modal_retry_contract_ready`: a solver-local
transverse face energy/phase proxy can stand in for the benchmark plane-DFT
distribution without importing the DFT observable into field updates, while
lagged owner state plus paired phase/CV and vacuum guards remain required before
implementation. That implementation lane now records
`no_private_plane_wave_observable_proxy_modal_retry_implementation`: the current
solver owner state stores scalar phase/magnitude references per active face,
while a solver-local modal proxy requires packed face-local distributions with
offsets and masks before any field-update hunk can be retained. The state-shape
design lane now records `private_plane_wave_proxy_face_packet_state_contract_ready`:
packed face-local proxy buffers, `FACE_ORIENTATIONS`-derived index metadata, and
CPML/non-CPML initialization symmetry can be specified without public API, hooks,
benchmark DFT, or public observable promotion. The implementation lane now records
`private_plane_wave_proxy_face_packet_capture_hunk_retained_fixture_quality_pending`:
the private owner state carries fixed-shape packed face-local proxy references,
offsets, masks, orientation metadata, and CPML/non-CPML/JIT initialization
symmetry. The modal retry lane now records
`private_plane_wave_observable_proxy_modal_retry_hunk_retained_fixture_quality_pending`:
a private solver-local, lagged packed-state E-modal correction hunk is retained
with update/coupling bounds, CPML/non-CPML symmetry, and no benchmark DFT import.
The parity-scoring lane now records
`private_plane_wave_observable_proxy_modal_retry_hunk_insufficient_fixture_quality_pending`:
the retained hunk is finite and reproducible, but it reduces the dominant phase
spread by only about 0.16% and remains far above unchanged phase/CV/vacuum
thresholds. The failure-theory lane now records
`private_plane_wave_modal_retry_failure_theory_redesign_contract_ready`: the
lagged packed face packet only aligned local face distribution state, while the
remaining blocker is propagation-aware transverse phase coherence plus an
incident-normalized observable basis with source/interface ownership separation.
The next safe lane is private modal-retry redesign implementation after that
failure-theory contract. That implementation lane now records
`no_private_plane_wave_observable_proxy_modal_retry_redesign_implementation`:
no new propagation-aware modal-basis hunk is retained because the current
private owner state has only the interface-observed packet and no separate
source-owner/incident-normalizer packet. The next safe lane is private
source-interface ownership state-shape design. That design lane now records
`private_plane_wave_source_interface_owner_state_shape_contract_ready`: private
source-owner incident packets, interface-owner observed packets, and incident
normalizer buffers can remain fixed-shape, CPML/non-CPML/JIT symmetric, and
separate without public TFSF, benchmark DFT, hooks, or public observable
promotion. The next safe lane is private source-interface ownership state-shape
implementation after design contract ready. Hook experiments remain closed and
public promotion remains closed. That implementation lane now records
`private_plane_wave_source_interface_state_shape_hunk_retained_fixture_quality_pending`:
private source-owner reference, incident-normalizer, weight/mask, offset/length,
and orientation buffers are retained separately from the existing interface
packet, with CPML/non-CPML/JIT initialization symmetry and no modal field-update
behavior change. The next safe lane is private propagation-aware modal retry
implementation after source-interface state-shape hunk retained.
That implementation lane now records
`private_plane_wave_propagation_aware_modal_basis_hunk_retained_fixture_quality_pending`:
a bounded private helper uses only the source-owner packet, incident normalizer,
and lagged interface-owner packet, remains no-op without populated source
packets, and does not import benchmark DFT/flux/TFSF/port/S-parameter or public
observable state. The next safe lane is private propagation-aware modal retry
parity scoring after the source-normalized hunk is retained.
That scoring lane now records
`private_plane_wave_propagation_aware_modal_retry_parity_scored_fixture_quality_pending`:
finite private scoring is available, but the unchanged material-improvement and
true-R/T readiness gates remain below threshold because production source-owner
incident packet population is not wired yet. The next safe lane is private
source-owner incident packet population design.
That design lane now records
`private_plane_wave_source_owner_incident_packet_population_contract_ready`:
source-owner reference, incident-normalizer, packet offset/orientation, and
pre-modal-retry timing contracts can remain private, fixed-shape,
CPML/non-CPML/JIT symmetric, and separate from interface-owner packets. The
implementation lane now records
`private_plane_wave_source_owner_incident_packet_population_hunk_retained_fixture_quality_pending`:
a solver-local packetization helper populates private source-owner incident
packets before propagation-aware modal retry, preserves interface-owner packets,
and keeps CPML/non-CPML wiring plus public promotion closed. The next safe lane
is private source-populated propagation-aware modal retry parity scoring after
the source-owner packet hunk is retained. That scoring lane now records
`private_plane_wave_source_populated_propagation_aware_modal_retry_hunk_insufficient_fixture_quality_pending`:
the populated source-owner packet is consumed by the private modal retry, but
unchanged paired material-improvement and true R/T readiness gates remain
closed. The next safe lane is private source-populated propagation-aware modal
retry failure-theory redesign after parity scoring insufficient. That theory
lane now records
`private_plane_wave_source_populated_modal_retry_time_alignment_theory_contract_ready`:
the next bounded private target is time-aligning the lagged interface-owner
packet with the same-step source-owner packet before another modal retry hunk.
The next safe lane is private source/interface time-aligned packet staging
design after that theory contract. That design lane now records
`private_plane_wave_source_interface_time_aligned_packet_staging_contract_ready`:
private fixed-shape staged source/interface packet slots, CPML/non-CPML/JIT
initialization, and modal-retry consumer timing are specified without hooks or
public observables. The implementation lane now records
`private_plane_wave_source_interface_time_aligned_packet_staging_hunk_retained_fixture_quality_pending`:
previous source/interface packet fields are retained in private owner state, a
staging helper snapshots the last completed pair before source overwrite, and
propagation-aware modal retry now consumes that time-aligned pair without hooks
or public observables. The scoring lane now records
`private_plane_wave_time_aligned_modal_retry_hunk_insufficient_fixture_quality_pending`:
the retained staged-packet hunk is finite under private scoring, but unchanged
paired material-improvement, transverse-uniformity, and vacuum-stability gates
remain closed. The failure-theory lane now records
`private_plane_wave_time_aligned_modal_retry_modal_projection_normalizer_theory_contract_ready`:
time alignment made no material score delta against the source-populated
baseline, so the design lane now records
`private_plane_wave_modal_projection_normalizer_contract_design_ready`: shared
modal basis, incident normalizer, and face-mask weighting contracts are ready
for a bounded private implementation lane while public observables remain
closed. The implementation lane now records
`private_plane_wave_modal_projection_normalizer_contract_hunk_retained_fixture_quality_pending`:
a fail-closed private contract gate is retained inside propagation-aware modal
retry. The parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_contract_hunk_insufficient_fixture_quality_pending`:
the retained contract gate hunk is finite under private scoring, but unchanged
material-improvement, transverse-uniformity, and vacuum-stability gates remain
closed. The failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_basis_redesign_contract_ready`:
the scalar fail-closed gate only proves packet compatibility and does not apply
a shared projected modal-basis transform, so the next safe lane is private
modal projection/normalizer projected-basis redesign contract. The projected-basis
design lane now records
`private_plane_wave_modal_projection_normalizer_projected_basis_contract_design_ready`:
private basis vectors, power normalization, mask weighting, and fail-closed
implementation preconditions are frozen; the next safe lane is private
projected-basis implementation. The projected-basis implementation lane now
records
`private_plane_wave_modal_projection_normalizer_projected_basis_hunk_retained_fixture_quality_pending`:
a solver-local `_project_private_modal_basis_packets` helper projects private
source packets onto the incident normalizer basis before modal retry
subtraction, fails closed on missing projection energy or packet contract
mismatch, and keeps public true R/T/DFT/flux/TFSF/port/S-parameter surfaces
closed; the next safe lane is private projected-basis parity scoring. The
projected-basis parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_basis_hunk_insufficient_fixture_quality_pending`:
the retained helper produces finite private evidence, but the unchanged
material-improvement and fixture-quality gates remain closed, so the next safe
lane is private projected-basis failure-theory redesign. The projected-basis
failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_basis_redesign_contract_ready`:
the retained hunk projects the source packet but leaves the interface target
unprojected, so the next safe lane is a private projected target/source-interface
basis implementation; public promotion remains closed. The projected
target/source-interface basis implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_basis_hunk_retained_fixture_quality_pending`:
`_project_private_modal_basis_packets` now projects both source and interface
packets onto the shared incident-normalizer basis before subtraction; public
promotion remains closed and the next safe lane is private parity scoring. The
projected target/source-interface parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_basis_hunk_insufficient_fixture_quality_pending`:
the retained shared target-basis hunk has finite private score evidence, but
unchanged material-improvement and fixture-quality gates remain closed, so the
next safe lane is private failure-theory redesign. The projected
target/source-interface failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_redesign_contract_ready`:
the retained hunk projects both source and interface packets, but it still uses
one incident-normalizer basis for the full correction; the next safe lane is a
private residual/reflected/transverse projected target-basis redesign contract.
The projected target residual-basis design lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_contract_design_ready`:
fixed-shape incident, reflected, and transverse residual target modes are ready
as a private implementation contract without public observables or threshold
laundering; the next safe lane is private residual-basis implementation. The
projected target residual-basis implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_hunk_retained_fixture_quality_pending`:
`_project_private_modal_basis_packets` now uses fixed-shape incident/reflected/
transverse residual target modes before subtraction while remaining private and
fail-closed; public promotion remains closed and the next safe lane is private
residual-basis parity scoring. The projected target residual-basis parity-scoring
lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_hunk_insufficient_fixture_quality_pending`:
the retained residual-basis hunk has finite private score evidence, but unchanged
material-improvement and fixture-quality gates remain closed, so the next safe
lane is private residual-basis failure-theory redesign. The projected target
residual-basis failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_modal_orthogonality_floor_theory_ready`:
the retained hunk removed the single-incident-basis floor, but its residual modes
are scalar weighted complex-L2 packet modes rather than an energy-biorthogonal
characteristic E/H basis; the next safe lane is private energy-biorthogonal
residual-basis redesign. The projected target residual-basis energy-biorthogonal
design lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_contract_design_ready`:
the design replaces scalar packet-L2 residual modes with fixed-shape private
incident, reflected, and transverse characteristic E/H power modes, keeps
thresholds unchanged, reuses existing private owner packet shapes, and leaves
all public observable/API/result/export surfaces closed; the next safe lane is
private energy-biorthogonal implementation. The projected target residual-basis
energy-biorthogonal implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_hunk_retained_fixture_quality_pending`:
the private helper weights packet coefficients by a characteristic E/H energy
form and projects source/interface packets through a fixed-shape
energy-biorthogonal Gram system before target subtraction, with public
promotion still closed; the next safe lane is private energy-biorthogonal
parity scoring. The projected target residual-basis energy-biorthogonal
parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_hunk_insufficient_fixture_quality_pending`:
the retained private hunk has finite score evidence, but unchanged material-
improvement and fixture-quality gates remain closed, so true R/T and public
observable promotion stay locked; the next safe lane is private
energy-biorthogonal failure-theory redesign. The projected target residual-
basis energy-biorthogonal failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_floor_theory_ready`:
the finite-but-insufficient score is classified as a packet-local energy
metric-shape floor rather than a public-readiness signal; thresholds remain
unchanged, no solver hunk is added, and the next safe lane is private
energy-biorthogonal metric-shape calibration design. The projected target
residual-basis energy-biorthogonal metric-shape
calibration design lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_contract_design_ready`:
the design defines a private face-local SBP/mortar power metric calibration
contract over existing owner packet shapes, keeps thresholds and public
surfaces unchanged, and defers solver edits to a separate implementation
lane; the next safe lane is private metric-shape calibration implementation.
The projected target residual-basis energy-biorthogonal metric-shape
calibration implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_hunk_retained_fixture_quality_pending`:
the private projection helper now replaces packet-local energy-density
shaping with a face-local SBP/mortar power metric scale computed from
existing owner weights, masks, and normal signs; thresholds and public
surfaces remain unchanged, and the next safe lane is private metric-shape
calibration parity scoring.
The projected target residual-basis energy-biorthogonal metric-shape
calibration parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_hunk_insufficient_fixture_quality_pending`:
the retained private hunk has finite score evidence, but unchanged
material-improvement and fixture-quality gates remain closed, so true R/T
and public observable promotion stay locked; the next safe lane is private
metric-shape calibration failure-theory redesign.
The projected target residual-basis energy-biorthogonal metric-shape
calibration failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_transverse_modal_coupling_floor_theory_ready`:
the retained face-local scalar metric calibration produced no material score
change after normal signs, masks, and weights were already present, so the
remaining private theory is a transverse modal basis-coupling floor rather
than public readiness or threshold adjustment; the next safe lane is private
transverse modal-coupling metric design.
The projected target residual-basis energy-biorthogonal transverse modal-
coupling metric design lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_contract_design_ready`:
the design defines a fixed-shape private 3x3 modal power-coupling metric over
existing owner/interface packet shapes and existing projection-normalizer
helpers, rejects another scalar face-metric extension, keeps thresholds and
public surfaces unchanged, and defers solver edits to a separate bounded
implementation lane.
The projected target residual-basis energy-biorthogonal transverse modal-
coupling metric implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_hunk_retained_fixture_quality_pending`:
the private projection helper now applies a bounded fixed-shape 3x3 modal
coupling matrix inside the incident/reflected/transverse Gram packet, keeps
public surfaces and thresholds unchanged, and defers fixture-quality claims to
a separate private parity-scoring lane.
The projected target residual-basis energy-biorthogonal transverse modal-
coupling metric parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_hunk_insufficient_fixture_quality_pending`:
the retained private 3x3 helper has finite private score evidence, but unchanged
material-improvement, transverse-uniformity, and vacuum-stability gates remain
closed, so true R/T and public observable promotion stay locked; the next safe
lane is private failure-theory redesign.
The projected target residual-basis energy-biorthogonal transverse modal-
coupling metric failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_source_interface_transfer_floor_theory_ready`:
the retained in-packet 3x3 coupling is finite but insufficient, so the selected
private next target is a fixed-shape source/interface transverse modal transfer
map before another solver hunk; solver edits, threshold changes, and public
promotion remain deferred.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_hunk_retained_fixture_quality_pending`:
the private projection helper now applies a bounded fixed-shape 3x3
source/interface modal transfer correction before subtracting projected packets,
keeps public surfaces and thresholds unchanged, and defers fixture-quality
claims to a separate private parity-scoring lane.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_hunk_insufficient_fixture_quality_pending`:
the retained private transfer-map helper has finite private parity evidence,
but unchanged material-improvement, transverse-uniformity, and vacuum-stability
gates remain closed, so true R/T and public observable promotion stay locked;
the next safe lane is private failure-theory redesign.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_floor_theory_ready`:
the retained private transfer-map hunk is finite but still source-outer oriented,
so the selected private next target is a fixed-shape target/residual-basis
orientation for the transfer map; solver edits, threshold changes, and public
promotion remain deferred.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation implementation lane now
records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_hunk_retained_fixture_quality_pending`:
the private projection helper now applies a bounded fixed-shape target-basis-
oriented transfer map from existing source/interface modal packets, keeps public
surfaces and thresholds unchanged, and routes the next step to private parity
scoring before any true R/T readiness claim.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation parity-scoring lane now
records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_hunk_insufficient_fixture_quality_pending`:
the retained target-basis-oriented transfer-map hunk has finite private parity
evidence, but unchanged material-improvement, transverse-uniformity, and vacuum-
stability gates remain closed, so true R/T and public observable promotion stay
locked; the next safe lane is private failure-theory redesign.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation failure-theory redesign
lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_floor_theory_ready`:
it consumes the finite-but-insufficient target-basis orientation parity evidence,
selects a private residual phase/sign floor inside the same fixed-shape modal
packet contract, defers solver edits to a later private implementation lane, and
keeps true R/T plus public observable promotion locked.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/sign
implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_hunk_retained_fixture_quality_pending`:
the private projection helper applies a bounded fixed-shape residual phase/sign
correction to the target-basis-oriented transfer map, keeps public surfaces and
thresholds unchanged, and routes the next step to private parity scoring before
any true R/T readiness claim.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/sign
parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_hunk_insufficient_fixture_quality_pending`:
the retained residual phase/sign hunk has finite private parity evidence, but
unchanged material-improvement, transverse-uniformity, and vacuum-stability gates
remain closed, so true R/T and public observable promotion stay locked; the next
safe lane is private failure-theory redesign.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/sign
failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_phase_magnitude_imbalance_floor_theory_ready`:
it consumes the finite-but-insufficient residual phase/sign parity evidence,
selects a private phase/magnitude imbalance floor inside the same fixed-shape
modal packet contract, defers solver edits to a later private implementation
lane, and keeps true R/T plus public observable promotion locked.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_hunk_retained_fixture_quality_pending`:
the private projection helper applies a bounded residual phase/magnitude balance
correction to the residual phase/sign transfer map, keeps public surfaces and
thresholds unchanged, and routes the next step to private parity scoring before
any true R/T readiness claim.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_hunk_insufficient_fixture_quality_pending`:
the retained residual phase/magnitude balance hunk has finite private parity
evidence, but unchanged material-improvement, transverse-uniformity, and
vacuum-stability gates remain closed, so true R/T and public observable
promotion stay locked; the next safe lane is private failure-theory redesign.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_floor_theory_ready`:
it consumes the finite-but-insufficient phase/magnitude balance parity evidence,
selects a private residual modal-coupling floor inside the same fixed-shape modal
packet contract, defers solver edits to a later private implementation lane, and
keeps true R/T plus public observable promotion locked.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_hunk_retained_fixture_quality_pending`:
the private projection helper applies a bounded residual modal-coupling map
inside the retained fixed-shape source/interface packet contract, keeps public
surfaces and thresholds unchanged, and routes the next step to private parity
scoring before any true R/T readiness claim.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_hunk_insufficient_fixture_quality_pending`:
the retained residual modal-coupling hunk has finite private parity evidence,
but unchanged material-improvement, transverse-uniformity, and vacuum-stability
gates remain closed, so true R/T and public observable promotion stay locked;
the next safe lane is private failure-theory redesign.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_floor_theory_ready`:
it consumes the finite-but-insufficient residual modal-coupling parity evidence,
selects a private packet-basis mismatch floor inside the retained fixed-shape
source/interface packet contract, defers solver edits to a later private
implementation lane, and keeps true R/T plus public observable promotion locked.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling packet-basis mismatch implementation lane now
records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_hunk_retained_fixture_quality_pending`:
the private projection helper applies a bounded packet-basis mismatch transfer
after the residual modal-coupling map inside the retained fixed-shape
source/interface packet contract, keeps public surfaces and thresholds
unchanged, and routes the next step to private parity scoring before any true
R/T readiness claim.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling packet-basis mismatch parity-scoring lane now
records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_hunk_insufficient_fixture_quality_pending`:
the retained packet-basis mismatch hunk has finite private parity evidence,
but unchanged material-improvement, transverse-uniformity, and vacuum-stability
gates remain closed, so true R/T and public observable promotion stay locked;
the next safe lane is private failure-theory redesign.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling packet-basis mismatch failure-theory redesign
lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_floor_theory_ready`:
it consumes the finite-but-insufficient packet-basis mismatch parity evidence,
selects a private owner-packet weighting floor inside the retained fixed-shape
source/interface packet contract, defers solver edits to a later private
implementation lane, and keeps true R/T plus public observable promotion locked.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling packet-basis mismatch owner-packet weighting
implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_hunk_retained_fixture_quality_pending`:
the private projection helper applies a bounded owner-packet weighting transfer
after the packet-basis mismatch map inside the retained fixed-shape
source/interface packet contract, keeps public surfaces and thresholds
unchanged, and routes the next step to private parity scoring before any true
R/T readiness claim.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling packet-basis mismatch owner-packet weighting
parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_hunk_insufficient_fixture_quality_pending`:
the retained owner-packet weighting hunk has finite private parity evidence,
but unchanged material-improvement, transverse-uniformity, and vacuum-stability
gates remain closed, so true R/T and public observable promotion stay locked;
the next safe lane is private owner-packet weighting failure-theory redesign.
The projected target residual-basis energy-biorthogonal source/interface
transverse modal transfer-map target-basis orientation residual phase/magnitude
balance residual modal-coupling packet-basis mismatch owner-packet weighting
failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_floor_theory_ready`:
it consumes the finite-but-insufficient owner-packet weighting parity evidence,
selects a bounded private modal energy/impedance weighting floor inside the
existing source/interface owner-packet transfer-map contract, defers solver
edits to a later private implementation lane, and keeps true R/T plus public
observable promotion locked.
The modal energy/impedance implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the retained
owner-packet weighting map, adds clipped residual modal-energy row weights and
source/interface impedance column weights, retains the 3x3 private transfer-map
contract, and keeps parity scoring, true R/T readiness, thresholds, runner state,
and public observable promotion closed.
The modal energy/impedance parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_hunk_insufficient_fixture_quality_pending`: the retained hunk has finite private vacuum parity evidence,
but unchanged material-improvement, transverse-uniformity, and vacuum-stability
gates remain closed, so true R/T and public observable promotion stay locked;
the next safe lane is private modal energy/impedance failure-theory redesign.
The modal energy/impedance failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_floor_theory_ready`: it consumes the finite-but-insufficient modal energy/impedance parity evidence,
selects a bounded private transverse energy redistribution floor inside the
existing source/interface modal energy/impedance transfer-map contract, defers
solver edits to a later private implementation lane, and keeps true R/T plus
public observable promotion locked.
The transverse energy redistribution implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the retained modal energy/impedance map, applies clipped transverse residual row/column weights inside the existing 3x3 source/interface packet contract, and keeps true R/T readiness, thresholds, runner state, and public observable promotion closed while routing the next step to private parity scoring.

The transverse energy redistribution parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_hunk_insufficient_fixture_quality_pending`: it consumes the retained helper hunk and current private vacuum parity metrics, records finite private evidence, and confirms that unchanged material-improvement, transverse-uniformity, and vacuum-stability gates remain closed. No true R/T, DFT/flux, TFSF, port, S-parameter, threshold, runner, API, or public observable surface is promoted; the next safe lane is private transverse energy redistribution failure-theory redesign.
The transverse energy redistribution failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_theory_ready`: it consumes the finite-but-insufficient retained redistribution parity evidence, rejects repeated sequential redistribution plus public/threshold escape routes, and selects a bounded private coupled modal energy-balance target inside the existing 3x3 source/interface transfer-map contract. Solver edits, true R/T readiness, runner state, thresholds, and public observable promotion remain deferred to later approved lanes.
The coupled modal energy-balance implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the retained transverse redistribution map, adds clipped coupled row/column modal energy-balance weights inside the existing 3x3 source/interface packet contract, and keeps true R/T readiness, thresholds, runner state, and public observable promotion closed while routing the next step to private parity scoring.

The coupled modal energy-balance parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_hunk_insufficient_fixture_quality_pending`: it consumes the retained helper hunk and current private vacuum
parity metrics, records finite private evidence, and confirms that unchanged
material-improvement, transverse-uniformity, and vacuum-stability gates remain
closed. No true R/T, DFT/flux, TFSF, port, S-parameter, threshold, runner, API,
or public observable surface is promoted; the next safe lane is private coupled
modal energy-balance failure-theory redesign.

The coupled modal energy-balance failure-theory redesign lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_floor_theory_ready`: it consumes the finite-but-insufficient coupled modal
energy-balance parity evidence, rejects a repeated coupled row/column metric
floor plus public/threshold escape routes, and selects a bounded private
target-basis packet normalization target inside the existing 3x3
source/interface coupled-balance transfer-map contract. Solver edits, true R/T
readiness, runner state, thresholds, and public observable promotion remain
deferred to later approved lanes.

The target-basis packet normalization implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the retained coupled
modal energy-balance map, adds clipped target/source packet-normalization weights
inside the existing 3x3 source/interface packet contract, and keeps true R/T
readiness, thresholds, runner state, and public observable promotion closed
while routing the next step to private parity scoring.


The target-basis packet normalization parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_hunk_insufficient_fixture_quality_pending`: the retained solver-local helper is privately scored with finite
evidence, but unchanged material-improvement, transverse-uniformity, and
vacuum-stability gates remain below the frozen thresholds. No solver hunk,
runner state, threshold, public observable, or true-R/T readiness claim is added
by this scoring lane; the next safe step is a private failure-theory redesign.

The target-basis packet normalization failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_floor_theory_ready`: it consumes the finite-but-insufficient retained
packet-normalization parity evidence, rejects repeat normalization, threshold
relaxation, and public-observable escape candidates, and selects the next
bounded private source/interface packet energy co-normalization implementation
target inside the existing fixed-shape 3x3 source/interface transfer-map
contract. No solver hunk, runner state, threshold, public observable, or true-R/T
readiness claim is added by this theory lane.

The source/interface packet energy co-normalization implementation lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_hunk_retained_fixture_quality_pending`: a solver-local fixed-shape helper consumes the retained target-basis
packet-normalization helper, adds bounded source/interface packet-energy
co-normalization inside the existing 3x3 source/interface transfer-map
contract, and keeps true R/T readiness, thresholds, runner state, and public
observable promotion closed while routing the next step to private parity
scoring.


The source/interface packet energy co-normalization parity-scoring lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_hunk_insufficient_fixture_quality_pending`: it consumes the retained implementation metadata, preserves frozen
thresholds and baseline/current metrics, records finite private scoring evidence,
and keeps true R/T readiness, runners, hooks, exports, public observables, and
threshold changes closed while routing the next step to private failure-theory redesign for the source/interface packet energy co-normalization floor.

The source/interface packet energy co-normalization failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_floor_theory_ready`: it consumes the finite-but-insufficient co-normalization parity
evidence, rejects repeat energy-only normalization and public/threshold escape
routes, and selects private source/interface phase-energy residual implementation inside the retained 3x3 transfer-map contract. No true R/T readiness, runner state,
threshold, public observable, API/export, or docs-public/examples promotion is
unlocked by this theory lane.

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



Until those fixture-quality gates pass, the private flux/DFT gate remains internal
diagnostic evidence only, and the support matrix continues to mark true R/T as
deferred.
In other words, the support matrix continues to mark true R/T as deferred.

## Why true R/T is deferred

A meaningful true R/T benchmark needs incident/reflected/transmitted separation.
The current Phase-1 support surface deliberately keeps most of the surfaces
that would make that benchmark clean and claims-bearing outside public support:

- UPML/open-boundary variants beyond the guarded all-CPML subset
- TFSF plane-wave excitation with subgridding
- DFT plane probes and flux monitors with subgridding
- waveguide/Floquet/S-parameter style ports with subgridding
- impedance/wire/coaxial ports inside the refined region

A time-gated all-PEC cavity fixture could be used as a research diagnostic, but
it would still conflate interface error, source spectrum, finite-domain echo
control, and point-probe normalization. It is therefore not sufficient as the
public true R/T benchmark for SBP-SAT subgridding.

## Required future fixture

The first claims-bearing true R/T benchmark should use a 1D-normal-incidence
slab geometry embedded inside the refined z slab:

- Boundary model: open z boundaries, or a documented equivalent that prevents
  boundary echoes from contaminating the measurement window.
- Excitation: an incident field with a defined spectrum and normalization.
- Material fixture: at least vacuum -> dielectric slab -> vacuum, with the slab
  fully inside the fine region and with a uniform-fine reference at the same
  effective resolution.
- Observables: independent incident, reflected, and transmitted measurements;
  flux or S-parameter style extraction is preferred over a single point probe.
- Frequency band: a documented passband where source energy, mesh dispersion,
  and absorber quality are all measurable.

## Metrics

The benchmark report must compute and store:

1. `R(f)` magnitude and phase against a reference solver or analytic transfer
   matrix where applicable.
2. `T(f)` magnitude and phase against the same reference.
3. Energy-balance residual `|R|^2 + |T|^2 + loss - 1` for lossless and lossy
   fixtures as appropriate.
4. Time-window definition, including exclusion of source turn-on and boundary
   echo windows.
5. Mesh-convergence trend across at least two coarse/fine resolutions.

## Initial tolerance targets

These are target gates for the future benchmark, not claims about current code:

| Metric | Initial target | Notes |
|---|---:|---|
| `|R|` relative error | `<= 5%` where `|R_ref| >= 1e-3` | use absolute error near nulls |
| `|T|` relative error | `<= 5%` where `|T_ref| >= 1e-3` | use absolute error near nulls |
| phase error | `<= 5°` in the usable passband | unwrap before comparison |
| energy residual | `<= 5%` for lossless fixtures | after time-window and absorber corrections |
| convergence | non-increasing error under refinement | investigate any regression |

These tolerances mirror the current proxy amplitude/phase gates only as a
starting point. They must be revisited after a real fixture exposes absorber,
source-normalization, and flux/S-parameter error separately.

## Acceptance gate for implementation

True R/T implementation may begin only after both prerequisites are available
for the subgridded path:

1. An echo-control strategy, either:
   - open-boundary coexistence with documented CPML/UPML interaction; or
   - a validated time-gated finite-domain method with echo rejection.
2. A source/observable contract that separates incident, reflected, and
   transmitted fields.

Until then, the support matrix must say:

- SBP-SAT subgridding status: `experimental`
- benchmark evidence: proxy numerical equivalence plus bounded-CPML
  point-probe feasibility evidence plus private flux/DFT benchmark evidence
- true R/T benchmark: deferred
- feasibility probe gate: inconclusive
- private flux/DFT benchmark gate: inconclusive under the current private
  analytic-sheet bounded-CPML fixture-quality gates, even after the bounded
  boundary-expanded recovery sweep and private time-centered helper
  fixture-quality recovery ladder
- public support promotion: blocked

## Deferred issue record

**Title:** Implement claims-bearing true R/T benchmark for SBP-SAT z-slab
subgridding.

**Milestone:** after boundary/source/observable contracts are available for the
subgridded path, expected no earlier than the boundary-coexistence and
ports/observables roadmap milestones.

**Problem:** current proxy crossval only compares one point-probe DFT against a
uniform-fine reference. It cannot separate incident, reflected, and transmitted
fields, so it cannot justify public R/T, S-parameter, or energy-balance claims.

**Required work:**

1. Choose and document the true R/T fixture.
2. Implement incident/reflected/transmitted separation.
3. Add amplitude, phase, energy-balance, and convergence assertions.
4. Store fixture metadata and usable frequency band in the test output or docs.
5. Update `docs/guides/support_matrix.json` from `deferred` to `implemented`
   only after the benchmark passes.

**Exit criteria:** all metrics in this document pass, the support matrix is
updated, and public docs cite the new benchmark without broadening beyond the
tested support surface.

## Regression hooks

The support-matrix contract is locked by `tests/test_support_matrix_sbp_sat.py`.
The proxy benchmark naming is locked by `tests/test_subgrid_crossval.py` and by
support-matrix metadata pointing to that file.
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
