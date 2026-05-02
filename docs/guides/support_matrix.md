# rfx Support Matrix

Status legend:
- **supported** — part of the current claims-bearing surface
- **shadow** — retained and tested, but not claims-bearing yet
- **experimental** — research-grade / partial / promotion pending
- **unsupported** — should fail clearly rather than degrade silently

## Current claims-bearing reference lane

**Lane:** uniform Cartesian Yee RF workflows

### Scope
- boundaries: `pec`, `cpml`, `upml`
- sources: point/current sources, lumped/wire ports, waveguide ports
- observables: time-series probes, flux monitors, S-parameters, Harminv resonances, NTFF/far-field where benchmarked
- materials: isotropic linear, conductive, and validated dispersive subsets
- workflows: cavity, waveguide, patch antenna, simple scattering, resonance, de-embedding, selected differentiable proxy objectives

## Lane summary

| Lane | Status | Current role | Notes |
|---|---|---|---|
| Uniform Yee RF lane | **supported** | claims-bearing reference lane | primary correctness surface |
| Nonuniform graded-z | **shadow** | preserved thin-substrate lane | no silent fallback; no new promotions until re-qualified |
| SBP-SAT / subgridding | experimental | research lane | not part of the claims-bearing surface |
| ADI | experimental | research lane | separate accuracy/stability envelope |
| Distributed | experimental | scaling lane | not part of correctness-bearing baseline |
| Floquet/Bloch | experimental | periodic/phased-array lane | promotion pending explicit benchmark ladder |

## SBP-SAT / subgridding experimental lane

Current retained subset:
- all-PEC outer boundaries
- mixed PEC/PMC reflector-face `BoundarySpec` subsets
- periodic axes when the refinement box is either interior to that axis or spans it end-to-end
- bounded CPML absorbing faces when the refinement box stays outside every active absorber pad plus a one-coarse-cell guard
- do not mix PMC faces with periodic axes, or CPML faces with reflector/periodic faces, in the same supported subset
- one axis-aligned refinement box only; CPML cases require the explicit absorber guard
- soft point sources and point probes only
- proxy numerical-equivalence comparison against a uniform-fine reference

Current policy:
- preserved as a research lane, not a claims-bearing public support surface
- unsupported combinations hard-fail instead of warning or silently dropping
- public support promotion is blocked until true R/T evidence exists
- a bounded-CPML point-probe feasibility probe now exists, but it is
  inconclusive and remains internal evidence only
- a private fine-owned flux/DFT benchmark gate now exists for internal true
  R/T diagnostics; the current recorded fixture uses private TFSF-style
  incident-field hooks plus a benchmark-only same-contract uniform reference,
  remains inconclusive because vacuum/reference parity and transverse-uniformity
  gates do not pass, now records `transverse_phase_spread_deg` as the dominant
  row-2 blocker, and now carries a private causal ladder classification of
  `sbp_sat_interface_floor` after source, plane-shift, and central-core aperture
  controls failed the paired material-improvement rule; the follow-up private
  solver/interface-floor investigation now subclasses the blocker as
  `coarse_fine_energy_transfer_mismatch` because a front/back signed-flux
  residual persists across predeclared boundary-expanded and nearer bounded
  interface candidates; the private tau-sensitivity repair sweep did not find
  a material repair (`tau=1.0` improved the residual by only 18.17% and
  regressed the paired vacuum-phase metric); slab R/T scoring is still
  intentionally skipped, and the gate does not enable public DFT planes, flux
  monitors, TFSF, S-parameters, or true R/T claims

### Benchmark evidence

| Evidence layer | Status | Artifact | Claim level |
|---|---|---|---|
| unit / integration | implemented | `tests/test_sbp_sat_api_guards.py`, `tests/test_sbp_sat_face_ops.py`, `tests/test_sbp_sat_3d.py`, `tests/test_sbp_sat_alpha.py`, `tests/test_sbp_sat_jit.py` | API guards, operator behavior, smoke stability, JIT seam |
| proxy crossval | implemented | `tests/test_subgrid_crossval.py` | single-probe DFT amplitude/phase vs uniform-fine reference; **not** physical R/T |
| box proxy crossval | implemented | `tests/test_sbp_sat_box_crossval.py` | internal arbitrary-box x/y-face plus edge/corner proxy fixtures; **not** public R/T |
| boundary proxy crossval | implemented | `tests/test_sbp_sat_boundary_crossval.py` | internal PMC reflector plus periodic full-axis/interior proxy fixtures; mixed PMC+periodic remains blocked; **not** public R/T |
| absorbing proxy crossval | implemented | `tests/test_sbp_sat_absorbing_crossval.py` | internal CPML interior-box decay and late-tail proxy fixtures; **not** public R/T or S-parameters |
| bounded-CPML point-probe R/T feasibility | inconclusive | `tests/test_sbp_sat_true_rt_feasibility.py` | internal measurement-contract probe only; **not** public R/T, S-parameters, flux, or port evidence |
| private analytic-sheet flux/DFT R/T benchmark gate | inconclusive | `tests/test_sbp_sat_true_rt_flux_dft_benchmark.py` | internal benchmark-only analytic-sheet history plus current private TFSF-style incident hook/accumulator evidence; no public source, DFT plane, flux monitor, TFSF, S-parameter, or true R/T promotion |
| true reflection/transmission | deferred | `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md` | no public R/T, S-parameter, or calibrated open-boundary claim yet |

Proxy tolerance is intentionally narrow and local: relative amplitude error
`<= 5%` and phase error `<= 5°` against a uniform-fine reference for the
current PEC-cavity proxy fixture. That tolerance checks numerical closeness to
the reference run; it does not validate incident/reflected/transmitted field
separation, energy balance, or calibrated S-parameters.

The private flux/DFT gate strengthens the internal accumulator regression by
checking multi-step, all-axis, windowed DFT semantics, retaining the prior
private analytic sheet/source history, and now adding a private TFSF-style
incident field plus a benchmark-only same-contract uniform reference.  The
current recorded status is still **inconclusive**: slab R/T scoring is
intentionally skipped because the same-contract private reference is present but
uniform/subgrid vacuum parity and transverse-uniformity gates remain below
threshold.  The row-2 metadata ranks the current full-aperture fixture blockers
instead of relaxing thresholds; the dominant blocker is
`transverse_phase_spread_deg`, followed by transverse magnitude spread and
vacuum magnitude parity.  The private solver/interface-floor investigation
now records `coarse_fine_energy_transfer_mismatch`: the signed front/back flux
ratio residual stays above the unchanged `0.02` parity threshold across the
predeclared boundary-expanded baseline and nearer bounded interface control,
while same-contract uniform self-error remains below threshold and direct
hook invariants still pass.  The bounded private repair stage then scored the
existing `tau` plumbing at `0.25`, `0.5`, `0.75`, and `1.0` without changing
the public default/API.  It records `no_material_repair`: the best candidate
(`tau=1.0`) reduced max front/back ratio error from `0.8813` to `0.7211`, below
the required 50% material-improvement gate, and introduced a paired
`vacuum_phase_error_deg` regression.  The private energy-transfer theory/design
review is now complete: it selects `discrete_eh_work_ledger_mismatch` as the
next kernel-repair candidate and records that no executable synthetic
diagnostics, solver edits, hook experiments, public API changes, or public
claims were added.  The follow-up private manufactured-interface energy-ledger
diagnostic is now executable and records `ledger_mismatch_detected`: matched
and zero-work probes stay private/CI-green, while the nonzero E/H face-work
probe reports normalized balance residual `0.1656` above the unchanged `0.02`
threshold.  The bounded private face-kernel feasibility stage then records
`no_signature_compatible_bounded_repair`: bounded current/minimum/reciprocal
face-jump candidates still fail the manufactured ledger threshold, while a
ledger-passing low-scale control is rejected as under-coupling.  The private
SAT face-coupling theory/redesign stage now records
`paired_face_coupling_design_ready`: a test-local same-step paired E/H
candidate closes the manufactured face ledger with normalized residual
`0.000578`, preserves matched-trace and zero-work gates, keeps coupling
strength near the current kernel, and requires paired E/H face context.
The private paired-face helper implementation gate now records
`production_context_mismatch_detected`: both non-CPML and CPML step paths apply
H SAT before the E update and E SAT after the E update, so the selected
same-step paired E/H candidate cannot access one co-temporal pre/post
current-SAT ledger state without time-centered staging.
The private time-centered E/H face-ledger staging redesign now records
`time_centered_staging_contract_ready`: same-call `H_pre_sat`/`H_post_sat` and
`E_pre_sat`/`E_post_sat` staging with a centered-H ledger closes the private
manufactured face ledger, passes the production-expressibility gate with named
CPML and non-CPML local slots, uses `FACE_ORIENTATIONS` only, and rejects
non-selected staging controls with explicit reordering, trace-availability, or
cross-step-state reasons.
The private time-centered paired-face helper implementation now records
`private_time_centered_paired_face_helper_implemented`: bounded private helper
functions in `sbp_sat_3d.py` capture same-call H/E face traces around the real
H-SAT/E-SAT slots and apply the selected `same_call_centered_h_bar` centered-H
correction after E SAT in both CPML and non-CPML paths, using a predeclared
bounded `0.02` relaxation to preserve existing proxy-regression gates.  This is a solver-internal
experimental behavior change only; runner, hook, API, `SimResult`, public
Result, default `tau`, DFT/flux/TFSF/port/S-parameter/true-R/T, and other public
observable surfaces remain closed.  Public promotion remains blocked until
fixture-quality plus R/T gates pass.  Periodic+CPML and public TFSF remain
rejected for this lane; the next safe lane is `private time-centered paired-face
helper fixture-quality recovery ralplan`, not a hook experiment, threshold
reinterpretation, helper-specific switch, or public TFSF promotion.
The private time-centered paired-face helper fixture-quality recovery lane now
records `measurement_contract_or_interface_floor_persists`: the finite C0/C1/C2/C3
candidate ladder keeps the original boundary-expanded fixture visible, treats
C1/C2 as measurement controls that cannot claim original-fixture recovery, and
rejects the only solver-touching C3 helper-relaxation candidate after rollback
to `0.02`.  All candidates remain above unchanged transverse-uniformity or
vacuum-stability thresholds, so public true-R/T/DFT/flux/TFSF/port/S-parameter
promotion remains closed.  The next safe lane is private
measurement-contract/interface-floor redesign, not a public observable promotion.
The private measurement-contract/interface-floor redesign lane now records
`persistent_interface_floor_confirmed`: the D0 authoritative integrated-flux
contract remains blocked, D2 phase-referenced modal coherence is not ready,
D3 local E/H impedance/Poynting normalization is not ready despite clean
intersection-mask provenance, and D4 current helper-state interface residual
evidence remains positive with prior committed manufactured-ledger context.
This is private diagnostic evidence only; public true-R/T/DFT/flux/TFSF/port/
S-parameter/API/result/hook promotion remains closed; public promotion remains closed.
The private interface-floor repair theory/implementation lane now records
`no_bounded_private_interface_floor_repair`: the F1
`oriented_characteristic_face_balance` candidate passes all-six-face orientation,
matched-trace, zero-work, coupling-strength, update-bound, and edge/corner
preacceptance guards, but it is algebraically equivalent to the current component
SAT update and leaves the manufactured ledger residual at `0.16562 > 0.02`.
No `sbp_sat_3d.py` repair hunk is retained, and public true-R/T/DFT/flux/TFSF/
port/S-parameter/API/result/hook promotion remains closed. The private higher-order
SBP face-norm/interface-operator redesign ladder now records
`no_private_face_norm_operator_repair`: the existing face restriction is already
the unmasked mass-adjoint of prolongation under the current diagonal face norms,
H1/H2 candidates fail the manufactured ledger or higher-order projection/noop
gates, and no solver hunk is retained. The next safe lane is
`private broader SBP derivative/interior-boundary operator redesign after face-norm/interface-operator ladder failed ralplan`.
That broader derivative/interior-boundary lane now records
`no_private_derivative_interface_repair`: the current ledger floor is reproduced
in the reduced private energy-identity fixture, a test-local face correction can
close that reduced identity, but the production lift is blocked by
`requires_global_sbp_operator_refactor`. No `sbp_sat_3d.py` or `face_ops.py`
solver/operator hunk is retained, the ledger threshold remains `0.02`, and
public true-R/T/DFT/flux/TFSF/port/S-parameter/API/result/hook promotion remains
closed. The private global SBP derivative/mortar operator architecture lane now
records `private_global_operator_3d_contract_ready`: A1-A4 identity evidence
passes for diagonal-norm SBP derivatives including Yee-staggered dual operators,
norm-compatible mortar projection, material/metric weighted tangential EM flux
closure, all-six-face edge/corner partition closure, and CPML/non-CPML SAT
staging evidence. No `sbp_sat_3d.py` solver hunk is retained, public true-R/T/DFT/flux/TFSF/port/S-parameter/API/result/hook
promotion remains closed. The private solver-integration gate now records
`private_solver_integration_requires_followup_diagnostic_only`: the
operator-projected face SAT adapter passes S1 preacceptance, but the S2
production-shaped dry run leaves the manufactured ledger residual above the
unchanged `0.02` threshold, so no `sbp_sat_3d.py` hunk is retained and the next
safe lane is private operator-projected face SAT energy-transfer redesign, not
public observable promotion. The private operator-projected energy-transfer
redesign now records `private_operator_projected_energy_transfer_contract_ready`:
the ratio-weighted scalar plus skew E/H operator work form closes the private
manufactured ledger below the unchanged `0.02` threshold, declared only a
separate solver-integration prerequisite at that stage, and retained no
`sbp_sat_3d.py`,
runner, hook, API, Result/SimResult, default-tau, true-R/T, DFT/flux/TFSF/port,
S-parameter, or public-promotion surface. The follow-up private bounded solver
integration now records
`private_operator_projected_solver_hunk_retained_fixture_quality_pending`: one
same-call solver-local skew E/H face helper is retained in `sbp_sat_3d.py` with
all-six-face `t1/t2` component slot-map, normal-sign orientation, CPML/non-CPML
parity, and edge/corner guards covered. Its post-existing-SAT solver call
disables scalar projection to avoid double-coupling; public observables and
public API/result/runner/hook surfaces remain closed. The private boundary
coexistence and fixture-quality validation lane now records
`private_boundary_coexistence_passed_fixture_quality_blocked`: direct step-path
probes exercise the retained helper under representative all-PEC, selected PMC,
periodic, and bounded-CPML accepted boundary arguments, while the canonical
`BoundarySpec` unsupported classes stay hard-fail. Unchanged fixture-quality
gates remain blocked by transverse uniformity and vacuum-parity errors, so the
follow-up private fixture-quality blocker repair lane now records
`private_fixture_quality_blocker_persists_no_public_promotion`: the F0-F4
finite ladder preserves the boundary-expanded baseline, reuses the existing
private fixture/source candidates and phase-referenced measurement diagnostics,
and selects the fail-closed terminal outcome because unchanged
transverse-uniformity and vacuum-parity gates still fail. The follow-up private
source/reference phase-front fixture-contract redesign lane now records
`private_source_phase_front_self_oracle_failed`: the uniform private
source/reference phase-front self-oracle itself exceeds unchanged phase-spread
and magnitude-CV thresholds before subgrid vacuum parity can be blamed. The
follow-up private analytic source phase-front self-oracle repair lane now
records
`private_analytic_source_phase_front_self_oracle_blocked_no_public_promotion`:
global temporal phase changes, sheet phase-center reasoning, the existing
center-core aperture proxy, and active-mask observable narrowing do not produce
a private uniform-reference phase-front self-oracle that passes unchanged
thresholds. The follow-up private analytic plane-wave source implementation
redesign lane now records `private_uniform_plane_wave_source_self_oracle_ready`:
a private prototype uniform plane-wave source self-oracle passes unchanged
phase-front thresholds without adding public TFSF/DFT/flux observables or
runtime/API surfaces. The follow-up private fixture contract recovery lane now
records `private_uniform_plane_wave_reference_contract_ready`: the plane-wave
self-oracle is accepted as a private same-contract uniform reference, but it is
not yet wired through subgrid-vacuum parity and does not unlock true R/T. The
follow-up private subgrid-vacuum plane-wave fixture lane now records
`private_plane_wave_subgrid_vacuum_fixture_blocked_no_public_promotion` because
that W1/R1 contract is not wired through the private subgrid-vacuum fixture
path. The follow-up fixture-path wiring lane now records
`private_plane_wave_fixture_path_wiring_blocked_no_public_promotion`: existing
private TFSF-style hooks are not the W1 plane-wave source contract, and this
lane cannot change forbidden runner/API surfaces. The follow-up adapter design
lane now records `private_runner_plane_wave_adapter_design_ready`, selecting the
existing private runner request/spec pattern rather than a direct JIT-only
bypass. The follow-up adapter implementation lane now records
`private_plane_wave_adapter_implemented_parity_pending`: the private request,
builder, JIT spec, subgrid helper, and same-contract reference helper can carry
the W1/R1 plane-wave contract. The follow-up private subgrid-vacuum parity
scoring lane now records
`private_subgrid_vacuum_plane_wave_parity_failed_no_public_promotion`: the score
is reproducible through the private adapter, but unchanged thresholds fail on
transverse phase spread, transverse magnitude CV, and vacuum stability. The
follow-up private blocker repair/design lane now records
`private_plane_wave_interface_floor_repair_design_required`: the failed parity
packet is preserved as the baseline, phase-front/source and measurement-contract
laundering candidates are rejected, and the next safe lane is a private
plane-wave interface-floor repair implementation plan with explicitly widened
production scope. That implementation lane now records
`no_private_plane_wave_interface_floor_repair`: no bounded local characteristic
or projection-only hunk can honestly claim material improvement over the failed
parity packet, so the next safe lane is private plane-wave interface-floor
architecture/root-cause redesign. That redesign lane now records
`private_plane_wave_interface_energy_form_root_cause_identified`: the source,
fixture-geometry, and projection-only explanations are rejected, and the next
safe lane is a private interface energy-form architecture repair design before
any new implementation hunk. That design lane now records
`private_plane_wave_interface_energy_form_implementation_contract_ready`: the
energy-potential and time-centered Poynting-work form are combined into a
future implementation contract with CPML/non-CPML, edge/corner, slow-metadata,
and public-surface guards. The follow-up implementation lane now records
`no_private_plane_wave_energy_form_implementation`: the frozen parity packet is
preserved, no bounded F1/F2 local solver hunk is retained as material
improvement, and the next safe lane is a private interface energy-form failure
theory/design review before true R/T readiness. That theory/design lane now
records `private_plane_wave_energy_form_redesign_contract_ready`: the local
implementation failure is explained as a missing operator-owned, time-centered
interface energy-state contract, and the next safe lane is a private
operator/mortar time-centered energy-form implementation plan. That
implementation lane now records
`no_private_plane_wave_operator_mortar_energy_form_implementation`: the current
solver staging does not expose a bounded owner for that time-centered interface
energy state, so the next safe lane is a private transverse phase-coherence
architecture redesign. That architecture lane now records
`private_plane_wave_phase_coherence_staging_contract_ready`: a single private
interface-state owner must preserve transverse phase spread and magnitude CV
jointly before any implementation hunk or true R/T readiness claim; public
promotion remains closed. The follow-up implementation lane now records
`no_private_plane_wave_phase_coherence_staging_implementation`: K1 would change
solver state shape and scan staging to introduce that private interface-state
owner, while K2 is only a scoring guard and cannot repair the solver path
without K1. The next safe lane is private plane-wave solver-wide
interface-state owner architecture redesign. That architecture lane now records
`private_plane_wave_solver_wide_interface_state_owner_contract_ready`: one
private owner and explicit scan/staging state-shape contract must be
implemented before phase-coherence parity repair can be claimed. The next safe
lane is private plane-wave solver-wide interface-state owner implementation;
that implementation lane now records
`no_private_plane_wave_solver_wide_interface_state_owner_implementation`: owner
state shape and same-step scan wiring require a private solver-state
propagation boundary contract for JAX pytree shape and runner/JIT
initialization before any bounded hunk can be retained. The next safe lane is
private plane-wave solver-state owner propagation boundary design. That design
lane now records `private_plane_wave_solver_state_owner_propagation_contract_ready`:
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
phase/CV and vacuum-regression gates. The next safe lane is private plane-wave
face-local modal correction architecture. That architecture lane now records
`private_plane_wave_face_local_modal_correction_contract_ready`: phase and
magnitude owner references map to private face-local tangential modal
contracts with paired phase/CV, vacuum-regression, and CPML/non-CPML symmetry
guards before any field-update hunk. The next safe lane is private plane-wave
face-local modal correction implementation after architecture contract ready.
That implementation lane now records
`no_private_plane_wave_face_local_modal_correction_implementation`: no bounded
field-update hunk is retained because paired phase/CV and vacuum-regression
gates were not honestly improved. The next safe lane is private plane-wave
face-local modal correction failure-theory redesign after the bounded
implementation failed. That redesign lane now records
`private_plane_wave_face_local_modal_failure_theory_contract_ready`: the failed
implementation is explained by same-step diagnostic owner-reference feedback
and characteristic-vs-observable modal-basis mismatch, so the next retry
requires a lagged owner reference and observable-aligned modal basis under
paired phase/CV and vacuum gates. The implementation retry lane now records
`no_private_plane_wave_face_local_modal_retry_implementation`: lagged owner
timing alone is insufficient, and the claims-bearing transverse plane-DFT
observable cannot be imported into solver-local field updates without private
observable-proxy modal-basis architecture. The next safe lane is private
plane-wave observable-proxy modal-basis architecture. That architecture lane now
records `private_plane_wave_observable_proxy_modal_retry_contract_ready`: a
solver-local transverse face energy/phase proxy can stand in for the benchmark
plane-DFT distribution without importing the DFT observable into field updates,
while lagged owner state plus paired phase/CV and vacuum guards remain required
before implementation. The next safe lane is private plane-wave
observable-proxy modal retry implementation. That implementation lane now
records `no_private_plane_wave_observable_proxy_modal_retry_implementation`:
the current solver owner state stores scalar phase/magnitude references per
active face, while a solver-local modal proxy requires packed face-local
distributions with offsets and masks before any field-update hunk can be
retained. The next safe lane is private plane-wave observable-proxy face-packet
state-shape design. That design lane now records
`private_plane_wave_proxy_face_packet_state_contract_ready`: packed face-local
proxy buffers, `FACE_ORIENTATIONS`-derived index metadata, and CPML/non-CPML
initialization symmetry can be specified without public API, hooks, benchmark
DFT, or public observable promotion. That implementation lane now records
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
thresholds. The next safe lane is private plane-wave observable-proxy modal retry
failure-theory redesign after packed-state parity scoring insufficient; public
promotion remains closed. That failure-theory lane now records
`private_plane_wave_modal_retry_failure_theory_redesign_contract_ready`: the
lagged packed face packet only aligned local face distribution state, while the
remaining blocker is propagation-aware transverse phase coherence plus an
incident-normalized observable basis with source/interface ownership separation.
The next safe lane is private plane-wave observable-proxy modal retry redesign
implementation after failure-theory contract ready; public promotion remains
closed. That implementation lane now records
`no_private_plane_wave_observable_proxy_modal_retry_redesign_implementation`:
no new propagation-aware modal-basis hunk is retained because the current
private owner state has only the interface-observed packet and no separate
source-owner/incident-normalizer packet. The next safe lane is private
plane-wave source-interface ownership state-shape design after modal retry
redesign implementation blocked; public promotion remains closed. That design
lane now records
`private_plane_wave_source_interface_owner_state_shape_contract_ready`: private
source-owner incident packets, interface-owner observed packets, and incident
normalizer buffers can remain fixed-shape, CPML/non-CPML/JIT symmetric, and
separate without public TFSF, benchmark DFT, hooks, or public observable
promotion. The next safe lane is private source-interface ownership state-shape
implementation after design contract ready; public promotion remains closed.
That implementation lane now records
`private_plane_wave_source_interface_state_shape_hunk_retained_fixture_quality_pending`:
private source-owner reference, incident-normalizer, weight/mask, offset/length,
and orientation buffers are retained separately from the existing interface
packet, with CPML/non-CPML/JIT initialization symmetry and no modal field-update
behavior change. The next safe lane is private propagation-aware modal retry
implementation after source-interface state-shape hunk retained; public
promotion remains closed. That implementation lane now records
`private_plane_wave_propagation_aware_modal_basis_hunk_retained_fixture_quality_pending`:
a bounded solver-local helper subtracts the private incident-normalized
source-owner packet from the lagged interface-owner packet before applying a
small tangential-E modal correction. It is no-op until source-owner incident
packets are populated, imports no benchmark DFT/flux/TFSF/port/S-parameter
state, and keeps true R/T readiness pending. The next safe lane is private
propagation-aware modal retry parity scoring after the source-normalized hunk
is retained; public promotion remains closed. That scoring lane now records
`private_plane_wave_propagation_aware_modal_retry_parity_scored_fixture_quality_pending`:
the retained hunk has finite private parity evidence, but unchanged material-
improvement and readiness gates do not pass because production source-owner
incident packet population is still absent. The next safe lane is private
source-owner incident packet population design; public promotion remains
closed. That design lane now records
`private_plane_wave_source_owner_incident_packet_population_contract_ready`:
source-owner reference, incident-normalizer, packet offset/orientation, and
pre-modal-retry timing contracts are private, fixed-shape, CPML/non-CPML/JIT
symmetric, and non-aliasing with the interface-owner packet. The implementation
lane now records
`private_plane_wave_source_owner_incident_packet_population_hunk_retained_fixture_quality_pending`:
a private packetization helper populates source-owner incident packets before
propagation-aware modal retry, preserves the interface-owner packet, and keeps
CPML/non-CPML wiring plus public promotion closed. The next safe lane is private
source-populated propagation-aware modal retry parity scoring after the
source-owner packet hunk is retained. That scoring lane now records
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

### Explicit unsupported combinations in the SBP-SAT lane

| Combination | Status | Expected behavior |
|---|---|---|
| UPML boundary | unsupported | hard-fail |
| CPML refinement box inside absorber guard | unsupported | hard-fail |
| per-face CPML thickness override with subgrid | unsupported | hard-fail |
| mixed reflector + CPML `BoundarySpec` | unsupported | hard-fail |
| mixed periodic + CPML `BoundarySpec` | unsupported | hard-fail |
| mixed PMC + periodic `BoundarySpec` | unsupported | hard-fail |
| periodic axis touched on only one side | unsupported | hard-fail |
| geometry-driven `xy_margin` auto-box refinement | unsupported | hard-fail |
| NTFF | unsupported | hard-fail |
| DFT planes | unsupported | hard-fail |
| flux monitors | unsupported | hard-fail |
| TFSF | unsupported | hard-fail |
| waveguide or Floquet ports | unsupported | hard-fail |
| lumped RLC | unsupported | hard-fail |
| coaxial ports | unsupported | hard-fail |
| impedance point ports or wire/extent ports | unsupported | hard-fail |

## Reference-lane support table

| Dimension | Supported subset |
|---|---|
| grid / runner | uniform Cartesian Yee |
| materials | isotropic linear, conductive, validated Debye/Lorentz/Drude subsets |
| absorbers | PEC, CPML, bounded UPML |
| sources | point/current, lumped port, wire port, waveguide port |
| observables | probes, flux, calibrated S-parameters, Harminv resonance, benchmarked NTFF |
| optimization-facing observables | validated proxy objectives only until explicitly promoted |

## Nonuniform graded-z shadow lane

Current retained subset:
- graded-z thin-substrate workflows with probes and current-style excitation
- smoke/convergence coverage in `tests/test_nonuniform_api.py` and `tests/test_nonuniform_convergence.py`
- selected dispersive-material smoke coverage

Current policy:
- preserved for continuity and qualification work
- **not** part of the claims-bearing reference lane
- no silent fallback / no silent feature dropping
- no new promotions until contract + benchmark ladder exist

### Explicit unsupported combinations in the nonuniform lane

| Combination | Status | Expected behavior |
|---|---|---|
| Floquet + nonuniform | unsupported | hard-fail |
| NTFF + nonuniform | unsupported | hard-fail |
| DFT planes + nonuniform | unsupported | hard-fail |
| TFSF + nonuniform | unsupported | hard-fail |
| Waveguide ports + nonuniform | unsupported | hard-fail |
| Lumped RLC + nonuniform | unsupported | hard-fail |

## Promotion rule

A lane or feature can be promoted to **supported** only when all of the following exist:
1. support-matrix entry
2. explicit source/observable contract where relevant
3. unit + integration tests
4. benchmark / convergence evidence
5. docs/examples/API wording aligned to the promoted scope
