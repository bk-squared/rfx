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

The immediate lane is now private plane-wave modal projection/normalizer
projected target residual-basis energy-biorthogonal source/interface transverse
modal transfer-map target-basis orientation residual phase/magnitude balance
residual modal-coupling packet-basis mismatch owner-packet weighting modal
energy/impedance transverse energy redistribution coupled modal energy-balance
target-basis packet normalization source/interface packet energy
co-normalization phase-energy residual implementation after the private
failure-theory redesign selected a bounded next target under unchanged
material-improvement, transverse-uniformity, and vacuum-stability gates.
The previous implementation lane records
`private_plane_wave_source_owner_incident_packet_population_hunk_retained_fixture_quality_pending`:
a solver-local packetization helper now populates private source-owner incident
packets before propagation-aware modal retry, preserves the interface-owner
packet, and keeps CPML/non-CPML wiring plus public promotion closed. The latest
scoring lane records
`private_plane_wave_source_populated_propagation_aware_modal_retry_hunk_insufficient_fixture_quality_pending`:
the populated source-owner packet is consumed by the private modal retry, but
unchanged material-improvement and true R/T readiness gates remain closed. The
failure-theory lane now records
`private_plane_wave_source_populated_modal_retry_time_alignment_theory_contract_ready`:
the next bounded private target is time-aligning the lagged interface-owner
packet with the same-step source-owner packet before another modal retry hunk.
The design lane now records
`private_plane_wave_source_interface_time_aligned_packet_staging_contract_ready`:
private fixed-shape staged packet fields, CPML/non-CPML/JIT initialization, and
modal-retry consumer timing are specified without hooks or public observables. The
implementation lane now records
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
baseline, so the next bounded private target is shared modal projection, packet
normalization, and face-mask weighting equivalence rather than another public
observable or true R/T readiness claim. The contract design lane now records
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
projected-basis redesign contract. The projected-basis design lane now records
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
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_hunk_insufficient_fixture_quality_pending`: it consumes the retained implementation metadata, preserves frozen thresholds and baseline/current metrics, records finite private scoring evidence, and keeps true R/T readiness, runners, hooks, exports, public observables, and threshold changes closed. The next safe step is private failure-theory redesign for the source/interface packet energy co-normalization floor.

The source/interface packet energy co-normalization failure-theory lane now records
`private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_coupled_modal_energy_balance_target_basis_packet_normalization_source_interface_packet_energy_conormalization_phase_energy_residual_floor_theory_ready`: it consumes the finite-but-insufficient co-normalization parity evidence, rejects repeat energy-only normalization and public/threshold escape routes, and selects private source/interface phase-energy residual implementation inside the retained 3x3 transfer-map contract. No true R/T readiness, runner state, threshold, public observable, API/export, or docs-public/examples promotion is unlocked by this theory lane.


The private owner state shape also propagates through CPML and non-CPML subgrid steps plus
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
The follow-up implementation attempt keeps true R/T readiness pending and records
`no_private_plane_wave_observable_proxy_modal_retry_redesign_implementation`:
no new propagation-aware modal-basis hunk is retained because the current
private owner state has only the interface-observed packet and no separate
source-owner/incident-normalizer packet. The source/interface state-shape design
lane now records
`private_plane_wave_source_interface_owner_state_shape_contract_ready`: private
source-owner incident packets, interface-owner observed packets, and incident
normalizer buffers can remain fixed-shape, CPML/non-CPML/JIT symmetric, and
separate without public TFSF, benchmark DFT, hooks, or public observable
promotion. The next lane may implement that private state shape before another
field-update hunk. That implementation lane now records
`private_plane_wave_source_interface_state_shape_hunk_retained_fixture_quality_pending`:
private source-owner reference, incident-normalizer, weight/mask, offset/length,
and orientation buffers are retained separately from the existing interface
packet, with CPML/non-CPML/JIT initialization symmetry and no modal field-update
behavior change. The next lane may use the retained source/interface state shape
for a private propagation-aware modal retry. That implementation lane now
records
`private_plane_wave_propagation_aware_modal_basis_hunk_retained_fixture_quality_pending`:
a bounded solver-local helper uses the private source-owner packet, incident
normalizer, and lagged interface-owner packet while remaining no-op without
populated source-owner incident packets and importing no public observable or
benchmark DFT/flux/TFSF/port/S-parameter state. The next lane is private
propagation-aware modal retry parity scoring. That scoring lane now records
`private_plane_wave_propagation_aware_modal_retry_parity_scored_fixture_quality_pending`:
finite private scoring is available, but material-improvement and true-R/T
readiness gates remain below threshold because production source-owner incident
packet population is not wired yet. The next lane is private source-owner
incident packet population design. That design lane now records
`private_plane_wave_source_owner_incident_packet_population_contract_ready`:
source-owner reference, incident-normalizer, packet offset/orientation, and
pre-modal-retry timing contracts can stay private, fixed-shape,
CPML/non-CPML/JIT symmetric, and non-aliasing with the interface-owner packet.
The implementation lane now records
`private_plane_wave_source_owner_incident_packet_population_hunk_retained_fixture_quality_pending`,
the source-populated scoring lane is insufficient, and the failure-theory lane
selects private source/interface time-aligned packet staging. That design
contract is now ready, so the next lane is private staging implementation.
The private solver-local
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
the face-packet implementation lane has retained the packed state/capture hunk,
and the modal retry lane has retained the private packed-state E-modal correction
hunk, but the parity-scoring lane has shown that hunk is insufficient under the
unchanged material-improvement rule. The failure-theory lane now defines the
private redesign contract: a later implementation must use a
propagation-aware, incident-normalized modal basis with source/interface
ownership separation. The redesign implementation lane has now failed closed on
the missing source-owner/incident-normalizer packet, so the next private design
step was source-interface ownership state shape. That design contract is now
ready, and the fixed-shape source-owner/interface-owner state has now been
retained without public promotion or modal field-update behavior changes. The
private source-owner incident packet population hunk is also retained and the
source-populated modal retry parity scoring lane is now insufficient. The
failure-theory lane selected time-aligned source/interface packet staging, and
the staging design contract is now ready. The next private gate is staging
implementation. The goal gate therefore remains claims-closed until a later
redesign/readiness lane proves the real solver path improves without breaking
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
