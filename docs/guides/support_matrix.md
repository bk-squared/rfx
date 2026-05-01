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
runtime/API surfaces. The next safe lane is private fixture contract recovery
using the plane-wave source self-oracle, not public promotion.

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
