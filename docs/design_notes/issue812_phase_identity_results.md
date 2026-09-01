# issue #812 P1 — self-referential phase gates (cv20, cv21): results

Companion to `issue812_phase_identity_predeclaration.md`, which fixed every
threshold below in the commit **preceding** the one that measured them.
Measured: 2026-09-01, branch `agent/regate-phase-identity`.

Append-only. Corrections go in a dated section at the bottom.

## 0. How these numbers were obtained

openEMS is absent from this host, so both cases exit 2 here and neither was
re-run. Every measurement is a **replay of committed field data through the real
witness functions** — the pattern
`test_matched_through_witness_run3_regression_measured_vs_analytic_beta` already
established in `tests/test_coax_two_port_referee_header.py`. Four committed
configurations were used, two per case:

| case | configuration | source |
|---|---|---|
| cv21 | registered mesh, `dx_scale = 1.0`, VESSL 369367251629 | the `_RUN3_*` literals in `tests/test_coax_two_port_referee_header.py` |
| cv21 | 1.5x refinement, `dx_scale = 2/3`, VESSL 369367251845 | `validation/crossval/_21_coax_two_port_referee_logs/mesh_refinement_369367251845_result.json` |
| cv20 | declared board (openEMS `h_sub` 254um), VESSL 369367251705 | `validation/crossval/_20_msl_phase_referee_logs/20260804T055009Z_result.json` |
| cv20 | #723 realized board (openEMS `h_sub` 300um), VESSL 369367256520 | `validation/crossval/_20_msl_phase_referee_logs/20260827T102342Z_result.json` |

No physics verdict changed in either case. The gates were blind; the answers
were not challenged and are not challenged here.

## 1. cv21 — coax two-port referee

### 1.1 The blindness, reproduced

Pinned by `test_matched_through_witness_is_identically_blind_to_a_coherent_beta_error`.

* The audit's construction (a synthetic through, `angle(S21) = -k*beta*L`
  exactly): `max_phase_dev_deg` = **0.000000** and `group_delay_dev_ps` =
  **0.000000** at `k` = 1.02 / 1.10 / 1.30 / 1.50 / 1.57, against the 30 deg /
  200 ps gate. The audit's figure reproduces exactly.
* On run-3's own committed data, with the propagation term scaled and the
  extraction residual left alone: the deviation is **bit-identical to its
  unperturbed value, 0.4365 deg**, at every one of those `k`. Same identity,
  stated on real data.

### 1.2 Criterion (A) — the case still passes, at two meshes

| configuration | `N` (annulus cells) | envelope `BOUND(N)` | `beta` dev | group-delay dev |
|---|---|---|---|---|
| registered, `dx_scale = 1.0` | 3.789288 | **0.157022** | 0.121179 | 0.125221 |
| 1.5x refinement, `dx_scale = 2/3` | 5.683932 | **0.086002** | 0.066247 | 0.067494 |

Margin on the `beta` leg is **1.2958x** (registered) and **1.2982x**
(refined); on the group-delay leg 1.2540x and 1.2742x.

State the reason plainly rather than presenting ~1.30x as a discovery: the
envelope's scale *is* the registered mesh's committed excess times the declared
headroom, so the `beta` margin is the headroom **by construction**. Both points
land just under 1.30 for the same small reason — the record's `excess_before`
(`0.1208`) and `excess_after` (`0.06616248`) are the run's own summary values,
while the gate takes the per-bin max over the gated central band (`0.121179`
and `0.066247`). Measured against the record's own values the margin is
1.29985x at both meshes; the 0.03% below 1.30 is `N_REF = 3.789` being the
record's rounded cell count against the layout's 3.789288.

What the two configurations establish independently is that the *functional
form* transports — the refined mesh's measured excess lands where the committed
convergence law says it should, so the envelope neither strands the registered
mesh nor goes slack under refinement.

The measured `beta` ratio is unchanged at 1.1205–1.1212 across the gated central
band: this lane moved what the gate compares against, not the physics.

### 1.3 Criterion (B) — the new gate fires on the audit's defect

Perturbation: `beta_measured -> k*beta_measured` **and**
`S21 -> S21*exp(-1j*(k-1)*beta_measured*L12)`, i.e. the line's real propagation
constant is wrong and the port measures the wrong value along with the wrong
through-path phase.

| `k` | old witness (E1) | new witness (E2), `beta` dev vs bound 0.157022 |
|---|---|---|
| 1.02 | PASS, 0.4365 deg | **PASS**, 0.1436 — below the pre-declared floor `k = 1.032334` |
| 1.10 | PASS, 0.4365 deg | **FAIL**, 0.233296 |
| 1.30 | PASS, 0.4365 deg | **FAIL**, 0.457532 |
| 1.50 | PASS, 0.4365 deg | **FAIL**, 0.681768 |
| **1.57** | PASS, 0.4365 deg | **FAIL**, 0.760250 (4.8x over) |
| 0.50 | PASS, 0.4365 deg | **FAIL**, 0.439739 |

Failure reason checked, not assumed: the `RuntimeError` text names
`analytic-beta witness failed`, `MEASURED beta`, `continuum coax TEM beta`, and
reports both `beta dev=` and `group-delay dev=` with `(ok=False)` — it is the
beta/group-delay legs that red, not a magnitude-band or passivity side effect.

`k = 1.57` also reds through the **real `_run_stage_b` wiring** (not the witness
in isolation), via the `_run_one_drive` replay pattern, and the raised error
still carries `partial_stage_b_data`. The same harness at `k = 1.0` gives
`sanity_passed = True` with `beta` dev 0.121179 — so the wiring passes and fails
for the right inputs.

### 1.4 The declared detection floor, restated as a limitation

`k = 1.02` does **not** fire, exactly as pre-declared. This is not a residual
gap to be closed by tightening: at 3.8 cells across the PTFE annulus the real
staircase bias is 12.08%, and no analytic gate can be tighter than the bias it
must tolerate. The floor is `k > 1.032334` / `k < 0.752106` at the registered
mesh and it **moves with the mesh**: at the committed 1.5x refinement the
envelope is 0.086002 against a 6.62% bias, which puts the floor at
`k > 1.018528` / `k < 0.857210` — i.e. **the refined mesh does catch the audit's
`k = 1.02`**, with no change to the gate. Refining the mesh is the only thing
that improves the floor, and the envelope follows automatically.

### 1.5 Evidence-level correction

**No gated Stage-B leg in cv21 is E4.** Stage B is a single-solver openEMS
fixture that reads no rfx S-parameters, so the second candidate reference —
"the committed external solver's own phase" — does not exist for this case. The
`E4` in its registration covers Stage A's external tutorial reproduce-gate only.
Stage B's gated phase claim is now **E2** (analytic beta) **plus E1**
(self-consistency, kept and unchanged). `evidence_levels` gains `E1`, with a
matching `self-invariant` reference entry, and the `claim_scope` says all of
this in full.

Stage A needed nothing: its own matched-through call already passes `beta=None`
(the analytic path), so that leg has always been E2 — which is also why the
audit found the defect only on Stage B.

## 2. cv20 — MSL phase referee

### 2.1 The blindness, reproduced — and a third instance of it

Pinned by `test_self_consistency_witness_is_blind_to_a_factor_two_phase_velocity_error`
and `test_dispersion_corrected_residual_is_blind_for_the_same_reason`.

* The audit's construction (scale the de-embedded phase, which scales the
  extraction residual with it): **0.2414 deg** against the 3.0 deg gate. The
  audit's figure reproduces exactly.
* The propagation-only construction: the deviation is **bit-identical to its
  unperturbed 0.12072 deg** at `k = 2.0` and `k = 0.5`.
* `residual_phase_diff_after_dispersion_deg` — the number the script's own
  docstring recommends as the honest cross-solver quantity — is **blind for the
  same reason**: `residual = raw_diff - (beta_openems - beta_rfx)*L12` subtracts
  a term built from `beta_rfx`. Measured: baseline max 0.715345 deg, and
  **0.715345 deg** again with the rfx phase velocity halved, `np.allclose` to
  1e-12. The raw difference on the same perturbation goes to 44.8146 deg. This
  is why the **raw** difference is what got gated.

### 2.2 Criterion (A) — the case still passes, on two committed runs

| configuration | `beta_rfx` vs HJ (tol 0.020) | `beta_openems` vs HJ (tol 0.020) | raw cross-solver phase (tol 3.0 deg) |
|---|---|---|---|
| run-2, #723 realized board | **0.00938** (2.1x margin) | **0.00307** (6.5x) | **0.3418 deg** (8.8x) |
| run-1, declared board | **0.00938** (2.1x) | **0.00494** (4.1x) | **0.3039 deg** (9.9x) |

The rfx side is the same committed fixture in both runs, so its figure is
identical by construction; the openEMS side differs because its board does
(254um in run-1, 300um in run-2), and each is judged against the closed form of
**its own** board. In production there is only one board — post-#723 Stage B
builds openEMS on the rfx fixture's realized geometry, which is what
`_run_stage_b` reads (`layout["w_trace_realized_m"]`,
`layout["h_sub_realized_m"]`, pinned by
`test_independent_phase_legs_are_wired_into_sanity_passed`). Run-1 predates that
match, so its replay supplies the declared-board `eps_eff` explicitly; that is
what makes it a genuinely second configuration rather than a re-run of the
first.

These are genuinely independent margins, unlike cv21's: the tolerance was
derived from a model-error budget (HJ accuracy + one-cell conductor thickness +
neglected dispersion = 1.77%, rounded to 2.0%) with no measured quantity in it,
and the measurement landed 2.1x inside.

### 2.3 Criterion (B) — both new gates fire

Perturbation: the rfx side's phase velocity halved coherently
(`beta_rfx -> 2*beta_rfx`, `S21_rfx` rotated by `exp(-1j*beta_rfx*L12)`).

| gate | `k = 2.0` | `k = 0.5` |
|---|---|---|
| E1 self-consistency (unchanged) | PASS, 0.1207 deg | PASS, 0.1207 deg |
| E2 analytic beta, tol 0.020 | **FAIL**, 1.018764 | **FAIL**, 0.495643 |
| E4 raw cross-solver phase, tol 3.0 deg | **FAIL**, 44.8146 deg | **FAIL**, 22.1883 deg |

Failure reasons checked: the E2 message names
`analytic-beta witness failed for solver 'rfx'` and
`Hammerstad-Jensen quasi-static` — it **attributes** to the rfx side; the E4
message names the two solvers' de-embedded phases and says explicitly that it
does **not** say which one is wrong. That division is deliberate: a cross-solver
disagreement gate that pretended to attribute would be the same overclaim this
issue is about.

### 2.4 Evidence-level correction, and one narrowed scope fence

cv20 was registered `["E2", "E4"]` while every gated leg was intra-solver
self-consistency; the E2 came from Stage A's notch oracle and the E4 from a leg
that was computed and printed but never gated. After this change the E4 is
**gate-backed for the first time**, a new E2 leg gates both solvers' `beta`
against a closed form, and `evidence_levels` gains `E1` with a matching
`self-invariant` reference entry naming the self-consistency witness.

One consequence to state rather than bury: the script's scope fence says it
"brackets, does not judge rfx's own numbers", and the new E4 gate means a
sufficiently large cross-solver phase disagreement now makes the script exit 1.
The fence is narrowed, deliberately and only this far — the gate asserts that
the two solvers agree, and its message refuses to say which is wrong. The E2 leg
is what attributes, and it attributes to whichever side is out of envelope,
including openEMS.

## 3. What was not changed

* No existing gate was widened, and none was tightened either.
  `B_PHASE_TOL_DEG = 3.0`, `B_GD_TOL_PS = 200`, `phase_tol_deg = 30`,
  `gd_tol_ps = 200`, every magnitude band and every passivity tolerance are
  byte-identical.
* No physics verdict moved. cv21's measured `beta` ratio is still 1.1205–1.1212;
  cv20's cross-solver phase agreement is still 0.342 deg in band.
* Neither solver was re-run, and no VESSL job is required by this lane. A future
  live run of either case exercises the new gates for the first time on fresh
  data; the replay evidence above is what stands until then, and it is the same
  class of evidence the repo already accepted for the run-3 fix.
* cv05's prose, cv09, cv10, cv02 and cv14 (Phase 0/1, PRs #814–#818) were not
  touched.

## 4. Filed elsewhere, not fixed here

Nothing physics-class was found.

One instrument-class observation, recorded so it is not lost:

* cv20's `self_consistency_rfx` failure is caught and recorded rather than
  raised (so the artifact completes), which is reasonable — but it means the
  rfx side's E1 leg is not in `sanity_passed`. The two new gates cover the rfx
  side directly, so this is no longer a hole in the coherent-beta class; it is
  still an asymmetry worth a deliberate decision, and it belongs to whoever owns
  cv20's exit-code policy, not to this re-gate.

One claim this note nearly made and does **not**, because the refuting search
was run: while reading the committed refinement artifact
`mesh_refinement_369367251845_result.json` it looked as though
`MESH_REFINEMENT_PREDECLARATION` was still `"UNRUN"` with
`measured_ratio_after = None` despite the run being committed — an
evidence-chain fill of Phase 0's class. That is **wrong**. The `UNRUN` in the
artifact is the dict as it stood *at run time*, snapshotted into the run's own
output, which is exactly the fill contract working. The record in `main` today
reads `status = "RUN"`, `measured_ratio_after = 1.0661624823818885`,
`vessl_run_id = "369367251845"`, `log_path` set, `verified_on = "2026-08-05"`,
and `implied_convergence_order = 1.4847707054524188`. Checked by loading the
module and printing the dict, not by reading the artifact a second time. The
envelope in §1 is pinned against that record's own
`implied_convergence_order` field for exactly this reason.

## 3. Corrections — round 2 (2026-09-01)

Append-only; §§0–2 above are the round-1 record and are not edited. Every number
below is an artifact key in
`validation/crossval/_issue812_phase_identity/regate_evidence.json`, written by
`scripts/diagnostics/build_issue812_phase_identity_evidence.py` (no FDTD; a
replay of the four committed configurations through the referees' own witness
functions) and kept current by
`tests/test_issue812_phase_identity_evidence.py`'s
`test_committed_artifact_equals_a_fresh_replay`.

**3.1 Blocker (cv20) — the inverted gating decision, still standing where it was
first written.** Round 1 gated the raw cross-solver `angle(S21)` difference but
left the superseded "REPORTED, not gated" decision at five live sites: the
manifest entry's `references[1].name`, the superseded REPORTED list inside the
same entry's `claim_scope`, the source comment that introduces the block whose
own witness call now gates the quantity, and the two module-docstring paragraphs
that report it.
Each now reads as the code behaves, and the decision itself is machine-readable at
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv20.cross_solver_raw_phase_difference_is_gated`.
The 2026-08-04 promotion records (the module docstring's `PROMOTED` section and
`MUST_MOVE_WHEN_VALIDATED`) still describe the pre-#812 decision and are
deliberately **not** edited: they record what the #490 reviewer approved on that
date, which remains true of that date.

**3.2 cv21 criterion (A) — the "establish independently" clause is withdrawn.**
§1.2's closing sentence claimed the two meshes independently establish that the
functional form transports; they do not, because the committed convergence order
*is* the two-point fit through the two committed excesses, so the refined-mesh
envelope equals the declared headroom times that mesh's own committed excess to
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.margin_is_the_declared_headroom_by_construction.bound_at_n_after_minus_headroom_times_excess_after`
with a recovered-order error of
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.margin_is_the_declared_headroom_by_construction.order_p_recovery_abs_error = 0`
— the refined-mesh check is arithmetically the registered-mesh statement, so
criterion (A)'s *margin* is established by one configuration, not two. What the
refined replay does independently confirm is narrower and still worth having:
the committed refined-mesh data still reads what its record says it reads, and
the witness runs on it unchanged.

**3.3 cv21 detection floor — two-sided, in every summary.** §1.3's `k = 1.02` row
names only the HIGH side; the floor is two-sided and the gate is blind for `k`
between
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.predeclared_k_lo = 0.752106`
and
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.predeclared_k_hi = 1.032334`
at the registered mesh, so a one-sided restatement understates the blind interval
by its whole LOW side.

**3.4 cv21 `E4` — withdrawn, and attributed where the comparison actually
happens.** §1.5 kept `E4` on the argument that it covered Stage A's tutorial
reproduce-gate; the refuting search says otherwise — the referee imports no rfx
module and reads no rfx fixture, so no leg puts an rfx quantity on either side of
a comparison (`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.e4_supporting_leg_count = 0`,
pinned by `test_cv21_registers_no_e4_because_no_leg_supports_one`) — so
`evidence_levels` is now `["E1", "E2"]` and the E4 is attributed to the
downstream `compute_coaxial_two_port` label-lift chain in
`docs/guides/sparameter_support_matrix.md`, which is where rfx's numbers are
actually compared against this referee's openEMS output.
