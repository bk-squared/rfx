# issue #812 P1 — self-referential phase gates (cv20, cv21): threshold pre-declaration

Lane: `phase-gates` (issue #812 audit pattern **P1**, the two cases the Phase-1 batch
#815–#818 did not touch).
Pre-declared: 2026-09-01, in the commit that carries this file, **before** any of the
numbers this lane will measure were computed.
Scope: `validation/crossval/20_msl_phase_referee.py`,
`validation/crossval/21_coax_two_port_referee.py`, and their header suites.

This note is append-only. Corrections go in a dated section at the bottom stating the
old value and why it was wrong.

---

## 1. The defect, restated from the audit

Both cases build the expected phase of a through line out of the **same run's own
measured propagation constant**:

```
expected_phase = -beta_measured * L        # cv21 _matched_through_witness (beta given)
expected_phase = -beta_re * l12_m          # cv20 _self_consistency_witness
measured_phase = unwrap(angle(S21))
```

`beta_measured` and `angle(S21)` are two readings of one field solve. A **coherent**
error in the line's propagation constant — the physical phase velocity is wrong, so the
port's measured `beta` and the through-path phase move together by the same factor `k` —
cancels identically:

```
measured_phase - expected_phase = -k*beta*L - (-k*beta*L) = 0
```

The audit measured the consequence:

* **cv21** — perturbing `k*beta` with line and port moved together at
  `k = 1.02 / 1.10 / 1.30 / 1.50 / 1.57` gives `max_phase_dev` **exactly 0.000 deg** and
  `gd_dev` **exactly 0.00 ps** at every `k`, against a 30 deg / 200 ps gate.
* **cv20** — a **factor-2 error in phase velocity** (beta and `angle(S21)` moved
  together) reads **0.2414 deg** against the **3.0 deg** gate: 12x inside.

The group-delay leg cancels for the same reason: it is a derivative of the same
identity. cv20's `residual_phase_diff_after_dispersion_deg` is blind for a *third*
instance of the same reason —
`residual = raw_diff - (beta_openems - beta_rfx)*L12` subtracts a term built from
`beta_rfx`, so doubling `beta_rfx` and the rfx phase together leaves the residual
unchanged. Only the **raw** cross-solver difference is sensitive.

## 2. What "independent" has to mean here

A reference is independent iff **no quantity produced by the run under judgement appears
on the reference side**. Two candidate reference classes exist for these two fixtures;
both are evaluated, and the choice per case is by derivation, not preference.

| candidate | cv21 (coax) | cv20 (MSL) |
|---|---|---|
| analytic quasi-TEM `beta` from declared geometry + materials | `beta = omega*sqrt(eps_r)/c` is **exact** for the continuum coax TEM mode — it depends on `eps_r` alone, not on `a`, `b`, or the mesh. Reference side contains zero run quantities. **Chosen.** | Hammerstad–Jensen `eps_eff` on the **realized** board (`h_sub = 300um`, `w = 600um`, `eps_r = 3.66`) — a closed form of declared geometry only. **Chosen.** |
| the committed external reference's own phase | **Does not exist.** cv21 Stage B is a single-solver (openEMS) fixture; it reads no rfx S-parameters at all. There is no second phase to compare against. | **Exists.** Stage B already reads the committed rfx fixture `tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json` and already computes `raw_phase_diff_deg` between the two solvers' de-embedded `angle(S21)` — **reported, never gated**. **Also chosen** (a second, genuinely E4 leg). |

Consequence for cv21: its Stage-B phase claim can be pinned to an analytic (E2) oracle
and to intra-run self-consistency (E1), but **not** to an external solver. Its
registered `evidence_levels` must say so.

## 3. cv21 — pre-declared threshold

### 3.1 Gated quantity (new)

```
beta_dev(f) = beta_measured(f)/beta_analytic(f) - 1
beta_analytic(f) = 2*pi*f*sqrt(B_PTFE_EPS_R)/c0          # exact continuum coax TEM
gate: max |beta_dev(f)| over the gated central band  <=  BOUND(N)
```
and, under the **same** fractional envelope, the through-path group delay measured from
`S21` alone against the analytic value:
```
gd_dev = gd_measured/gd_analytic - 1,   gd_analytic = L12*sqrt(eps_r)/c0
gate: max |gd_dev| over the gated central band  <=  BOUND(N)
```

### 3.2 The envelope, and why it is not fitted

`beta_measured` at this fixture's registered mesh is a **real ~12% above** the continuum
value. That is not a defect: it is Yee staircasing of a curved PTFE annulus spanned by
only ~3.8 cells, already diagnosed, attributed, and — decisively — **shown to converge**
by the committed mesh-refinement run. So the envelope must be mesh-dependent, and its
scale comes from committed prior provenance, not from anything this lane measures.

Prior provenance, all present in `main` today
(`MESH_REFINEMENT_PREDECLARATION` in `validation/crossval/21_coax_two_port_referee.py`,
pre-declared 2026-08-04, filled from VESSL 369367251845):

| symbol | value | source (verbatim literal in main) |
|---|---|---|
| `EXCESS_REF` | `0.1208` | `MESH_REFINEMENT_PREDECLARATION["excess_before"]` |
| `N_REF` | `3.789` | `MESH_REFINEMENT_PREDECLARATION["annulus_cells_before"]` = `(2.055-0.635)/0.37474` |
| `P` | `1.4847707054524188` | `MESH_REFINEMENT_PREDECLARATION["implied_convergence_order"]` — the committed two-point order from VESSL 369367251845 (`status = "RUN"`) |

Declared envelope:

```
N       = (B_B_MM - B_A_MM)/dx_mm                 # annulus cells, pure geometry
BOUND(N) = HEADROOM * EXCESS_REF * (N_REF/N)**P
HEADROOM = 1.30                                   # DECLARED HERE
```

* **Form** (`(N_REF/N)**P`) is the committed convergence law, not a new model.
* **Scale** (`EXCESS_REF` at `N_REF`) is committed prior provenance.
* **`HEADROOM = 1.30`** is the only new number. It is a declared safety factor, chosen
  before measurement as one round step above unity — not derived from any residual.
  Its cost is stated in §3.3 rather than hidden.

The property the audit asked for and the old gate did not have: **this envelope shrinks
when the solver's mesh improves.** `BOUND` at the registered mesh is `0.157040`; at the
committed 1.5x refinement (`N = 5.6835`) it is `0.086011`. A gate whose width tracks
`round-up(measured x 1.5)` never tightens; this one tightens as `N**-1.485`.

### 3.3 Declared detection floor (stated, not discovered)

With the registered mesh's real excess at `EXCESS_REF`, a coherent `k*beta` perturbation
fires the gate when

```
|k*(1+EXCESS_REF) - 1| > BOUND   ->   k > 1.032334  or  k < 0.752106
```

So the new gate **cannot** discriminate `k = 1.02`. That floor is a physical property of
this mesh's staircase envelope, not a choice: no honest analytic gate at 3.8 annulus
cells can be tighter than the staircase bias itself. It **does** discriminate the audit's
`k = 1.10 / 1.30 / 1.50 / 1.57` and the `k = 0.5` (factor-2) case. Criterion (B) for this
lane is `k = 1.57`.

### 3.4 Falsifier (pre-declared)

1. **(A)** The gate must PASS on the committed run-3 registered-mesh data
   (the `_RUN3_*` literals in `tests/crossval/test_coax_two_port_referee_header.py`, VESSL 369367251629) and on
   the committed 1.5x-refinement run
   (`_21_coax_two_port_referee_logs/mesh_refinement_369367251845_result.json`), with the
   margins reported.
2. **(B)** The gate must FAIL on `s21 -> s21*exp(-1j*(k-1)*beta_measured*L12)` together
   with `beta_measured -> k*beta_measured`, at `k = 1.57`, and the RuntimeError text must
   name the **measured-vs-analytic beta** comparison — not the magnitude band, not
   passivity.
3. If (A) fails at either mesh, **STOP**. Do not widen `HEADROOM`. Report it.

## 4. cv20 — pre-declared thresholds

### 4.1 Gated quantity 1 (new, E2): each solver's `beta` vs Hammerstad–Jensen

```
eps_eff_hj = (er+1)/2 + (er-1)/2 * (1 + 12*h/w)**-0.5      # realized h, w
beta_analytic(f) = 2*pi*f*sqrt(eps_eff_hj)/c0
gate: max |beta_solver(f)/beta_analytic(f) - 1| over the 3.0-4.5 GHz gate band
      <= B_BETA_ANALYTIC_TOL_FRAC = 0.020
```
applied to **both** `beta_rfx` and `beta_openems`.

Derivation of `0.020` — a linear worst-case sum of the four terms by which the
zero-thickness quasi-static closed form is known to differ from what either FDTD actually
simulates. `beta ∝ sqrt(eps_eff)`, so a fractional `eps_eff` error halves into `beta`.

| term | size in `eps_eff` | size in `beta` | source |
|---|---|---|---|
| Hammerstad–Jensen quasi-static model accuracy | ±1.0% | **±0.50%** | `rfx/microstrip.py` module docstring ("accurate to roughly 1 %") |
| finite conductor thickness `t = dx = 50um`, not in the zero-thickness form | `-(er-1)(t/h)/(4.6*sqrt(w/h))` = `-0.0681` = -2.41% | **-1.21%** | Bahl–Garg thickness correction |
| frequency dispersion neglected by the quasi-static form, at the band top 4.5 GHz | Getsinger `f_p = Z0/(2*mu0*h) = 70.4 GHz`, `G = 0.6+0.009*Z0 = 1.078` → +0.13% | **+0.06%** | Getsinger dispersion model |
| — | | **sum 1.77%** | |
| declared envelope, one round step up | | **2.00%** | |

The declared envelope is 2.0%, i.e. `1.13x` the derived budget. It is not fitted: it is
`ceil` of a sum computed from the board's declared geometry before the measurement.

The three budget terms are exactly the three exclusions `rfx/microstrip.py`'s own
*Accuracy* section names — conductor thickness, dispersion, surface roughness — with
roughness zero here because both solvers model ideal PEC. The board sits inside the
model's stated validity range (`w/h = 2.0` in `[0.05, 20]`, `eps_r = 3.66 <= ~13`), so
the 1% figure is the applicable one.

Disclosure, because it matters for the burned-data rule: the committed run-2 artifact
`_20_msl_phase_referee_logs/20260827T102342Z_result.json` was read during §1's audit
reproduction, so its in-band beta ratios were visible when this budget was written. The
budget above is derived independently of them and lands **2x larger** than the largest
in-band deviation that artifact carries; had it been fitted, it would have been tighter,
not looser.

### 4.2 Gated quantity 2 (new, E4): raw cross-solver phase difference

```
raw_phase_diff_deg(f) = degrees(angle(exp(1j*(unwrap(angle(S21_rfx)) - unwrap(angle(S21_openems))))))
gate: max |raw_phase_diff_deg| over the 3.0-4.5 GHz gate band
      <= B_CROSS_SOLVER_PHASE_TOL_DEG = 3.0
```

This is the leg the script currently computes and deliberately declines to gate, on the
argument that the raw difference conflates a *physical*, mesh-dependent `eps_eff`
difference with the reference-plane claim. That argument is correct about what the number
contains and wrong about the conclusion: the conflated physical term is **boundable**, and
the alternative (`residual_phase_diff_after_dispersion_deg`) is provably blind to the very
defect this lane exists to catch (§1). So the raw difference is gated, at a tolerance
sized to contain the physical term.

Derivation of `3.0 deg`:

| term | deg at `beta_max = 155.92 rad/m`, `L12 = 5 mm` | source |
|---|---|---|
| inter-solver realized-`h_sub` difference, ±1 cell (50um) | `0.7109%` of `beta*L12` = **0.3176 deg** | Hammerstad–Jensen sensitivity `d(beta)/beta` for `h = 300 -> 250um` |
| inter-solver realized-`w_trace` difference, ±1 cell | `0.3267%` of `beta*L12` = **0.1459 deg** | same, `w = 600 -> 550um` (the larger of the two ±1-cell signs) |
| reference-plane positional uncertainty, ±4 cells total (both ports) | **1.787 deg** | the script's own committed `GATE-BUDGET DERIVATION`, unchanged |
| — | **sum 2.2505 deg** | |
| declared tolerance, one round step up | **3.00 deg** | |

3.0 deg coincides with the existing `B_PHASE_TOL_DEG`; that is a consequence of sharing
the reference-plane term, not a copy. Note this gate **cannot attribute** a failure to
either solver — its message must say so, and the §4.1 analytic gate is what attributes.

### 4.3 Falsifier (pre-declared)

1. **(A)** Both gates must PASS on the committed run-2 artifact
   (`_20_msl_phase_referee_logs/20260827T102342Z_result.json`, the #723 realized-board
   re-run) **and** on the committed run-1 artifact (`20260804T055009Z_result.json`, the
   declared-board configuration, `h_sub` realized `254um`) — two independent
   configurations, with margins reported.
2. **(B)** With the rfx side's phase velocity halved coherently
   (`beta_rfx -> 2*beta_rfx`, `S21_rfx -> S21_rfx*exp(-1j*beta_rfx*L12)`), **both** new
   gates must FAIL, and the old `_self_consistency_witness` must still PASS at
   ~0.24 deg — i.e. the test records the blindness alongside the fix.
3. If (A) fails on either configuration, **STOP**. Do not widen either tolerance.

## 5. Evidence-level relabelling (part of the fix, not separate)

| case | leg | today | after |
|---|---|---|---|
| cv21 | Stage A reproduce-gate `ZL` vs Coax.m closed form | E2 | E2 (unchanged) |
| cv21 | Stage A matched-through phase vs **analytic** beta | E2 (already independent) | E2 (unchanged; documented as such) |
| cv21 | Stage B matched-through phase vs **measured** beta | claimed under E2/E4 | **E1** — intra-run self-consistency, explicitly |
| cv21 | Stage B `beta` vs analytic coax TEM | — | **E2** (new) |
| cv20 | Stage A notch frequency vs Hammerstad–Jensen | E2 | E2 (unchanged) |
| cv20 | Stage B each solver's phase vs its own beta | claimed under E2/E4 | **E1** — intra-run self-consistency, explicitly |
| cv20 | Stage B `beta` vs Hammerstad–Jensen | — | **E2** (new) |
| cv20 | Stage B raw cross-solver `angle(S21)` | reported, ungated | **E4** (new, gated) |

`evidence_levels` in `validation/crossval/manifest.json` gains `E1` for both cases, with
a matching `self-invariant` reference entry (the manifest contract test requires the
pairing). cv20 keeps `E4` and it becomes gate-backed for the first time. cv21 keeps `E4`
for the Stage-A external tutorial reproduce-gate only, and its `claim_scope` states
explicitly that **no gated Stage-B leg is E4**.

## 6. What this lane does not do

* It does not widen any existing gate. `B_PHASE_TOL_DEG = 3.0`, `B_GD_TOL_PS = 200`,
  `phase_tol_deg = 30`, `gd_tol_ps = 200` and every magnitude band are untouched.
* It does not re-run either solver. openEMS is absent from this host; both cases exit 2
  here. Every measurement is a replay of committed field data through the real witness
  functions — the pattern
  `test_matched_through_witness_run3_regression_measured_vs_analytic_beta` already
  established in this repo.
* It does not touch cv05's prose, cv09, cv10, cv02 or cv14 (Phase 0/1, PRs #814–#818).

## 7. Corrections — round 2 review (2026-09-02)

Append-only; nothing above is edited. No threshold, window or declared constant moves.

**7.1 §3.3's "that floor is a physical property of this mesh's staircase envelope, not a
choice" is withdrawn.** The floor is `k_hi = (1 + HEADROOM·EXCESS_REF)/(1 + EXCESS_REF)`,
and `HEADROOM = 1.30` is the declared choice. The same formula on the same committed data
at other headrooms, with criterion (A) re-checked at each:

| headroom | `k_hi` at the registered mesh | (A) still passes |
|---|---|---|
| 1.30 (declared) | `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.headroom_dependence[0].k_hi = 1.032334` | `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.headroom_dependence[0].criterion_a_still_passes_registered` |
| 1.10 | `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.headroom_dependence[1].k_hi = 1.010778` | `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.headroom_dependence[1].criterion_a_still_passes_registered` |
| 1.04 | `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.headroom_dependence[2].k_hi = 1.004311` | `validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.headroom_dependence[2].criterion_a_still_passes_registered` |

The only floor physics forces at this mesh is the tightest envelope the committed
registered-mesh data itself admits (per-bin max over the beta and group-delay legs),
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.physics_forced_registered.bound_frac = 0.125221`, which puts it at
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.physics_forced_registered.k_hi = 1.003945` — an implied headroom of
`validation/crossval/_issue812_phase_identity/regate_evidence.json::cv21.detection_floor.physics_forced_registered.implied_headroom = 1.0366`. The audit's
`k = 1.02` is missed because of the 1.30 headroom, not because of the 3.8-cell annulus.
"No analytic gate can be tighter than the bias itself" stands; the sentence built on it did not.

`HEADROOM` is **not** tightened here: that would be a post-measurement threshold change, the
exact move this lane forbids. A tighter headroom may be pre-declared for a future round, with
its own (A)/(B) record.

**7.2 Two source-constant pointer spans rewritten, form only.** §3 carried two backtick spans
written as a `.py` path, a double colon and a constant name, pointing at source constants. The #829 numeric-provenance gate treats
any backtick span carrying a double colon as an artifact reference and rejects one that does not parse,
so both are rewritten in place as "`NAME` in `path.py`" — the pointer, the constant and the
sentence are unchanged. This is the one in-place edit above this section, and it changes no
claim.
