# Pre-declaration — the slab family's per-bin window, derived per arm

**Status**: pre-declaration. Written and committed BEFORE any window value or
any arm verdict was computed. Appended sections carry the measurement.

**Scope**: the per-bin and band-mean windows the E2 gates of
`22_dispersive_slab_fresnel` and `23_lossy_slab_fresnel` enforce — `G1_R`,
`G1_T`, `G1_A`, `G2_R`, `G2_T`, `G2_A`. Nothing else moves. The E4 (Meep) legs
are out of scope and §7 says why, with the measurement a follow-up needs.

**Reads**: `docs/design_notes/20260903_lattice_witness_standard.md` (the same
error budget, applied to a different reference), issue #928's 2026-09-11 and
2026-09-16 comments, `validation/crossval/_04_fresnel_results/RECOMPUTE.md`.

---

## 1. The defect, in one paragraph

Both cases gate `|X_rfx(f) − X_TMM(f)| ≤ W_BIN + w_ade,X(f)` per bin, for
`X ∈ {R, T, A}`. `W_BIN` is `round-up(1.5 × cv04's per-bin max |R+T−1|)`.
`|R+T−1|` measures energy that left the accounting — a truncated record, an
absorber that reflected. The residual the window has to cover is lattice
dispersion: the Yee grid makes the slab look slightly electrically thicker, so
the Fabry–Pérot fringes sit at slightly the wrong frequency. That moves `R` and
`T` and leaves `R+T` alone. The window is sized by one quantity and spent on
another, and it is measured on a different board besides — cv04's lossless
ε′ = 4 slab, judging three dispersive and three lossy ones.

It has been passing because cv04's record was short: a 719-step record leaked
0.0487, which multiplied up to a 0.074 window, which happened to be wide enough.
cv04's settled record on the shipped absorber (revision r2, 2026-09-16) leaks
0.00037, which gives 0.001 — and then every declared arm fails, because the
thing it must cover was never the leak. 19 of the 29 committed records flip.

## 2. The replacement

For an arm with declared model `m` and declared parameters `p`, run at cell size
`dx` and timestep `dt` on the family rig, for each observable `X ∈ {R, T, A}`
and each frequency bin `f`:

```
W_X(f)  =  M · [ W_lat,X(f)  +  W_wit,X(f) ]
```

### 2.1 `W_lat,X(f)` — the arm's own lattice − continuum difference

```
W_lat,X(f) = | X_lattice(f; m, p, d, dx, dt) − X_TMM(f; m, p, d) |
```

`X_lattice` is `dispersive_eps.yee_lattice_slab_rt_model(f, m, p, d, dx, dt)`:
the exact time-harmonic solution of the 1-D Yee lattice the run actually
stepped, with `round(d/dx)` slab nodes carrying the **discrete-time**
permittivity `eps_numerical_ade(f, m, p, dt)` of the update that ran.
`X_TMM` is the continuum transfer matrix at the exact `ε(f)`, which is what the
gate compares against. `A = 1 − R − T` on both sides, so `W_lat,A` is a direct
difference, not a triangle bound.

Every input is the arm's own: its `ε(f)`, the family thickness `d = 10 mm`, its
own `dx` rung and its own `dt`. Nothing is read from cv04, from another arm, or
from any measurement of the run being judged.

**The model for cv22's poles already exists** and is already load-bearing
elsewhere: `yee_lattice_slab_rt_model` speaks debye, lorentz, drude and
conductive, `lattice_witness.lattice_rta` calls it for every slab case
including cv22's three arms, and the lattice-witness standard's §5.2 is the
cv22 rungs measured through it. This pre-declaration builds no new model; it
takes the difference the standard already computes and spends it on the
continuum gate, which is the one place the family still borrows.

**The parameters are the DECLARED ones, never the run's.** A falsifier record
stores `params` (declared) and `params_run` (defective) separately and the
evaluator is handed `params`. A wrong model therefore cannot widen the window
that is supposed to catch it — the same rule `evaluate_e4` already applies to
the Meep mapping.

### 2.2 `W_wit,X(f)` — the derived floor

`W_lat` alone is not a window: it says how far the lattice sits from the
continuum, not how far the MEASUREMENT sits from the lattice. That second
distance is bounded by the lattice-witness budget, which is already committed,
already reviewed and already the gate `GL1` enforces:

```
W_wit,R(f) = 2 √R_lat (δ_scat + δ_round) + 2 R_lat δ_inc      (and T likewise)
W_wit,A(f) = W_wit,R(f) + W_wit,T(f)
```

computed by `lattice_witness.budget_terms` + `windows_from_terms` from the
record's own settling tails (`tail.scat_refl_rel`, `tail.total_trans_rel`), its
incident purity (`tail.purity_inc_rel`), its ring-down rate
(`run.record.rate_ring_1_s`, taken with any slower fitted tail), its step count
and the float32 epsilon. Standard §3 derives it; §0–§3 of that note is the
argument that none of its inputs is the residual being gated.

This is the **floor the task asks for, derived rather than picked**: a per-bin
window cannot sit below the distance the record's own finiteness puts between
the measurement and the lattice. It is not a separate clamp — it is the second
term of an exact decomposition:

```
| X_rfx − X_TMM |  ≤  | X_rfx − X_lat |  +  | X_lat − X_TMM |
                    ≤       W_wit       +       W_lat
```

the first inequality by the triangle inequality, the second by `GL1`. So the
window is the sum, and a floor is what the sum degenerates to wherever `W_lat`
vanishes (cv23's `tand3` transmission, where `T_TMM ≤ 0.003` and the lattice
term is 3e-5).

### 2.3 `M` — the governance multiplier

`M = tests._gate_policy.ENVELOPE_GATE_MULTIPLIER` (1.5), imported, never
restated as a literal. It is the repo-wide slack over a derived envelope and is
the one object a reviewer has to edit to widen anything here.

**No quantization.** `gate_from_envelope` rounds up to `1/quantum`, and at the
cases' declared `quantum = 1000` that raises every bin whose derived window is
below 1e-3 to 1e-3. On the fine rungs most bins are there (cv23 `tand0p1_dx4`'s
lattice term is 8.6e-4 at its worst bin and 2.4e-4 in the band mean), so
quantizing would widen the gate by up to two orders of magnitude exactly where
the derivation is tightest — the opposite of what the no-silent-loosening rule
protects. Quantization exists to make a **hand-pinned scalar** readable; these
windows are recomputed from the record at evaluation time and are written down
as a literal nowhere, so there is nothing to make readable and the rounding can
only loosen. The band-mean windows are not quantized either, for the same
reason. This is a DEPARTURE from `gate_from_envelope` and it is a tightening in
every bin; the multiplier is still the shared one.

### 2.4 `w_ade` is NOT additive any more

Today the per-bin window is `W_BIN + w_ade,X(f)`, with
`w_ade,X = |X_TMM(ε_ade) − X_TMM(ε)|` the material model's discrete-time error.
`X_lattice` is built **on `ε_ade`** — `yee_lattice_slab_rt_model` takes its slab
nodes from `eps_numerical_ade(f, m, p, dt)` — so the time-discretization term is
already inside `W_lat`. Adding `w_ade` on top double-counts it.

It is not a nesting, and the note says so rather than claiming one: in 14 of the
29 committed records there are gated bins where `W_lat < w_ade` (worst: 5 bins
of cv22's Drude `R`, ratio 7.9). Those are bins where the spatial and the
temporal lattice terms partially **cancel**. The lattice model represents that
cancellation; a sum of two magnitudes cannot. Dropping `w_ade` is therefore a
tightening in those bins and the right answer in all of them, because the
decomposition in §2.2 closes without it.

The settling allowance is likewise not a separate additive term — it is
`W_wit`, §2.2.

### 2.5 The band mean (G2)

```
W_mean,X = M · mean_gated[ W_lat,X(f) + W_wit,X(f) ]
```

the same quantity, averaged over the same gated bins the mean residual is taken
over. **Decision: `W_MEAN_R` / `W_MEAN_T` / `W_MEAN_A` move to the per-arm
quantity too; they do not stay cv04-derived.** The objection in §1 applies
verbatim to the mean: r1's `mean_dR = 0.0066` and `mean_dT = 0.011` are cv04's
ε′ = 4 board's own continuum residual, and using them to judge six other boards
is the same borrowing one gate over. Leaving the per-bin gate fixed and the
band-mean gate borrowed would close the instance and leave the class.

`W_BIN_A = 2 · W_BIN` and `W_MEAN_A = W_MEAN_R + W_MEAN_T` — today's triangle
sums — are replaced by the direct `A` difference of §2.1 plus
`W_wit,A = W_wit,R + W_wit,T`. The reported-only tighter pair (`A_tight_ok`)
keeps its present cv04-r1 derivation; it is not a gate and moving it would add a
schema change to a gate change.

## 3. What this is NOT

- Not a re-pin. No window value is written into any module, test or fixture.
- Not an adoption of cv04 r2 by cv22/cv23. Their `CV04_ADOPTION` stays at r1,
  and after this change r1 feeds only the E4 legs (§7). The cv04-side adoption
  of r2 is a separate change by construction — the same-commit guard in
  `tests/contracts/test_calibration_envelope_ownership.py` refuses a diff that
  touches `envelope.json` and an adoption record together.
- Not a change to `GL1` / `GL2`, to the settling witness, to the passivity
  gate, to the record-length recipe, or to any Meep-leg number.

## 4. The decision rule

For each of the 29 committed records (12 declared-arm records, 17 falsifier /
rung records — enumerated in §6), re-judge every E2 gate under `W_X(f)` and
`W_mean,X` and table the verdict before and after.

- If a **declared** arm fails any E2 gate: STOP. Do not widen, do not re-pin.
  Report the per-bin trace and the mechanism.
- If a **wrong-model falsifier record's overall verdict** stops being FAIL:
  STOP, same treatment. A per-gate flip inside a record that still fails is
  reported, not a stop — a falsifier's job is to fail the record.
- If a verdict flips in a direction §5 did not predict, §5's table records the
  miss. It is not edited.

## 5. Pre-declared expectations

### 5.1 The declared arms — PASS, with a margin ≥ 1.5, by arithmetic

`GL1` — `|X_rfx − X_lat| ≤ W_wit` per bin — holds on every one of the 12
declared-arm records, worst per-bin ratio 0.30 (cv23 `tand3_dx4`, T; standard
§5.1, and re-run here on every committed cv22 and cv23 record). Given that, §2.2
gives `|X_rfx − X_TMM| ≤ 0.30 · W_wit + W_lat ≤ W_wit + W_lat = W_X / 1.5`, so
each declared arm passes `G1` with at least the multiplier in hand. `G2` follows
the same way on the band mean. **Prediction: all six declared arms and all six
dx-ladder rungs pass every E2 gate, unchanged from today.**

### 5.2 The wrong-model falsifiers — the record still FAILS

`GL1` fails on all 13 wrong-model falsifier records, with worst per-bin ratios
from 23 (cv22 `debye_tau_x2`) to 12151 (cv23 `tand3_sigma_zero`). At the worst
bin, `|X_rfx − X_lat| ≥ k · W_wit` with `k ≥ 23`, so
`|X_rfx − X_TMM| ≥ k·W_wit − W_lat`, and the record fails `G1` there whenever
`(k − 1.5)·W_wit > 2.5·W_lat`. **Prediction: every wrong-model falsifier record
keeps an overall FAIL verdict, and several gain gates they do not fail today**
(`debye_tau_x2` `G1_R`, `tand0p1_sigma_x1p5` `G1_R`/`G1_A`,
`tand1_sigma_x1p5` `G1_T`/`G1_A`, `tand3_sigma_x1p5` `G1_T`/`G1_A`), because the
window that was covering them was cv04's 0.074.

### 5.3 The four Meep-leg falsifier records — E2 unchanged

`rfx__falsifier_meep_lorentz_{no_2pi,gamma_half}` and
`rfx__falsifier_meep_tand1_sigma_{2pi,no_eps}` run the DECLARED rfx arm; only
their Meep leg is defective. **Prediction: their E2 gates all stay PASS and
their overall verdict stays FAIL on E4**, which this change does not touch.

### 5.4 The one I am least sure of

**`rfx__falsifier_tand0p1_sigma_x1p5`, gate `G1_R`** (and `G1_A` with it).
It is the smallest parameter perturbation in the falsifier set (σ × 1.5 on the
least lossy arm) sitting on the arm with the family's largest lattice term
(`W_lat,R` max 1.4e-2, four times its own `W_wit,R` max of 3.5e-3) — so it is
the one record where the new window is widest relative to the defect it must
catch. It passes `G1_R` today (it needs 0.0154 against 0.074) and I predict it
FAILS under the new window, but of the 29 this is the prediction I would bet
least on. Its record fails either way — `G1_T`, `G2_T` and `G2_A` are red today.

Second-least sure, for a different reason: **cv22's `debye` declared arm, worst
bin.** Its window WIDENS there. Its measured settling tail is the family's
slowest (standard §14.1: 1.5× slower than the derivation), so `W_wit,R` reaches
5.6e-2 at the top of the gated band, where the differentiated-Gaussian incident
amplitude has fallen to ≈ 8 % of its peak and the budget divides by it. The
derived window at that bin is ≈ 1.5 × (5.6e-2 + 4.0e-3) ≈ 0.09, against today's
flat 0.074. That is a **1.2× widening in one bin of one arm**, and it is what
the derivation gives — I am pre-declaring it rather than discovering it, and
§6's table reports the per-bin before/after so the direction is visible
everywhere else (the same arm's median bin tightens by more than an order of
magnitude). If a falsifier survives BECAUSE of that bin, the decision rule in §4
applies and this pre-declaration is wrong.

### 5.5 What would refute this pre-declaration

1. A declared arm failing an E2 gate.
2. A wrong-model falsifier record reaching an overall PASS.
3. `W_lat` failing to reproduce cv04's own r2 residual in the limiting case:
   the standard's `|rfx − lattice| = 1.7e-5` against `|rfx − continuum| =
   0.0073` band-mean means `W_lat` must account for ≈ 99.7 % of cv04's residual
   on its ε′ = 4 arm. If the same function, evaluated on cv04's rung, does not,
   the model is not the one the decomposition assumes. Checked in §6.
4. Any bin where `W_wit` is not finite and positive — the decomposition has no
   floor there and the gate would be undefined.

## 6. The measurement

*(appended after the note was committed; §5 is not edited)*

## 7. What this does NOT fix — the Meep legs

`W_BIN`, `W_MEAN_R` and `W_MEAN_T` survive as cv04-r1-derived scalars, used by
`evaluate_e4` alone (`G4_*`, `G5_*`). They are not moved here, for a measured
reason rather than for scope: the term a `G4` window must cover is **Meep's**
own discretization error, which is neither cv04's nor the rfx arm's. Adopting
cv04 r2 there would put `W_BIN = 0.001` against measured `G4` residuals of
0.0011–0.0132 and fail every committed Meep leg; applying the rfx arm's window
there fails cv22's Drude `G4_T` (0.0115 measured against a 0.0077 window),
because rfx runs at 1 mm and Meep at 0.25 mm and their lattice terms are not the
same number.

A per-arm Meep derivation does exist and it works. Measured here, on the six
declared arms, so a follow-up starts from evidence rather than from scratch —
`W_lat,meep` from the same lattice model at Meep's `(dx, dt)` with
`eps_numerical_meep`, plus the one-cell thickness excess cv23's §12 already
hypothesises (`meep_thickness_excess_rta`, E nodes inclusive at both faces):

| arm | res | measured max \|R_meep − R_TMM\| | `W_lat,meep` R | one-cell thickness R | 1.5 × sum, R |
|---|---|---|---|---|---|
| cv22 debye | 40 | 0.00717 | 0.00027 | 0.01031 | 0.01560 |
| cv22 lorentz | 40 | 0.01159 | 0.00030 | 0.01572 | 0.02367 |
| cv22 drude | 40 | 0.00384 | 0.00009 | 0.00792 | 0.01189 |
| cv23 tand0p1 | 80 | 0.01324 | 0.00022 | 0.01307 | 0.01985 |
| cv23 tand1 | 40 | 0.00509 | 0.00048 | 0.00519 | 0.00805 |
| cv23 tand3 | 40 | 0.00110 | 0.00117 | 0.00048 | 0.00200 |

The thickness term alone reproduces the measured Meep residual to within a
factor of 1.5 on all six arms, and the derived sum covers every one. That is
enough to make the follow-up a pre-declaration with a falsifier set rather than
an exploration — and not enough to adopt it here, because taking a
"measured-in-round-2 evidence, not a pre-declared window term" (cv23 note §12's
own words) straight into a live gate on six points, with the Meep falsifiers
un-retabled, is the self-certification this issue was filed about. It is named
here as the remaining borrowing, with its numbers, rather than left silent.
