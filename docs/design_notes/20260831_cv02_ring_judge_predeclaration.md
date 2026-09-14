# cv02 ring-resonator judge — re-gate pre-declaration (issue #812, Phase 1)

Date: 2026-08-31 · Lane: `agent/regate-cv02` · Case:
`validation/crossval/02_ring_resonator.py` (claims-bearing, E4, the repo's only
curved-boundary external-solver case).

**Append-only.** Corrections are added as new sections; nothing above a
correction is edited.

## 1. What is wrong with the shipped judge

The shipped PART 3 matcher and the shipped SUMMARY gate are the same number.

```python
        if diff < best_diff:          # best_diff starts at 1.0
            best_diff = diff
    if best_idx is not None and best_diff < 0.05:      # matcher window
        matched.append(...)
...
    errs = [abs(rf - mf) / mf * 100 for mf, _, rf, _ in matched]
    mean_err = np.mean(errs)
    if mean_err < 5.0:                                 # headline gate
```

`errs[i]` is `best_diff_i * 100` for exactly the pairs the matcher admitted,
and the matcher admits a pair only when `best_diff_i < 0.05`. So every element
of `errs` is `< 5.0` by construction and therefore `mean(errs) < 5.0` by
construction. **The headline gate is entailed by the matcher window.** It is a
tautology of the assignment step, not a measurement of the solver.

Issue #812 measured the entailment: **200,000 random trials through the
verbatim judge, maximum `mean_err` ever observed 4.9997%, zero failures of the
mean gate.** The only surviving discriminator is `len(matched) >= 2`, and that
one is *weakened* by the same window — a mode rfx places more than 5% away is
silently deleted from the comparison rather than counted against it, so a
large, real frequency error makes the reported `mean_err` *smaller*.

Second gap, from the same audit: **`Q` is extracted on both sides, printed on
both sides, and gated nowhere.** The case is the repo's only curved-boundary
E4; a staircased dielectric ring's radiation loss is precisely the quantity a
curved boundary gets wrong, and no gate looks at it.

## 2. Pre-declared replacement judge

Five gates, evaluated only when the external reference (Meep) is present.
Reference-absent behaviour is unchanged: exit 2, inconclusive, not a pass.

### Mode admission (both sides)

- **rfx side (unchanged):** `Q > 1` and `amplitude > 1e-10`, over the harminv
  search band `[fcen - df/2, fcen + df/2]`.
- **reference side (new, symmetric):** `Q > 1`, over the same band. The
  reference side previously had no floor at all, so a Meep harminv artefact
  with `Q < 1` used to enter the comparison as a full-weight mode.

### G1 — unmatched-mode failure (new)

Modes are paired by a **one-to-one assignment that minimises total relative
frequency distance** (`scipy.optimize.linear_sum_assignment` over the
`n_ref x n_rfx` matrix of `|f_rfx - f_ref| / f_ref`). **No tolerance enters the
assignment.** Every admitted reference mode is either given a distinct rfx
partner or recorded as `UNMATCHED`.

**Gate: any `UNMATCHED` reference mode is a FAIL.**

This is the decoupling. The matcher now answers "which rfx mode corresponds to
this reference mode", a question about ordering; the verdict answers "how far
apart are they", a question about accuracy. The 5% number appears in exactly
one of the two, so it can no longer entail itself.

Surplus rfx modes (rfx modes assigned to no reference mode) are **reported,
not gated** — a genuine mode below Meep's harminv sensitivity is not by itself
an rfx defect, and inventing a failure mode this lane cannot verify against a
live Meep would violate criterion (A).

### G2 — mode count (unchanged: `>= 2`)

At least two admitted reference modes must be present and assigned. Unchanged
in value; it now sits on top of an assignment that cannot delete a bad mode.

### G3 — mean relative frequency error `< 5%` (value unchanged)

Taken over **all** assigned pairs, not over a window-filtered subset. Same
published threshold as today
(`docs/public/guide/benchmarks.mdx`: "Mean mode-frequency error `< 5%`"). Not
widened. Its meaning changes from tautological to falsifiable purely by G1
removing the window that entailed it.

### G4 — max relative frequency error `< 5%` (new, same value)

The published claim reads per mode ("ring-resonator resonant-mode frequencies
... mean mode-frequency error < 5%"), and a mean over three modes has a null
space: one mode wrong by 12% averages to 4% against two exact ones and passes
G3. G4 applies the **already-published 5% budget** to each mode instead of
inventing a new number. This is a tightening; no existing gate is widened.

### G5 — Q gate, window derived from the record length and the reference (new)

*Derivation (first principles, no measured rfx Q enters).* A record of length
`T` cannot resolve exponential decay rates finer than `1/T` — the same
record-length limit that sets the `1/T` Fourier frequency resolution: two
envelopes whose rates differ by less than `1/T` differ by less than one factor
of `e` across the whole record and are not separable. Take that as the floor on
the estimable decay rate:

    delta_alpha  =  1 / T

A mode of quality factor `Q` at frequency `f` has amplitude decay rate
`alpha = pi f / Q`, i.e. amplitude e-folding time `tau = Q / (pi f)`. Hence the
record's own contribution to the fractional Q tolerance is

    delta_Q / Q  =  delta_alpha / alpha  =  Q / (pi f T)  =  tau / T

with `tau` and `f` taken from **the reference mode**, and `T` the actual
harminv record length of the run (`len(signal) * dt`, printed by the script).
Neither the measured rfx Q nor the measured rfx frequency appears anywhere in
the window.

*Gate form.* Symmetric in ratio, so it catches an rfx Q that is too **low**
(over-damped: leaky boundary, spurious loss) as well as too high:

    | ln( Q_rfx / Q_ref ) |  <=  ln( 1 + tau_ref / T )

*Which modes are gated.* A mode is Q-gated iff the record actually observed its
decay:

    T / tau_ref  >=  1/4

i.e. the record spans at least a quarter of an amplitude e-folding (>= 22% of
observed amplitude decay). **Modes below this are reported, never gated.**

Provenance of the `1/4`, stated plainly: issue #812 published, before this lane
opened, `T/tau = 0.376` for mode 2 ("the record resolves it") and `T/tau =
0.086` for mode 3 ("must be excluded or the gate measures record length rather
than physics"). Any cut in the open interval `(0.086, 0.376)` implements that
published finding; `1/4` is the round geometric fraction inside it. The cut is
prior-provenance, not fitted here. Its consequence is a stated instrument
property rather than a bare constant: because the admitted window is at most
`tau/T <= 4`, **the loosest Q gate this rule can ever issue still rejects a
factor-3.35 Q error, and every admitted mode rejects a 5x Q error.** A gate
that could not do that would not be evidence, which is the whole reason mode 3
is excluded rather than gated loosely.

Note that #812's `0.376 / 0.086` were computed with rfx's own Q. G5 uses the
**reference** Q, which moves the two numbers to roughly `0.43 / 0.10` — the
same side of `1/4` in both cases, so the published exclusion is unaffected by
the substitution.

## 3. Two-sided acceptance, declared before measuring

- **(A)** On today's code the case must still pass: the assignment must pair
  every reference mode, and G2–G5 must hold on the rfx modes today's
  `02_ring_resonator.py` actually produces.
- **(B)** The new judge must **fail** on defects the shipped judge passes:
  1. a reference mode rfx misses entirely (shipped: `matched = 2 >= 2`, mean of
     the survivors tiny, PASS);
  2. one mode displaced well beyond the matcher window (shipped: that mode is
     deleted from `matched`, so the reported mean *improves*, PASS);
  3. a mode-2 `Q` wrong by 5x (shipped: `Q` is gated nowhere, PASS).
- **(C)** The 200,000-trial harness of #812 must be re-run against **both**
  judges. The shipped judge must reproduce "max `mean_err` 4.9997%, mean gate
  never fails"; the new judge must show `mean_err >= 5%` occurring and the
  verdict failing at a substantial rate. Same trial stream, both judges.

A change satisfying only (A) is cosmetic; only (B) means the case is broken.

## 4. Disclosure (burned-data discipline)

Written before any judge code exists, and before the numbers below were
compared. Two things were already in view when this note was written and are
disclosed rather than hidden:

- Today's rfx mode list on this host, from one unmodified run of the shipped
  script (`f = 0.118068, 0.147213, 0.175298` in Meep units, `Q = 86.5, 357.6,
  1864.1`), used to confirm the case is alive and to size the runtime. No
  threshold above is a function of these numbers.
- The Meep tutorial's **published** harminv output for this exact geometry
  (`f = 0.118101575, 0.147162556, 0.175246751`; `Q = 80.683, 316.293,
  1677.485`), which is the reference this case is defined against. G5's window
  is a formula in the reference `Q` and `T`; it is not a number chosen after
  seeing an agreement.

The three thresholds that carry a numeric value are: `5%` (already published,
unchanged, now applied twice), `>= 2` (already published, unchanged), and
`1/4` (prior-provenance from #812, bracketed by numbers that issue published
before this lane opened). The Q **window** carries no chosen value at all — it
is `tau_ref / T`, computed per mode per run.

## 5. External-reference status on this host

Meep is not installed here (`ModuleNotFoundError: No module named 'meep'`), so
the E4 leg exits 2 and the live cross-check cannot be executed in this lane.
Criterion (A) is therefore demonstrated by feeding the judge **today's measured
rfx modes** together with the **published tutorial reference**, in a committed
test, and is labelled as such. It is not a claim that a live Meep run was
performed here.

---

## Correction 1 (2026-08-31, after implementation, before the measurement run)

Section 2, G5 stated:

> the loosest Q gate this rule can ever issue still rejects a factor-3.35 Q
> error, and every admitted mode rejects a 5x Q error.

**The first clause was wrong.** 3.35 is the bound for *mode 2 of this
particular run* (`tau_ref/T = 2.35`, so the admitted ratio band is
`1 + 2.35 = 3.35`), not the bound the rule guarantees in general. The general
bound follows from the admission cut alone: `tau/T <= 1 / 0.25 = 4`, so the
widest band the rule can ever issue is `1 + 4 = 5`.

Corrected statement, and the one the test
`test_no_admitted_q_gate_tolerates_more_than_a_factor_five` checks:

> **Every Q-gated mode rejects any Q error strictly larger than a factor 5**,
> and the mode this run actually gates most loosely (0.147) rejects anything
> beyond a factor 3.35.

The gate itself is unchanged — only the claim made about it. No threshold
moved; nothing was widened.

## Correction 2 (2026-08-31) — the trial stream, not the judge

The first tautology-harness stream drew reference modes across the full
`[0.10, 0.20]` harminv band with a `0.015` minimum separation. Both choices
were wrong for the *control* arm of the experiment (defect-free trials, which
must not fail):

1. a reference mode near a band edge could have its defect-free partner pushed
   *outside* the harminv search band, where it is legitimately unmatched —
   measured 146 spurious control failures in 952 defect-free trials;
2. `0.015` is smaller than twice the largest in-window shift
   (`2 x 0.05 x 0.19 = 0.019`), so two adjacent reference modes could cross
   under defect-free perturbation and the assignment correctly swapped them —
   measured 3 further control failures in ~9,500.

Fixed by drawing reference modes from the band inset by the matcher window
(`[F_MIN/0.95, F_MAX/1.05]`) with `MIN_SEPARATION = 0.021` and `n_ref` in
`{2,3,4}`. Both were defects in the *experiment*, not in the judge: with them
fixed, the new judge fails **zero** defect-free trials while the shipped judge
fails zero of everything.

## Correction 3 (2026-08-31) — G4 is load-bearing, and here is the proof

G4 (per-mode max) was declared as a tightening whose motivation was the mean's
null space. That motivation is now measured rather than argued: an rfx mode
displaced **-12%** from the 0.175 reference — a defect the shipped judge scores
as a clean 0.03% run — produces `mean = 4.02%`, which **passes** G3's published
5%. Without G4 the decoupling alone would not have caught it. Recorded in
`test_b2_displaced_mode_passes_the_shipped_judge_and_fails_the_new_one`.

## Correction 4 (2026-09-13) — G5's gate form is superseded, and its "no chosen value" claim is withdrawn

Two statements in this note about the Q gate were wrong. Both are corrected in
code; this section is the record, and the sections above are left as written.

**(a) The gate form in §2 G5 was wrong on one side (#945, fixed in PR #999).**
G5 declared

    | ln( Q_rfx / Q_ref ) |  <=  ln( 1 + tau_ref / T )

i.e. `Q_rfx/Q_ref` in `[1/(1+s), 1+s]` with `s = tau_ref/T`. That is not the
image of the rate interval G5 derives it from. `Q = pi f / alpha` is monotone
**decreasing** in the decay rate, so `alpha/alpha_ref` in `[1-s, 1+s]` maps to

    Q_rfx / Q_ref  in  [ 1/(1+s), 1/(1-s) ]

The lower bound matched; the upper did not, since `(1+s)(1-s) = 1-s^2 < 1`. The
symmetric form rejected a mode whose decay rate differed by exactly the `1/T`
the note declares as the tolerance, and it imposed a finite ceiling in the
`s >= 1` regime — where the declared rate interval reaches zero and the
transformed interval has no finite upper bound at all. `s >= 1` is not
hypothetical here: the committed live board's mode 2 runs at `s = 2.8295`
(`validation/crossval/_02_ring_resonator_results/crossval.json`, the
`q_window` field of the second assignment row).

The shipped gate is now the exact transform, in
`ring_mode_judge.rate_interval_to_log_q_bounds`. §4's factor-five statement
still holds on the low-Q side (the admission cut still bounds `s <= 4`, so the
tolerated over-damping is still at most `1+s = 5`); on the high-Q side the
bound is `1/(1-s)` below `s = 1` and unbounded above it, so the phrase "every
admitted mode rejects a 5x Q error" now applies to the over-damped direction
only.

**(b) §4's "The Q window carries no chosen value at all" is withdrawn (#907).**
The sentence is true about the window's *arguments* — no measured rfx quantity
enters it, so it is not fitted to the agreement it judges, and that property is
retained. It is false about the window's *form*. The premise §2 G5 argues from,
"a record of length `T` cannot resolve exponential decay rates finer than
`1/T`", was measured and refuted for this estimator: a damped exponential fixes
its exponent from adjacent-sample ratios, and there is no Fourier separation
limit to import. rfx's matrix-pencil harminv with `decimate=False` returns Q to
~3e-12 relative error at `T/tau = 0.0822`, a third of the `1/4` admission cut
(#907, comment of 2026-09-10).

**(c) A second empirical prop, added under (b) and withdrawn 2026-09-13.**
This section briefly continued: "what does degrade at short records is the
decimated path cv02 actually runs" — with `3.49 %` at `T/tau = 0.0822` and
`0.24 %` at the cut, from the same #907 comment. Independent review could not
reproduce those figures and neither could a re-measurement here. At
`T/tau = 0.0822` the record is too short for a decimation stage to leave the
requested pencil capacity, so `decimate='auto'` decimates by nothing and there
is no decimated path there to degrade; where a stage does fire the decimated
path is not worse than `decimate=False`, and every measured relative Q error on
the ladder is orders of magnitude below what was asserted. The ladder, both
frequency bands and the per-rung decimation plans are committed as
`tests/fixtures/cv02_ring_judge/harminv_decimation_ladder.json` (generator
`scripts/diagnostics/cv02_harminv_decimation_ladder.py`), pinned by
`test_the_decimation_penalty_claim_does_not_reproduce`.

Whether the REAL cv02 record — multi-mode, with source contamination still in
the analysed window — degrades under decimation is **unmeasured**; the ladder
is a clean single exponential and does not settle it.

So `tau_ref/T` is a **policy envelope with provenance**, not a derived bound,
and that provenance is #812's published bracket alone. The `1/4` admission cut
keeps its prior-provenance standing from #812; the sentence that also claimed
independent support for it from the decimation measurement is withdrawn with
(c).

**The obvious alternative was checked and is not available.** The natural
replacement for a declared envelope is the estimator's own fit residual.
rfx's harminv reports a field named `error`, but it is not a residual:
`rfx/harminv.py` computes `err = 1 - min(|lam|, 1/|lam|)`, and for a decaying
pole `|lam| = exp(-alpha * dt_eff)`, so

    error  ==  1 - exp(-decay * dt_eff)

exactly — measured to bit equality on a clean damped exponential at cv02's own
step, on both the decimated and undecimated paths, in
`test_harminv_error_field_is_the_decay_restated`. It is a monotone function of
the reported decay rate, i.e. of `1/Q` at fixed `f`, and says nothing about how
well the pole fits the record. A tolerance built from it would scale with the
very quantity the Q gate judges — #812's self-referential gate class. Meep's
`err` is a signal-processing residual on the other side and is not retained by
this case at all. Item 1 of #907's re-scoped ask therefore needs a measurement
campaign, not a field that already exists.

**What replaces the claim.** The gate's inputs are now three named objects with
an explicit epistemic status, in `ring_mode_judge.Q_GATE_INGREDIENTS`, printed
with every report and persisted under `gate_limits.q_gate_ingredients`:

| ingredient | kind | what it is |
|---|---|---|
| `estimator_uncertainty` | declared-policy | the scale `s = tau_ref/T` |
| `rate_to_q_transform` | derived | the exact interval inversion, (a) above |
| `discretization_budget` | absent | no permitted rfx-vs-Meep disagreement has ever been declared |

Because the third does not exist, a cv02 `q` PASS is a two-solver **consistency
heuristic**, not a bound on either solver's Q accuracy. That is the reading
recorded when #907 was closed on 2026-09-13, with the three research items
(the estimator's real SNR / model-order uncertainty, a source-free Meep
reference record, and a discretization budget against the exact annulus)
deferred to a pre-declared campaign rather than settled by choosing a floor —
a floor taken from the observed gap would have made the gate certify the
agreement it exists to test.

**(d) The retained evidence record predates (a)–(c) and keeps the old wording,
on purpose.** `validation/crossval/_02_ring_resonator_results/crossval.json` is
the run of 2026-09-06T17:07:57Z (commit `296cabad`). It is the only copy that
predates the (a) transform, so re-driving today's judge on it is the falsifier
for "the transform moved a verdict that was already committed"; regenerating it
would turn that check into the judge against its own output. It is therefore
pinned as-is by `CV02_RECORD_PIN` and
`test_the_committed_record_is_still_the_pre_transform_one`, which red loudly if
the record moves.

The cost is that the withdrawn framing survives in the reader-facing artifact:
that record's `gate_limits.note` still reads, verbatim, "the Q window is
tau_ref/T per mode, derived from the reference Q and THIS record length -- not
a chosen number". Read it as **superseded** by
`ring_mode_judge.Q_GATE_INGREDIENTS`, which every record written after
2026-09-13 carries instead. The half about the
window's *arguments* stands — no measured rfx quantity enters it. The half
about its *form* does not: `tau_ref/T` is declared policy with #812's bracket
behind it, not a derived bound. No number in that record is affected; the note
describes provenance, and the gates it reports are the ones today's judge
reproduces.

**What did not change.** No gate moved. The verdict on the committed board is
unchanged: re-driving the judge on that record's own mode pairs and record
length reproduces all five gates PASS. The run-length contingency #907
describes is still present and still pinned by
`test_verdict_lane_q_gate_is_run_length_contingent`, which remains a
characterization test of current behaviour.

---

## Correction 5 (2026-09-14) — the transform is applied at each pair's own two frequencies (#945, reopened)

Correction 4(a) fixed the shape of the Q interval and left a second error in
the same line: the inverted interval was compared against a Q ratio measured
at a *different* frequency from the reference's. #945 was reopened on
2026-09-13 with the counterexample below. The sections above are left as
written; this is the record.

**The algebra, in three lines.** A mode's amplitude decay rate is
`alpha = pi f / Q`, so

    alpha_rfx / alpha_ref  =  (f_rfx / f_ref) * (Q_ref / Q_rfx)

    ln(alpha_rfx/alpha_ref)  =  ln(f_rfx/f_ref) - ln(Q_rfx/Q_ref)

and the declared interval `alpha_rfx/alpha_ref in [1-s, 1+s]` therefore maps to

    ln(Q_rfx/Q_ref)  in  [ ln(f_rfx/f_ref) - log1p(s),
                           ln(f_rfx/f_ref) - log1p(-s) ]

i.e. Correction 4(a)'s interval **shifted by `ln(f_rfx/f_ref)`**. The inversion
`Q/Q_ref in [1/(1+s), 1/(1-s)]` is the special case `f_rfx = f_ref`.

**What the missing term did.** `judge()` compared `ln(Q_rfx/Q_ref)` against the
unshifted interval, so the whole of a frequency disagreement — which cv02 gates
separately, at 5% — was charged to the Q gate as well. Reproduction on
`origin/main` at commit `88651a1c`: `f_ref = 1`, `Q_ref = 100`, `T = tau/0.02`
(so `s = 0.02`), and an rfx mode 4.900% low in frequency whose decay rate is
**exactly** the reference's, `alpha_rfx/alpha_ref = 1.000000` — dead centre of
the declared rate interval:

    freq error = 4.900%   (max_err gate < 5.0%: PASS)
    ln(Q_rfx/Q_ref) = -0.050241
    bounds          = [-0.019803, +0.020203]
    ln(f_rfx/f_ref) = -0.050241   <-- the missing term
    q_gated = True   q_pass = False

`ln(Q_rfx/Q_ref)` equals the frequency term exactly, because at an equal decay
rate a 4.9% lower frequency *is* a 4.9% lower Q. The gate rejected a mode
sitting on the reference's own decay rate, for a frequency error its own
frequency gate admits. After the fix the bounds read `[-0.070044, -0.030039]`
and `q_pass = True`.

**The tolerance did not move; the comparand did.** `s = tau_ref/T` is still
built from the reference alone (`q_window`), so the envelope is still not
fitted to the agreement it judges — §4's surviving claim about the window's
*arguments* is untouched. What now carries both solvers' measurements is the
quantity being compared: a decay-rate ratio is a function of both `(f, Q)`
pairs because a rate is. Ingredient 2 in `Q_GATE_INGREDIENTS` states this: the
transform is derived, and it is evaluated at each mode's own realized
frequency pair.

Each gated row now records what was subtracted, so the artifact is readable
without redoing the algebra: `q_log_freq_term = ln(f_rfx/f_ref)`,
`q_log_rate_ratio_signed = ln(alpha_rfx/alpha_ref)`, and `q_log_lower` /
`q_log_upper` — the **shifted** bounds that actually judged the row. Both keys
are additions; nothing was renamed or removed.

**What did not change.** No gate constant, window or tolerance moved. Re-driving
today's judge on the committed record's own stored inputs
(`_02_ring_resonator_results/crossval.json`, commit `296cabad`,
2026-09-06T17:07:57Z) reproduces all five gates PASS and every row's `q_pass`,
before and after: the two gated rows' frequency terms are `+0.000515` and
`+0.000306` against windows `0.7982` and `2.8295`, three to four orders of
magnitude of margin. That record is still pinned as-is by `CV02_RECORD_PIN`,
for the reason Correction 4(d) gives — it is the only copy predating the
transform, so re-driving the judge on it is the falsifier for "the transform
moved a committed verdict", and regenerating it would void the falsifier
rather than refresh it.

The run-length contingency #907 describes is untouched, and
`test_verdict_lane_q_gate_is_run_length_contingent` stays a characterization
test of current behaviour: on that fixture the frequency term is `-2.8e-4`
against a window that shrinks from `0.7472` to `0.0642`, so the longer record
still fails. The frequency term is not the fix for #907 and is not offered as
one.

Pinned by `test_issue945_a_frequency_error_is_not_charged_to_the_q_gate`
(the counterexample above), `test_issue945_the_q_gate_still_bites_at_exact_frequency`
(the mirror: at an exact frequency, a rate 1.5x the declared scale off still
fails, both sides), `test_issue945_the_row_reports_the_frequency_term_and_the_rate_term`
and `test_issue945_the_frequency_term_is_refused_on_an_unphysical_frequency`.
