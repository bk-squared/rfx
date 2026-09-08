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

## Correction 4 (2026-09-08) — G5's derivation is withdrawn; the gate is not

Section 2, G5 presented its window as a derivation *from first principles*:

> A record of length `T` cannot resolve exponential decay rates finer than
> `1/T` — the same record-length limit that sets the `1/T` Fourier frequency
> resolution [...] Take that as the floor on the estimable decay rate.

and Section 4 concluded:

> The Q **window** carries no chosen value at all — it is `tau_ref / T`,
> computed per mode per run.

**The second sentence stands. The first does not.** Both are corrected here;
**no threshold, formula or gate value changes**, and the judge's verdicts are
bit-identical to those it produced before this correction.

### 4.1 The `1/T` rate-resolution premise is false for this estimator class

`1/T` is a *Fourier separation* statement about telling two nearby components
apart. It is not a floor on the exponent of a single clean damped exponential,
and rfx's `rfx/harminv.py` is the **Matrix Pencil Method**, which fixes that
exponent from adjacent-sample ratios. Measured through rfx's own harminv on a
synthetic single damped sinusoid (`f = 1 THz`, `Q = 1500`, 14 400 samples at
`dt = 1 fs`, so `T/tau = 0.0302` — an eighth of G5's own `1/4` admission cut,
2.97 % total amplitude decay over the record):

| additive noise sigma | recovered Q | relative Q error | rate error in units of `1/T` |
|---|---|---|---|
| 0 (noiseless) | 1500.000000 | 2.1e-10 | 6e-12 |
| 1e-4 | 1499.755499 | 1.6e-4 | 5e-6 |
| 1e-3 | 1498.698910 | 8.7e-4 | 3e-5 |

On that record `1/T` is itself 33x the decay rate being measured, i.e. under
the withdrawn reading the rate would not be determined at all; the estimator
determines it four to eleven orders of magnitude more finely. Upstream Meep's
Harminv is filter diagonalisation — a different estimator again — so the two
sides of this comparison do not share one uncertainty law either.

**Consequence for the wording, not for the gate:** `tau_ref/T` and the `1/4`
e-folding cut are **policy choices with board provenance** (the `1/4`'s
provenance, `(0.086, 0.376)` from #812, is unaffected and still stands), not
derived bounds. Section 4's claim that the window carries no *chosen value* —
no hand-picked number, no measured rfx quantity, computed per mode per run —
remains true and is still the property that keeps the envelope from being
fitted to the agreement it judges.

### 4.2 The "discretization offset" attribution is UNRESOLVED

Text added to this lane's code after the original note (in
`ring_mode_judge.q_window` and in `02_ring_resonator.py`) explained the
rfx-vs-Meep Q gap as *a discretization offset (staircased ring boundary,
subpixel treatment), hence roughly constant in `T`*. **That attribution is
withdrawn**, and nothing is asserted in its place.

The argument behind it — the solvers' close frequency agreement bounds how far
their effective geometries can differ, hence bounds the Q gap — varies one
parameter at a time. Radius and index trade off against each other in
frequency while adding in `Q`. Against the nominal ring
(`R_in = 1, R_out = 2, n = 3.4`), the single perturbed annulus

    R_in = 0.975,  R_out = 1.975,  n = 3.443263353299652

evaluated in the analytic 2D TM annulus (Bessel/Hankel matching, outgoing
exterior) moves the three modes' frequencies by `-0.00886 %` / `0 %` (`n` was
chosen to null this one) / `+0.00392 %` — all inside the two solvers' own
observed disagreement on the live board, `0.0515 %` / `0.0306 %` / `0.0359 %`
— while moving `ln Q` by `+0.0660` / `+0.0895` / `+0.1132`, i.e. 1.1x–1.9x the
measured gaps `0.0582` / `0.0479` / `0.0761`. **This is a counterexample to a
bound, not a claim that either solver's effective geometry is that annulus.**

The decomposition `ln(Q_rfx/Q_ex) - ln(Q_meep/Q_ex) = ln(Q_rfx/Q_meep)` cannot
settle it: `Q_ex` cancels identically, so the identity holds for every
positive value and corroborates nothing about the cause.

What would settle it: a per-solver convergence study (Q versus
cells-per-wavelength at fixed geometry, with and without subpixel averaging)
plus each solver's own Q swept against run length at fixed resolution.

### 4.3 The rate-to-Q transform is asymmetric (issue #945)

G5's gate form, `|ln(Q_rfx/Q_ref)| <= ln(1 + s)` with `s = tau_ref/T`, admits Q
ratios `[1/(1+s), 1+s]`. Its *motivation* bounds the decay rate, and at fixed
`f`, `Q = pi f / alpha`, so `alpha_rfx` inside `[alpha_ref(1-s),
alpha_ref(1+s)]` corresponds to Q ratios `[1/(1+s), 1/(1-s)]` — a different
interval, with no finite upper bound once `s >= 1`. Worked through the shipped
`q_window`: at `f_ref = 1/pi`, `Q_ref = 10`, `T = 20` (so `s = 0.5`), a mode
whose decay rate differs by exactly the admitted `1/T` (`alpha 0.100 -> 0.050`,
hence `Q 10 -> 20`) scores `|ln ratio| = 0.693147` against threshold
`ln(1.5) = 0.405465` and **FAILS**. The live board's mode 2 sits at
`s = 2.8295`, inside the unbounded regime. Tracked as issue #945; the
comparison is unchanged.

### 4.4 Numbers in this note are the tutorial board

Sections 1–4 and Corrections 1–3 quote the Meep **tutorial** board
(`f = 0.118101575, 0.147162556, 0.175246751`; `Q = 80.683, 316.293,
1677.485`), which was the only reference available when this lane opened. PR
#937 committed a **live** Meep 1.34.0 run at
`validation/crossval/_02_ring_resonator_results/crossval.json`
(`T = 260.9798793571893`, settling witness `-8.714255697 dB`):

| mode | `f_ref` | `Q_ref` | `Q_rfx` | freq err % | `T/tau` | raw `tau/T` | `abs lnQ` | Q-gated |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.11800975 | 77.2293 | 81.8569 | 0.0515 | 1.2528 | 0.7982 | 0.0582 | yes |
| 2 | 0.14716790 | 341.4140 | 358.1786 | 0.0306 | 0.3534 | 2.8295 | 0.0479 | yes |
| 3 | 0.17524601 | 1747.8797 | 1619.8639 | 0.0359 | 0.0822 | 12.1648 | (0.0761) | no |

Only **two of three** modes carry a Q verdict on the live board (mode 3's
`abs lnQ` is parenthesised: the record stores `null`, because below the `1/4`
cut the judge never forms the ratio). Correction 1's "the mode this run
actually gates most loosely (0.147) rejects anything beyond a factor 3.35" is
a **tutorial-board** statement; on the live board that mode's band is
`1 + 2.8295 = 3.83`. Correction 1's general bound — every Q-gated mode rejects
a Q error strictly larger than a factor 5 — is unaffected.

### 4.5 What is open

Issue **#907** is re-scoped to "`q_window` presents a policy as a derivation":
either derive an uncertainty model for the Q comparison, or keep the envelope
as policy and justify its size on stated grounds. Issue **#945** is the
asymmetry in 4.3. The earlier prescription in the code — "give the Q window a
floor encoding the expected discretization Q gap" — rested on 4.2 and is
deleted with it. No gate value changed in this correction.
