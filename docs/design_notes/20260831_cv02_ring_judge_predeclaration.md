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
