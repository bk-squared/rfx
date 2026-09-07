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

## Correction 4 (2026-09-07) — G5's window applied the resolution premise to one of two records (issue #907)

Section 2, G5 stated:

> with `tau` and `f` taken from **the reference mode**, and `T` the actual
> harminv record length of the run (`len(signal) * dt`, printed by the script).

**The derivation is right; it was applied to one of the two measurements it
compares.** `|ln(Q_rfx/Q_ref)|` differences two numbers, and *both* are harminv
readings off a finite record. G5 gave `Q_rfx` its `tau/T_rfx` and silently
treated `Q_ref` as exact. It is not: `Q_ref` carries `tau/T_ref` by the identical
argument. A difference cannot be sharper than its blunter term, so the
comparison's resolution is

    T_cmp   =  min(T_rfx, T_ref)
    dQ/Q |comparison  =  tau_ref / T_cmp

and the admission cut, which asks whether the record observed this mode's decay,
must ask it of the same record:

    gated  iff  T_cmp / tau_ref  >=  1/4        (the 1/4 is unchanged)

The gate form is unchanged and still symmetric in ratio, so an rfx `Q` that is
too **low** is caught exactly as before:

    | ln( Q_rfx / Q_ref ) |  <=  ln( 1 + tau_ref / T_cmp )

### What the defect looked like

The pre-#907 window shrank as `1/T_rfx` without limit while the physics did not,
so a longer, better-settled record reddened a stable case. Measured on the
committed frozen board (`MEEP_REFERENCE` / `RFX_TODAY`), driving `origin/main`'s
judge directly:

| `T_rfx` | gate `q` | rows failing `q` |
|---|---|---|
| 291 (committed) | PASS | 0 |
| 3385 (1 e-fold of the slowest in-band tau) | **FAIL** | 1 |
| 15600 (the −40 dB record) | **FAIL** | 2 |
| →∞ | **FAIL** | 3 |

The repo's own ring-down rule asks for records extended toward −40 dB settling,
so following one repo rule turned a claims-bearing gate red for a reason that
was not physics.

### Why `T_ref` does not break G5's anti-tautology property

G5's central design property is *"Neither the measured rfx Q nor the measured rfx
frequency appears anywhere in the window."* The amended window's inputs are
`f_ref`, `Q_ref` (reference modes), `T_rfx` (a run length, not a measurement of
the ring) and `T_ref` (a reference-run parameter, the same class of input as
`f_ref` and `Q_ref`). **The sentence survives verbatim.** It is pinned by
`test_the_window_is_a_function_of_the_reference_and_the_records_only`, which
sweeps the rfx `Q` over four decades and asserts every row's window, e-folding
count and gating decision are bit-identical — a property measured by driving
`judge()`, not a frozen column and not a shared helper.

**Explicitly rejected, and why it must stay rejected:** using rfx's own measured
Q stability (0.22% between `T = 291` and 1101, 1.7% between 1575 and 3281) as
the floor. Three independent reasons, any one sufficient. (i) It injects the
measured quantity the gate judges, which is #812's failure mode reintroduced on
the Q axis after being removed from the frequency axis. (ii) Run-to-run
stability is *precision*, not *accuracy* — a solver can be perfectly
self-consistent and uniformly wrong, which is the case this gate exists to
catch. (iii) It does not even work: 0.0022 is three orders below the 0.0696 gap,
so the gate would still red at `T_rfx = 3385`. The stronger reading — set the
floor to the measured `|ln Q|` gap — is pinning a gate to the result it judges
and is forbidden outright.

### The composition is `min`, and that is the tighter of two defensible choices

Two independent uncertainties can be composed by taking the larger (which is
what `min` of the two records yields) or by adding them. **`min` is chosen
because it is the tighter of the two defensible compositions** — it is the
conservative choice at every input, so it cannot be pass-motivated. Two
consequences follow, and they are consequences, not reasons:

* the pre-declared factor-5 property survives with **Correction 1's original
  proof**, because the cut and the window read the same `T_cmp`: gated ⟹
  `T_cmp/tau ≥ 1/4` ⟹ `tau/T_cmp ≤ 4` ⟹ `window_ln ≤ ln 5`. With the sum
  (`tau/T_rfx + tau/T_ref`) the bound would be `ln 9` and the published
  instrument property would fail;
* at the committed operating point (`T_rfx = 291 < T_ref`) nothing changes at
  all: `T_cmp = 291`, today's value, and the verdict is bit-identical.

### Monotonicity — the no-silent-loosening statement

`min(T_rfx, T_ref) ≤ T_rfx` for every input, so the amended rule **weakly
loosens both the window and the admission cut, and tightens nothing, anywhere**.
Pinned by `test_the_amendment_loosens_weakly_and_tightens_nothing`, which
recovers the pre-#907 rule exactly by passing `T_ref = T_rfx` and compares row by
row. No threshold value moved: `5%`, `>= 2` and `1/4` are untouched, and the
window still carries no chosen number.

The one deliberate **tightening** in this change is not in the window: an empty
gated set is now inconclusive rather than a pass (below).

### Detection power that remains

| | mode 1 | mode 2 | mode 3 |
|---|---|---|---|
| pre-#907 @ `T = 291` (committed) | rejects > 1.747× | > 3.351× | ungated |
| amended, `T_ref = 300`, any `T_rfx` | > 1.725× | > 3.280× | ungated |
| amended, `T_ref = 450`, any `T_rfx` | > 1.483× | > 2.520× | ungated |

Detection power at the operating point is preserved to 1–2%. Two-sidedness is
untouched. The loosest window the rule can issue on any board at any pair of
record lengths is still `ln 5`, and
`test_no_admitted_q_gate_tolerates_more_than_a_factor_five` now *drives* that
property — a board taken to the admission cut (`tau_ref = 4·T_cmp`) issues
exactly `ln 5`, admits 4.99× and rejects 5.001× — instead of restating an
algebraic identity about a constant.

**What is given up, stated rather than buried.** Relative to the pre-#907 rule
*at long records* this is a large loosening: at `T = 15600` the old windows were
0.0138 / 0.0429 (nominally rejecting a 1.4% / 4.4% Q error) and the amended
floors are 0.5451 / 1.1880 (rejecting 73% / 228%) — roughly a 50× reduction in
nominal power at long records. That old tightness was fictitious, because the
reference never resolved its Q that finely; but the reduction is real and is
recorded here. Separately, mode 3 is now **never** gated (`T_ref/tau = 0.0985`
at 300, 0.1575 at 450, both below `1/4`), where the old rule would have newly
gated it at `T_rfx ≥ 762`. Gating a mode the reference observed for 9.8% of one
e-folding would measure Meep's run length — #812's published exclusion, applied
consistently — but it *is* a reduction relative to the old long-`T` behaviour.

**Consequence: lengthening the REFERENCE record is now the sole remaining lever
on this gate's power.** That is a property of the amended rule, not a follow-up
wish: raising the Meep run from 300 to ~3000 would take mode 1's window from
0.545 to 0.071 and bring mode 3 into gating range. It re-measures the reference
Q's, so the frozen board moves with it. Nothing on the rfx side can tighten this
gate any further.

### `T_ref` provenance, and the direction of the choice

`T_ref = 300.0` on the frozen board is **published, not chosen**. `MEEP_REFERENCE`
is the Meep "Modes of a Ring Resonator" tutorial's harminv output verbatim
(`meep.readthedocs.io`, Python_Tutorials/Basics — the three lines
`0.118101575043663 / 80.683059081382`, `0.147162555528154 / 316.29272471914`,
`0.175246750722663 / 1677.48461212767` appear there to every digit), and that
tutorial's run call is

```python
sim.run(mp.at_beginning(mp.output_epsilon),
        mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+0.1), fcen, df)),
        until_after_sources=300)
```

— Harminv wrapped in `mp.after_sources`, so its record is exactly the 300 time
units following source-off, and the tutorial's own prose says "we have only
analyzed it for about 300 time units".

**Rule for the ambiguous case, pre-declared here: an ambiguous `T_ref` resolves
to the longest defensible record.** A *shorter* `T_ref` gives a *looser* floor,
so an ambiguity must never be resolved in the direction that widens the gate.
On this board the rule is not needed (the provenance is citable), and the one
other candidate — 450, the whole run, which is what `02_ring_resonator.py`'s own
un-wrapped Harminv sees — describes the repo script's *live* lane rather than
the tutorial these frozen numbers come from. Both give the same verdict at every
record length (`test_the_verdict_does_not_depend_on_which_reference_record_is_taken`),
and 450 is the tighter of the two. On the live lane the script therefore derives
`T_ref` from the reference run itself (`sim_meep.meep_time()`), which is both the
honest reading — it is literally what that harminv was fed — and the
conservative one.

### An empty gated set is INCONCLUSIVE, not a pass

`gates["q"] = all(row.q_pass is True for row in rows if row.q_gated)` is `True`
over an empty set. Measured on this board by driving the judge: below
`T_cmp = 54.4` no mode is gated, and an rfx `Q` wrong by **100× on every mode**
yields `passed = True`. The hazard pre-dates #907 through `T_rfx`, but `T_rfx` is
the knob the ring-down rule tells you to *increase*, so the old rule drifted
toward more gating. The amendment adds `T_ref` as a second path into the same
state, and it is the path nobody re-runs: shortening the Meep reference run to
save CPU would silently disable the Q gate.

**Pre-declared here: a verdict in which no reference mode is Q-gated is
inconclusive, not a pass.** It is realised as the crossval script's existing
exit 2 ("inconclusive, NOT a pass"; CI must not treat it as green), reported by
`Verdict.q_vacuous` and printed by `format_report`, and pinned by
`test_an_empty_gated_set_is_inconclusive_not_a_pass` and
`test_the_script_exits_2_on_a_vacuous_q_gate`.

It is deliberately **not** a sixth judge gate. A hard "at least one gated mode"
rule would fail ~14% of the pre-declared 200,000-trial stream (reference `Q` is
drawn log-uniform on `[10, 5000]`, so a trial whose reference modes are all
high-`Q` gates nothing), *including defect-free trials* — and criterion (C)
published that the new judge fails zero defect-free trials. Turning a
pre-declared control property red to close a reporting hole would be the wrong
trade.

### What the cv02 Q gap actually is

The pre-#907 `q_window` docstring asserted the gap was "a discretization offset
(staircased ring boundary, subpixel treatment, hence a slightly different
radiation Q)". A semi-analytic exact solve of the same annulus (2-D TM(Ez), 4×4
Bessel/Hankel matching determinant, complex root; committed as
`scripts/diagnostics/cv02_annulus_exact_oracle.py`, reported-only, gating
nothing) settles what can and cannot be said. It is validated by Maxwell scale
invariance — scaling both radii together must leave `Q` invariant and scale `f`
by exactly −1; both recovered to 5–6 digits from independent numerical
derivatives — and it reproduces the case (`f_exact` 0.118192 / 0.147431 /
0.175779, `Q_exact` 77.26 / 343.92 / 1634.21).

**The decomposition (this is the finding).** Signed log errors against the exact
annulus:

| m | `ln(Q_meep/Q_exact)` | `ln(Q_rfx/Q_exact)` | difference | measured rfx-vs-Meep gap |
|---|---|---|---|---|
| 3 | +0.0434 | +0.1130 | **+0.0696** | +0.0696 |
| 4 | −0.0837 | +0.0390 | **+0.1227** | +0.1227 |
| 5 | +0.0261 | +0.1316 | **+0.1055** | +0.1055 |

To four decimals on all three modes: **the observed gap IS the difference of the
two codes' own discretization Q errors at resolution 10.** The gap is
attributed, not unexplained.

**The narrowed falsification.** Along the mode branch (root re-solved at each
perturbation, not the fixed-`k` WKB shortcut, which is 10–20× larger and of the
opposite sign) `∂lnQ/∂R_out = +0.28 / +0.24 / +0.19` per `a`, so the two
solvers' 0.028% frequency agreement bounds an effective-radius offset at
`|Δln Q| ≤ 1.5e-4` and an effective-index offset at `≤ 2.6e-3` — 40 to 700×
below the gaps. What that supports, and all it supports: *the gap cannot be
produced by an effective radius or index shift of the ideal annulus of a
magnitude the frequency agreement permits.* It is **not** a falsification of
discretization as a class. The model's own self-test refutes that stronger
reading: fed Meep's own frequency error against the exact annulus (0.076% /
0.182% / 0.303%) it predicts a Q error of at most 3.8e-3 / 1.3e-2 / 2.7e-2,
against measured 0.0434 / 0.0837 / 0.0261 — an under-prediction of 11× / 7× /
1×. Radiation `Q` is out-coupled flux through a staircased curved boundary; it is
not a smooth function of the effective annulus the frequency sees, so the branch
sensitivity matrix is the wrong instrument for that question.

### Honesty about what a pass on this gate means

The window is a **worst-case resolution envelope**, not an accuracy bound. The
same oracle measures the reference's *actual* Q accuracy on this board at 4.3% /
8.4% / 2.6% — roughly 17× better than the ±72% / ±228% / ±1016% envelope,
because filter diagonalisation routinely beats `1/T` on an isolated high-SNR
mode. So the honest statement is not "a 7% / 12% disagreement is not evidence of
anything"; it is: *the reference is demonstrably far better than the envelope,
the floor's looseness is the price of having no accuracy bound, and that is a
known weakness of this gate which #907 does not remove.*

And no reference-side-only accuracy bound can replace it here. rfx's own error
against the exact annulus (11.3% / 3.9% / 13.2%) **exceeds the reference's on
two of three modes**; the strongest such construction the evidence supports —
resolution term plus the reference's measured error against the exact annulus —
gives 0.0573 / 0.1266 / 0.2045 at the −40 dB record against gaps of 0.0696 /
0.1227 / 0.1055, i.e. it fails mode 1 outright and clears mode 2 by 3%. **The
cv02 Q gate passes because the envelope is loose, not because rfx's radiation Q
is demonstrated accurate.** Anyone reading the close of #907 as a clean bill of
health on rfx's radiation Q is reading it wrong.

### Disclosure (burned-data discipline)

The measured gaps 0.070 / 0.123 / 0.105 were in view throughout, as were the
pre-#907 pass/fail outcomes at `T = 291 / 3385 / 15600`. **No number in this
amendment was chosen after seeing them.** The amendment introduces no constant
at all: `T_cmp = min(T_rfx, T_ref)` is a symbol, `T_ref` is a published property
of the reference run, and every threshold (`5%`, `>= 2`, `1/4`) is unchanged.
The test that it is not re-aimed is quantitative: the amended thresholds sit
7.8× (mode 1) and 9.7× (mode 2) above the gaps they must admit, so the gap would
have to grow 7.8× before mode 1 reddened — a fitted floor sits just above the
number it must admit, and this one does not. For contrast, the best
accuracy-based alternative sits at 0.82× / 1.03× and fails anyway.

Candidate selection was outcome-aware in exactly two places, both disclosed: the
exact-annulus discretization bound was rejected partly because it does not fix
the defect (rejecting a candidate that does not do the job is the task), and the
sum composition was rejected in favour of the strictly tighter `min` (a
tightening cannot be pass-motivated).

### What this correction does NOT change

- No gate value moves: `5%` (twice), `>= 2`, `1/4` are all as pre-declared.
- G1–G4 are untouched, in form and in outcome.
- G5's gate form, its two-sidedness and its anti-tautology property are
  unchanged; only *which record* the resolution term reads changes.
- Correction 1's factor-5 proof stands as written.
- The committed verdict at `T_rfx = 291` is bit-identical, and so is every
  pre-existing number in `tests/fixtures/cv02_ring_judge/tautology_trials_200k.json`
  (regenerated and diffed: the only changes are two ADDED keys,
  `reference_record_T_meep_units` and a `q` breakout in `failures_by_gate`).
  The fixture is unaffected because its `RECORD_T = 291` is below every
  candidate `T_ref`, so `T_cmp = 291` either way — *not* because
  `q_log_ratio ≡ 0`: the stream draws "missing" modes, an unmatched-but-gated
  row leaves `q_pass = None`, and the new `q` breakout counts 50,593 such
  failures.
- The record-length planner (`plan_record` / `RecordPlan` / `ModeSettling` /
  `resolvable_tau_bound`) recommends the same lengths as before. It serves the
  settling witness on the no-verdict lane, where there is no reference at all,
  so "a longer record observes more of the decay" remains true there without
  qualification. What changed is only the *reported* resolvable-tau bound on the
  verdict lane, which now follows `T_cmp` — otherwise the report would call a
  mode resolvable that the gate does not gate.
- Meep is not installed on this host, so the verdict lane was not executed end
  to end for this correction. Everything above is computed from the committed
  frozen mode pair, from direct drives of the judge, and from the exact-annulus
  oracle; the live lane's `T_ref` must be read off a Meep-equipped host before
  the ≈450 figure is quoted as measured rather than derived.
