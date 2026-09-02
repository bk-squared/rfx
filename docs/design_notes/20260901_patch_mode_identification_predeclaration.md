# cv05 + cv15: replacing the self-confirming resonance selector with mode-resolved identification

**Status:** PRE-DECLARATION (this note and the gate code it describes are
committed BEFORE the measurements that judge them).
**Opened:** 2026-09-01 · **Issue:** #812, cv05 + cv15 lane · **Author:** implementation agent
**Append-only.** Corrections are added as new sections stating the old value and why it was wrong.

---

## 1. The defect being repaired

Both patch cross-vals pick their gated resonance the same way:

* `validation/crossval/05_patch_antenna.py` — rfx side and openEMS side both do
  `modes_good.sort(key=lambda m: abs(m.freq - f_resonance_an)); best = modes_good[0]`,
  and then print `harminv_err_pct = |f_res - f_resonance_an| / f_resonance_an`.
* `validation/crossval/15_patch_antenna_rt5880.py` — `_harminv_f0()` does
  `good.sort(key=lambda m: abs(m.freq - fr_an))` and returns `good[0]`.

The anchor is both the selector and the referee. Whichever ring-down mode
happens to sit nearest the closed form is promoted to "the" resonance, so the
reported distance is bounded by the spacing of the cavity spectrum rather than
by the physics, and a build whose design mode has drifted far from its declared
place silently re-anchors onto a **different member** of that spectrum.

## 2. What replaces it

`validation/crossval/comparators/patch_mode_identification.py`.

The declared geometry fixes a whole spectrum, not one number:

```
f_mn = (c/2) * hypot( m / (a_eff sqrt(eps_a)) , n / (b_eff sqrt(eps_b)) )
eps_eff(width) = (er+1)/2 + (er-1)/2 (1 + 12 h/width)^(-1/2)
dL(width)      = 0.412 h (eps_eff+0.3)(width/h+0.264) / [(eps_eff-0.258)(width/h+0.8)]
a_eff = a + 2 dL(b),   b_eff = b + 2 dL(a)
```

`(1,0)` and `(0,1)` reproduce each script's own single-mode Balanis closed form
**exactly** (verified: cv05 `f_resonance_an` and cv15 `f_res_analytic()[0]` are
bit-for-bit the `(1,0)` entry); `(1,1)` is the standard separable-cavity
combination. Members are restricted to each case's own harminv extraction band,
because a member outside the band cannot be measured.

Gate clauses:

* **G1** — every measured mode inside the identification span is assigned to a
  declared member within the tolerance, **injectively**.
* **G2** — the DESIGN member has exactly one mode assigned to it; that mode is
  the reported resonance. No anchor distance selects it.
* **G3** — at least one further identified member resolves the **other**
  in-plane axis (TM01 / TM11 / TM02), so the verdict rests on a mode **pair**,
  not a scalar.

## 3. The threshold, derived (no measured frequency enters it)

For two adjacent declared members `f1 < f2`, the tolerance windows
`[f1/(1+t), f1(1+t)]` and `[f2/(1+t), f2(1+t)]` stay disjoint iff
`(1+t)^2 < f2/f1`. So the **largest tolerance for which "nearest declared
member" is unique** is

```
tol = sqrt( min_adjacent(f2/f1) ) - 1
```

Evaluated on the declared geometry alone:

| case | declared members in band | adjacent ratios | **tol** | identification span |
|---|---|---|---|---|
| cv05 (eps_r 4.3, h 1.5 mm, a 29.5 mm, b 38 mm, band 1.5–3.5 GHz) | TM010 1.91491, TM100 2.42351, TM110 3.08874 GHz | 1.265598, 1.274489 | **12.4988 %** | [1.70216, 3.47479] GHz |
| cv15 (eps_r 2.2, h 3.175 mm, a 40 mm, b 50 mm, band 1.6–3.4 GHz) | TM010 1.97405, TM100 2.41560, TM110 3.11961 GHz | 1.223674, 1.291446 | **10.6198 %** | [1.78454, 3.45091] GHz |

This is an **identification** tolerance and is deliberately **looser** than the
closed form's own 5–8 % accuracy. Passing it is not an accuracy claim and no
accuracy claim may be read from it; cv05's `harminv_err_pct` stays REPORTED and
NOT GATED, exactly as before — what changes is that it is now computed on the
mode-resolved design mode, so a captured neighbour can no longer make it small.

Modes outside the identification span are printed and **not** gated: they belong
to higher members the in-band declared set does not model.

## 4. Pre-declared falsifiers

| # | case | injected defect | required verdict | required reason |
|---|---|---|---|---|
| F1 | cv05 | realized patch resonant length shortened so the true design mode reads **+24 %** against the declared-geometry anchor | FAIL | `G2`: declared DESIGN member TM100 has no measured mode within 12.4988 % |
| F2 | cv15 | one in-plane patch dimension mis-realized by more than the tolerance | FAIL | `G2` (as F1) |
| F3 | cv05, cv15 | correct build | PASS | every measured in-span mode identified, design member found, second axis resolved |

## 5. What this instrument CANNOT see — stated before measuring, not after

Every observable in §2 is a ratio of a measured frequency to a declared one.
Under a **common-mode dilation** of the whole cavity spectrum, `f_mn -> s f_mn`,
every identification residual moves by exactly `s - 1` and **every dimensionless
spectral observable — the mode-pair ratio included — is unchanged**.

Issue #740's vacuum ground cell is precisely such a defect: it dilutes the
cavity's series capacitance uniformly, so it rescales the spectrum instead of
reshaping it. The audit's proposed remedy for cv15 was a mode-pair **ratio**
band; a ratio band therefore **cannot** fire on #740 (withdrawn in §6.9 — the interval is non-empty; §6.9 and §6.10), and this note records that
prediction before §6's measurement is quoted anywhere.

Defects of that class belong to the realized-geometry checks
(`assert_realized_stack`, PR #768), not to a spectral test. The two instruments
are complementary and neither substitutes for the other:

* spectral identification sees **per-axis** errors (a mis-realized in-plane
  dimension, a captured neighbouring mode) and is blind to uniform rescaling;
* the realized-wall-plane check sees **stack** errors and is blind to in-plane
  ones.

## 6. Measurement log

Filled in by the commits that follow this one. Every number here must be
re-derivable from the file it cites.

**Provenance of §3's numbers, stated plainly.** The tolerances and spans in §3
are functions of the DECLARED constants only (`eps_r`, `h`, `a`, `b`, the
extraction band) through the closed form in §2 — re-derivable with
`identification_tolerance(members_in_band(declared_cavity_spectrum(...)))` and
pinned by `tests/test_patch_mode_identification.py`. No measured frequency
enters them. Baseline characterisation runs of both cases were made before this
note (they are what established that cv05's ring-down does **not** contain the
TM010 cross mode, §6.1); they are recorded below, and they did not set any
threshold. The falsifier runs of §4 are made **after** this commit.

---

### 6.1 cv05 — the cross mode is NOT in the ring-down (baseline characterisation)

`python validation/crossval/05_patch_antenna.py` (CPU, 205 s to the openEMS
probe, then `exit 2` — CSXCAD/openEMS are absent on this host, so PART 2 and
PART 3 could not run and the case's own end-to-end verdict is **not**
demonstrable here). PART 1's ring-down, Q > 2 and amplitude > 1e-8:

| f (GHz) | Q | nearest declared member | residual |
|---|---|---|---|
| 2.33227 | 42.3 | **TM100** 2.42351 | **−3.76 %** |
| 3.02744 | 74.3 | TM110 3.08874 | −1.98 % |
| 3.60524 | 105.0 | — (above the span) | not gated |

**TM010 (1.91491 GHz) is absent.** The committed configuration is mirror
symmetric in y — patch, ground plane and substrate are centred on `dom_y/2` and
the source sits at `feed_y = dom_y/2` — and the y-resonant mode is odd about
that plane, so it is not excited. The audit's cv05 sentence describes capture by
"the TM001 cross mode"; on this run the mode available to capture the anchor is
**TM110**, and §6.3 shows it doing exactly that. This is a correction of the
mechanism, not of the finding: the selector is self-confirming either way.

The run is **UNDER-SETTLED**: the settling witness reads −21.5 dB against a
−40 dB bar, and cv05 prints that witness without gating it. Every cv05
frequency here is therefore identification evidence, not accuracy evidence.

### 6.2 cv05 criterion (A) — the correct build passes with margin

Same run, through the new gate: **PASS**. TM100 found at 2.33227 GHz,
residual −3.76 % against the derived 12.4988 % tolerance — a **3.3×** margin —
and TM110 resolves the second in-plane axis (G3). The reported resonance is
**the same 2.33227 GHz the anchored selector returned**, so no number cv05
publishes moves.

### 6.3 cv05 criterion (B) — a mis-realized resonant length fails, by name

Injection: a `runpy` harness patches `rfx.Box`'s x extent for the patch
conductor only. Every declaration — `L`, `f_resonance_an`, the mode set —
still says 29.5 mm; the solver builds something else. Three live FDTD runs:

| realized L | design mode | vs declared TM100 | captured by | verdict |
|---|---|---|---|---|
| 29.5 mm (correct) | 2.33227 GHz | −3.76 % | TM100 | **PASS** |
| 22.5 mm | 2.87308 GHz | **+18.55 %** | TM110 (−6.98 %) | **FAIL** |
| 21.0 mm | 3.11259 GHz | **+28.43 %** | TM110 (+0.77 %) | **FAIL** |
| 38.0 mm | 1.81365 GHz | **−25.16 %** | TM010 (−5.29 %) | **FAIL** |

Every failure prints the same reason:

> `declared DESIGN member TM100 (2.4235 GHz) has NO measured mode within
> 12.50% -- the ring-down carries [... -> TM110]. The design resonance was not
> found; it was not merely mis-measured.`

**On the audit's exact +24 % point.** It is not realizable on cv05's dx = 1 mm
mesh: one cell of patch length is ≈ 4–5 % in frequency here, so the reachable
neighbours are +18.55 % (22 cells) and +28.43 % (21 cells) — both measured,
both FAIL. The +24 % point itself is pinned algebraically instead: a design
mode at 1.24 × TM100 sits 2.7 % below TM110 and 24 % from TM100, so it is
captured by TM110 and TM100 is reported missing
(`test_design_mode_24_percent_high_fails_because_the_member_is_not_found`).

**What the OLD selector did on these runs.** `argmin |f − f_analytic|` returned
the drifted mode itself (there was no closer neighbour), i.e. it *printed*
+18.55 / +28.43 / −25.16 % — and cv05 gated **none** of it: `pass_vs_analyt` is
excluded from `all_ok` by an explicit comment. The −25.16 % run is the audit's
capture mechanism made concrete: that mode lands **inside the declared TM010
member's window at −5.29 %**, so had TM010 been excited the selector would have
returned it and reported near-agreement.

### 6.4 cv15 criterion (A) — the refreshed leg passes every gate

`compare()` on the committed legs: **ALL GATES PASSED, exit 0.**

```
[PASS] f0 agreement (rfx ring-down vs openEMS): 0.69% <= 8%
[PASS] stack geometry fidelity (realized wall planes, #740): ... two_plane
[PASS] mode-resolved identification (design mode FOUND, #812): tol 10.62%
       (derived); 1.8936->TM010 -4.07%, 2.3139->TM100 -4.21%,
       3.0840->TM110 -1.14%, 3.7126->UNIDENT
[PASS] settling witness (open CPML, -40 dB bar): -54.0 dB
[PASS] rfx passivity (max|S11| <= 1.05): max|S11| = 0.997
[PASS] openEMS passivity (max|S11| <= 1.05): max|S11| = 0.992
[PASS] broadside D envelope (<= 3 dB): rfx 7.24 vs oems 7.34 dBi, dD = 0.09 dB
```

Worst identification residual 4.21 % against the derived 10.6198 % — a **2.5×**
margin.

### 6.5 cv15 criterion (B) — the #740 defect does NOT fire, and here is the proof

Live reproduction of the pre-#768 realization through cv15's **production**
builder, `build_rfx_sim(two_plane=False)`:

| member | correct (two_plane) | #740 (one_plane) | ratio |
|---|---|---|---|
| TM010 | 1.893648 GHz | 2.026796 GHz | 1.070313 |
| TM100 | 2.313947 GHz | **2.471858 GHz** | 1.068243 |
| TM110 | 3.083971 GHz | 3.286636 GHz | 1.065716 |

The reproduction's TM100, 2.4718581 GHz, is the committed pre-fix leg's
`f_harminv_hz` 2.4718581 GHz to **6 × 10⁻⁹** relative — the same defect.

Mean dilation **1.06809**, half spread **0.0023**. The defect is a *dilation*,
not a *reshaping*: each member's identification residual moves by the dilation
(+2.67 / +2.33 / +5.35 %), all inside the 10.6198 % tolerance, and the mode-pair
ratio moves by **0.19 %** (1.22197 → 1.21958). So:

* the mode-resolved identification **PASSES on the #740 defect** — `mode_id_ok:
  true` on the live reproduction;
* the audit's proposed **mode-pair RATIO band cannot fire on #740 either**, at
  any width that admits the correct build;
* this was pre-declared in §5 **before** the reproduction was run, and is now
  measured.

What does fire is `assert_realized_stack`, which refuses the build outright:

> `assert_realized_stack: realized electric-wall plane(s) do not match the
> declared stack -- no electric wall at z_sub_lo=7.9375 mm (k=18, ok=False)
> ... Refuse to quote f0 for a cavity taller than the declared substrate
> (issue #740).`

**Verdict on falsifier F2/#740: STOP, with proof.** Criterion (B) for cv15 as
the audit framed it — "the pair gate must fire on #740" — is **not met and
cannot be met by any dimensionless spectral instrument**. The committed pre-fix
leg does fail the live judge (it always did, on the wall-plane check, and now
additionally on the spectral gate's SCHEMA clause because it carries no mode
list) — but that is a *schema* failure, not a physics measurement of its
spectrum, and this note refuses to count it as (B). Numbers and the refusal
string are committed in
`tests/fixtures/patch_mode_identification/cv15_ringdown_spectra.json` so the
claim can be refuted without a solve.

### 6.6 cv15's rfx leg was STALE — refreshed here, with the delta bisected

The gate needs the leg to carry its mode list, so the leg was regenerated.
Doing so exposed that the committed leg no longer reproduced. Bisected with
three control runs of the identical command:

| code point | max\|S11\| | f_dip | dip depth | f0 (ring-down) | Q | D |
|---|---|---|---|---|---|---|
| `1f005d0` (the leg's own commit, 2026-08-29, PR #768) | 0.786967 | 2.3100 GHz | −4.4298 dB | 2.3139475 GHz | 18.897344 | 7.242423 dBi |
| committed leg as it stands | 0.786966 | 2.3100 GHz | −4.4298 dB | 2.3139474 GHz | 18.897387 | 7.242423 dBi |
| `ad13b4c^` (post-#776, pre-#777, 2026-08-31) | **1.546926** | 2.1800 GHz | +1.6670 dB | 2.3139475 GHz | 18.897344 | 7.242423 dBi |
| `main` today | 0.996699 | 2.3200 GHz | −0.3448 dB | 2.3139475 GHz | 18.897344 | 7.242423 dBi |

* The leg **reproduces at its own commit** (|S11| agrees to 4.2 × 10⁻⁷,
  float32 noise) — its provenance is sound; it is stale, not unprovenanced.
* The **ring-down half is invariant across all four**: f0 to 2.5 × 10⁻⁸, Q to
  2.3 × 10⁻⁶, settling to 9 × 10⁻⁵ dB, directivity **bit-identical**. So the
  FDTD field solve did not move; the change is entirely in the wire-port
  S-parameter extraction.
* The |S11| change is bracketed by the two merged wire-port PRs between the
  leg and today — **#776 `7c80714`** (whole-port driven diagonal, 2026-08-30)
  and **#777 `ad13b4c`** (uniform-lane POST flip + decomposer recalibration,
  2026-08-31).
* **This is physics/extractor territory, not this lane's**, and it is reported
  rather than adjudicated here. Two things in it deserve a separate issue:
  at `ad13b4c^` cv15's own passivity gate would have **FAILED** at
  max|S11| = 1.5469, and on today's `main` the margin has fallen from 0.263 to
  **0.053** below the 1.05 bar.

Refreshed leg: `validation/crossval/_15_patch_results/rfx.json`, produced by
`python 15_patch_antenna_rt5880.py rfx --num-periods 45.0 --n-freqs 181 --gain`
at commit `5b6db32`, now carrying a `provenance` block. Gated consequences of
the refresh: the f0 gate is unchanged (it reads the ring-down, which did not
move — 0.69 % vs openEMS either way); passivity stays PASS; dip depth is not
gated.

### 6.7 CORRECTION (2026-09-01, same day) — two digits in §6.5

**Old text, §6.5:** "the mode-pair ratio moves by **0.19 %** (1.22197 →
1.21958)".

**Why it was wrong.** Those two ratios were computed by hand from the
4-decimal-GHz values printed in an exploratory probe log, then written out to
six digits as if they had that precision. Recomputed from the committed
fixture `tests/fixtures/patch_mode_identification/cv15_ringdown_spectra.json`
itself:

```
TM100/TM010, correct build : 1.2219522385412454
TM100/TM010, #740 defect   : 1.2195891710298377
change                     : -0.19338 %
```

**Corrected values: 1.221952 → 1.219589, a change of −0.1934 %.** The
conclusion is unchanged — the ratio is invariant to the defect at the 0.2 %
level, two orders below any window that admits the correct build — but the
digits are now the file's, not a hand transcription of a rounded log line.

Nothing else in §6 was transcribed this way: every other number here was
printed at full precision by the run or recomputed from the committed file, and
all of them were re-verified against those files after §6 was written.

### 6.8 CORRECTION (2026-09-01, same day) — a withdrawn negative claim in §6.3, and a unit slip in §6.6

Three items. The first is the one that matters.

**(1) WITHDRAWN — "the +24 % point is not realizable on cv05's dx = 1 mm mesh".**

> Old text, §6.3: "**On the audit's exact +24 % point.** It is not realizable on
> cv05's dx = 1 mm mesh: one cell of patch length is ≈ 4–5 % in frequency here,
> so the reachable neighbours are +18.55 % (22 cells) and +28.43 % (21 cells) —
> both measured, both FAIL."

**That claim is false and is withdrawn in full.** It was a negative existence
claim ("not realizable") published from the two injections I happened to have
run, without the census that could refute it. The census — the patch Box's
realized x-cell count on cv05's own grid, no FDTD, now committed as
`_realized_x_cell_census` in
`tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json` — says:

```
declared 29.5 mm -> 29 cells      declared 22.0 mm -> 22 cells
declared 23.0 mm -> 23 cells      declared 21.65 / 21.5 / 21.0 / 20.5 mm -> 21 cells
declared 22.5 mm -> 23 cells      declared 38.0 mm -> 38 cells
```

A **22-cell** realization exists between the two I had run, and the old text
even mislabelled the 22.5 mm injection as "22 cells" when it rasterizes to 23.
Running it (live FDTD, same harness):

| realized | design mode | vs declared TM100 | captured by | verdict |
|---|---|---|---|---|
| 23 cells (declared 22.5 mm) | 2.87308 GHz | +18.55 % | TM110 (−6.98 %) | FAIL |
| **22 cells (declared 22.0 mm)** | **2.98956 GHz** | **+23.36 %** | TM110 (−3.21 %) | **FAIL** |
| 21 cells (declared 21.0 mm) | 3.11259 GHz | +28.43 % | TM110 (+0.77 %) | FAIL |

So the audit's +24 % point **is** essentially realizable: +23.36 % is the
closest the lattice offers, 0.64 percentage points from +24 %, and it FAILS
with the same message. Criterion (B) for cv05 is met at the point the audit
named, not merely bracketed around it. The per-cell step is ≈ 4.8–5.1
percentage points here (measured: 18.55 → 23.36 → 28.43 %), not the "4–5 % of
frequency" the old text asserted without measuring.

*Why this happened, since the issue has been bitten by it twice already:* I
searched for a realization at +24 % by running two injections and, when neither
landed there, concluded none could. That is the search that fails to confirm,
not the search that refutes. The refuting search — enumerate the realized cell
counts — costs no FDTD at all.

**(2) Unit slip, §6.6.** Old text: "settling to 9 × 10⁻⁵ dB". The settling
witness values are −53.98492 dB (three code points) and −53.98978 dB (the
committed leg): the difference is **4.9 × 10⁻³ dB absolute, 9 × 10⁻⁵
relative**. The "9 × 10⁻⁵" figure was right; the unit label on it was not.

**(3)** §6.3's summary table and the audit-point paragraph are superseded by
the four-row table above; the 38.0 mm row (−25.16 %, captured by TM010 at
−5.29 %) is unchanged.

### 6.9 CORRECTION (2026-09-01, round 2) — the ratio-band absolute is withdrawn, cv15 lands as an honest STOP, and the regenerated leg is reverted

Three items, under the numeric-provenance discipline #812 adopted for this
round: the numbers below live in committed JSON and are named by artifact key,
not restated as digits.

**(1) WITHDRAWN — the ratio-band absolute (§5 and §6.5).**

> Old text, §6.5: "the audit's proposed **mode-pair RATIO band cannot fire on
> #740 either**, at any width that admits the correct build" — and, in the same
> section, "cannot be met by **any** dimensionless spectral instrument".
> Old text, §5: "a ratio band therefore **cannot** fire on #740".

**False, and withdrawn in full.** A reviewer refuted it. Same failure mode as
§6.8: a negative existence claim published from the search that failed to
confirm rather than the one that could refute. The refuting search costs no
FDTD — it is closed form over numbers already committed — and is now itself
committed as `scripts/diagnostics/build_patch_mode_pair_ratio_band_census.py`
→ `tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json`,
re-derived field by field in
`test_mode_pair_ratio_band_census_reproduces_from_the_committed_spectra`.

What the measurement supports: a declared-anchored band
`|r_measured/r_declared − 1| ≤ w` on the TM100/TM010 pair admits the correct
build **and** rejects the #740 realization for every `w` in
[`tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json::declared_anchored_band.min_half_width_admitting_correct_build = 0.001407`,
`tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json::declared_anchored_band.max_half_width_still_rejecting_740 = 0.003338`). That interval
is non-empty (`tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json::declared_anchored_band.admissible_interval_is_nonempty`), and
a band at its geometric midpoint is exercised in both directions by
`test_a_band_from_the_census_interval_does_separate_the_two_realizations`. §5's
model statement — that an **exact** common-mode dilation leaves every
dimensionless observable unchanged — is still true; the error was applying it to
#740, whose dilation is only near-common
(`tests/fixtures/patch_mode_identification/cv15_ringdown_spectra.json::measured_common_mode_dilation.half_spread`), so
the pair ratio does move.

**(2) cv15 lands as an honest STOP; round 1's cv15 re-gate is withdrawn.**
Criterion (B) requires the new gate to fail on the defect the audit measured
cv15 blind to, which for cv15 is #740. It does not: the mode-resolved
identification PASSES on the live reproduction through cv15's production
`build_rfx_sim(two_plane=False)` (§6.5, unchanged). Per (1) a ratio band *can*
be made to separate the two realizations, but not one this lane may adopt:
both interval endpoints are properties of the two measurements the band would
then judge (burned-data, the burned-data rule (a threshold may not be set from the measurements it will judge; the lane spec's rule 0.2.2, kept outside the repo)); the interval's upper endpoint is
`tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json::declared_anchored_band.upper_endpoint_over_identification_tolerance = 0.0314` of the
tightest window derivable from declared geometry alone; and no committed run
establishes the run-to-run reproducibility of the measured pair ratio at that
scale. So the correct statement is **not** "no spectral instrument can meet (B)"
but "**(B) is not met by any window whose provenance this lane can supply**".

Consequently `validation/crossval/15_patch_antenna_rt5880.py` is reverted to
`main`. cv15 keeps its anchored `_harminv_f0()` selector and its existing
gates; the audit's cv15 finding stays **OPEN**, and this note is the record of
why. A gate that meets only criterion (A) is cosmetic and is not shipped.
`test_cv15_declared_constants_are_the_scripts_own` fails if cv15 regrows the
withdrawn `_mode_id_ok` gate, so re-opening this STOP is a deliberate act.

**(3) The regenerated cv15 rfx leg is reverted.** Round 1 rewrote
`validation/crossval/_15_patch_results/rfx.json` on today's `main` (§6.6) so the
now-withdrawn gate could read a mode list. With no gate to feed, the
regeneration has no purpose and the leg is restored to #768's committed
version, which owns it. No evidence is lost: the reproduction's ring-down and
the committed leg's are the same solve to the bound asserted by
`test_cv15_reproduction_ringdown_matches_the_committed_leg`, and the mode list
lives in the `two_plane_ground.modes` list of `tests/fixtures/patch_mode_identification/cv15_ringdown_spectra.json`, whose `source`
field now names the run rather than the leg. §6.6's wire-port |S11| bisection
stands as a **report** — it is extractor territory, not this lane's, and it
adjudicates nothing here.

**Unchanged: cv05.** Its re-gate ships. (A) and (B) are re-verified in this
round against the committed fixtures — the correct build's design-member
residual is more than 3× inside the derived tolerance, and the audit's
+24 % point fails by name at the closest realization the lattice offers
(the `runs.patch_len_22p0mm` block of `tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`, with the reachable set in
the `_realized_x_cell_census` block of `tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`). Every digit in those two sentences is re-derived
from the fixture inside the assertions, not transcribed.

**What would be needed to close cv15's (B) later.** The reviewer's reproduction
is the starting point: `build_rfx_sim(two_plane=False)` reproduces the pre-#768
frequencies, so the defect is available on demand. Closing (B) needs, in order:
(i) a mode-pair ratio window **derived** from declared geometry or first
principles at the scale of `tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json::declared_anchored_band.max_half_width_still_rejecting_740 = 0.003338`,
which the closed form's own 5–8 % accuracy does not supply; and (ii) a
committed reproducibility census of the measured pair ratio across settling
length and mesh showing scatter below
`tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json::declared_anchored_band.min_half_width_admitting_correct_build = 0.001407`. Absent both,
#740's detector is and remains `assert_realized_stack` (PR #768).

**Dangling pointer noted.** §6.5 quotes "`mode_id_ok: true` on the live
reproduction". That field was written by the gate withdrawn in (2) and no
longer exists in cv15. The fact it recorded is unchanged and is now asserted
directly on the committed spectra by
`test_cv15_740_defect_is_a_common_mode_dilation`
(`assert ident_bad.ok` — the spectral identification passes on the #740
realization). §6.5 is left as written; this is the correction.

### 6.10 Round-2 review (2026-09-02) — two blockers, what closed them, what is recorded

An independent reviewer re-derived the cv05 spectrum (TM010 1.914913 / TM100 2.423510 /
TM110 3.088737 GHz), the tolerance (12.4988 %), every fixture row including the
+23.36 % point, the x-cell census (rebuilt from the grid without FDTD), and every key in
the ratio-band census; confirmed the note is append-only and the windows were fixed
(9f5c450) before the measurements (9c03c02); confirmed cv15's script and committed leg
are byte-identical to `main`; and returned FIX-THEN-SHIP.

**(B1) cv05's openEMS-leg identification sat in `all_ok` with no run ever showing it
pass.** §6.1's PART 2/3 never ran here (no openEMS on this host), so
`pass_mode_id_oe` was a gate added to the verdict by construction. It is not dropped —
it is measured: `scripts/vessl_cv05_openems_and_ringdown_fixture.yaml` runs the shipped
script through all three parts on the lab cluster's openEMS image, capturing the JSON
the script already emits (`openems_mode_id_ok`). Recorded in 6.11 when it lands; until
then this section says the gate is unobserved.

**(B2) The withdrawn absolute survived in a fixture key.**
`tests/fixtures/patch_mode_identification/cv15_ringdown_spectra.json::measured_common_mode_dilation.note` still said "every
dimensionless spectral observable is blind to it" after §6.9 withdrew it; it now says the
dilation is *nearly* common (half-spread 0.0023) and that the ratio moves by
`tests/fixtures/patch_mode_identification/cv15_mode_pair_ratio_band.json::measured_anchored_band.defect_offset_from_correct_build = 0.001934`.

**Also closed.** "harminv_err_pct stays REPORTED and NOT gated" was false in effect: G2
bounds the design mode's error to the log-symmetric window [−11.11 %, +12.50 %], so an
error beyond it fails the case by identification; the docstring, the print and this
note now say so (the lower edge had never been stated). The cv05 fixture's producer was
an uncommitted runpy harness; `scripts/diagnostics/build_cv05_ringdown_spectra.py` is
that harness, committed, driven by a `RFX_CV05_PATCH_L_MM` hook in the script, and the
same VESSL job rebuilds the fixture and compares it to the committed copy. Nothing
outside this note and the test said what changed: the manifest, README and public rows
for cv05 now describe the identification gate, and those for cv15 state the blindness of
the 8 % gate to #740 (+6.09 %, cited by key) and the STOP. §5's "cannot fire" sentence
carries an in-place forward pointer to §6.9 (form only); the out-of-repo "SPEC-00 §0.2.2"
pointer is replaced by the rule it named (a threshold may not be set from the
measurements it will judge) in the note, the test and the census artifact. The ten
citation spans that did not parse under the #829 gate are rewritten with repo-relative
paths and values.

**Recorded, not changed.** Identification does not enforce spectral *ordering*: a
simultaneous x- and y-mis-realization that moves TM010 into TM100's window and the design
mode into TM110's would pass G1–G3 — a two-parameter coincidence, stated as the
instrument's limit. The public benchmarks table's patch far-field row still quotes the
retired pre-#693 envelope (sign-flipped against the committed test constants); that is
on `main`, outside this lane, and is left for its owner.
