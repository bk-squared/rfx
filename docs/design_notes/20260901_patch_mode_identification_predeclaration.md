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
band; a ratio band therefore **cannot** fire on #740, and this note records that
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
