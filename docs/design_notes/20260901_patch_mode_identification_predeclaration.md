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
