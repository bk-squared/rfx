# Landing pre-declaration — NTFF half-cell collocation (#820)

**Status:** PRE-DECLARATION. No `rfx/` change is proposed by this note and none has
been made. It says what landing the correction would move, and which of those
movements is an oracle rather than a fixture. · **Opened:** 2026-09-14 ·
**Branch:** `diag/820-rcs-translation` · **Issue:** #820 · **PI:** step 2 approved
2026-09-14 (measure in a scratch copy, stop before touching `rfx/`).

Measurements live in `scripts/diagnostics/issue820_results/`; the scratch extractor
is `scripts/diagnostics/issue820_ntff_collocation.py`, whose module docstring holds
the arm pre-declaration and its gates.

---

## 1. The defect, in one paragraph

`compute_far_field` gives all four stored components of a face one position
(`_face_positions`, `rfx/farfield.py:344`) and samples them on one plane
(`accumulate_ntff`, `rfx/farfield.py:247-268`), while the Yee grid puts them at six
different half-cell locations:

| | offset from the node, in cells |
|---|---|
| `ex` | (½, 0, 0) |
| `ey` | (0, ½, 0) |
| `ez` | (0, 0, ½) |
| `hx` | (0, ½, ½) |
| `hy` | (½, 0, ½) |
| `hz` | (½, ½, 0) |

The **temporal** half-step between E and H is already corrected
(`rfx/farfield.py:238`, `phase_h` uses `t + dt/2`). The **spatial** one is not. At 41
cells per wavelength the resulting phase between the `J` and `M` terms of a face is
**4.3902°**.

Two independent reasons to take it seriously:

1. **The repo already applies this correction elsewhere.** `rfx/simulation.py:1802-1811`
   averages H at `idx−1` and `idx` to co-locate it with E before forming a Poynting
   cross-product, with a comment saying why; `rfx/nonuniform.py:2292-2295` mirrors it.
   The NTFF surface integral is the one flux-like surface integral in the repo without
   it.
2. **It is not mirror-symmetric.** On a y face the stored H sits at `j+½` on *both*
   `y_lo` and `y_hi` — half a cell inside the box on one, half a cell outside on the
   other. That breaks the y symmetry of the extractor, which is a candidate for the
   **odd** transverse residue #820 measures.

---

## 2. Every consumer that would move

Grepped `compute_far_field`, `compute_far_field_jax`, `_face_positions`,
`_face_positions_jax`, `_surface_currents`, `_surface_currents_jax` across the tree.

### 2.1 `rfx/` call sites

| file:line | what it does | exposure |
|---|---|---|
| `rfx/rcs.py:473`, `:504`, `:592` | `compute_rcs` — pattern, the #280 vacuum reference, and the backscatter bin | every RCS number in the repo |
| `rfx/rcs.py:153` | `compute_rcs_jax` via `compute_far_field_jax` | the differentiable RCS objective |
| `rfx/antenna.py` | `radiation_pattern` / `directivity` are documented as built on `compute_far_field` | antenna gain and directivity |
| `rfx/optimize_objectives.py:371`, `:392`, `:397` | far-field objectives incl. the sphere-integral normalisation | inverse design |
| `rfx/visualize.py:1337`, `:1349` | pattern plotting | figures only |
| `rfx/api/_preflight.py` | NTFF validator | preflight messages |
| `rfx/__init__.py:80`, `:294` | public export | API surface |

`_face_positions_jax` (`rfx/farfield.py:621`) carries the same single-position scheme,
so **the differentiable path inherits the defect and would have to be corrected in
lockstep** — a fix applied only to the numpy twin would silently split the two,
which is the `#743` class this repo has already been bitten by.

### 2.2 Tests that read the transform directly (11)

```
tests/crossval/test_patch_canonical_farfield_e4.py
tests/locks/test_ntff_directivity_validation_battery.py
tests/oracle/test_farfield.py
tests/unit/api/test_visualize_realized_grid.py
tests/unit/nonuniform/test_nonuniform_api.py
tests/unit/farfield/test_farfield_chunking.py
tests/unit/farfield/test_farfield_nonuniform.py
tests/unit/farfield/test_farfield_inplane_nonuniform.py
tests/unit/autodiff/test_directivity_gradient.py
tests/unit/autodiff/test_ad_surface_contract.py
tests/unit/autodiff/test_rcs_jax_differentiable.py
```

62 test files reference NTFF or RCS at all.

### 2.3 Committed artifacts that would need regenerating

```
tests/fixtures/patch_canonical_farfield_e4
tests/fixtures/patch_mode_identification
tests/fixtures/rcs280_reference_subtraction
tests/fixtures/rcs_cube_bem
tests/fixtures/rcs_dielectric_sphere_mie
tests/fixtures/rcs_mie_e4
tests/fixtures/rcs_mie_ka_sweep
tests/fixtures/rcs_sphere_mie
tests/fixtures/rcs_sphere_three_way
```

and the crossval records `validation/crossval/_16_ka_sweep_results/rfx.json`,
`_17_dielectric_results/rfx.json`, plus cv15 / cv16 / cv17 and
`validation/tmtt_paper/beam_steering_superstrate.py`.

`rcs_mie_ka_sweep` is the expensive one: its gate constants `GATE_COARSE_DB = 3.3`
and `GATE_FINE_DB = 4.0` are `round-up(measured clearance-scan envelope × 1.5)`, so a
regenerated fixture re-derives the envelope and the gate test asserts the relation.
That is a full `--write-fixture` run (13 clearances × the gated bins), not an edit.

---

## 3. The one consumer with a known exact answer — and its limits

`tests/locks/test_ntff_directivity_validation_battery.py` (#302) pins the **absolute**
power scale of the NTFF chain against a value that is exact in the continuum:

> `P_ntff / P_flux = 1/2 exactly in the continuum, independent of dt, source`
> `spectrum, and frequency bin`
> base rung dx = 3.0 mm, 400 steps: `P_ntff/P_flux = 0.4947778515` (off by −0.0052)

A −1.04 % power deficit is the right order for a 4.3902° collocation phase
(`cos 4.3902° = 0.99706` → −0.59 % in power), which makes this the most interesting
consumer to re-measure.

**But it is not a clean convergence oracle, and the battery says so itself.** Its dx
ladder reads `0.4948 / 0.5128 / 0.5215` — it crosses 0.5 and keeps going — and the
test file states the consequence in its own words (`:217-219`):

> the ladder ... **deliberately does NOT assert convergence to 0.5**. The
> source-derived 0.5 +/- 0.05 gate lives ONLY on the base-rung fast test above.

So something other than a fixed-rung phase error also drives this ratio with dx.
What can honestly be asked of it:

- **At the fixed base rung**, a collocation phase is a systematic of known sign and
  size, so a correct fix should reduce `|ratio − 0.5|` there. That is suggestive.
- **Across the ladder**, nothing may be asserted, because the uncorrected ladder is
  already non-monotone about 0.5.

Also recorded because the repo rule requires it: this fixture's own preflight emits
λ/4 advisories ("NTFF face x_lo is 9.00mm from port/source ... below λ/4 = 24.98mm
... NTFF will integrate reactive near-field"), so the battery is measured inside the
reactive near field by construction.

## 4. Pre-declared landing gates (for the PI to accept or reject; not run here)

**Revised 2026-09-14 after the verification review. `L3` is deleted: it gated on
`|rfx − Mie|`, and that column has since been shown not to be a discriminator at
this rung.** The sign control — the same correction applied with the deliberately
WRONG sign — beats the correct one on Mie distance by about 0.7 dB at two absorber
depths while being worse on the transverse null (§5.4). At 6.4 cells per radius the
0.69–0.73 dB baseline distance is dominated by staircase error, so a landing gate
built on it would reward the wrong fix. **No Mie-based gate survives in this
programme.**

| # | gate | bar |
|---|---|---|
| L1 | `P_ntff/P_flux` (#302 battery), **base rung only** | `\|ratio − 0.5\|` must not increase at dx = 3.0 mm. Reported, not gated, on the ladder — the uncorrected ladder is already non-monotone about 0.5 and the battery declines to assert convergence (§3) |
| L2 | numpy and JAX twins | corrected `compute_far_field` and `compute_far_field_jax` must agree to the same tolerance `test_farfield_chunking` already pins (bit-identical where it is today) |
| ~~L3~~ | ~~cv16 `\|rfx − Mie\|`~~ | **DELETED** — not a discriminator (see above and §5.4) |
| L4 | the #820 observable | transverse-null p-p and x-sweep `r` at the bars the scratch arms used. The null is the primary: its continuum value is exactly 0.000 dB, so it needs no reference |
| L5 | every moved artifact | regenerated by its own producer, with the old and new values tabled side by side in the PR body — no artifact replayed against a changed extractor |
| **L6** | **sign and direction, two ways** | (a) the JAX twin must move **identically** to the numpy one, not merely within tolerance — a fix that splits them re-opens the `#743` class; (b) the #302 base-rung `\|ratio − 0.5\|` must move **toward zero**, **sign only, magnitude not gated**, because §3 shows the ratio is not a convergent. A landing that moves it away is refused regardless of what it does to #820 |

**`C = Σ|B_f| / |Σ B_f|` is not a gate** anywhere in this programme. It diverges as
the extraction becomes perfect, so it cannot distinguish "better" from "worse"
(correction C2 on the issue).

## 5. Results (measured 2026-09-14, scratch copy, CPU)

Artifacts: `scripts/diagnostics/issue820_results/*_collocation_*.json` and
`*_transverse_*.json`. Baseline reproduces the probe bit-for-bit
(`0.000e+00` dB) at both rungs; the vacuum control keeps 58.4-59.2 dB of
target-to-empty separation on every arm; no warnings were emitted anywhere
(verified with `warnings.simplefilter("always")`, not assumed).

### 5.1 The transverse null, before any correction

The continuum value is exactly 0.000 dB, so every number here is error.

| rung | grid, steps | ring-down | axis | p-p (dB) | odd | even | odd/even |
|---|---|---|---|---|---|---|---|
| r1_ka1 | 91³, 700 | −63.07 dB | y | **0.7020** | 0.6883 | 0.1139 | 6.044 |
| r1_ka1 | 91³, 700 | −63.07 dB | z | 0.1933 | 0.1553 | 0.0807 | **1.924** |
| r1_box45 | 91³, 700 | −63.07 dB | y | 0.7422 | 0.7309 | 0.1054 | 6.934 |
| r2_ka2 | 104³, 700 | −48.61 dB | y | 0.5561 | 0.5561 | 0.0887 | 6.270 |
| r3_fine | 163³, 1400 | −66.22 dB | y | 0.2135 | 0.2130 | 0.0274 | 7.782 |

| gate | reading | verdict |
|---|---|---|
| (t1) odd/even ≥ 3 on **both** axes | y 6.044, z 1.924 | **CARRIER-INCONSISTENT** |
| (t2) dx halved at fixed physical geometry | 0.2135/0.7020 = **0.304** (0.322 sampling-matched), band 0.30–0.70 | **CARRIER-CONSISTENT, first order in dx** |
| (t3) box re-centred 45.5 → 45.0 | odd span 0.6883 → 0.7309, **+6.2 %** | **box offset is NOT the carrier** |

### 5.2 The two corrections

| arm | null p-p (ka=1) | shrink | T | `r` (x sweep) | R | σ(0) dBsm | from Mie | move |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.7020 | 1.000× | FALSE | 0.1189 | FALSE | −24.7679 | +0.7342 | — |
| **arm (i)** H averaged | 0.4716 | 1.488× | INCONCLUSIVE | **0.0798** | INCONCLUSIVE | −25.1377 | **+0.3644** | −0.3698 |
| **arm (ii)** per-component phase | 0.6609 | 1.062× | **FALSE** | **0.1602** | **FALSE** | −24.3796 | **+1.1225** | +0.3883 |
| **arm (iii)** both | 0.4813 | 1.458× | INCONCLUSIVE | **0.0767** | INCONCLUSIVE | −25.2362 | **+0.2659** | −0.4683 |

Same at ka = 2.0: arm (i) null 0.4407 (1.262×), Mie +1.0065 → +0.5191; arm (ii)
null 0.4888 (1.138×), Mie → +1.5211; arm (iii) null 0.4367 (1.273×), Mie → +0.5042.

Mie is evaluated at the realized `a_eff` (the #725 convention, `ka_eff` 0.9893 /
1.9966). The control bar 0.622 dB is the committed fixture's own `delta_db`, which
is the **pre-a_eff** number — stated because the two conventions differ by 0.11 dB
and mixing them silently is exactly the provenance mistake this lane is about.

### 5.3 What this says about landing

- **Arm (ii) alone must not land.** It makes all three observables worse, at both
  rungs, and the pre-declaration said in advance that a correction moving the
  answer away from Mie is a finding rather than something to absorb.
- **Arm (i) / (iii) are a real but partial correction** — about a third of the
  effect, repeatable at two electrical sizes, improving Mie agreement by 0.37 to
  0.50 dB — and both are INCONCLUSIVE against their own 3× bar. That is not
  nothing, and it is not a fix.
- **Something else carries the rest**: first order in dx, odd in y, four times
  weaker in z, untouched by re-centring, and still present after both corrections.
- So `L1`–`L5` are not yet worth spending: there is no candidate landing whose
  gate this lane can predict it would pass. Naming the remaining carrier comes
  first.

### 5.4 Second review: the Mie column is withdrawn, and the dx order is fitted

**The sign control, re-derived in this lane at two absorber depths.** Arm (ii)
applied with the *conjugate* (wrong) phase:

| depth | arm | null p-p (dB) | \|rfx − Mie\| (dB) |
|---|---|---|---|
| CPML 16 | baseline | 0.6404 | +0.6858 |
| CPML 16 | arm (ii), **correct** sign | 0.6345 | **+1.0357** |
| CPML 16 | arm (ii), **wrong** sign | 0.6465 | **+0.3104** |
| CPML 24 | baseline | 0.5887 | +0.6193 |
| CPML 24 | arm (ii), **correct** sign | 0.5907 | **+0.9463** |
| CPML 24 | arm (ii), **wrong** sign | 0.5977 | **+0.2691** |

At both depths the Mie column rewards the **wrong** sign by ~0.7 dB while the
transverse null prefers the **right** one. That is why L3 is deleted and why every
"closer to Mie" claim in this lane is withdrawn, including arm (i)'s.

**Arm R3b — the dx order, with clearance, absorber and record held in physical
units.** The earlier single halving was confounded: it left `CPML_LAYERS` at 8
*cells*, halving the absorber's physical thickness along with dx.

| rung | grid, steps | CPML | dx | ring-down | null p-p | n_occupied | a_eff/a |
|---|---|---|---|---|---|---|---|
| res 41 | 91³, 700 | 8 | 2.437e-3 | −63.07 dB | 0.6841 | 1127 | 0.98864 |
| res 61 | 135³, 1041 | 12 | 1.638e-3 | −68.59 dB | 0.4131 | 3821 | 0.99826 |
| res 81 | 177³, 1382 | 16 | 1.234e-3 | −67.77 dB | 0.2953 | 8952 | 0.99848 |

Two-point exponents **1.269** and **1.184**; three-point least squares
**p = 1.236**, worst log residual **0.0093**. Declared band [0.75, 1.25] →
**FIRST ORDER**. A half-cell collocation phase is first order in `k·dx/2`, so a
collocation term **can** be the leading error — the confounded halving had implied
1.717 and pointed the other way. The realized body changes across rungs
(`n_occupied`, `a_eff/a` above); that is disclosed, not controlled.

**Arm (i)'s own absorber control FAILS.** Its null shrink factor had to stay within
±15 % of its 8-cell value if co-location were what it buys:

| CPML cells | baseline null | arm (i) null | shrink | deviation |
|---|---|---|---|---|
| 8 | 0.6841 | 0.4716 | 1.4504× | — |
| 16 | 0.6404 | 0.3827 | **1.6733×** | **+15.4 %** |
| 24 | 0.5887 | 0.3313 | **1.7769×** | **+22.5 %** |

Both outside, monotone. Part of what arm (i) buys is bound up with the absorber —
on four of six faces its adjacent plane sits on the first interior cell against the
CPML (correction N2). Recorded without spin: the drift runs the *wrong way* for the
simplest version of that story, since a deeper and better absorber should shrink an
absorber-contamination advantage rather than grow it, and the baseline null also
falls with depth (0.6841 → 0.5887). The gate result stands as measured.

### 5.5 Consequence for landing

Unchanged from §5.3 and now better supported: **do not spend these gates yet.**
The dx order says a collocation term is admissible as the *leading* error, which is
new and favourable. But arm (i) has not shown that co-location is what its
improvement buys, arm (ii) is refuted on the longitudinal observables, and the
reference that looked like corroboration was measuring staircase. The next lane
needs to separate co-location from the absorber-interface plane — an arm (i)
variant whose adjacent plane does **not** touch the absorber (a larger
`ntff_offset`, or a deeper box) is the obvious construction — before any `rfx/`
change is worth pricing.
