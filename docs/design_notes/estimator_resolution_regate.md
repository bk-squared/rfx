# Estimator resolution re-gate — cv06b and cv07 (issue #812, mechanism P3)

Append-only. Sections 1-4 are the **PRE-DECLARATION**: every numeric window
in them was frozen in commit `fc6ca0a`, before any measurement that judges it.
Section 5 records what was then measured; it never edits a window. A
correction states the old value and why it was wrong. Three in sections 1-3:
the repository's own 70.7 MHz bin figure (section 1), this note's first
attribution of the -30.7 vs -31.23 dB difference (section 1), and its
description of a 0.5 dB prominence as "12 % in |S21|" (section 3, T5) — all
marked in place. **Section 6 appends round 2's, including one WITHDRAWN
construction.** No WINDOW has moved, in either round.

Lane: `agent/regate-estimator-resolution`. Scope: **instrument only.** No
physics verdict of either case is challenged, moved, or re-pinned here.

---

## 1. The finding, re-derived rather than quoted

The #812 audit's P3 paragraph says both cases take `argmin` on a grid coarser
than the tolerance they declare. Re-derived on this checkout:

* **cv06b.** `validation/crossval/06b_msl_notch_filter_uniform.py` calls
  `sim.compute_msl_s_matrix(n_freqs=100, ...)`. That entry point sweeps
  `jnp.linspace(freq_max / 10, freq_max, n_freqs)` (`rfx/api/_sparams.py`,
  the `freqs_arr` assignment in `compute_msl_s_matrix`), i.e. 0.7 - 7.0 GHz in
  100 points = **63.6364 MHz/bin**, which is **1.754 %** at the 3.627 GHz
  notch. The gated notch frequency is `f[argmin(|S21|_dB)]`, so the reported
  `err_pct` is quantised to multiples of 1.754 % about the analytic anchor.
  Confirmed against committed data: `tests/fixtures/msl_notch_e4/
  msl_stub_notch_rfx_dx50.json` carries `freqs_ghz[0] = 0.7`,
  `freqs_ghz[1] - freqs_ghz[0] = 0.0636364` GHz.

  **CORRECTION to a number already in the repository.** The case's own
  docstring section "THE NOTCH-FREQUENCY ROW IS BIN-LIMITED" states
  *"`compute_msl_s_matrix(n_freqs=100)` over the 7 GHz band gives 70.7 MHz
  bins = 1.95 % at 3.627 GHz"*. **That is wrong**: 70.7 MHz is
  7.0 GHz / 99, i.e. it assumes the sweep starts at DC. The sweep starts at
  `freq_max / 10`, so the bin is 6.3 GHz / 99 = **63.6364 MHz = 1.754 %**.
  The paragraph's *conclusion* is unaffected and still holds (one bin is still
  wider than the 1.40 % error being reported), but the width was overstated by
  11 %. Corrected in this lane's edit to that docstring.

* **cv06b depth gate.** `pass_notch_depth = s21_notch_db < -10`. For an ideal
  shunt open stub the transmission zero is a true zero, so the *sampled*
  minimum is set by how close a bin lands to it, not by the notch's quality.
  With `S21 = 2 / (2 + j r tan((pi/2) f/f0))` and `r = Z0_line / Z_stub = 1`
  (cv06b's stub and main line realise the SAME 635.0 um width — the case's own
  `fidelity_report()` quote), the worst case is a bin half a bin off `f0`:
  `theta = (pi/2)(1 + h/(2 f0))` with `h = 63.6364` MHz, `f0 = 3.6424` GHz
  gives `|S21| = 2/sqrt(4 + tan^2 theta) = 0.02745` = **-31.23 dB**, i.e.
  **21.2 dB inside the -10 dB gate**. The gate cannot fail while a notch
  exists at all.

  **On the audit's own figure for this quantity.** *(CORRECTION. The
  `fc6ca0a` version of this paragraph said the 0.5 dB gap "is the choice of
  `f0` — bin centre 3.627 vs refined 3.642 GHz". That is wrong, and it was
  wrong in the same way the #812 process note warns about: asserted without
  evaluating the alternative. Evaluated, `f0` moves the number by 0.12 dB
  across the whole plausible range. Replaced by the following.)*
  #812 published -30.7 dB.
  The derivation above is independent and lands at **-31.20 dB** at
  `f0 = 3.627` GHz (bin centre), -31.23 dB at 3.6424 GHz (refined) and
  -31.32 dB at 3.679 GHz (the analytic anchor) — the choice of `f0` moves it
  by 0.12 dB across that whole range, so it does NOT explain the 0.5 dB gap.
  Reaching -30.7 dB from the ideal r = 1 model would need a bin of 67.4 MHz
  rather than 63.6364 MHz. The likeliest explanation is that the audit
  evaluated the worst case on the measured notch shape rather than the ideal
  closed form, but that is a guess and is labelled as one: **the origin of
  the 0.5 dB difference is not established here.** Both numbers give the same
  verdict — the gate sits >20 dB from its own worst case — and this lane uses
  its own -31.23 dB, not the audit's, wherever the figure is quoted.

* **cv07.** `validation/crossval/07_sheen_lpf.py` sweeps
  `np.linspace(F_LO, F_MAX, n_freqs)` = `linspace(0.5, 20.0, 120)` GHz =
  **163.866 MHz/bin** = **2.081 %** at the 7.874 GHz zero, and gate C1 judges
  `abs(got - want)/want * 100 <= COMMITTED["null_tol_pct"] = 1.0`. A
  bin-quantised estimate against a 1.0 % window can only report 0.000 % or
  >= 2.081 %: **the declared threshold is unexercisable**.

* **cv07 blindness, reproduced.** Erasing the LOWER doublet member (fill the
  6.399 - 7.874 GHz interval with a straight line in dB through those two
  untouched anchor bins) leaves *every currently gated quantity bit-identical*:
  argmin 7.8739 GHz, depth -39.20 dB, passband mean 0.9378, max column power
  0.9995, correction footprint 0/120. All 17 gates pass on a leg missing one
  of the two transmission zeros the case exists to characterise.

---

## 2. What is adopted

Sub-bin **log-parabolic vertex refinement** — the method already committed at
`scripts/diagnostics/build_sheen_lpf_palace_referee.py::_min_in_window` and
`scripts/diagnostics/build_msl_notch_palace_referee.py::_notch` — factored into
`validation/crossval/comparators/spectral_features.py` so the crossval cases
and the referee producers cannot drift apart. The factored
`refined_extremum()` reproduces the committed referee fixture's
`referee.fdtd_doublet_ghz` for both solvers **bit-for-bit**, which is the
integrity check that the factoring changed nothing.

Alongside it, two estimators the cases previously had no equivalent of:

* `band_at_level()` — the width of a stopband at a stated dB level, with both
  edges linearly interpolated between bracketing bins (sub-bin, not a bin
  count). This is what replaces cv06b's unfailable depth gate.
* `level_crossing()` — a sub-bin -3 dB corner frequency. This is what makes
  cv07's corner a gated quantity.

And one instrument gate:

* `half_grid_witness()` — split the sweep into its two interleaved
  half-density sub-grids and refine the same feature on each. **The two
  sub-grids are disjoint in frequency, so a bin-quantised estimator's two
  answers are ALWAYS at least one full-grid bin apart** (measured: exactly
  1.0000 bin on both committed cv07 legs). A `spread < 1 full-grid bin` test
  is therefore *unpassable by the estimator the audit found* and passable only
  by a genuinely sub-bin one. It is an in-run proof of the resolution claim,
  not an assertion of it.

---

## 3. Pre-declared windows

### cv06b

**T1 — notch-frequency accuracy vs analytic: 15 % -> 4.0 % (a TIGHTENING).**

The oracle `F_NOTCH_AN = c / (4 L_stub sqrt(eps_eff_HJ))` is a fringing-free,
junction-free quarter-wave open stub. Three corrections it omits, each
evaluated on the REALIZED board (`u = W_realized/H_SUB = 635.0/254.0 = 2.500`,
`eps_eff = 2.882252`, `h = 254 um`, `L_stub = 12.000 mm`, `dx = 63.5 um`):

| term | model | value | % of L_stub |
|---|---|---|---|
| open-end fringing | Hammerstad-Bekkadal `dL/h = 0.412 (e+0.3)/(e-0.258) (u+0.264)/(u+0.8)` | 106.288 um | 0.8857 % |
| shunt-T reference plane on the stub arm | bounded by `0.5 * W_realized` | 317.50 um | 2.6458 % |
| stub-length rasterisation | half a cell, `0.5 * dx` | 31.75 um | 0.2646 % |
| **worst-case sum** | | | **3.7961 %** |

Window **4.0 %** (round up, two significant figures). Derived from geometry
and closed-form discontinuity models only; no measured cv06b output enters it.

Passability, from prior provenance (NOT from the run this will judge): the
committed 2026-08-27 GPU log reports 1.40 % on the bin-quantised estimator,
and the sub-bin correction measured on the sibling committed fixture
`tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json` is +15.12 MHz
(+0.42 %); so the refined figure is expected in the 1.0 - 1.4 % range,
>= 2.8x inside the window.

**T2 — -10 dB stopband fractional bandwidth: `[0.80, 1.20] x 0.210274`.**

For an ideal shunt open stub across a matched line,
`S21 = 2 / (2 + j r tan(theta))`, `theta = (pi/2)(f/f0)`,
`r = Z0_line / Z_stub`. `|S21| = -10 dB` <=> `r |tan theta| = 6` <=>
`f/f0 = 1 -+ (2/pi) atan(r/6)`, so the fractional -10 dB bandwidth is
`(4/pi) atan(r/6)`. cv06b's stub and main line realise the **same** 635.0 um
width (both verified in the case's own `fidelity_report()` quote), so
**r = 1 exactly by construction** and `BW_frac = (4/pi) atan(1/6) =
0.210274`.

Window derivation: the gate must fail on a build whose stub coupling is
degraded by 25 % (`r <= 0.75`), for which
`BW_frac(r)/BW_frac(1) = atan(0.125)/atan(1/6) = 0.7530`, i.e. -24.7 %.
A **+-20 %** window fires at `r <= 0.79` and at `r >= 1.28` (an
over-broadened, lossy notch). Unlike the depth it replaces, this quantity
CANNOT be satisfied by a bin landing near a zero: a shallow stub narrows the
band whatever the sampling does.

Passability, from prior provenance: the committed dx=50 um sibling fixture — a
real rfx run of the same open-stub notch on the same 63.6364 MHz grid — reads
`BW_frac = 0.20001`, ratio **0.9512**, i.e. 4.9 % low and 4x inside the window.

**T3 — half-grid resolution witness: `spread < 1.000` full-grid bin.**
Structural (see section 2); a quantised estimator scores exactly >= 1.0000.
Prior provenance for a correct build: dx=50 um sibling fixture 0.604 bin;
cv07 rfx leg 0.521 bin; cv07 openEMS leg 0.119 bin.

**The `s21_notch_db < -10 dB` gate is RETAINED, not removed and not widened.**
It stays as a reported witness with its -31.23 dB blindness stated beside it.

### cv07

**T4 — refined transmission-zero lock: +-0.50 %,** on four quantities (rfx
lower/upper, openEMS lower/upper), against values **already committed** in
`tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json`
(`referee.fdtd_doublet_ghz`): rfx 6.943990 / 7.925928 GHz, openEMS
7.030670 / 7.994749 GHz. These are the referee producer's own committed
output, not new pins by this lane, and this lane's factored estimator
reproduces all four bit-for-bit.

Window derivation: the smallest geometry error the dx=200 um mesh can express
on the dimension that sets the stopband zeros — one cell on the 20.320 mm
wide-patch transverse extent — is **0.984 %**. On the committed rfx leg's own
grid the log-parabolic estimator reports a +1.000 % frequency-axis shift as
+0.953 % and a -1.000 % shift as -1.060 %. A **0.50 %** window therefore fires
on a one-cell error with >= 1.9x margin. Its lower bound is the estimator's
reproducibility on a fixed committed grid, which is exact (a deterministic
function of a committed file).

**The pre-existing 1.0 % argmin lock (gate C1) is KEPT unchanged.** Nothing is
widened; one lock is tightened by being computed on a better estimator, and
the old one stays as the bin-level lock it always was.

**T5 — transmission-zero count in 5 - 15 GHz == 2,** for both solvers.
A zero = a local minimum of `|S21|` with depth `<= -20 dB` and prominence
`>= 0.5 dB` against the shallower of its flanking maxima. **-20 dB is not a
new constant**: it is gate A4's existing threshold in this same file
("openEMS shows a deep stopband zero", `oems_null_db <= -20.0`). 0.5 dB
prominence is a **5.9 %** change in `|S21|` (12 % in power) — *(CORRECTION:
`fc6ca0a` wrote "a 12 % change in |S21|"; 0.5 dB is 12 % in POWER and 5.9 % in
amplitude. The window is unchanged; only the sentence describing it was
wrong.)* — an order above the extraction noise
the legs' own committed evidence bounds (rfx raw passivity excess <= 0.0145;
openEMS column power inside the documented 1.10 envelope) — and it exists only
to stop a dense sweep's shoulder ripple from counting.

**T6 — -3 dB corner-frequency lock: +-0.25 %.**
`fc` = the lowest frequency above 2.0 GHz at which `|S21|` falls to
`(mean |S21| over 0.5 - 3.0 GHz) / sqrt(2)`, linearly interpolated between the
bracketing bins.
Window derivation: the corner of this LPF is set by the wide patch's shunt
capacitance, `C` proportional to patch area and `fc` to `1/sqrt(LC)`, so a
one-cell (0.984 %) transverse-dimension error moves `fc` by **0.49 %**.
A **0.25 %** window fires on a one-cell error with 2x margin.

**T7 — half-grid resolution witness on the gated upper zero: `spread < 1.000`
full-grid bin.** Same structural derivation as T3.

---

## 4. Falsifiers this lane must demonstrate (criterion B)

| case | defect | must fail |
|---|---|---|
| cv07 | lower transmission zero erased (deterministic dB-linear fill between two untouched anchor bins) | T5 (count 1 != 2) and T4 (lower zero) |
| cv07 | -20 % corner-frequency error, isolated: monotone piecewise-linear frequency warp that moves the -3 dB corner 5.5036 -> 4.4029 GHz while leaving the passband (<= 3.0 GHz) and both transmission zeros (>= 6.399 GHz) untouched | T6 only — and T4/T5/C1/C3/D must still PASS, which is what proves the corner was an ungated quantity and now is not |
| cv06b | one-cell stub-length error, `STUB_LEN 12.0000 -> 11.9365 mm` (one dx) — a sub-bin defect: true notch moves +0.53 %, i.e. 0.30 bin | the reported deviation must MOVE by ~0.5 % instead of staying at the bin-quantised value; T1 is an accuracy window against an oracle uncertain at ~3.8 %, so a one-cell error is made *visible*, not fatal, and this is stated rather than gated |
| cv06b | shallow-notch build: stub width 635.0 -> 317.5 um (5 cells, on-lattice), `Z_stub` ~ 68.9 ohm, `r` ~ 0.67 | T2 (`BW_frac` ratio ~ 0.674, -32.6 %, outside +-20 % with 1.6x margin) |

cv06b's two build-level falsifiers need the 5,729,080-cell dx=63.5 um solve
this case ships (329.2 s on one RTX4090; the same mesh was abandoned
UNFINISHED at 2h52m on a 32-core CPU pod — `validation/crossval/manifest.json`
`cpu_runner.excluded_reason`). They are emitted as a VESSL job, not run here.


---

## 5. MEASURED (appended 2026-09-01, after the windows above were frozen in `fc6ca0a`)

Environment: `~/Documents/rfx/.venv` on an Apple M1 Max (10 cores, 64 GB),
CPU JAX. Every number below is reproducible from committed files by the two
falsifier scripts named beside it.

### 5.1 cv07 — criterion (A): the case still passes, with margin

`python validation/crossval/07_sheen_lpf.py` -> **exit 0, 28/28 gates PASS**
(was 17 gates). The refinement is not cosmetic: it moves the estimates off the
bin centre by

| leg | zero | bin argmin | sub-bin refined | shift |
|---|---|---|---|---|
| rfx | lower | 6.8908 GHz | 6.9440 GHz | +0.325 bin (+0.766 %) |
| rfx | upper | 7.8739 GHz | 7.9259 GHz | +0.317 bin (+0.656 %) |
| openEMS | lower | 7.0325 GHz | 7.0307 GHz | -0.075 bin (-0.026 %) |
| openEMS | upper | 7.9831 GHz | 7.9947 GHz | +0.477 bin (+0.146 %) |

i.e. the bare argmin was mislocating the rfx zeros by about a third of a bin —
a **0.656 - 0.766 %** error that the old 1.0 % window could not express,
because on a 2.081 %/bin grid the *reported* deviation could only be 0.000 %
or >= 2.081 %. (openEMS's 801-bin sweep is 6.7x finer, so its argmin error is
correspondingly smaller: -0.026 % and +0.146 %.)
All four refined values reproduce the committed referee fixture to the digit
(gate C4b, and `tests/test_spectral_feature_estimators.py`).

The in-run witness C7 reads **0.5213 bin** (rfx) and **0.1192 bin** (openEMS)
against its 1.000 threshold, while the bare argmin on the same two sub-grids
reads **exactly 1.0000 bin** on both legs — the structural claim in section 2,
measured.

### 5.2 cv07 — criterion (B): the new gates fail on the audit's defects

`python scripts/diagnostics/cv07_estimator_falsifiers.py` -> exit 0. Verbatim:

```
[baseline]   exit 0   28 gates   ALL PASS

[erased_zero]  exit 1   28 gates
   [FAIL] C4 rfx lower zero, sub-bin refined: 7.5462 vs committed 6.9440 GHz
          (8.673% <= 0.5%); bin argmin was 7.3824 GHz, sub-bin shift +1.000 bin
   [FAIL] C5 rfx transmission-zero count: 1 == 2 in 5-15 GHz at <= -20 dB:
          7.9038 GHz (-39.2 dB, prom 38.5 dB)
   -> OK: failed exactly the 2 gate(s) it was built to fail; the other 26 still pass

[corner_m20]  exit 1   28 gates
   [FAIL] C6 rfx -3 dB corner frequency: 4.4231 vs committed 5.5036 GHz
          (19.632% <= 0.25%), referenced to passband mean 0.9378
   -> OK: failed exactly the 1 gate(s) it was built to fail; the other 27 still pass
```

Read both rows carefully:

* **erased_zero** fails for the RIGHT reason — C5 says the count is 1, C4 says
  the lower zero is not where it was. And the *other 26 gates pass*, which
  includes all 17 that existed before this lane: the audit's "17/17 PASS on a
  leg missing a transmission zero" is reproduced here, in the same run that
  shows it now failing.
* **corner_m20** fails C6 and **nothing else** — C1, C4, C5, C3 and the whole
  D evidence chain still pass. That is the proof that the -3 dB corner was an
  ungated quantity: a 19.6 % error in the filter's defining number moved
  nothing the case previously read.

**Honest limit on corner_m20.** This is NOT the audit's own leg. #812 reported
a -20 % corner defect that "passes all 17 script gates and 7 referee gates"
with "two spurious zeros appearing". A naive global `f -> f/0.8` compression
of the committed leg does **not** reproduce that: it drags the argmin to
6.3992 GHz and fails today's C1 at 18.7 % (measured). The audit's construction
is not recoverable from what it published, so this lane built its own — a
monotone piecewise-linear frequency warp that isolates the corner — and states
that substitution rather than implying the audit's leg was re-run. The warp
reproduces the audit's stated PROPERTY (a -20 %-class corner error invisible
to every pre-existing gate); the realised corner lands at 4.4231 GHz rather
than the 4.4029 GHz the warp targets, a 0.46 % resampling residual of the
construction itself, recorded rather than tuned away.

### 5.3 cv06b — what was demonstrated on CPU, and what was not

cv06b's own mesh is 5,729,080 cells and GPU-lane (329.2 s on one RTX4090; the
same mesh was abandoned UNFINISHED at 2h52m on a 32-core CPU pod — the
manifest's `cpu_runner.excluded_reason`). Grid shape re-confirmed on this
checkout: `sim._build_grid().shape = (553, 280, 37)`.

So the judgement was factored out of the solve. `evaluate(freqs, s21_mag,
z0_real, f_notch_analytic)` in the case computes every gated quantity as a
pure function of the sweep, and `scripts/diagnostics/cv06b_estimator_
falsifiers.py` drives it with **real committed rfx data**: the sibling fixture
`tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json`, a real rfx run of
the same open-stub notch through the same `compute_msl_s_matrix` on the same
63.6364 MHz grid (its board is the dx=50 um / 300 um-substrate sibling, so its
analytic anchor is that board's, 3.7110 GHz).

`python scripts/diagnostics/cv06b_estimator_falsifiers.py` -> exit 0.

**A — the estimator gates pass on real committed rfx notch data.**
refined 3.64240 GHz (+0.238 bin off the argmin 3.62727), err 1.85 %
(< 4.0), BW ratio **0.9512** (window 0.80-1.20), witness **0.6037** bin
(< 1.000), depth -32.9 dB. G4 (Re(Z0) in 40-65 ohm) is *not* judged by this
replay: the fixture records 31.38 ohm for that sibling BOARD, which is a board
property and none of this harness's business.

**B — a sub-bin frequency error is now visible.** The same sweep, frequency
axis warped:

| true shift | OLD bin argmin | NEW sub-bin refined |
|---:|---:|---:|
| +0.100 % (0.057 bin) | **+0.0000 %** | +0.1081 % |
| +0.200 % (0.114 bin) | **+0.0000 %** | +0.2437 % |
| +0.265 % (0.152 bin, half a cell of stub) | **+0.0000 %** | +0.3511 % |
| +0.529 % (0.303 bin, ONE cell of stub) | +1.7544 % | +0.8325 % |
| +0.750 % | +1.7544 % | +1.0833 % |
| +1.000 % | +1.7544 % | +1.2877 % |
| +1.750 % | +1.7544 % | +1.7480 % |
| -0.529 % | **+0.0000 %** | -0.3193 % |
| -1.000 % | **+0.0000 %** | -0.6646 % |

That is the audit's staircase, measured: the old estimator reports **exactly
0.0000 %** for every defect smaller than its quantum and then jumps a whole
bin, overstating a one-cell stub error 3.3x (+1.75 % for a +0.53 % truth).
The refined estimator responds monotonically to all of them. It is not
unbiased — at +0.529 % it reads +0.83 %, a 1.6x gain error where the notch
crosses a bin boundary and the parabola stencil changes — and that is recorded
rather than smoothed over. **Resolution, not accuracy, is what this fixes.**

**C — WITHDRAWN in round 2. See section 6.1.** The construction below built
the defect out of the quantity being judged; the table is left in place as the
record of what was published, and is superseded by section 6.1's
geometry-built replacement. Do not cite it.

~~A shallow-notch build fails the width gate while the depth gate does not.~~
The measured sweep rescaled by `M_r/M_1`, the ratio of ideal
shunt-open-stub responses (so the passband is untouched and only the notch's
depth and width move):

| r = Z0_line/Z_stub | depth | BW ratio | closed form | G2 | -10 dB depth gate |
|---:|---:|---:|---:|---|---|
| 0.90 | -32.0 dB | 0.8583 | 0.9016 | PASS | PASS |
| 0.80 | -31.0 dB | 0.7609 | 0.8026 | **FAIL** | PASS |
| 0.75 | -30.4 dB | 0.7158 | 0.7530 | **FAIL** | PASS |
| 0.67 | -29.5 dB | 0.6417 | 0.6734 | **FAIL** | PASS |
| 0.50 | -26.9 dB | 0.4826 | 0.5034 | **FAIL** | PASS |

The retained -10 dB depth gate passes by more than 16 dB on **every** one of
these, including the 50 %-degraded stub — the blindness #812 measured, shown
next to the gate that now catches it. Measured firing point: `r <= ~0.83`,
slightly better than the `r <= 0.79` the +-20 % window was declared to give,
because this board's own baseline already sits 4.9 % low. Declared window
unchanged.

**D — a bin-quantised estimator cannot pass the resolution witness.** Replaying
the same baseline with the vertex refinement disabled scores **exactly 1.0000
bin** against the `< 1.000` threshold: FAIL. The witness is a proof, not a
claim.

### 5.4 cv06b — what is NOT demonstrated here (the honest gap)

Criterion (B) at BUILD level — a real solve carrying the defect — is **not**
done in this lane. `scripts/diagnostics/cv06b_build_falsifiers.py` runs the
three builds (shipped geometry; `STUB_LEN` 12.0000 -> 11.9365 mm, one dx;
`W_STUB` 635.0 -> 317.5 um, 5 cells on-lattice) and is wired into
`scripts/vessl_cv06b_estimator_falsifiers.yaml`, which also runs the case
itself for criterion (A). **Until that job runs, cv06b's re-gate stands on
(A)-by-prior-provenance and (B)-by-replay-on-real-committed-data, not on a
fresh solve.** Stated as a gap, not papered over.

Predictions the job will judge (recorded now so they cannot be adjusted after):
* baseline: G1 err in 1.0-1.4 % (< 4.0), G2 BW ratio 0.90-1.00, G3 witness
  < 1.0, all gates PASS;
* `stub_1cell`: refined notch moves by >= 0.265 % (half the +0.529 % truth) —
  visible, and NOT expected to fail G1, whose oracle is uncertain at 3.8 %;
* `stub_narrow`: G2 BW ratio ~ 0.64 (0.673 closed form x this board's ~0.95
  offset) -> FAIL with ~1.25x margin below 0.80, while the -10 dB depth gate
  still PASSES.

### 5.5 Scope discipline

No physics verdict changed. No gate was widened. cv06b's `< -10 dB` depth gate
and cv07's 1.0 % argmin lock both survive verbatim as the witnesses they
always were. The one gate that moved, cv06b's 15 % -> 4.0 %, moved *inward*
and every term of the new number is a closed-form discontinuity model
evaluated on the realized board.

Regression surface touched: `scripts/diagnostics/report_msl_envelope.py`
mirrors cv06b's windows (it parses stdout rather than importing the case), so
its gate keys moved with them and `tests/test_physics_gate_reporting.py` gained
falsifier coverage for the tightening; and
`tests/_example_fidelity_lib.py::CLASSIFICATION` gained the new pure-numpy
comparator module.


---

## 6. ROUND 2 (appended 2026-09-01). Two blockers closed; no window moved.

Numbers in this section are **not restated from memory**: each is a key in a
committed artifact, regenerated by the script named beside it and re-derived
in-test by `tests/test_cv06b_shallow_stub_model.py`. Prose that needs a
quantity names the key.

Artifacts:
* `A6` = `tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json`
  — `scripts/diagnostics/cv06b_estimator_falsifiers.py`
* `A7` = `tests/fixtures/cv07_estimator_regate/cv07_estimator_falsifiers.json`
  — `scripts/diagnostics/cv07_estimator_falsifiers.py`
* `AB` = `cv06b_build_falsifiers_summary.json` — emitted by the VESSL job of
  section 6.2, **not yet produced**.

### 6.1 cv06b criterion (B): the shallow-notch defect was near-circular. Rebuilt.

**Withdrawn.** Round 1's case C degraded the stub by multiplying the measured
sweep by `M_r/M_1`, the ratio of the ideal shunt-open-stub responses G2's
window is derived from. That forces the falsified curve's -10 dB bandwidth to
be `(4/pi) atan(r/6)` times the baseline **by construction**, so G2 firing on
it demonstrated an algebraic identity, not detection power. The window is
unchanged; only the falsifier is.

**Replacement.** `scripts/diagnostics/cv06b_shallow_stub_model.py` builds
`|S21|` of the same board from geometry: Hammerstad-Jensen `Z0`/`eps_eff` per
line (the repository's own `hammerstad_jensen_z0_eps_eff`), Getsinger
dispersion, dielectric + conductor loss, Hammerstad-Bekkadal open end, and a
2x2 ABCD cascade referenced to the **50 ohm port** rather than to the line.
The only defect input is the **stub width in cells**
(`A6::case_C_shallow_notch_from_geometry.construction.defect_input`). No
cv06b gate constant is read anywhere in it, and that is checked
mechanically on the module's parsed AST with docstrings stripped
(`test_construction_reads_no_cv06b_gate_constant`) — including a ban on
`atan`, so the closed form cannot be rebuilt under another name.

**Why that is not the same formula in disguise, quantified.** At the shipped
stub width the model does **not** return G2's reference bandwidth:
`A6::case_C_shallow_notch_from_geometry.independence.model_departure_from_gate_reference_pct`.
A construction that reproduced the reference would read 0.00 % there;
`test_model_does_not_reproduce_the_gate_reference` fails the suite if it ever
gets within 1 %.

**Why it is nevertheless this board.** At the same shipped width the model
agrees with the **committed measured** dx=50 um sibling sweep on all three
features it is judged by — `independence.model_vs_measured_bw_ratio_pct`,
`independence.model_vs_measured_f_notch_pct`, and
`independence.model_notch_depth_db` against
`independence.measured_notch_depth_db`. Bounds are pinned at 2 % / 5 % / 1 dB
in `test_model_reproduces_the_committed_measured_sweep`.

**Two-sided, as criterion (B) requires.** G2 **passes** at the shipped width
and down to `A6::…narrowest_passing_stub_cells`, and **fails** from
`A6::…first_firing_stub_cells` down, while the retained -10 dB depth witness
passes on **every** row — the blindness #812 measured, shown beside the gate
that now catches it. G1 also passes on every row, so the failure is
attributable to the width gate alone. Every row is re-derived in
`test_g2_is_two_sided_on_the_geometry_ladder` and
`test_artifact_rows_match_a_fresh_derivation`.

**What this is still not:** a solve. It is a circuit model of cv06b's board,
independent of the gate but not of transmission-line theory. The FDTD version
is section 6.2's `stub_narrow` leg.

### 6.2 cv06b criterion (A): a VESSL job, reported — not a claim

cv06b is `role: claims-bearing` and its two new pass-criteria gates enter the
verdict at `validation/crossval/06b_msl_notch_filter_uniform.py`'s `evaluate`.
Round 1 demonstrated (A) on the dx=50 um **sibling** fixture, not on cv06b's
own board. That stand-in is withdrawn as evidence for (A).

`scripts/vessl_cv06b_estimator_falsifiers.yaml` runs, on one RTX4090:

1. the CPU replay, re-emitting `A6` and `diff`-ing it against the committed
   copy (so the harvested run proves the committed artifact is not stale);
2. **criterion (A)** — the shipped `06b_msl_notch_filter_uniform.py` on its
   own board, its exit code captured to `cv06b_baseline_run.exit`;
3. **criterion (B) at build level** — `cv06b_build_falsifiers.py`: `baseline`,
   `stub_1cell` (`STUB_LEN` minus one `DX`), `stub_narrow` (`W_STUB` = 5·`DX`),
   one geometric input changed each time, reduced to `AB`.

Until that job runs and `AB` exists, **cv06b's criterion (A) is UNDEMONSTRATED
on its own board** and this lane says so rather than borrowing the sibling's
result. Section 4's predictions stand as written and are what `AB` will judge.
The reporting and JSON half of that job is exercised with the solve stubbed
out (`tests/test_cv06b_build_falsifier_plumbing.py`) so a formatting crash
cannot cost four GPU solves.

### 6.3 cv07: re-verified, design unchanged

cv07's half was not blocked. Re-run on this checkout: criterion (A) is
`A7::baseline` (`n_gates`, `all_pass`), criterion (B) is `A7::defects`
(`erased_zero`, `corner_m20` — each with `expected_failures`,
`observed_failures`, `unexpected_failures`, `n_still_passing`). Every window
derivation section 3 states for cv07 is now recomputed into
`A7::window_derivations`: `sweep_bin_mhz`, `sweep_bin_pct_at_bin_argmin`,
`one_cell_transverse_pct`, `estimator_response_to_frequency_warp_pct`,
`one_cell_corner_shift_pct`, `prominence_0p5_db_in_amplitude_pct` and
`…_in_power_pct`. All reproduce the values section 3 declared; no window
moved and no gate changed.

**One source-comment correction.** `07_sheen_lpf.py`'s `zero_prominence_db`
comment read "12% in |S21|". 0.5 dB is 12 % in **power** and 5.9 % in
amplitude — the same slip section 3 T5 already corrected in this note, still
live in the code. Fixed in place, window unchanged.

### 6.4 What changed in the reporting surface

* `06b_msl_notch_filter_uniform.py` now **computes** the depth gate's blind
  margin (`worst_sampled_notch_db`, added there) instead of printing a typed
  "21.2 dB", so the margin cannot go stale if the sweep length changes.
* `report_msl_envelope.py` restates three cv06b windows because it parses
  stdout instead of importing the case;
  `test_report_mirrored_cv06b_windows_match_the_case_that_owns_them` pins them
  to the case that owns them.
* The one shallow-notch row `tests/test_physics_gate_reporting.py` types into
  a synthetic stdout is asserted to BE a row of `A6`
  (`test_cv06b_shallow_row_used_above_is_the_committed_artifact_row`).

