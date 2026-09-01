# cv04 (multilayer Fresnel) — fringe-structure re-gate: threshold pre-declaration

Issue: #812 (crossval gate audit), cv04 row — "the E4 label is carried by an
import, not by a verdict", plus audit pattern **P2** (band-mean collapse over an
interference pattern).

**This note is committed BEFORE any measurement of the new metrics.** Every
number below is derived from geometry, the discrete Yee dispersion relation, the
spectral resolution of the committed configuration, or prior committed
provenance in the script itself. None of them is fitted to a value produced by
the new gate.

Append-only. Corrections go in a new section at the end, stating the old value
and why it was wrong.

---

## 1. What the audit found, re-confirmed on this branch

`validation/crossval/04_multilayer_fresnel.py` declares two references, both
marked `required_for_script_pass: true` in `validation/crossval/manifest.json`:
an analytic normal-incidence transfer-matrix Fresnel R/T, and "Meep slab R/T".
The case is `role: claims-bearing` with `evidence_levels: [E2, E4]`.

Two defects, both re-read in the source on this branch:

1. **No Meep number enters any verdict.** `rfx_self_ok` is assembled at
   PART 2 (before PART 3 runs Meep at all) from `t_ok`, `r_ok`, `c_ok`,
   `cons_max_ok`, `tail_ok` — all rfx-vs-analytic or rfx-only. PART 3 computes
   `T_err_meep`, `R_err_meep`, `cons_meep` and PART 5 computes
   `T_rfx_meep_diff`, `R_rfx_meep_diff`, and every one of them is **printed
   only**. The exit code is `0` iff `rfx_self_ok and HAVE_MEEP`: the *import*
   of Meep is required, the *numbers* are not. `docs/public/guide/benchmarks.mdx`
   already says this out loud ("Meep values are informational ... numeric Meep
   agreement is not gated"), so the manifest's `required_for_script_pass: true`
   on the external-solver reference is true only in the degenerate
   import-succeeded sense.
2. **The analytic reference decides only through a band mean.** `t_ok`, `r_ok`
   are `mean(|T_rfx - T_an|) < 0.05` and `mean(|R_rfx - R_an|) < 0.05` over the
   masked band. The mean's null space is every zero-mean shape error, and the
   gated quantity here *is* an interference pattern — the audit measured that
   R_max may be 22.3% low (0.3600 -> 0.2797), eps 12.33% wrong (4.0 -> 4.4933),
   or d 8.0% wrong, and the case still reports PASS.

**Negative claim checked, and refuted.** The obvious excuse — "pymeep is absent
on every runner, so the reference cannot decide" — is **false**.
`.github/workflows/validation.yml:196-220` defines the `crossval-external` job
(scheduled Monday `0 6 * * 1` + `workflow_dispatch`), which installs
`pymeep` from conda-forge and runs every manifest case with
`scheduled_external_order is not None` — cv04 is one of them
(`scheduled_external_order: 4`). So the external reference *does* run
periodically, and it is legitimate to make its numbers bind. This lane
therefore takes the "make the declared reference decide" branch, not the
"correct the manifest" branch.

Meep is not installed on this host (no conda/mamba; `import meep` ->
`ModuleNotFoundError`), so the Meep leg's falsifier is demonstrated at
gate-logic level against synthetic arrays, and the live-solver demonstration is
owed to the scheduled lane. That limitation is stated in the results note; it
is not claimed as a demonstration.

## 2. Committed configuration (read from the script, not measured)

| quantity | value | source |
|---|---|---|
| `eps_slab` | 4.0 | script:39 |
| `n = sqrt(eps)` | 2.0 | script:40 |
| `d_slab` | 10.0 mm = 10 cells | script:41 |
| `dx` | 1.0 mm | script:43 |
| `dt` | 2.335067793382187e-12 s | `Grid(...).dt`, script:100 |
| Courant `S = c dt / dx` | 0.7000533 | derived |
| `n_steps` | 719 | script:117-119 |
| `nfft` | 8192 | `2**ceil(log2(719)) * 8`, script:262 |
| spectral bin `df = 1/(nfft dt)` | **52.277 MHz** | derived |
| evaluated band | 3-15 GHz masked to 2% incident amplitude | script:279 |

Free spectral range of the etalon: `FSR = c / (2 n d) = 7.495 GHz`.

Analytic normal-incidence extrema of `R(f)` for a lossless slab in air:

- `R` maxima at `delta = (m + 1/2) pi` -> `f = (m + 1/2) * FSR`
- `R` minima at `delta = m pi` -> `f = m * FSR`
- `R_max = ((n^2 - 1)/(n^2 + 1))^2 = 0.36` exactly; `R_min = 0` exactly.

Inside the evaluated band there are exactly **three** interior extrema:

| index | kind | f_analytic | analytic value |
|---|---|---|---|
| 0 | max | 3.7475 GHz | 0.36 |
| 1 | min | 7.4950 GHz | 0.00 |
| 2 | max | 11.2425 GHz | 0.36 |

Fringe contrast `C = R_max - R_min = 0.36`.

## 3. The new metric

Replace nothing; **add** a fringe-resolved comparison alongside the existing
mean gates (existing gates are never widened, per rfx discipline). The metric is
deliberately not a band mean:

1. **Reference-blind extremum detection.** Locate interior local extrema of the
   measured `R(f)` over the evaluated band with a prominence floor, with **no
   reference to the analytic extremum positions** — so the gate cannot be
   entailed by its own search window (the cv02 failure mode).
2. **Count/order gate.** The detected sequence must have the same length and the
   same max/min alternation as the analytic set.
3. **Position gate.** Each detected extremum's frequency, refined to sub-bin
   resolution by a 3-point parabolic vertex fit, must lie within `W(f)` of its
   analytic partner.
4. **Value gate.** Each detected extremum's refined vertex value must lie within
   `V` of the analytic value at that extremum.

## 4. Threshold derivation — positions

Two error sources are derivable from first principles for correct code.

**(a) Spectral quantization.** The vertex is refined by parabolic fit, but bound
the residual conservatively at half a bin: `df/2 = 26.14 MHz`.

**(b) Discrete-Yee numerical dispersion in the slab.** The 1-D Yee dispersion
relation is `sin(omega dt / 2) = S_m sin(k~ dx / 2)` with `S_m = c dt / (n dx)`
the in-medium Courant number (`S_m = 0.3500267` in the slab). A fringe extremum
sits where the *numerical* round-trip phase satisfies `k~(f) d = m pi / 2`-type
conditions, so the measured extremum moves to the `f` solving `k~(f) d =
k_exact(f_an) d`. Solving that exactly (Brent, not the Taylor expansion):

| f_analytic | f_numerical (Yee) | shift |
|---|---|---|
| 3.7475 GHz | 3.7441 GHz | **-3.4 MHz** (-0.090%) |
| 7.4950 GHz | 7.4680 GHz | **-27.0 MHz** (-0.361%) |
| 11.2425 GHz | 11.1512 GHz | **-91.3 MHz** (-0.812%) |

**Window.**

```
W(f) = SAFETY * ( df/2 + |dispersion_shift(f)| )
SAFETY = 2
```

`SAFETY = 2` is not tuning: it covers **one un-derived systematic of the same
magnitude as the largest derived term** — the material-interface staircase /
half-cell E/H stagger at the two slab faces, whose leading behaviour I have not
derived here. It is frozen at 2 and will not be raised after measurement.

Committed-config values (computed from the formula above, before measurement):

| extremum | W |
|---|---|
| 3.7475 GHz | **59.1 MHz** |
| 7.4950 GHz | **106.3 MHz** |
| 11.2425 GHz | **234.9 MHz** |

**Detection power (also computed before measurement, from geometry alone).**
A defect that changes `n d` moves every extremum by `f * (1 - n0 d0 / (n1 d1))`:

| defect | shift @3.7475 | shift @7.495 | shift @11.2425 | vs W |
|---|---|---|---|---|
| eps 4.0 -> 4.4933 (+12.33%, the audit's number) | -211.7 MHz | -423.4 MHz | -635.1 MHz | **3.6x / 4.0x / 2.7x** |
| d +8.0% (the audit's number) | -277.6 MHz | -555.2 MHz | -832.8 MHz | 4.7x / 5.2x / 3.5x |
| d +1 cell (+10%, smallest the grid expresses) | -340.7 MHz | -681.4 MHz | -1022.0 MHz | 5.8x / 6.4x / 4.4x |

So the position gate fires on all three named defects with margin, at every one
of the three fringes.

## 5. Threshold derivation — extremum values

Budget for a *correct* run, additive:

1. **Dispersion-induced contrast change.** Using the same exact Yee relation for
   both media, the effective indices at the top gated fringe are
   `n_eff(slab) = 2.016652`, `n_eff(air) = 1.001187`, giving
   `R_max = 0.36545`, i.e. **+0.0055** (and +0.0024 at 7.495 GHz, +0.0006 at
   3.7475 GHz).
2. **Finite-run truncation of the etalon ringdown.** Prior committed provenance
   in this script (the rung-C4 paragraph at script:297-305, job
   369367246779) records that extending the run to nx=1500 / 1940 steps shifts
   the band-mean `|dT|`, `|dR|` by **< 0.005**. An extremum is a single bin
   rather than a band mean, so allow **3x** that: **0.015**.
3. **Vertex quantization.** `R ~ R_max sin^2(delta)`, `d delta/df = 2 pi n d/c =
   4.192e-10 rad/Hz`; over half a bin `d delta = 0.0110 rad`, so the value error
   is `R_max * (d delta)^2 = 4.3e-5`. **Negligible.**

Sum = 0.0205. Applying the same frozen `SAFETY = 2`:

```
V = 0.04        (= 11.1% of the analytic fringe contrast C = 0.36)
```

**Detection power.** The audit's amplitude defect `R_max 0.3600 -> 0.2797` is a
change of **0.0803 = 2.0 x V** — it fires. (The eps defect additionally raises
the true `R_max` to 0.40439, `0.0444 = 1.11 x V`, but eps is caught primarily
and robustly by the position gate above.)

`V` also serves as the **prominence floor** for extremum detection: a ripple
smaller than the amplitude resolution of the measurement is not a resolvable
feature, so features with prominence `< V` are not counted as extrema. One
constant, one derivation.

## 6. Threshold derivation — the Meep leg

The Meep leg's job is to stop the E4 label from being carried by an import.
Three binding checks when `HAVE_MEEP`:

1. `max |R_meep - R_an|` and `max |T_meep - T_an|` over Meep's valid band
   `<= MEEP_ABS_LIMIT`.
2. `max |R_rfx - R_meep|` and `max |T_rfx - T_meep|` over Meep's valid band
   `<= MEEP_CROSS_LIMIT`.
3. Meep's own fringe extrema, located on **Meep's native flux grid** (200 points
   over ~2.5-17.5 GHz, spacing ~75 MHz) with the same reference-blind detector,
   must match the analytic set in count/order and in position within
   `W_meep(f)`.

Derivation. The pointwise `|dR|` induced by a position error `df` is bounded by
the maximum fringe slope `|dR/df|_max = R_max * 2 pi n d / c = 0.151 / GHz`.
Allowing each solver its own position budget `W(f_top) = 234.9 MHz` gives a
pointwise amplitude allowance of `0.151 * 0.2349 = 0.0355`, on top of the
amplitude budget `V = 0.04`:

```
MEEP_ABS_LIMIT   = 0.08          (0.0355 + 0.04, rounded up)
MEEP_CROSS_LIMIT = 0.16          (two independent solvers -> 2 x MEEP_ABS_LIMIT)
W_meep(f)        = SAFETY * ( df_meep/2 + |dispersion_shift(f)| ),  df_meep from
                   the actual Meep flux-grid spacing at runtime
```

`MEEP_CROSS_LIMIT = 0.16` is loose in absolute terms and is honestly reported as
such: it is the first gate to ever put a Meep *number* in cv04's verdict, and it
is sized so that a correct Meep run cannot red the scheduled lane for a
budget I could not measure on this host. It still fires hard on the audit's
defects — an eps +12.33% error moves the rfx curve by 635 MHz at the top fringe,
which at 0.151/GHz is a pointwise divergence of order the full fringe swing
(>= 0.3), i.e. `>= 1.9 x MEEP_CROSS_LIMIT`.

**Explicitly not claimed:** the Meep leg is *not* demonstrated against a live
Meep run in this lane. Its falsifier is demonstrated against synthetic arrays
through the same pure gate function the script calls. The live demonstration is
owed to `crossval-external`.

## 7. Falsifiers to be run after this commit

(A) the case still passes on today's correct code, with margin; and (B) each
gate fails on the specific defect the audit measured it blind to, for the right
printed reason:

1. **eps +12.33%**: FDTD slab built with `eps = 4.4933`, analytic reference left
   at 4.0. Expect: old gates PASS (reproducing the audit), new position gate
   FAIL at all three fringes.
2. **d + 1 cell** (+10%, the smallest thickness error the grid can express).
   Expect: old gates PASS, new position gate FAIL.
3. **R_max -22.3%**: scattered spectrum scaled by `sqrt(0.2797/0.36) = 0.88151`.
   Expect: old gates PASS, new value gate FAIL at both maxima, position gate
   silent (positions are untouched) — a directional check that the two gates
   are independent.

## 8. Documents that become false when this lands, and must change with it

- `docs/public/guide/benchmarks.mdx:54` — "Meep values are informational ...
  numeric Meep agreement is not gated".
- `validation/README.md:38` — the reference column.
- `validation/crossval/manifest.json` — `evidence_levels` / gate description
  stay as they are only if the Meep numbers really do bind after this change.

---

## 9. CORRECTION (appended after the first measurement) — the detector, not the windows

**Old design (section 3.1 above), and why it was wrong.** The pre-declaration
said the extremum detector would find "interior local extrema of the measured
`R(f)` over the evaluated band with a prominence floor", with the prominence
floor set to `V = 0.04`. Measured on the committed run, that detector found
**only one** of the three extrema — the minimum at 7.4706 GHz — and reported a
count/order mismatch, i.e. it failed criterion (A) on correct code.

The cause is a property of prominence, not of the physics. `scipy.signal.
find_peaks` measures a peak's prominence against the lowest contour that
separates it from a higher peak, **using the array boundary when there is no
higher peak on that side**. The evaluated band is `3.0321–11.8666 GHz`, which
truncates the outer shoulder of both maxima:

- max at 3.7475 GHz — the band starts at 3.0321 GHz where the analytic
  `R = 0.36 sin^2(delta)` is already 0.3278, so its left-side prominence is
  only **0.032**, below the 0.04 floor;
- max at 11.2425 GHz — the band ends at 11.8666 GHz where `R = 0.337`, so its
  right-side prominence is only **0.023**.

Both true maxima were therefore rejected. This is an instrument defect in the
detector; the two gate windows `W` and `V` are untouched by it.

**New design (implemented).** The analytic slab partitions the band into
half-fringe cells of width `FSR/2 = 3747.5 MHz`, one per extremum, centred on
the analytic extremum. `FSR/4 = 1873.75 MHz` is the largest half-width that
cannot contain two extrema of the same kind, so the cells tile the band without
overlap — pure geometry, no tuning. The measured extremum is the arg-extremum
inside its own cell, parabolically refined; if it lands within
`PIN_MARGIN_BINS = 2` bins of a cell edge the gate FAILS with a
`fringe CONTAINMENT ... PINNED` reason rather than reporting a boundary as an
extremum.

**Why this still cannot entail the verdict** (the cv02 failure mode). The search
half-width is 1873.75 MHz; the widest gate window is 234.9 MHz — 8.0x tighter
(16x on cell width). A defect that moves a fringe by more than `W` but less than
the cell half-width is *found* and reported as a POSITION failure; a defect that
moves it out of its cell is reported as a PINNING failure. Both regimes fire, so
"found in the cell" does not imply "passes the gate".
`test_cv04_fringe_gate_is_not_entailed_by_its_own_search_window` and
`test_cv04_fringe_gate_fails_rather_than_reports_a_pinned_extremum` pin both
halves.

`PROMINENCE_FLOOR` is removed. It was a detector parameter, never a gate
window; no threshold moved.

## 10. RESULTS

Command (all runs, CPU, this host, Meep absent):

```
PYTHONPATH=<worktree> .venv/bin/python validation/crossval/04_multilayer_fresnel.py
```

### (A) The case still passes on correct code, with margin

`exit 2` (unchanged: Meep unavailable is SKIP, not PASS). All pre-existing gates
unchanged and passing. The new gate:

| extremum | f_analytic | f_measured | dev | W | used |
|---|---|---|---|---|---|
| max | 3.7475 GHz | 3.7175 GHz | **-30.0 MHz** | 59.0 MHz | 0.51x |
| min | 7.4950 GHz | 7.4706 GHz | **-24.4 MHz** | 106.4 MHz | 0.23x |
| max | 11.2425 GHz | 11.3084 GHz | **+65.9 MHz** | 234.9 MHz | 0.28x |

| extremum | R_analytic | R_measured | dev | V | used |
|---|---|---|---|---|---|
| max @3.7175 | 0.3600 | 0.3598 | **-0.0002** | 0.04 | 0.005x |
| min @7.4706 | 0.0000 | 0.0000 | **+0.0000** | 0.04 | 0.000x |
| max @11.3084 | 0.3600 | 0.3645 | **+0.0045** | 0.04 | 0.11x |

Tightest margin: the low-frequency maximum at 1.97x. That fringe has the
smallest window (59.0 MHz) because its derived dispersion shift is only
3.4 MHz, so the window is essentially one spectral bin.

### (B) The new gate fires on the defects the audit measured it blind to

The audit's probes perturb the **reference**, holding the measurement fixed —
which is the correct scope, because cv04's other gates (`cons_rfx`, the tail
witness) compare rfx against rfx and are structurally incapable of seeing a
disagreement with a reference. Reproduced verbatim:

| probe | pre-#812 gates | new fringe gate |
|---|---|---|
| **eps 4.0 -> 4.4933 (+12.33%)** | `T` mean err **0.0494** < 0.05 PASS; `R` mean err **0.0449** < 0.05 PASS; `R+T` mean dev 0.0091 PASS; max\|R+T-1\| 0.0487 <= 0.06 PASS; tail PASS -> **exit 2, "rfx accuracy: PASS"** | **FAIL** — POSITION +181.7 MHz (**3.09x** W), +399.0 MHz (**3.83x**), +700.9 MHz (**3.09x**), plus one VALUE 1.12x |
| **d +8.0%** | `T` mean err **0.0476** PASS; `R` mean err **0.0464** PASS; max\|R+T-1\| 0.0487 PASS -> **"rfx accuracy: PASS"** | **FAIL** — POSITION +247.6 MHz (**4.30x**), +530.8 MHz (**5.57x**), +898.6 MHz (**4.56x**) |
| **R_max 0.3600 -> 0.2797 (-22.3%)** | `T` mean err 0.0110 PASS, `R` mean err **0.0452** < 0.05 PASS — **both reference-comparing gates blind** | **FAIL** — VALUE -0.0805 (**2.01x** V) and -0.0768 (**1.92x**), POSITION silent as designed |

That `T` mean err lands at 0.0494 against a 0.05 limit confirms the auditor
bisected the perturbation to the gate boundary, which is why "12.33%" carries
four significant figures.

**Honest scope note on the R_max probe.** The pre-existing per-bin conservation
ceiling (`max|R+T-1| <= 0.06`, issue #341) *does* fire on the amplitude probe
(0.1291), because scaling `R` alone breaks the internal energy identity. So that
defect class is not entirely undetected on today's code — but no gate that
*compares against the declared reference* saw it, and the ceiling would be blind
to a normalization error that scaled `R` and `T` consistently.

**Physical-injection direction, for completeness.** Building the FDTD slab with
`eps = 4.4933` (rather than perturbing the reference) also trips the pre-existing
mean gates (`T` 0.0547, `R` 0.0577) and the conservation ceiling (0.1001),
because a wrong slab degrades energy balance as well. The audit's "and pass" is
specific to the reference-perturbation direction; both directions trip the new
fringe gate (physical injection: 4.60x / 4.21x / 2.37x).

### Falsifiers pinned in the fast lane

`tests/test_crossval_gate_logic.py` imports the same pure functions the crossval
script calls (it does not replicate them, so it cannot drift) and pins: the three
frozen windows, acceptance of an exact analytic slab, rejection of each of the
three audit defects, rejection of a one-cell thickness error, the pinning
failure on a structureless curve, non-entailment by the search window, and both
legs of the external-solver pointwise gate including that it is a maximum and
not a mean. 20 tests, all passing.

## 11. What this lane does NOT demonstrate

The Meep leg is **not** exercised against a live Meep run. Meep is not
installable on this host (no conda/mamba; `pymeep` has no pip wheel). Its
falsifier is demonstrated at gate-logic level against synthetic arrays through
the same function the script calls
(`test_cv04_external_pointwise_gate_rejects_a_disagreeing_solver`), and the
live demonstration is owed to the scheduled `crossval-external` job. The
windows there (`MEEP_ABS_LIMIT = 0.08`, `MEEP_CROSS_LIMIT = 0.16`) are derived
in section 6 and are deliberately loose: they are sized so a correct Meep run
cannot red the Monday lane for a budget that could not be measured here, while
still firing at >= 1.9x on the audit's defects.

One clarification to section 6: it wrote `W_meep(f) = SAFETY * (df_meep/2 +
|dispersion_shift(f)|)` without saying whose grid the dispersion shift belongs
to. The implementation uses **Meep's own** grid — same `dx = 1 mm`, Meep's
default Courant 0.5, so `dt_meep = 0.5 dx / c` — which is the correct reading.
No number in section 6 changes.

## 12. Physics observation, filed NOT fixed

The measured fringe positions deviate from the *discrete-Yee prediction* by more
than the prediction itself at the two maxima:

| extremum | Yee prediction | measured | residual vs prediction |
|---|---|---|---|
| max @3.7475 GHz | -3.4 MHz | -30.0 MHz | **-26.6 MHz** |
| min @7.4950 GHz | -27.0 MHz | -24.4 MHz | +2.6 MHz |
| max @11.2425 GHz | -91.3 MHz | +65.9 MHz | **+157.2 MHz** |

The *minimum* tracks the dispersion prediction to 2.6 MHz (10% of it). The two
*maxima* do not, and the top one is displaced in the opposite direction. A
plausible mechanism is that a maximum of `R` is where the etalon ringdown
contributes most, so truncating the run at 719 steps (the tail witness still
reads 3.6% of incident peak) biases the peak location, while a minimum — where
`R -> 0` and the ringdown is destructive — is unaffected. That is a hypothesis,
not a measurement: it predicts the residual shrinks with run length, and the
committed rung-C4 provenance (nx=1500 / 1940 steps) is the run that would test
it. This is physics/measurement, not instrument, and every value stays inside
the pre-declared window, so it is recorded here and left for a separate lane.
