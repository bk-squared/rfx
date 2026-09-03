# WR-90 chain battery — pre-declaration for the SECOND run (v1.8 WP2 re-measurement)

Status: pre-declaration. Committed **before** any S-parameter of the corrected port is
measured, so that in git history every re-declared position, every prediction and every
outcome branch provably predates the first number of the second run. This note carries no
measured S value from the corrected port; the measurement lands in a later PR as a NEW
artifact (§7), and the first run's artifact
`tests/fixtures/waveguide_chain_battery/fixture.json` is neither edited nor deleted.

Parent document: **`docs/design_notes/waveguide_chain_battery_predeclaration.md`** (the first
pre-declaration, PR #861, read at `78885c10`). Most of this contract is inherited from it
verbatim and is listed rather than copied in §3. Read the parent first; this note is only
the delta plus the predictions.

Governing documents: `docs/design_notes/v18_waveguide_s_chain_plan.md` (WP2, decisions 2, 4
and 6) and `docs/design_notes/chain_closure_contract.md` (criteria 1–3). Builder (constructs,
never runs): `tests/_waveguide_chain_battery_fixture.py`. Gate arithmetic:
`tests/_waveguide_chain_battery_gates.py`. Replay: `tests/oracle/test_waveguide_chain_battery.py`.
Every number below was recomputed on main `0141f39e` (2026-09-03) in a worktree off that commit.

**Under R2 a pre-declaration is a one-run artifact.** The first note governed one run and is
spent. It is not amended, and its numbers are not re-used to authorize a second run; this
note is the contract for run number two and nothing else.

What this note does not do: run anything, loosen any quoted tolerance, edit the frozen
fixture, change the builder (§4.6 says why the builder edit belongs to the measurement PR),
introduce a source construction, import `rfx/probes/refplane.py` into the waveguide path, or
use `normalize=True`.

---

## 1. Memory (R1)

Line numbers are those of `docs/agent-memory/rfx-known-issues.md` in the primary checkout on
2026-09-03.

- **`:99`** — the waveguide chain-status row, rewritten against the measured battery:
  "the de-embedding β is wrong because `_discrete_te_mode_profiles` … solves the Neumann
  eigenproblem on the FULL cell-centred operator over the aperture's `nu` entries, and the
  aperture defaults to the whole transverse field extent — N+1 for an N-cell guide", and
  "The wrong-sign witness reads 0.734° and also fails, for an arithmetic reason: at 9.8 GHz
  the predicted rotation crosses ±180° where both sign hypotheses coincide mod 360°."
  *This note acts on both sentences*: PR #889 corrected the aperture (§2.2), and §4 re-declares
  the shift so the wrong-sign witness stops being degenerate. Neither is a tolerance change.
- **`:64`** — "#729 node-vs-cell miscounts on three committed surfaces." *Consistent*: the
  aperture defect PR #889 fixed is a fourth site of that class, and its own R3 line says so.
  The class entry is why a fifth site is worth looking for, not why this run is safe.
- **`:3110-3122`** — "WAVEGUIDE end-to-end AD — RESOLVED 2026-05-25 (G-AD-WIRE-WG2, commit
  f6dc361)". *Consistent*: this run re-measures objectives on a working tape; it
  re-implements no assembly. The expected AD-vs-FD order in §5 is taken from there.
- **`:3413-3420`** — "The two-run diagonal formula `S11 = (b_dev − b_ref) / a_ref` does NOT
  cancel Yee dispersion … ±10–20 % |S11| error with `normalize=True`." *Consistent*: lanes
  stay `normalize=False` and `normalize="flux"`.
- **`:4129`** — "The offset is a **time-convention conjugation** … conjugate the Meep data."
  *Consistent*: the only phase referee here is rfx's own analytic Airy oracle; no external
  phase data enters.
- **`:4336`** — "Do-not-repeat: never anchor a drive-side wave at the port cell." *Consistent*:
  probe planes stay 25.40 mm inward of each port plane at every rung, unchanged from the
  parent note.
- **`:3735-3752`** — the `aperture_dA` DROP-weight workaround. *Consistent, and checked*:
  PR #889 records `aperture_dA.sum()` unchanged at 2.3225759287e-04 m², because the DROP
  weight had already zeroed the extra row and column — only the eigen-operator's size moved.
  This run does not reopen the weight.
- **`:3546`** — "Production `compute_waveguide_s_matrix` on PEC-short geometry returns spread
  0.0004 at R=1 — Meep-class." *Consistent*: the ladder magnitude floor 0.005 is unchanged.
- **auto-memory `project_issue527_f32_comparator`** — "a comparator-validity assert placed
  BEFORE the accuracy gate". *Consistent*: every FD leg still asserts its ULP span first.

Nothing in the ledger contradicts a second run on a corrected instrument. Grep terms:
`chain battery`, `#868`, `#869`, `wrong-sign`, `reference_plane`, `settling`, `aperture`.

---

## 2. What changed since the first run, and why this is a new measurement rather than a retry

### 2.1 The R2 question, answered first

R2 forbids a second attempt that shares one mechanism hypothesis, one observable and one
intervention family with the first. This run shares none of those with a retry, because
**the instrument changed between the two runs.** Two defects in the measuring apparatus were
found, root-caused and fixed on main; the first artifact is a faithful record of an apparatus
that no longer exists. Re-running is not "the same attempt with better luck" — it is the only
way to learn what the corrected apparatus reads.

The line that would make this illegitimate is easy to state, so it is stated: **if this run's
numbers come back unwelcome, there is no third run.** A third run needs a third named
mechanism defect in the instrument, in writing, found before the run and not inferred from
the numbers it produced.

### 2.2 The two corrections

**PR #881, `d6a3df5d` (closed issue #869) — the settling witness.**
`settling_db_from_port_records` (`rfx/sources/waveguide_port.py`) took its worst
`10·log10((end+tiny)/(peak+tiny))` over every port record, including records that had fallen
off the bottom of float32. Behind a PEC short that broke the witness in both directions:
at the fine rung the four far-port records are exactly zero (`peak = 0.0`, `n_nonzero = 0` of
2849 steps), the ratio evaluates to 1 and the witness reported **0.00 dB**; at the mid rung
the same records are subnormals and the witness **PASSED at −40.85 / −40.91 dB**, 0.9 dB
inside the bar, on subnormal noise, while the signal-carrying records read −94.55 dB. The
witness now skips a record whose tail is below the float32 normal minimum and names the ones
it skipped; the floor is derived from the bar itself
(`finfo(dtype).tiny · 10^(40/20)` = 1.1754944e-36 for float32), not chosen. **The −40 dB bar
did not move.** All 18 settling verdicts in the committed fixture now pass, re-derived from
the same stored per-record peaks with no re-run.

**PR #889, `0141f39e` (issue #868, still open) — the port's transverse eigenproblem.**
`_range_to_slice` (`rfx/api/_compile.py`) and `_range_to_slice_nu` (`rfx/runners/nonuniform.py`)
return NODE-index spans; they were handed straight to `WaveguidePort.u_slice` / `.v_slice`,
which are FIELD-ARRAY slices with one transverse unknown per entry. So the Neumann
eigenproblem described a guide one cell wider than the walls make, and
`WaveguidePortConfig.f_cutoff` — consumed by `_compute_beta` for the reference-plane rotation
and by `_compute_mode_impedance` for `Z_TE` — belonged to that wider guide.

That is worth stating in the form that matters for a convergence study: **the pre-fix operator
was a second-order-accurate discretization of the wrong guide.** Its width error is one cell
out of N, i.e. `1/N ∝ dx`, so its cutoff error against the real guide was **first order and
stayed first order at every rung** — refining the mesh could never remove it. The frozen
fixture shows exactly that signature in a quantity nobody tuned: the empty guide's spurious
`max_f |S11|` on the `normalize=False` lane reads 0.1354 / 0.0653 / 0.0320 at dx = 2.54 /
1.27 / 0.635 mm — ratios 2.07 and 2.04, first order, on a structure whose true reflection is
zero. Corrected, `f_cutoff` becomes the guide's own discrete cutoff and the mode profile
becomes the exact sampled TE10 (correlation 0.9856 → 1.0000).

### 2.3 The frozen fixture's own evidence that it recorded the old port

Read from `tests/fixtures/waveguide_chain_battery/fixture.json`, key `port_cutoff.per_rung`,
against the guide each rung actually realizes (`cells[].guide_cells_yz`, first entry — the
broad-wall cell count `N`, pinned at 9 / 18 / 36 by
`tests/unit/geometry/test_waveguide_chain_battery_geometry.py`):

| rung | guide cells `N` | `port_cutoff_effective_width_cells` | `fc_port_hz` | thru S21 phase fit `fc_fit_hz` | `rms_deg_at_port_cutoff` | `rms_deg_at_discrete_guide` |
|---|---|---|---|---|---|---|
| coarse | 9 | **10.0412** | 5.877188 GHz | 6.523 GHz | 8.613° | 0.0797° |
| mid | 18 | **19.0217** | 6.204954 GHz | 6.549 GHz | 5.084° | 0.0173° |
| fine | 36 | **37.0111** | 6.378004 GHz | 6.555 GHz | 2.753° | 0.0041° |

The effective width is `N + 1` to four decimal places at every rung, and the cutoff the
extractor used is 2.7 % (fine) to 9.9 % (coarse) below the cutoff the guide itself propagates
with, as measured by the thru's own S21 phase at an rms of 0.004°. Built on main `0141f39e`
in this worktree, the same reader now returns exactly **9.00000 / 18.00000 / 36.00000 cells**
and `f_cutoff` = **6.523901 / 6.548821 / 6.555060 GHz**, equal to the closed-form discrete
cutoff `(2/dx)·sin(π·dx/2a)·c/2π` of the guide's own N cells.

So the committed fixture is a frozen record of a port that no longer exists. Its REPLAY layer
still reads it and its 24 red verdicts stand as history. Its LIVE layer re-measures and can no
longer reproduce it; three live tests are `xfail(strict=True)` naming a pending
re-measurement (`tests/oracle/test_waveguide_chain_battery.py`, `_LIVE_STALE_VS_FIXTURE` and
the plane-shift reason).

### 2.4 Census as of main `0141f39e`

185 stored verdicts: 105 pass, 56 report_only, **21 fail**, **3 not_interpretable** →
**24 red in four families** (26 in five before #881). The four:
plane-shift rotation (12 keys), flux-lane forward identity (8), the AD-vs-FD zero-derivative
leg (1), and three uninterpretable ladders (3).

---

## 3. Inherited unchanged from the first pre-declaration

Everything in this list is **unchanged**, and the pointer is the parent note's section. The
default is that none of it changes: holding the fixture fixed across the two runs is what
makes them comparable, and a change here would confound the instrument correction with a
geometry change.

| what | value | parent §  |
|---|---|---|
| Guide | WR-90, a = 22.86 mm (y), b = 10.16 mm (z); CPML on x, PEC on y and z faces | §2.1 |
| Ladder | dx = a/9, a/18, a/36 = 2.540 / 1.270 / 0.635 mm; guide 9/18/36 × 4/8/16 cells | §2.2 |
| x layout | domain 0.12192 m; ports 0.01270 / 0.10922; default reference planes 0.02032 / 0.10160 (passed explicitly); probe planes 0.03810 / 0.08382 | §2.3 |
| DUT set | thru (non-vacuity control only, #395), `pec_like` PEC-short at 0.05842–0.06350, εr = 4 slab at 0.05588–0.06604 | §2.3 |
| θ windows | slab's own cells; PEC-short vacuum window 0.04826–0.05842; θ0 = 0 (eps), 0.05 S/m (sigma); FD steps 0.05 and 0.005 | §5(a) |
| Absorber rule | `CPML_LAYERS = ceil(0.75·λ_g(f_low)/dx)` at preflight's wall-to-wall numerical cutoff | §2.4 |
| Band | 17 bins, 8.4 → 11.6 GHz, 0.2 GHz apart, centre bin 8 = 10.0 GHz | §2.5 |
| Drive | f0 = 10.0 GHz, bandwidth 0.5, modulated Gaussian, TE(1,0), `mode_profile="discrete"`, `n_modes=1`, `num_periods = 40` | §2.5 |
| Lanes | `normalize=False` and `normalize="flux"` only; `Simulation` default float32; FD legs under a per-test x64 context | §2.5 |
| Claims rung | fine (0.635 mm) for the referee and physics gates; legs rung fine | §2.6 |
| Settling bar | ≤ −40 dB per drive, doubled record where a drive exceeds it, both numbers written | §2.5 |
| All gates | rotation 3° / 6°, wrong-sign floor 10°, ladder floors 0.005 / 1°, ratio window [0.15, 0.70], AD-vs-FD rel ≤ 0.05, ULP floor 1e4, forward identity rtol 1e-5 / atol 1e-7, column power 1.02, reciprocity 0.01, PEC-short 0.99–1.03, Airy 0.05 / 15° | §5, §6 |

**Checked, not assumed.** The absorber derivation reads preflight's wall-to-wall span, not the
port's eigen-aperture, so PR #889 does not move it. Rebuilt on main `0141f39e`:
`fc_TE10, numerical = 6.557140 GHz` at all three rungs, CPML = **17 / 34 / 68** layers
(43.18 mm at every rung), grids (83, 10, 5) / (165, 19, 9) / (329, 37, 17), `n_steps` =
713 / 1425 / 2849 — identical to the parent's §2.2 and §2.4 tables. The geometry guard
`tests/unit/geometry/test_waveguide_chain_battery_geometry.py` is 15 passed in 13.2 s on this
worktree. The preflight findings of §2.6 are therefore expected verbatim and are re-asserted
by the replay's `EXPECTED_PREFLIGHT_CODES` / `EXPECTED_PREFLIGHT_FRAGMENTS`; a changed
preflight set is a finding, not a fixture edit.

**One thing that is inherited but should not be**: the parent's §2.4 explains its CPML
derivation in terms of "the port's numerical TE10 cutoff", and its Erratum item 2 already
records that two different "numerical cutoffs" exist. The absorber uses preflight's
wall-to-wall reader, which was always right; only the port config's `f_cutoff` was wrong. This
note repeats the distinction because it is the single easiest thing to conflate when reading
the two documents side by side.

---

## 4. The re-declared reference-plane shift — the one genuinely new declaration

### 4.1 Why the first pair could not be rescued by any port correction

The first run declared Δ_L = +10.16 mm (4 coarse cells) and Δ_R = −12.70 mm (5 coarse cells).
With the continuous β of the guide's own cutoff (c/2a = 6.557140 GHz), the predicted rotations
across the band are:

| entry | lever | 8.4 GHz | 10.0 GHz | 11.6 GHz | crosses 180° in band? |
|---|---|---|---|---|---|
| ∠S11 | 2Δ_L = 20.32 mm | 128.11° | 184.23° | 233.49° | **yes** |
| ∠S22 | 2\|Δ_R\| = 25.40 mm | 160.13° | 230.29° | 291.86° | **yes** |
| ∠S21 = ∠S12 | Δ_L + \|Δ_R\| = 22.86 mm | 144.12° | 207.26° | 262.68° | **yes** |

The wrong-sign discriminator asks whether the measurement sits far from the *opposite*-sign
prediction: it scores `min_bins |wrap(measured + predicted)|`, which for a correct extractor
is `min_bins |wrap(2·predicted)|`. Where `2βΔ ≈ 180°` the two hypotheses coincide modulo 360°
and that quantity collapses to zero. All three entries pass through 180° inside this band, so
**the discriminator is degenerate by arithmetic, at any cutoff, for any port**. That is why the
first run read `wrong_sign_resid_min = 0.734°` against a 10° floor on all four (dut, lane)
cases, identically — no port correction rescues it, and none was ever going to.

### 4.2 The lattice search — the new pair is derived, not picked

Shift distances stay integer multiples of the coarse cell (2.54 mm) so that all three rungs
realize them identically and the parent note's legibility rule survives. Requirement: the
doubled rotation `2·(2βΔ)` must stay clear of every multiple of 360° across the whole band by
more than the committed 10° floor, and the rotation itself must stay clear of 0° (where the
shift would be indistinguishable from no shift at all). Computed with continuous β at
c/2a over the 17 declared bins:

| Δ / coarse cells | Δ (mm) | 2βΔ over the band | wrong-sign margin `min\|wrap(2·2βΔ)\|` | verdict |
|---|---|---|---|---|
| 1 | 2.54 | 32.03° … 58.37° | **64.05°** | admissible |
| 2 | 5.08 | 64.05° … 116.74° | **126.51°** | admissible |
| 3 | 7.62 | 96.08° … 175.12° | 9.77° | **fails the 10° floor** |
| 4 | 10.16 | 128.11° … 233.49° | 4.57° | fails (the first run's Δ_L) |
| 5 | 12.70 | 160.13° … 291.86° | 1.99° | fails (the first run's Δ_R) |
| 6 | 15.24 | 192.16° … 350.23° | 19.53° | rejected: 2βΔ comes within 9.8° of a full turn at 11.6 GHz, where the shift is nearly a no-op |

Only k = 1 and k = 2 are admissible, so **the largest admissible unequal pair on this lattice
is {2, 1} coarse cells**, and it is unique. The pair stays unequal for the reason the parent
note gives at §5(b): a sign error on one port must not be cancellable by the other. With
Δ_L = 2Δ_R, a swap of the two ports' shifts is also caught — it would move each entry by
2β·2.54 mm = 32°…58°, ten to twenty times the 3° gate.

### 4.3 Declared values

> **Δ_L = +5.08 mm (2 coarse cells), Δ_R = −2.54 mm (1 coarse cell), both inward.**
> Shifted reference planes: **left x = 0.02540 m, right x = 0.09906 m.**

Realization at each rung — the shifts and the planes are exact on all three lattices:

| quantity | metres | coarse cells (2.540 mm) | mid cells (1.270 mm) | fine cells (0.635 mm) |
|---|---|---|---|---|
| left shifted plane | 0.02540 | 10 | 20 | 40 |
| right shifted plane | 0.09906 | 39 | 78 | 156 |
| Δ_L | +0.00508 | 2 | 4 | 8 |
| Δ_R | −0.00254 | 1 | 2 | 4 |

Geometry check — both planes sit between their own port and the DUT, in the same port-to-probe
interval as their defaults, so nothing about the extraction window changes:

| clearance | left plane 0.02540 | right plane 0.09906 |
|---|---|---|
| to its port plane (0.01270 / 0.10922) | 12.70 mm (5 coarse cells) | 10.16 mm (4 coarse cells) |
| to its probe plane (0.03810 / 0.08382) | 12.70 mm inward of the plane | 15.24 mm outward of the plane |
| to the nearest slab face (0.05588 / 0.06604) | 30.48 mm | 33.02 mm |
| to the nearest PEC-short face (0.05842 / 0.06350) | 33.02 mm | 35.56 mm |
| to the CPML interface (x = 0 / 0.12192) | 25.40 mm | 22.86 mm |

Both planes are ≥ 30 mm from any DUT face and ≥ 22 mm from the absorber, so no evanescent
DUT field and no CPML region is inside a de-embedding span. The default planes (0.02032 /
0.10160) are unchanged and remain the raw record planes with `ref_shift = 0`.

### 4.4 Predicted rotation under the corrected port

Continuous β at c/2a (the 6° gate's reference) and the Yee-discrete β at the fine rung
(the 3° gate's reference):

| entry | lever | 8.4 GHz | 10.0 GHz | 11.6 GHz | span | wrong-sign margin |
|---|---|---|---|---|---|---|
| ∠S11 | 2Δ_L = 10.16 mm | 64.05° | 92.11° | 116.74° | 64.05 … 116.74 | 125.5 … 126.5° |
| ∠S22 | 2\|Δ_R\| = 5.08 mm | 32.03° | 46.06° | 58.37° | 32.03 … 58.37 | 64.05 … 64.07° |
| ∠S21 = ∠S12 | Δ_L + \|Δ_R\| = 7.62 mm | 48.04° | 69.09° | 87.56° | 48.04 … 87.56 | 96.08 … 96.10° |

Yee-vs-continuous over these distances is 0.380° / 0.093° / 0.023° at coarse / mid / fine, so
both the 3° and the 6° gate are reachable at every rung by a wide margin.

**Correction against the proposal this note was asked to check.** The proposed spans
(64.1–116.7° and 32.0–58.4°) and the proposed 64.0° margin are confirmed. The proposed
126.6° margin for Δ_L is **not** what the arithmetic gives: it is 126.51° with continuous β
and 125.46 / 126.25 / 126.45° with the Yee β at coarse / mid / fine. The computed value is
used below. The difference is immaterial to the decision (both are 12× the floor) and is
recorded so the note and the fixture cannot disagree later.

### 4.5 Model validation — the prediction machinery reproduces the first run exactly

The predictions in §5 come from an analytic model of the shift algebra: measured rotation
`= 2β(f_cutoff_port)·Δ`, predicted `= 2β(c/2a)·Δ`, both through the same `_compute_beta`
the gate module uses. Run against the FIRST run's inputs (Δ_L = 10.16 mm, Δ_R = −12.70 mm,
the old port's fine-rung `f_cutoff` = 6.378004 GHz) it returns:

| quantity | model | frozen fixture |
|---|---|---|
| `resid_yee_max` | 6.6020° | **6.602°** |
| `resid_cont_max` | 5.9083 / 6.5648° (max) | **6.5648°** |
| `wrong_sign_resid_min` | 0.7343° | **0.7343°** |

Three digits on all three, so the model is not being fitted to the answer it is about to
predict. The same model, run against the corrected `f_cutoff` and the FIRST run's shifts,
gives `resid_yee_max` = 1.279° (coarse) and 0.318° (mid) — the two numbers PR #889 measured
live and wrote into the replay's `xfail` reason. That is a second, independent confirmation.

### 4.6 What this requires in code, and why this PR does not do it

The shift constants live in the builder (`tests/_waveguide_chain_battery_fixture.py`,
`_K_SHIFT_LEFT = 12` and `_K_SHIFT_RIGHT = 35`, which must become 10 and 39). Changing them
here would immediately red the frozen fixture's replay, because
`test_fixture_constants_match_builder` asserts the artifact's `reference_planes_shifted_m`
against the live builder constants — and the frozen artifact will never carry the new pair.
So the builder edit belongs to the measurement PR, together with the two-artifact scheme of
§7 that keeps the guard binding for both. This PR changes documentation only.

---

## 5. Predictions and outcome branches

Every leg below states the quantity, the gate that judges it (unchanged from the parent note),
the predicted value with where the prediction comes from, and what each outcome means. A
prediction with only a pass branch is not a pre-declaration, so every leg has at least two
branches and the "much better than predicted" branch is written wherever it is informative.

### 5.1 ∠S rotation against the Yee-discrete and continuous β

Gate: `resid_yee_max ≤ 3°` (`tests/unit/sparams/test_waveguide_phase_gate.py:259`) and
`resid_cont_max ≤ 6°` (`PHASE_TOL_DEG`, `:63`). Both unchanged.

Predicted, from §4.4/§4.5 with the corrected `f_cutoff` and the new shifts:

| rung | `resid_yee_max` (gate 3°) | `resid_cont_max` (gate 6°) |
|---|---|---|
| coarse | **0.512°** | **0.669°** |
| mid | **0.127°** | **0.164°** |
| fine (claims / legs rung) | **0.032°** | **0.041°** |

The parent's first run measured 6.602° / 6.565° at the fine rung. The predicted 0.032° is a
factor 206 improvement, and it is not a tuning claim: it is what remains once the de-embedding
β and the propagating β are the same guide's, namely the difference between the discrete
cutoff of an N-cell guide and the analytic c/2a, which is O(dx²).

- **Residual ≤ 0.1° at the fine rung** — as predicted. The shift algebra and the port's β are
  the same quantity to the resolution of this fixture; criterion 3(b)'s rotation leg closes.
- **Residual between 0.1° and 3°** — green, but not as predicted. Something rung-dependent
  remains: read `resid_port_beta_max` (the residual against the port's own β, which was
  ≤ 6.3e-5° in the first run). If that is still ~1e-5°, the shift is still an exact
  `exp(∓jβ_port·s)` and the gap is between `f_cutoff` and the guide — report the two cutoffs
  and the thru phase fit before any other move.
- **Residual still above 3°** — the aperture was NOT the whole mechanism. Do not touch the
  gate. The artifact carries `resid_port_beta_max` and the thru S21 phase fit at four
  candidate cutoffs (`port_cutoff.per_rung`); those decide whether the remaining error is in
  β, in the shift arithmetic, or in the mode projection. This is a STOP for the family, not a
  tolerance question.
- **Residual far below the prediction — say ≤ 1e-3° at every rung** — suspect the measurement,
  not the physics. `2β(discrete guide)·Δ` and `2β(c/2a)·Δ` differ by a computable amount
  (0.512° at coarse); a measured residual orders below that would mean the gate is comparing
  a quantity against itself, e.g. the fixture writing the port's own β into both the
  prediction and the measurement. Check that `fc_predeclared_hz` in the artifact still reads
  6.557140 GHz and not the port's cutoff.

### 5.2 The wrong-sign discriminator — which must now fire

Gate: `wrong_sign_resid_min > 10°` (`phase_gate:266`). Unchanged.

Predicted: **64.07 / 64.06 / 64.05°** at coarse / mid / fine — set by the ∠S22 entry, the
smallest lever. Per entry at the fine rung: ∠S11 126.45°, ∠S21 = ∠S12 96.08°, ∠S22 64.05°.
First run: 0.734°.

- **≥ 60°** — as predicted; the discriminator is live for the first time in this battery and a
  flipped shift sign cannot hide.
- **Between 10° and 60°** — green but unexplained. The margin is a pure function of β and Δ,
  both declared here, so a value in this band means the measured rotation is not tracking the
  prediction (see 5.1) rather than that the witness is weak.
- **≤ 10°** — the shifts did not land where this note declares. First check
  `plane_shift.reference_planes_shifted_m` in the artifact against 0.02540 / 0.09906, then the
  builder constants. A red here after §4 is a fixture-construction failure, not a physics
  result.
- **Independent corroboration, unchanged**: the cheap refute
  (`plane_shift.cheap_refute`, `test_cheap_refute_flipped_shift_sign_makes_the_rotation_gate_red`)
  re-runs the plane-shift stage with the sign of `_shift_modal_waves` flipped in a local copy
  and requires the rotation gate to go red. The first run measured a minimum residual of
  119.49° under the flip. Predicted again red, by ≥ 60°; a flip that stays inside 3° means the
  gate does not bind and nothing else in §5.1 is readable.

### 5.3 Settling witness

Gate: `settling_db ≤ −40 dB` per drive, on the claims-bearing record, re-derived through the
post-#881 arithmetic. Unchanged.

Predicted: **every drive between −78 and −102 dB at `num_periods = 40`**, and **no cell
triggering the 80-period rerun**. Basis: the first run's own per-record numbers over normal
records — thru −84.9 / −94.6 / −98.0 dB, slab −79.5 / −93.5 / −99.8 dB, PEC-short −81.3 /
−94.6 / −100.0 dB at coarse / mid / fine. The PEC-short fine rung, the cell that read 0.00 dB,
is predicted at **−98 to −100 dB** on both lanes. The corrected port changes the mode
projection, not the ring-down of a CPML-terminated guide, so these should move by less than a
few dB.

- **All 18 cells report a real dB below −40** — as predicted; the witness measures the
  ring-down rather than a float32 artefact, and the parent note's §2.5 requirement is met
  without a rerun.
- **Exactly 0.00 dB anywhere** — this now means something different from what it meant in the
  first run. #881 skips every record whose tail is below the float32 normal minimum and
  returns NaN plus a warning when *all* records are skipped, and NaN cannot pass a `≤ −40 dB`
  gate. So a 0.00 dB can only come from a record whose peak and tail mean are both normal
  numbers and equal — a run that did not decay at all. That is a genuine truncation or
  instability and blocks the run; it is not a bookkeeping artefact to re-derive around.
- **A drive between −40 and 0 dB** — the declared rerun path fires: repeat that cell at
  `num_periods = 80` at the same absorber, write both numbers, and the 40-period number of
  that cell stops being claims-bearing. `num_periods` is never tuned per cell silently.
- **A drive far below −102 dB at the coarse or mid rung** — worth a look rather than a
  celebration: the coarse rung has 713 steps and 17 CPML layers, and a sudden 20 dB
  improvement there without a geometry change would suggest the record being scored is not the
  one the drive wrote.

### 5.4 Empty guide `|S11|` and column power on the `normalize=False` lane

Not gated (the thru is the non-vacuity control only, #395), but the most direct read of the
extractor's own mismatch, and therefore the leg that most directly tests §2.2.

`max_f |S11|` on the empty guide, `normalize=False`:

| rung | first run (old port) | measured by #889 (live CPU) | predicted here |
|---|---|---|---|
| coarse | 0.0762 … 0.1354 per bin | **0.007 … 0.041** | (measured) |
| mid | 0.0653 (max) | — | 0.010 (if O(dx²)) / 0.020 (if O(dx)) |
| fine | 0.0320 (max) | — | **0.0026 (if O(dx²))** / 0.010 (if O(dx)) |

Column power `max_f Σ_i |S_i1|²` on the empty guide, `normalize=False`: first run 1.018253 /
1.004082 / 1.000983; #889 measured 1.006 at coarse. Predicted fine **≈ 1.0004**, from the
first run's own ~4× per halving (#873) applied to the corrected coarse excess.

- **Fine-rung `|S11|` at or below 0.004, falling ~4× per halving** — as predicted, and the
  strongest single confirmation that the removed error was the first-order one-cell width
  term. The remaining reflection is then second order, i.e. ordinary Yee `Z_TE` error (#873).
- **Fine-rung `|S11|` near 0.010, falling ~2× per halving** — a first-order term survives the
  aperture correction. #889 fixed a site of the node-vs-cell class (ledger `:64`); a surviving
  first-order term says there is a fifth site. Report the ratio sequence, do not adjust
  anything.
- **Fine-rung `|S11|` worse than the frozen 0.0320** — the correction made the extractor worse
  at the claims rung. Blocking; the run does not close anything until it is explained.

### 5.5 `pec_short` + `normalize=False` column power — an expected worsening, bounded in advance

Gate: `column_power_max < 1.02` (`port_validation_battery.py:307`). Unchanged, and **not**
moved for this.

#889 measured this quantity getting **worse**: 1.000116 → **1.004715** at coarse and
1.000032 → **1.001162** at mid, a ratio of 4.06 between the two, i.e. second order in dx.
The reason is not a new error but the removal of a cancellation: the pre-fix `Z_TE` error and
the modal `V/I` decomposition error had opposite signs on a total reflector, and the sum
happened to sit closer to unity than either term. Declaring it here means a reviewer of the
measurement PR reads it as the predicted consequence of a fix, not as a regression.

Predicted: **coarse 1.0047, mid 1.0012, fine ≈ 1.0003**, extrapolating the measured second-order
sequence one more rung.

- **Fine rung ≤ 1.001, and the sequence still ~4× per halving** — as predicted. The
  corresponding `|S11| = √1.0003 = 1.00015` stays inside the referee's 0.99–1.03 window, and
  the coarse worst case `√1.004715 = 1.00236` does too, so §5.7's PEC-short referee is
  unaffected.
- **Fine rung between 1.001 and 1.02** — green against the committed gate but worse than
  predicted; report the per-bin column power and the ratio sequence. A first-order sequence
  here contradicts the mechanism above and is a finding.
- **Any rung ≥ 1.02** — the committed passivity gate is red on a passive structure. Blocking.
  Quote the offending bins with the preflight context verbatim, per the repo's passivity rule;
  do not attribute it to the extractor before checking the energy/instability witnesses.

### 5.6 Families expected to be UNCHANGED

Declaring these explicitly is the point: it turns an unexpected change into a finding instead
of a relief.

**(i) Forward identity on the `normalize="flux"` lane — 8 legs, expected to stay RED.**
Gate `rtol=1e-5, atol=1e-7` (`test_waveguide_flux_ad.py:104`). First run: `max_scaled_diff`
1.440 (slab) and 1.065 (PEC-short), i.e. absolute 0.9–1.1e-5, while the same traced call under
a scoped x64 context agrees with the untraced one to 1.5e-15…2.2e-14. Mechanism: float32
reassociation of a 2849-step Poynting DFT under the reverse-mode tape. The port aperture does
not enter it, so predicted **unchanged at 1.0–1.5 scaled**.
  - *Stays 1.0–1.5* — as predicted; closing this needs the x64 declaration decision of §6, not
    a run.
  - *Flips green* — **this is not evidence of a fix.** The quantity is a noise measure sitting
    a few percent above its own gate; a green here is inside the run-to-run spread of the same
    defect. Treat a green as unresolved until the x64 witness shows the float32 and x64 primals
    agreeing at the 1e-7 level, not at 1e-5.
  - *Above 10 scaled* — a real forward-identity break introduced between the two runs.
    Blocking; criterion 1 fails on that lane by more than a reassociation argument can carry.

**(ii) The AD-vs-FD zero-derivative leg — expected to stay RED, and its sibling may move.**
`pec_short | flux | eps | s11_mag2`: first run `g_ad = +2.683e-5` (float32) against
`g_fd = −7.245e-7`, `rel = 38.03`, FD span 6.53e8 ULP. The objective's derivative is physically
zero (|S11| = 1 for a lossless window in front of a PEC), the float32 AD noise floor exceeds
the O(1e-6) residual, and the x64 AD gives −9.821e-7 — the same sign and order as FD.
Predicted **unchanged: red, with `|g_ad|` of order 1e-5 and `rel` ≫ 0.05.**

  The sibling leg on the `normalize=False` lane **passed** in the first run
  (`rel = 1.092e-2`, `g_ad = +7.800e-5`, `g_fd = +7.716e-5`) — and it passed because the old
  port's `Z_TE` mismatch gave that lane a genuinely non-zero |S11| sensitivity. §5.4 predicts
  that spurious mismatch shrinks by roughly an order at the fine rung, so this leg's true
  gradient shrinks with it. Three branches, all pre-declared:
  - *it skips under the ULP floor* — the parent note's §5(a) expectation ("expected skip,
    declared now") is finally realized; record it as such;
  - *it stays green with a smaller `g_fd`* — the residual derivative is still resolvable;
  - *it goes red like the flux lane* — the SAME mechanism, a float32 AD noise floor above a
    near-zero derivative, now reaching the second lane. That is a finding about float32 AD on
    a zero-derivative objective, **not** a regression caused by the port fix, and the x64
    witness (`ad_vs_fd[].x64_witness`) is what distinguishes the two.

  The other 14 AD-vs-FD legs are predicted green with `rel` in 1e-4 … 1.2e-2 against the 0.05
  gate (first run: 1.0e-4 … 1.1e-2, ledger `:3110` expectation 1e-3). Gradient *magnitudes* on
  the `normalize=False` lane will move with `Z_TE`; the gated quantity is the AD-FD agreement,
  which should not.

**(iii) The three uninterpretable ladders — expected to stay "not interpretable".**
`slab_s11_mag|false` (ratio 0.037 at 8.4 GHz), `slab_s21_mag|false` (0.098 at 8.6 GHz),
`pec_short_s11_phase_deg|flux` (0.077 at 10.0 GHz), all outside the declared [0.15, 0.70]
window on a conditioned bin. A ratio far *below* 0.25 means the coarse→mid delta is dominated
by an error that dies faster than dx² between mid and fine — i.e. the coarse rung is not on
the same asymptotic branch as the other two, which at 5.1 cells/λ_eff inside the εr = 4 slab
is what §2.6 of the parent note expected. That is a mesh property, and the port correction does
not touch it. **The numbers will move** (the `false` lane's magnitudes carry `Z_TE`), but the
verdict class is predicted unchanged.
  - *All three stay not-interpretable* — as predicted; the fix is one more rung (§6), not
    this run.
  - *One or more becomes interpretable* — a finding to explain, not a relief: it would mean
    the coarse rung's dominant error was the port's first-order cutoff term rather than the
    slab's under-resolution, which contradicts the reading above and changes what the ladder
    is measuring.
  - *A previously interpretable ladder becomes uninterpretable* — same treatment. Seven of the
    ten are interpretable today; the population is the claim, not any single row.
  - The non-increase gate itself (`fine_delta ≤ coarse_delta + floor`, floors 0.005 and 1°)
    passed on all ten observables in the first run and is predicted to pass again. A red there
    is a real convergence failure and blocks.

### 5.7 Referee and physics gates at the claims rung

All gates unchanged. Predicted from the first run plus the `Z_TE` correction:

| quantity | gate | first run (fine rung) | predicted |
|---|---|---|---|
| PEC-short \|S11\| range | 0.99 … 1.03 | 0.999967 … 1.000008 | 0.9999 … 1.0002, still well inside |
| slab vs Airy, magnitude | ≤ 0.05 | 0.02072 (`false`) / 0.00903 (flux) | 0.010 … 0.021 (`false`), ≤ 0.012 (flux) |
| slab vs Airy, phase | ≤ 15° | 7.37° / 6.58° | 5 … 8°, improving on the `false` lane |
| column power, both DUTs | < 1.02 | 1.000975 (slab `false`) | ≤ 1.002 |
| magnitude reciprocity | < 0.01 | 2.585e-3 | ≤ 4e-3 |
| complex reciprocity | ≤ 0.01 | 6.983e-3 (`false`) / 3.28e-4 (flux) | ≤ 8e-3 / ≤ 5e-4 |

The `false`-lane referee numbers are predicted to *improve* because the ~2 % `Z_TE` error at
the fine rung is removed; the flux lane is power-based and predicted to move less. Column power
does **not** move in one direction for both DUTs: §5.4 predicts the empty guide and the slab
improve, §5.5 predicts the PEC-short gets worse (a cancellation removed), and the ≤ 1.002 above
bounds both. A run in which every column-power number moves the same way contradicts that
reading and is a finding.
- *Any referee gate red* — blocking, and the first thing to check is the comparator: the Airy
  oracle and its `exp(−2jβ_v d_L)` / `exp(−jβ_v(d_L+d_R))` shift convention are unchanged
  here, so a red says the extraction moved, not the oracle.
- *Complex reciprocity above 0.01* — reported, not absorbed, exactly as the parent note
  declared. The warn-only advisory landed at 0.011 (PR #882) derived from the first run's
  claims-rung envelope; a second run measuring materially worse than 6.983e-3 puts that
  advisory's derivation back on the table and must be said out loud in the measurement PR.

### 5.8 The success criterion for this run

Three tests are `xfail(strict=True)` today, pointing at this re-measurement. **They are the
criterion**; nothing else in this note substitutes for them.

| test | today | required after the run |
|---|---|---|
| `test_live_cells_reproduce_the_fixture_cpu[coarse, mid]` | xfail: live differs from the frozen fixture by up to 1.6e-1 against `LIVE_ABS_S_TOL = 1e-4` | pass against the NEW artifact |
| `test_live_cells_reproduce_the_fixture_fine_rung` (gpu) | xfail, same reason | pass against the NEW artifact |
| `test_live_plane_shift_rotation_coarse_rung[false, flux]` | xfail: `wrong_sign_resid_min` 1.801° against the 10° floor with the old shifts | pass — predicted 0.512° / 0.669° / 64.07° at coarse |

`LIVE_ABS_S_ENVELOPE = 5.000e-6` and the derived `LIVE_ABS_S_TOL = 1e-4` are committed values
from a measured CPU-vs-GPU envelope. They are **not** moved. If the new run's cross-backend
envelope exceeds 5e-6, the excess is reported with both backends' numbers; it is not absorbed
by widening the pin.

---

## 6. What this run cannot settle

- **cv11 and the WR-90 broad-E5 band envelopes.** They live on the external VESSL lane
  (`scripts/vessl_crossval_external.yaml`, weekly `crossval-external`) and their numbers will
  move under the corrected aperture, but this run does not re-measure them. #889 recorded that
  they do not *degrade* (cv18 unchanged to ≤ 2e-5, broad-E5 and the NU gates 70 passed), which
  bounds the risk without re-pinning anything. Re-capturing them is a separate lane with its
  own pre-declaration, and the #812 artifact lane stays untouched either way.
- **The flux lane's forward identity.** §5.6(i) predicts it stays red at 1.0–1.5 scaled. That
  is a **declaration decision, not a measurement**: either criterion 1 on the flux lane is
  declared under x64 (the lane has followed `JAX_ENABLE_X64` since PR #862 and the x64 witness
  agrees to 1.5e-15), or the float32 reassociation is bounded by a derived, pre-declared
  tolerance. No amount of re-running decides it. It belongs to a WP-level decision PR.
- **The three uninterpretable ladders.** Making them readable needs a fourth rung at
  **a/72 = 0.3175 mm** (b = 32 cells), the only admissible next rung under the integer-N rule.
  This run **does not add it**, and the note does not argue that it should: a/72 quadruples the
  cell count and doubles the step count against the fine rung, the three affected observables
  are all on lanes and rungs the claims-bearing set does not rest on, and adding a rung in the
  same run as an instrument change would confound the two. If the corrected run leaves them
  uninterpretable — which §5.6(iii) predicts — the a/72 rung is the next pre-declared step,
  with its own note.
- **Issue #873** (`normalize=False` reporting 1.8e-2 column power on an empty guide) is
  *informed* by §5.4 but not closed by it: this run measures the corrected empty-guide numbers,
  and the direct comparison of the lane's modal `Z_TE` against the analytic value — the
  missing instrumented step — is still missing.
- **Whether the waveguide family is chain-closed.** Even a clean sweep of §5 leaves the flux
  lane's criterion 1 open (above). The contract is explicit: failure of any single pass
  condition means the family is not chain-closed, and "chain-closed (v1.8)" is not written for
  this family on the strength of this run alone.

---

## 7. What this run writes, and how the two artifacts are told apart

**The frozen artifact stays exactly as it is.** `tests/fixtures/waveguide_chain_battery/fixture.json`
is not edited, not re-pinned, not renamed and not deleted. It keeps `schema_version` 1, its
185 verdicts, its 24 red, and its `predeclaration_sha` 78885c10 pointing at the parent note.
It is the record of a port that no longer exists, and it is the evidence for §2.3.

**The new run writes a new file**:
`tests/fixtures/waveguide_chain_battery/fixture_guide_cell_aperture.json`, named for the
physics that distinguishes it — the port's transverse mode solved on the guide's own N cells —
with `schema_version` 2, the same schema otherwise
(`tests/fixtures/waveguide_chain_battery/README.md`, extended in the measurement PR for the
two new keys below), `predeclaration` pointing at **this** note and `predeclaration_sha` the
commit that lands it, which must predate `provenance.commit`. Driver and lane are unchanged:
`scripts/diagnostics/waveguide_chain_battery_measure.py --fixture-out <new path>` under
`scripts/vessl_waveguide_chain_battery.yaml`.

**Three mechanisms tell them apart, and none of them is a filename convention:**

1. **The instrument, read from the artifact itself.** `port_cutoff.per_rung.*.port_cutoff_effective_width_cells`
   is 10.0412 / 19.0217 / 37.0111 in the frozen record and must be **9.000 / 18.000 / 36.000**
   in the new one, with `fc_port_hz` = 6.523901 / 6.548821 / 6.555060 GHz and
   `rms_deg_at_port_cutoff` collapsing from 8.613 / 5.084 / 2.753° to
   `rms_deg_at_discrete_guide` (0.0797 / 0.0173 / 0.0041°). The replay asserts this per
   artifact, so a future run that regressed the aperture cannot be filed as the corrected one.
2. **The declared shift pair.** The builder gains both pairs as named constants — the first
   run's as `half_turn_pair` (0.03048, 0.08890), so called because `2βΔ` passes through a half
   turn inside the band, and this run's as `sign_discriminating_pair` (0.02540, 0.09906). Each
   artifact records `shift_pair_name`; `test_fixture_constants_match_builder` resolves the pair
   by that name and asserts the artifact's `reference_planes_shifted_m` equals it. An artifact
   with no `shift_pair_name` (i.e. `schema_version` 1 — the frozen file, left byte-identical)
   resolves to `half_turn_pair` by default. The guard therefore keeps binding for both
   artifacts and cannot silently accept a third pair.
3. **`supersedes`.** The new artifact carries
   `supersedes: "tests/fixtures/waveguide_chain_battery/fixture.json"` with a one-line reason,
   so a reader who opens either file is told which is live.

The live layer (`test_live_cells_reproduce_the_fixture_*`, `test_live_plane_shift_rotation_coarse_rung`)
re-points at the new artifact and its `xfail(strict=True)` marks come off — which strict xfail
requires the moment they pass. The frozen artifact keeps only the replay tests that do not read
the live builder shift constants; its 24 red verdicts stay red, with their `xfail` reasons
extended by one clause naming this note.

Lane placement, unchanged from the parent: the replay layer is JSON arithmetic and stays fast;
the measurement is minutes on a GPU and stays in the slow lane / the tracked VESSL YAML. The
whole-battery wall time goes in the measurement PR body (first run: 1157.6 s).

**Pins.** The frozen artifact's pins (`gradient_invariance_gate` 0.001 from envelope
1.7938e-7, `richardson_quantum` 100 / 10, `monotone_quantum` 100) are **not** moved. The new
artifact carries its own pins derived by the same `gate_from_envelope` policy from its own
measured envelopes, in a separate pin step, and the measurement PR states both numbers
side by side. If the new envelope is larger than the frozen one, that is reported as a
finding; the frozen pin is not loosened to accommodate it.

---

## 8. Must not

- **No second run to improve a number.** One run, this contract. A further run needs a third
  named instrument defect, found and written down before it, not inferred from these results.
- **No gate, tolerance or golden moved.** Every value in §3 and §5 is a committed gate. A red
  gate needs a written root cause first; `LIVE_ABS_S_ENVELOPE` and the fixture pins are
  included in this.
- **The frozen fixture is not edited, re-pinned, renamed or deleted**, and its red verdicts are
  not re-derived away.
- No `normalize=True` anywhere in the battery.
- No import of `rfx/probes/refplane.py` into the waveguide path.
- No new source construction; reuse the port's `cfg.e_inc_table` / `h_inc_table`.
- No ladder rung that is not `a/N` at integer N; no DUT face moved off the 2.54 mm lattice; no
  fourth rung added in this run (§6).
- `num_periods` is not tuned per cell without writing both numbers.
- The #812 lane is untouched: `wr90_rectangular_broad_e4_comparison.json`, crossval
  cv02/03/04/09/10/14/20/21, cv06b.

---

R3: memory=rfx-known-issues.md:99 (chain-status row — the N+1 aperture mechanism and the ±180° wrong-sign arithmetic, both acted on here), :64 (#729 node-vs-cell class), :3110-3122, :3413-3420, :4129, :4336, :3735-3752, :3546 + project_issue527_f32_comparator | R2-attempts=0 in this lane (no measurement; the second RUN is authorized by a changed instrument — PR #881 d6a3df5d and PR #889 0141f39e — not by a repeat of the first mechanism hypothesis) | falsifier=rebuilt the fixture on main 0141f39e and read the port back: effective aperture width is now exactly 9.00000 / 18.00000 / 36.00000 cells and f_cutoff 6.523901 / 6.548821 / 6.555060 GHz (was 10.0412 / 19.0217 / 37.0111 cells and 5.877 / 6.205 / 6.378 GHz in the frozen artifact), while fc_TE10_numerical stays 6.557140 GHz and CPML stays 17/34/68 — and tests/unit/geometry/test_waveguide_chain_battery_geometry.py is 15 passed, so the inherited geometry of §3 is unchanged
