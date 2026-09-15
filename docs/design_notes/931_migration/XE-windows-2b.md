# X-E phase-2b ingest — pre-declared windows, and the four run records

Written during the phase-2b ingest of group X-E (cv20-cv24, `validation/research`,
`validation/tmtt_paper`) on branch `feat/931-crossval-E`, after merging
`feat/931-lattice-ownership` @ `770c4e6c`.

**Nothing in this note edits a window, a gate constant or a tolerance.** Where a
window's input is stale, that is said and the run that would refresh it is named.
Where a gate is red, the cause is named and the gate is left red.

---

## 1. `thru_feedpost` — the pre-declared windows, re-derived only through the
## predeclaration's own procedure

Source of the procedure:
`docs/design_notes/thru_feedpost_twoseg_predeclaration.md` §1 (hygiene rule), §2
(model bounds), §4 (falsifier windows). The constants live in
`validation/research/thru_feedpost_deembed.py` and
`validation/research/thru_feedpost_twoseg_extraction.py`.

### 1.1 The binding constraint on any re-derivation

§1: *"Every window below is derived from geometry/first principles, from prior
ledger / #313 instrument provenance, or from the NEW independent fixtures —
**never from the numbers those windows will judge**."*

So the one thing that must not happen is: run this lane's `--extract` arm on the
post-#931 board, read its Zc, and set `F_I3_ZC`/`ZC_CENTER` from it. That is the
burned data by name. The `--extract` arm exists in the script and is cheap; it is
**not** the re-derivation input, and running it does not authorise moving a
window.

### 1.2 What the branch changed about this fixture (build-time, no solve)

Measured on this branch with the shared realizer
(`validation/crossval/comparators/realized_conductors.py`, which delegates to
`tests/_realized_geometry.py`), on `build_thru`:

    realized wall planes (z)  [2] at 0.001000 m  = H          (one plane: sheet)
    realized Ex footprint     11 node rows, y 0.007500 -> 0.012500 m
    node span                 5.000 mm  = the drawn W
    n_rows * dx               5.500 mm

The plane is unchanged (z = H is the node the old single wall landed on). The
**width gained one row**: the old half-open node sampler dropped the hi row and
realized 10 rows. A sheet footprint is closed, so it realizes 11.

The line is air microstrip (h = 1.0 mm above the `z_lo` PEC wall, no dielectric),
so its Zc moves with W/h and nothing else. Hammerstad-Jensen air-microstrip
closed form, both width conventions (the geometric-vs-electrical width question
is itself unsettled on this branch — see the cv06b 571.5-vs-635 um entry in the
merge report):

| convention | W old -> new | Z0_HJ old -> new | move |
|---|---|---|---|
| geometric node span | 4.5 -> 5.0 mm | 53.20 -> 49.37 ohm | -7.20% |
| electrical `n_rows*dx` | 5.0 -> 5.5 mm | 49.37 -> 46.07 ohm | -6.68% |

Robust to the convention: **the line's Zc drops about 7%.**

### 1.3 `F_I3_ZC = (44.0, 53.0)` ohm — the procedure, and why it cannot be
### re-derived here

§4 F-I3: *"every measured Re(Zc) in [44, 53] ohm (attempt-2 F-J2 derivation:
#313 mid-line class +-10% for plane definition and low-f dispersion)"*.

The declared input is the #313 Phase-0 measured mid-line class, which the same
file freezes as `ZC_CENTER, ZC_CORNERS = 48.25, (47.9, 48.6)`. The arithmetic
reproduces the committed window exactly, and does so under either reading of
"the class", so the procedure is not ambiguous where it matters:

    from the corners:  ceil(47.9 * 0.9) = ceil(43.11) = 44.0
                       floor(48.6 * 1.1) = floor(53.46) = 53.0
    from the centre:   ceil(48.25 * 0.9) = ceil(43.425) = 44.0
                       floor(48.25 * 1.1) = floor(53.075) = 53.0

Note the rounding direction: **inward**, to whole ohms. The procedure tightens at
the rounding step; it never widens. Any re-derivation must keep that.

**The input is stale and this lane cannot refresh it.** `ZC_CORNERS` is a
measurement of the #313 refplane instrument's own fixture, not of this one, and
that fixture's realization also moves under the ownership contract. Re-deriving
`F_I3_ZC` therefore requires the #313 Phase-0 line-constant measurement to be
re-made on the post-#931 realization — owned by the refplane / #313 lane, not by
X-E — and then the two lines of arithmetic above, unchanged.

**What that will probably do, stated in advance so the outcome is falsifiable.**
Carrying the -7% of section 1.2 through the committed class gives an expected new
class of about 44.5-45.4 ohm. That is INSIDE the current window (44.0, 53.0), but
against its low edge: the margin at the bottom goes from 3.9 ohm to roughly
0.5 ohm. So the honest reading is that the window is not loose, it is nearly
exhausted on one side, and the correct response to a future F-I3 firing on the
low side is a re-derived window from a re-measured #313 class — **not** a widened
one, and **not** one set from this lane's own Zc.

### 1.4 The rest of the set — which inputs move and which do not

| window | value | declared input | moves under #931? |
|---|---|---|---|
| `ZC_CENTER`, `ZC_CORNERS` | 48.25, (47.9, 48.6) ohm | #313 Phase-0 measured class | **yes** — the line is ~7% lower; needs the #313 re-measurement |
| `BETA_CENTER`, `BETA_CORNERS` | 1.055, (1.048, 1.062) | #313 Phase-0 measured class | probably slightly (a wider line raises the air-line beta only through fringing); same owner |
| `F_I3_ZC` | (44.0, 53.0) ohm | the above, x[0.9, 1.1], rounded inward | via its input only |
| `F_I3_BETA_FAC` | (1.00, 1.10) | lo: air quasi-TEM floor (first principles). hi: measured class 1.062 "plus low-f headroom" | lo: **no**, ever. hi: the headroom is a declared judgement, not a formula — it is not mechanically re-derivable and must not be re-derived by inventing one |
| `F_I1_IM_RE` | 0.03 | the refplane module's own measured class boundary (`_ZC_IM_RE_WARN_RATIO`) | **no** — an instrument-cleanliness class, independent of this board |
| `F_I2_*`, `F_I4_*` | 1.2 ohm, 0.02 | 2x a "0.6 ohm" plane-to-plane spread class | **no** — a spread, not a level. (Observation while checking: §4 names a "0.6 ohm class" while the corner span it cites, 47.9-48.6, is 0.7 ohm. The window value 1.2 is not exactly reproducible from the quoted numbers; recorded, not acted on.) |
| `F_A1_MAX/RMS` | 0.025 / 0.012 | four modelled residual terms (loss, junction C, bin scatter, DFT) | **no** — model-error budget, no realized dimension in it |
| `F_A2_MAX/RMS` | 0.06 / 0.03 | #313 plane-path amplitude/phase classes + the F-A1 terms | **no**, same reason |
| `F_P_L_NH` | (0.20, 0.50) nH | #318 ledger class -25%, thin-wire estimate +25% | **no** — first principles for a 1 mm post at r = 67.5 um |
| `F_P_TAU_PS` | (1.67, 8.34) ps | h_eff in [0.5, 1.5] mm from a +-1-cell dead/live ambiguity, + <=2 cells of fringing; [0.5 mm/c, 2.5 mm/c] | **no**, and the reason is now stronger, see below |
| `F_V1_*`, `F_V2_*` | corr 0.90, cond 60/15, sigma 0.10/0.15/0.20 | model-derived expectations vs the refuted attempt-2 class | **no** |
| `F_C_L_NH`, `F_C_TAU_PS` | 0.11 nH, 2.6 ps | injected systematic classes + 3 sigma | **no** |
| `F_D2_REDUCTION` | 0.5 * 0.2910 | held-gate provenance (PR #777 §6) | **no** |
| `F_L1/L2/L3/L5`, `F_X1/X2/X3` | as committed | attempt-1 geometry/first principles | **no** |

On `F_P_TAU_PS`: its `+-1 cell` term was written because *"the top post cell is
dead inside the PEC trace"* under a one-cell PEC Box. Base commit `6d66ac65`
(the half-open wire-port extent, #931 R8) measured this exact fixture and found
`n_live = 2` under both the old accidental arrangement and the corrected rule, so
the gap V integral still spans 2 live cells and the derivation's input is intact.
The ambiguity term could arguably now be tightened, which would make the window
narrower; §4 freezes it, so it stays as committed.

### 1.5 `issue764`, `issue683`, `issue770`, `#786`

Same shape, listed for completeness and not re-derived here: their geometry
changed (a volume that gained its far face; a midpoint recipe deleted), their
committed numbers are stale, and each window's input is a measurement nobody has
re-made. None of them was submitted as a run by this group, and none of their
constants is touched.

---

## 2. cv20 — the fixture is ingested; the six remaining reds are a comparator
## input, not a gate

After committing the run-369367259200 fixture,
`tests/crossval/test_msl_phase_referee_header.py` went **14 failed / 49 passed ->
6 failed / 57 passed**. All six fail inside one witness,
`_analytic_beta_witness`, with one number:

    measured beta vs Hammerstad-Jensen closed form: max|dev| = 2.1306%
    gate B_BETA_ANALYTIC_TOL_FRAC = 2.000%

The witness feeds the closed form `h = meta['h_sub_realized_m'] = 300 um`. Under
the ownership contract that is no longer the height of **dielectric**: the trace
is a one-cell VOLUME occupying the top substrate cell, 250 -> 300 um, so the
dielectric under the strip is 250 um and the strip is inlaid flush with the
substrate top. Recomputed with the fixture's own new key
`meta['trace_wall_planes_realized_z_m'][0] = 0.00025` and nothing else changed
(the committed beta array, the same `_hammerstad_jensen_eps_eff`, the same gate
band):

| h fed to the oracle | eps_eff | max|dev| | gate 2.00% |
|---|---|---|---|
| 300 um (`h_sub_realized_m`, as gated today) | 2.832693 | **2.1306%** | FAIL |
| 250 um (realized lower wall plane) | 2.872970 | **1.4122%** | pass |
| 254 um (declared) | 2.869386 | 1.4755% | pass |

The tolerance does not need to move for the second row to pass, and it is not
moved. Its own budget re-derives consistently at h = 250 um: the Bahl-Garg
finite-thickness term `-(er-1)(t/h)/(4.6*sqrt(w/h))` goes -0.0681 -> -0.0747 in
eps_eff, i.e. -1.21% -> -1.31% in beta, and the linear sum 1.77% -> 1.87%, still
under the declared 2.00% ("one round step up"). Also note the sign: the budget
allows only +0.56% on the HIGH side (HJ +-0.50%, dispersion +0.06%), so the
observed +2.13% is 3.8x the positive budget — it is not a near-miss to be
rounded away.

**Why nothing is changed here, and why the Stage B re-run is HELD.** The same
question decides what openEMS builds, and the two must not be answered
separately. `_build_stage_b_thru` today builds the substrate `0 -> h_sub` and the
trace `h_sub -> h_sub + t_metal`, i.e. **300 um of dielectric with a 50 um strip
proud on top**, while rfx realizes **250 um of dielectric with the strip inlaid
into the top cell**. Those are two different boards — a 20% difference in
substrate height on a fixture whose own preflight measures Z0 at 0.6% per 1% of
h. The file already asserts that both solvers build the same *metal thickness*
(`t_metal` vs `layout['t_metal_realized_m']`); it does not assert the same
*dielectric height under the strip*, and that is the gap.

So the cv20 openEMS Stage B re-run was **not submitted** in this pass. The
trace_y part of the instruction (950/1550 -> 900/1500 um) needs no code change —
`_build_stage_b_thru` already reads `trace_y_{lo,hi}_realized_m` from the fixture
meta, so it flows automatically. What needs a decision first, by the cv20 owner
with the merge agent, is one of:

* **(a) inlay the openEMS strip** — `pec.AddBox([..., h_sub - t_metal], [...,
  h_sub])` instead of `[h_sub] -> [h_sub + t_metal]`, so openEMS builds rfx's
  board; then feed the oracle `h = 250 um` for both solvers and add the missing
  assertion (dielectric height under the strip must equal the fixture's realized
  lower wall plane). One line plus an assertion; no gate constant moves.
* **(b) declare rfx's board the proud one** — which means the rfx fixture is the
  thing that is wrong and the board must be redrawn, an option this lane already
  rejected with a measurement (dx = 50.8 um destroys the on-lattice
  `ref_plane_shift`).

A run submitted before that choice would produce a claims-bearing referee
artifact comparing two different boards, and the committed
`_20_msl_phase_referee_logs/*_result.json` history is superseded, never edited —
so a wrong one is permanent. Holding costs one deferred run; submitting costs a
GPU-hour and a misleading record.

One qualification, so the finding is not overstated: **the two solvers were not
building the same object before #931 either.** Under the old rule the rfx trace
realized as a single zero-thickness wall — no metal thickness at all — while
openEMS has always built a 50 um strip proud on top of the substrate. What the
contract changed is the rfx side: the strip is now 50 um of real metal inlaid
flush with the realized substrate top. The `t_metal` assertion this branch added
to `_build_stage_b_thru` closes the THICKNESS half of that old gap; the
dielectric-height-under-the-strip half is still open, and the analytic-beta
witness is where it now shows up (it passed at h = 300 um while the strip was
zero-thickness and buried; it does not at 2.13% now that the strip is a
conductor occupying that top cell). So this is a pre-existing comparator
weakness that #931 exposed, not a defect #931 introduced — which is an argument
for deciding it deliberately, not for deciding it inside an ingest.

### 2.1 Resolved 2026-09-07 (branch `feat/931-core-last`): each leg is judged
### against the board its own solver built; the openEMS inlay is still held

The half of option (a) that needs no run is done. `_stage_b_layout` gains
`h_dielectric_under_strip_realized_m`, read from the fixture's own
`trace_wall_planes_realized_z_m[0]` when `trace_realization_kind` is
`volume` and cross-checked against `h_sub_realized_m - t_metal_realized_m`;
`_run_stage_b` feeds it to the rfx analytic-beta leg and keeps
`h_sub_realized_m` for the openEMS leg, which is the board
`_build_stage_b_thru` actually builds. Sharing one `eps_eff` was judging
rfx's beta against openEMS's substrate height.

Measured (2026-09-07), gate `B_BETA_ANALYTIC_TOL_FRAC` untouched at 2.000 %:

| leg | h fed | max abs dev |
|---|---|---|
| rfx, current fixture | 250 um | 1.4122 % |
| openEMS, run-2 artifact | 300 um | 0.3068 % |

The pre-declared budget was re-derived at 250 um, as this section predicted
before the change: Bahl-Garg 0.0121 -> 0.0130, Getsinger 0.0006 -> 0.0005,
sum 0.0177 -> 0.0185, still under 0.0200. The budget got TIGHTER against
the gate, not looser. `EXTERNAL_PHASE_REFERENCE_PREDECLARATION` records the
new terms and names both boards.

**The split does not hide the board mismatch, and the mismatch got worse.**
The E4 raw cross-solver phase witness compares the two solvers' de-embedded
angles directly and is untouched at 3.0 deg. On the re-solved fixture it
reads 0.5308 deg against the pre-re-solve 0.3418 deg: the margin fell from
8.8x to 5.7x. That is where the 300-vs-250 um disagreement shows, and it is
gated.

**Still held, and owed:** `_build_stage_b_thru` still puts openEMS's strip
PROUD on 300 um, and the assertion this section asks for (dielectric height
under the strip equal on both sides) is NOT added -- adding it would red the
lane before anyone can run it, and changing the build would leave the
committed `_20_msl_phase_referee_logs/*_result.json` describing a board the
script no longer builds. Both wait on the cv20 owner submitting the Stage B
re-run.

Bookkeeping the split forced, all of it the same principle -- a beta array
is judged against the board that produced it:

* the committed run-1/run-2 artifacts carry the PRE-#931 fixture's
  `beta_rfx_real` (one zero-thickness wall on the full 300 um substrate),
  so their rfx leg stays at 300 um. Judging it at 250 um reads 0.23 %
  instead of 0.94 % -- a better-looking number for a board that run never
  had. `run1_declared_board` and `run2_realized_board` in
  `regate_evidence.json` are unchanged by this commit, which is the check
  that the split was applied in the right places.
* the header test's replay harness fakes openEMS with run-2 data but feeds
  the CURRENT fixture, so its three legs are neither block's. The artifact
  gains `run2_openems_with_current_rfx_fixture` for it.
* the #830 signed-envelope replay also reads run-2's `beta_rfx_real`; at
  250 um that beta falls inside the envelope and the G1 finding disappears
  by arithmetic. It stays at 300 um.

**One inconsistency left standing on purpose, with the measurement that says
it is safe to leave.** The #830 SIGNED beta-envelope leg (reported, not
gated) still builds its envelope and its `eps_eff` from
`h_sub_realized_m` = 300 um, so inside one artifact the gated E2 leg now
judges rfx's beta at 250 um while the reported signed leg judges the same
array at 300 um. Moving it was measured rather than argued, on the current
fixture:

| h | admissible band | rfx signed deviation | verdict |
|---|---|---|---|
| 300 um (as shipped) | [-0.017176, +0.005624] | [0.019908, 0.021306] | above hi by 0.015682 |
| 250 um | [-0.018157, +0.005483] | [0.012733, 0.014122] | above hi by 0.008639 |

The finding does not flip: rfx is above hi either way, and the excess
exceeds that board's own half-cell rasterization band either way (0.004954
at 300 um, 0.005861 at 250 um), so the attribution sentence is the same
too. The excess roughly halves. #830 is an open question with its own
committed record and its own owner, so the leg is left where it is and this
table is the input a future owner needs; it is not a defect this branch
should decide.

---

## 3. tmtt beam-steer — ingested as a record; nothing to commit

Run **369367259206** (gpu-a6000-1, SMOKE=0, rc 0). Its only produced file is
`validation/tmtt_paper/beam_steering_superstrate.png`, which `.gitignore` line 64
(`**/*.png`) excludes, so no file is committed. The numbers, against this
script's own pre-change baseline run (harvest
`/root/workspace/claude-workspace/rfx/runs/issue931-baseline-tmtt-beam-steer-20260907T055211Z`,
same image and preset, rc 0; the harvest carries no run id) rather than against
the paper headline:

| quantity | baseline | post-#931 |
|---|---|---|
| bare plate-backed dipole D(30 deg) | +5.86 dBi | +5.86 dBi |
| wedge init D(30 deg) | +3.73 dBi | +3.81 dBi |
| optimized D(30 deg) | +11.47 dBi | **+11.55 dBi** |
| realized peak | +11.61 dBi at 27.5 deg | +11.58 dBi at **30.0 deg** |
| broadside | -3.73 dBi | **-10.01 dBi** |
| reflector realized wall plane | (not printed) | `[32]` at 109.9239 mm = declared `plate_z` |

Pre-declared expectation was "the plate's realized wall plane is unchanged, so
the dipole is still lambda/4 above the metal; expect the numbers to move
slightly; a large move is a finding". +0.08 dB on the headline and an unchanged
bare reference is the small move. The peak landing exactly on the 30 deg target
and the deeper broadside null are the aperture correction (31 -> 30 cells =
149.896 mm = the declared 1.5 lambda) doing what it was supposed to.

The 9.5 / 9.45 dBi in the module docstring are the T-MTT paper's published
numbers for a different configuration, not this script's baseline; comparing the
post run against them would have manufactured a 2 dB "finding" that does not
exist.

---

## 4. tmtt msl-stub — rc = 1 twice, classified

Runs **369367259205** (post-rebase, 12:40 UTC) and its pre-rebase twin, both
rc = 1, both failing the same two gates:

    G1  Adam cost decrease >= 0.3 dB:      0.12 dB  (FAIL)   baseline 5.58 dB
    G3  imperative notch depth <= -15 dB:  -0.0 dB  (FAIL)   baseline -44.7 dB
    G2, G4 PASS on both.

**Not environment.** The pre-change baseline run of the same script, same image,
same preset, same machine, hours earlier: overall PASS, imperative notch
-44.7 dB at 6.129 GHz.

**Not a gate to loosen.** The observable is gone, not out of tolerance. Over the
whole brute-force scan L = 4 -> 12 mm the AD arm's |S21|^2 only moves 1.96e-2 ->
4.0e-2 (-17.1 to -14.0 dB): a 0.12 dB dip where the baseline had a 5.6 dB one.
Widening G1/G3 to cover that would be pinning "there is no notch" as a pass.

**So: a realization regression in this lane's own redraw (sheet feed line +
volume stub), and it needs an investigation with a named falsifier before any
re-solve** (R2, RF/EM intensifier: one clean attempt per mechanism hypothesis).

Two things already measured, so the next attempt does not repeat them:

* **The junction is NOT disconnected.** The hypothesis that the stub's root sits
  one cell off the realized sheet was tested at build time (no solve) on the
  imperative board at L_opt = 7.271 mm: the Ex rows at the trace-x midline run
  j = 20 -> 82 with **no gap greater than one cell**, y 1.524 -> 9.398 mm
  continuous. Refuted; do not re-try it.
* **The passivity warning is not the new thing.** "80 of 80 frequency bins
  non-passive as extracted" fires in the baseline too (worst sigma_max 1.006,
  vs 1.002 post). Red herring.

What it needs, in order, none of it a re-solve:

1. Dump the imperative arm's |S21|(f) across the band. The gate reads
   `argmin(db_imp)` only; a band-flat |S21| ~ 1 after passivity projection is an
   extraction signature, not a notch measurement, and the two cannot be told
   apart from one number.
2. Run this file's own `assert_soft_pec_equals_hard` at the **failing** L
   (7.271 mm). It is hard-coded to 7.0 mm, so it currently certifies that the two
   arms build the same metal at a length the run never used.
3. Only then declare a falsifier and re-solve.

Context that the investigation should carry: the realized through-line trace is
5 node rows (635 um by `n_rows*dx`, 508 um by node span) against a declared
600 um, and the run's own preflight flags both conductor Boxes off-lattice
(stub x-extent residual 44 um = 7.33%, trace y-extent 35 um = 5.83%) — the
gcd(254, 600) = 2 um problem this lane already recorded as not redrawable.

---

## 5. cv23 — the control that is deliberately not committed

Run **369367259202**, rc 0, all gates PASS. Produced vs committed
`validation/crossval/_23_lossy_results/rfx.json`: max rel diff **2.038e-11**
(`arms/tand0p1/lattice/W_lat_R[121]`, 3.4047885075771944e-06 ->
3.4047885076465834e-06), and **no key added or removed by the producer**. The
only schema difference runs the other way: the committed file carries a
hand-added `tail.fit_note` recording a post-hoc envelope refit, which the
produced file has no reason to write. Replacing it would delete that provenance
to buy a 2e-11 change, so nothing is committed for cv23.

For the record, the same comparison on the other two controls (both committed,
for producer-added schema keys only): cv22 max rel diff 8.505e-16, cv24 4.2e-14
on the mode frequencies.
