# SPEC-02 portgrid M0+M1 — measurement results (append-only)

Lane: `agent/portgrid-m0m1` · Tracker: #781 · Pre-declaration: `portgrid_m0m1_predeclaration.md`
(windows frozen at commit `0b91ec0`; correction 1 — F-M1b gating geometry, windows unchanged —
at `e000020`, before the F-M1b measurement ran). Date: 2026-08-29 (KST).

All runs: CPU JAX 0.10.2, f64, `PYTHONPATH=<worktree> ~/Documents/rfx/.venv/bin/python`.
Raw JSON: `portgrid_m1a_energy_audit.json`, `portgrid_m1b_reflection.json`,
`portgrid_m1b_diag_island12.json` (this directory).

## Verdict summary

| Falsifier | Window | Measured (worst case) | Verdict |
|---|---|---|---|
| F-M0-a adjoint residual | ≤ 1e-13 | all (r,m,ℓ) combinations pass (32 param cases + vjp) | PASS |
| F-M0-b supply-rate ≡ 0 | ≤ 1e-13 | 64 random draws × 12 cases pass | PASS |
| F-M0-c certificate vs paper conditions | see note §6 | dt_max_cert ≥ classical CFL; PD/indefinite crossing at 2/s_max; σ<0 flagged; B=L(LᵀB) exact | PASS |
| F-M1a 10⁶-step energy non-growth | ≤ +1e-8 | r=4: +1.7e-15; r=5: +3.9e-15 | PASS |
| F-M1b interface reflection | ≤ −45 dB @[2,20] GHz AND ≤ −35 dB @[2,30] GHz | r=3: −44.4 / −33.4 dB; r=6: −43.7 / −32.6 dB (r=2: −45.9 / −34.9) | **FIRED** |
| F-M1-grad AD vs central FD | ≤ 1e-6 | 1.54e-10 (h=2e-5; sweep 7.5e-9 / 1.5e-10 / 1.1e-9) | PASS |
| F-M1-vjp P-adjoint reverse structure | ≤ 1e-12 | jacrev ≡ jacfwd on full step; interface block = cb/r replication exactly | PASS |

Test battery: `validation/research/portgrid/` — 58 passed
(`pytest validation/research/portgrid -o addopts="" -q`).

## F-M1a detail (script `m1_energy_audit.py`, 10⁶ steps each arm)

| arm | dt | n_off | E_ref | max rel growth after source-off | max |drift| | wall |
|---|---|---|---|---|---|---|
| r=4 (paper-exact) | 7.384e-13 s | 4116 | 4.9667e-10 J/m | +1.665e-15 | 4.16e-15 | 36.8 s |
| r=5 (odd lane)   | 5.907e-13 s | 5145 | 7.7604e-10 J/m | +3.864e-15 | 3.86e-15 | 50.2 s |

Conservation is at f64 round-off level over 10⁶ steps — consistent with the exact-arithmetic
conservation prediction (Theorem 1 + lossless interconnect (63)) and far inside the +1e-8
window. CPU wall time ≈ 40–50 s per arm → no VESSL yaml needed for M1 (pre-declared 20-min
rule not triggered).

## F-M1b detail (script `m1_reflection.py`) — FIRED, with cause analysis

Declared fixture (700 mm guide, 20×20 mm island, coarse Δ = 1 mm):

| r | max|S11| [2,20] GHz (win −45 dB) | max|S11| [2,30] GHz (win −35 dB) |
|---|---|---|
| 2 | −45.87 (pass) | −34.92 (miss by 0.08) |
| 3 | −44.44 (miss by 0.56) | −33.43 (miss by 1.57) |
| 4 | −43.99 | −32.95 |
| 5 | −43.79 | −32.75 |
| 6 | −43.69 | −32.64 |

The window is violated; per the burned-data discipline the verdict is FIRED and the window is
NOT adjusted. Post-verdict diagnostics (new measurements, windows untouched):

1. **Chain null**: r=1 through the identical measurement chain gives max|S11| = −295.9 dB
   (max time-domain difference 3.0e-16) — the measurement pipeline contributes nothing.
2. **Spectral shape**: (r=3) −79 dB @2 GHz, −50 dB @10 GHz, −45 dB @20 GHz, −44 dB @24 GHz —
   matching the paper's Fig. 9 bottom-panel class — then a confined feature at 26–28 GHz
   peaking −33.4 dB @27.8 GHz, falling back to −40 dB @29–30 GHz. The violation lives
   entirely in the 26–30 GHz band, where the coarse grid is 10–11.5 cells/λ.
3. **Island-size arms**: 12×12 mm island: peak −30.3 dB @29.9 GHz; 30×20 mm: −30.6 dB
   @28.5 GHz — the high-band feature is not perimeter-proportional (perimeter hypothesis
   rejected); its magnitude ~−30…−33 dB near 10 cells/λ regardless of island size.
4. **Resolution scaling**: same physical fixture at coarse Δ = 0.5 mm (20 cells/λ @30 GHz),
   r=3: band max −45.3 dB, @28 GHz −45.5 dB, @20 GHz −57.6 dB — a uniform ≈ −12 dB drop per
   mesh halving, i.e. amplitude ∝ (Δ/λ)². This is exactly the second-order
   dispersion-mismatch scattering the scheme is expected to have; no anomalous convergence.
5. **Window provenance defect**: the paper's Fig. 9 bottom panel is axis-clipped at −40 dB;
   near 30 GHz the printed curve touches the axis top, so the paper's true 26–30 GHz values
   are not readable below −40 dB. The pre-declared −35 dB @30 GHz boundary treated the
   clipped −40 dB reading as data with a 5 dB allowance; measurements show the scheme's real
   band-edge (10 cells/λ) reflection class is ≈ −33 dB in this fixture. r-dependence is
   nearly flat (0.7 dB from r=3 to r=6), matching the paper's near-overlapping curves.

**Interpretation offered to review (not a re-judgment):** stability, losslessness,
convergence order, and AD structure all behave exactly as the papers predict; the fired
falsifier localizes to a window-calibration error at the band edge (figure-derived number
below the figure's own axis clip, and no cells-per-wavelength normalization). A re-derived
window (e.g. paper-comparable claim restricted to ≥ 11.5 cells/λ ⇔ ≤ 26 GHz at Δ = 1 mm, or
thresholds anchored to the measured (Δ/λ)² law) would require fresh pre-declaration and
fresh data per SPEC-00; that decision belongs to the adversarial reviewer / PI, not this lane.

## F-M1-grad / F-M1-vjp detail

- g_AD = −0.279058865214…; central FD sweep rel. mismatch {7.5e-9, 1.54e-10, 1.1e-9} for
  h/θ ∈ {1e-4, 1e-5, 1e-6} → min 1.54e-10 (window 1e-6; also inside the spec's M3a 1e-8
  class). θ = block εr inside the fine island with a face on the interface row; the
  parameter flows through the interface ε̂ coefficients (eq. (58)) as well as fine interior
  updates.
- jacrev(step) ≡ jacfwd(step) to ≤ 1e-12·scale on the full tiny-state Jacobian; the forward
  interface block d(coarse ifc Ex)/d(fine boundary Hz) equals cb·(segment mean) and its
  reverse counterpart is the cb/r replication — the P-adjoint of the averaging, as predicted
  by T_c2f = P_f⁻¹T_f2cᵀP_c.

## Limitations recorded

- Fine islands must be strictly interior (≥1 coarse cell of host on every side); a
  boundary-touching island aliases the interface update onto PEC rows and behaves as a
  near-total barrier (measured −0.4 dB "reflection"). Guarded with a ValueError +
  regression test. Multi-island, odd-anisotropic ratios: M4 scope.
- The 2-D prototype accepts any integer r ≥ 1 (2-D paper rule); the 3-D odd-only rule is
  carried by `require_odd` and binds at M2.
- dt remains 0.99 × fine CFL (explicit non-goal here; dt relief is SPEC-03).

## PR #90 cross-reference (spec M1 requirement)

- Source area scaling: M1 sources live on the coarse grid only; the fine/coarse cell-area
  ratio r² never enters. The PR #90 failure mode (raw per-cell pulse re-used across
  resolutions) is structurally absent; the note in the pre-declaration binds any future
  fine-region source (scale by cell area).
- Interface signs: derived once from the continuous curl updates and verified three
  independent ways (r=1 null test at 1e-12, 10⁶-step energy audit at 1e-15, certificate).
  No sign tuning was needed at any face — unlike SBP-SAT penalty terms.
- No boundary-derivative staggering-scale ambiguity arises: the half-cell factors come
  mechanically from eqs. (50)/(52).

## Correction 2 (2026-08-29, append-only — adversarial-review blocking finding)

The F-M1b cause analysis above is WITHDRAWN. It rested on a misreading of
arXiv:1606.08761 Fig. 9 (bottom panel): the claim "axis-clipped at -40 dB;
true 26-30 GHz values are not readable" is factually wrong. Vector-data
extraction of the figure PDF (review, independently performed) shows the
y-axis spans -110..-30 dB (frame top -30.0 dB; -40 dB is merely the topmost
tick LABEL) and every curve is fully readable: the curves end at
-34.28 dB (r=6), -34.62 dB (r=4), -36.5 dB (r=2) at 30 GHz, and the worst
curve's maximum over [2, 20] GHz is -51.23 dB.

Consequences, recorded for the next lane's design (the FROZEN verdict here
is untouched — F-M1b FIRED as declared and stays FIRED; no window moves):

1. The pre-declaration's own "-35 dB [2,30]" window violated its own
   ">= 5 dB above the worst Fig.-9 curve" rule (it sits 0.7 dB BELOW the
   paper's worst curve). The window was mis-derived at declaration time.
2. Under a correctly derived paper-class window (worst curve + 5 dB:
   [2,30] GHz ~ -29.3 dB; [2,20] GHz ~ -46.2 dB), the measured band-edge
   values (-32.6..-33.4 dB) would PASS, but the measured [2,20] GHz values
   (-43.7..-45.9 dB) would still FIRE for every r.
3. Therefore the real paper-class discrepancy is NOT a band-edge
   calibration defect, and the previously proposed remedy (restrict to
   >= 11.5 cells/lambda) is withdrawn. The discrepancy is a ~6-7 dB-higher
   MID-BAND reflection plateau vs the paper, most plausibly a
   fixture-class difference: this lane measured a 20x20 mm island at
   normal incidence in a PEC/time-gated box with the probe 80 mm from the
   face, vs the paper's four-rod scatterer enclosed by the subgrid with
   PML termination.
4. The next attempt's fresh pre-declaration must adjudicate
   scheme-vs-fixture with a paper-faithful fixture (rod-class scatterer
   enclosed by the subgrid; absorbing termination or a derivation of the
   PEC/time-gating equivalence) and windows derived from the CORRECT
   figure reading via the >= 5 dB rule.

## Correction 3 (2026-08-29, append-only — test-count prose slip, reviewer nb)

The line "Test battery: `validation/research/portgrid/` — 58 passed" above conflates
two states of the branch. At the commit whose measurements this note reports
(`e000020`, where F-M1a and F-M1b ran) the battery collected **57** tests; the 58th
(`test_island_must_be_strictly_interior`) landed only WITH this results note's own
commit (`638a3a5`), as part of the boundary-touching-island guard recorded under
"Limitations". Verified by re-collecting at both commits (57 / 58). The 58-test
battery is the correct post-guard state; no measured number changes.

## F-M1b RETRY results (2026-08-29, append-only) — paper-faithful fixture

Pre-declaration: `portgrid_m1b_retry_predeclaration.md` (windows frozen at `3415e37`;
Correction R1 — floor-arm gate ray derivation, window unchanged — at `1da7745`, before
the floor arm ran). Implementation at `f4e7aaf`. All runs CPU JAX f64,
`m1b_retry.py`; raw JSON: `portgrid_m1b_retry_{floor,null,interface,rods}.json`.

### Verdict summary

| Falsifier | Window (frozen) | Measured (worst case) | Verdict |
|---|---|---|---|
| F-M1b-abc (PML floor) | ≤ −50 dB on [2,30] GHz, dt(r=2) and dt(r=6) | −94.0 dB (both) | PASS |
| chain null r=1 (full retry chain) | ≤ −200 dB | −306.9 dB (time-domain max diff 1.3e-16) | PASS |
| **F-M1b-r2 (primary, interface-only)** | [2,20] ≤ −46.24 dB AND [2,30] ≤ −29.29 dB, every r ∈ {2,3,4,5,6} | worst r=6: −56.44 / −40.44 dB | **PASS (all r)** |
| F-M1b-rod (secondary, r=6 vs all-fine) | linear mismatch ≤ 0.0941 over [2,30] GHz | 0.0299 | PASS |

Per-r interface-only detail (windows −46.24 / −29.29 dB):

| r | max dB [2,20] GHz | max dB [2,30] GHz |
|---|---|---|
| 2 | −58.67 | −42.64 |
| 3 | −57.21 | −41.20 |
| 4 | −56.75 | −40.75 |
| 5 | −56.55 | −40.55 |
| 6 | −56.44 | −40.44 |

Rod-arm context (non-falsifier, linear mismatch vs our all-fine r=6, same dt):
r=2: 0.0663, r=4: 0.0285, r=6: 0.0299, all-coarse: 0.0437; max linear |S11_allfine| =
0.207 (−13.7 dB). Paper-extracted classes for comparison: r=2/4/6: 0.062/0.063/0.053,
all-coarse 0.156. Our all-coarse mismatch is smaller than the paper's class — note the
paper ran all-coarse at the COARSE grid's own CFL while ours shares dt(r=6), and its rod
staircasing at 1 mm is not specified in the paper; recorded as context only.

### Interpretation (verdict under frozen windows; no re-judgment of phase 1)

- **F-M1b retry PASSES for every r with ≥ 10 dB margin** — on the paper-faithful fixture
  the measured interface-only reflections (−56.4..−58.7 dB over [2,20] GHz;
  −40.4..−42.6 dB band max) sit BELOW the paper's own extracted curves (−51.2..−53.5 and
  −34.3..−36.5 dB). The scheme, as implemented from eqs. (55)/(56)/(58)/(61), meets the
  paper's reported interface-reflection class outright.
- Correction 2's fixture-class hypothesis is thereby CONFIRMED with the fixture excuse
  removed in the other direction: the phase-1 FIRE (which stands, as declared) was a
  property of the phase-1 fixture (single-point Hz probe recording non-TEM scattered
  modes — the parallel-plate n=2 mode propagates above 7.5 GHz — plus PEC/time-gated box
  and a 20×20 mm island at broadside), not of the scheme. The retry's TEM-projection
  probe (y-averaged Ey column, the paper's probe-line reading) and PML termination
  reproduce the paper's class.
- r-dependence is again nearly flat (2.2 dB spread over r=2..6), matching the paper's
  near-overlapping curves; the odd ratios r=3,5 interpolate the even ones smoothly.
- **M1 is recorded COMPLETE**: F-M1a (energy, roundoff), F-M1b (retry, paper-faithful
  fixture, PASS), F-M1-grad / F-M1-vjp (AD contract) all hold; the phase-1 F-M1b FIRE
  remains on the record as an honest fixture-sensitivity finding with its diagnostics.

### Regressions after the material/PML additions (`f4e7aaf`)

- Battery: 65 passed (58 prior + 7 new wiring falsifiers), `-o addopts="" -q`.
- F-M1a 10⁶-step arms re-run post-change: see `portgrid_m1a_energy_audit_regression.json`
  (r=4 and r=5 must stay ≤ +1e-8; values recorded below).
- dx-scaling diagnostic now COMMITTED as `m1_reflection.py --scale N` (reviewer nb);
  `--scale 2 --ratios 3` reproduces the phase-1 Δ=0.5 mm arm: band max [2,30] GHz
  −45.31 dB (phase-1 ad-hoc run recorded −45.3), [2,20] GHz max −56.38 dB
  (`portgrid_m1b_diag_scale2.json`).
- Material-path falsifiers (pre-declared §4): vacuum-maps-vs-default exact (≤1e-14);
  r=1 lossy island ≡ uniform lossy Yee (≤1e-12); Sec. V-B-class lossy traverse across
  the interface: energy monotone non-increasing within +1e-13·E_ref after source-off
  (σ̂ terms of (61) active), with real dissipation observed.

### F-M1a regression numbers (post-`f4e7aaf` re-run, 10⁶ steps each)

| arm | max rel growth after source-off | max |drift| | wall | verdict (win +1e-8) |
|---|---|---|---|---|
| r=4 | +1.457e-15 | 4.16e-15 | 56.5 s | PASS |
| r=5 | +3.864e-15 | 3.86e-15 | 80.3 s | PASS |

Identical roundoff class as the phase-1 values (+1.67e-15 / +3.86e-15): the material/PML
additions did not perturb the verified lossless path (its default code path is unchanged;
only the energy-sum association differs at the 1-ulp level).

## Correction 4 (2026-08-30, append-only) — the retry's observable is a TEM projection: interface claims narrowed to the witness, absolute cross-check recorded, remaining review items landed

Adversarial-review BLOCKING finding, accepted without re-litigation. Pre-declaration of the
measurements quoted here: `portgrid_m1b_retry_predeclaration.md` **Correction R3** (committed
`5c50017`, before any of them ran). Raw JSON: `portgrid_m1b_retry_xcheck.json`; paper-side
provenance `portgrid_fig9_allfine_absolute.json` (same instrument, same tick calibration as
the frozen `portgrid_fig9_extraction.json`, which re-runs byte-identical).

### 4.1 What stands, and what is withdrawn

**Stands.** The F-M1b-r2 verdict (PASS at every r under the frozen −46.24 / −29.29 dB
windows) is legitimate and is NOT reopened. The pre-declaration declared the observable —
"the y-AVERAGE of Ey on the x = 19 mm column" — before measuring, the implementation honours
it exactly (`sim2d.py` lines 575 / 627, `jnp.mean(ey_new[probe_col, :])`), and not one
measured number changes. The rod, floor and null verdicts likewise stand.

**Withdrawn.** Every claim that read that verdict as a statement about the paper, or about
the scheme in general. The y-average is the TEM (n = 0) modal projection: with PEC plates at
y = 0, H it annihilates every cos(nπy/H), n ≥ 1. That is exactly right for the INCIDENT
field a y-uniform source launches, and wrong for the REFLECTED field, which an interface or a
scatterer populates with higher-order content. |S11| built on it is a TEM→TEM reflection
coefficient, and the pre-declaration's §2 reasoning ("projects out every cos(nπy/H) mode …
leaving the TEM mode") silently applied the incident-field argument to the reflected field.

### 4.2 Claim-by-claim replacement

**(a) Line 197, WITHDRAWN:** "…sit BELOW the paper's own extracted curves (−51.2..−53.5 and
−34.3..−36.5 dB). The scheme … meets the paper's reported interface-reflection class
outright."

Replacement: *Under this lane's declared TEM-projected observable the interface arm PASSES
the frozen F-M1b-r2 windows at every r ∈ {2,3,4,5,6} (worst r = 6: −56.44 / −40.44 dB against
−46.24 / −29.29 dB). Read on the SAME runs through a point probe the same arm gives:*

| r | y-mean (declared observable) | point, guide centre line y = H/2 | point, y ≈ H/4 |
|---|---|---|---|
| 2 | −58.67 / −42.64 | −49.86 / −34.05 | −54.94 / −36.70 |
| 4 | −56.75 / −40.75 | −47.93 / −32.15 | −53.00 / −34.79 |
| 6 | −56.44 / −40.44 | −47.62 / −31.84 | −52.69 / −34.46 |

(max dB over [2,20] GHz / over [2,30] GHz; frozen windows −46.24 / −29.29; paper's worst
extracted curve −51.24 / −34.29.)

*The centre-line readings are still INSIDE the frozen windows, but by 1.4 dB and 2.6 dB at
r = 6 instead of ≈ 10 dB, and they are ABOVE the paper's worst extracted curve by 3.6 dB and
2.5 dB. Choosing the projection moves this fixture's |S11| by up to 8.8 dB at fixed geometry.
No claim that our interface reflection sits below the paper's is supported by this lane. What
is supported: the arm passes its frozen windows under the observable it declared, and passes
them under a centre-line point probe as well, with the margin reduced to 1.4 / 2.6 dB.*

**(b) Lines 205–206, WITHDRAWN:** "The retry's TEM-projection probe (y-averaged Ey column,
the paper's probe-line reading) and PML termination reproduce the paper's class."

Two defects: it asserts the paper's projection ("the paper's probe-line reading"), which
cannot be read off Fig. 8 — the figure shows a probe LINE, not what is done with it — and
§4.3 below shows the y-average does NOT reproduce the paper's absolute class.

Replacement: *The retry fixture differs from phase 1 in TWO classes at once — geometry
(paper-faithful guide, island, PML termination) and observable (TEM projection vs phase 1's
single-point Hz probe). This lane measured the pair jointly and did not decompose them. The
observable contribution is now measured separately and is up to 8.8 dB at fixed geometry —
several times the 1.3 / 2.4 dB by which phase 1 exceeded its own windows. Correction 2's
fixture-class hypothesis therefore remains supported in DIRECTION, but this lane does not
establish how much of the phase-1 FIRE was geometry; the phase-1 FIRE stands as recorded and
is not converted into an exoneration.*

**(c) PR body, WITHDRAWN:** "paper-class interface reflection" and "the scheme is
exonerated". Replacement text for the PR body (this lane cannot write to GitHub; recorded
here for the PI to paste):

> F-M1b retry: the interface arm PASSES the frozen windows (−46.24 / −29.29 dB) at every
> r ∈ {2,3,4,5,6} **under this lane's declared TEM-projected observable** (y-mean of the Ey
> probe column). Read through a centre-line point probe on the identical runs it still
> passes, by 1.4 / 2.6 dB at r = 6, and sits above the paper's worst extracted curve. An
> absolute cross-check against the paper's own all-fine curve shows our y-averaged |S11| is
> 5.5–6.1 dB below the paper's for a run containing no subgridding at all, so the paper's
> reported |S11| is not a TEM projection and our numbers are not directly comparable to it.
> No exoneration of the scheme is claimed: what is claimed is that the frozen falsifier did
> not fire under the observable it was declared with.

### 4.3 F-M1b-abs — all-fine vs paper, ABSOLUTE (the check that would have caught this)

Our all-fine arm is a uniform r = 6 grid with the paper's four copper rods and **no
subgridding anywhere** — plain Yee. The paper's Fig. 9 top-panel all-fine curve is the same
quantity for the same fixture. Pure Yee against pure Yee carries no scheme content, so any
gap is fixture, observable or extraction. Anchors (dB), band [2, 29.8] GHz as extracted:

| curve | max | @10 GHz | @25 GHz | @29 GHz |
|---|---|---|---|---|
| paper all-fine (Fig. 9 top) | −8.03 | −18.42 | −10.94 | −8.78 |
| ours, y-mean (declared) | −13.67 | −24.50 | −16.41 | −14.59 |
| ours, point y ≈ H/4 | −7.86 | −24.51 | −10.78 | −9.19 |
| ours, point y = H/2 | −4.22 | −19.30 | −8.48 | −6.67 |

- Under the y-mean every anchor is **5.5–6.1 dB below the paper** (−5.64 / −6.08 / −5.47 /
  −5.81), outside the lane's ±5 dB class rule, for a run with no subgridding in it.
- The incident (reference) trace at the probe column is y-uniform to **1.0–1.3e-16**
  relative, so the |S11| DENOMINATOR is identical under every projection. The entire gap is
  reflected-field content the y-average discards.
- **Supported conclusion: the paper's reported |S11| is not a TEM (y-averaged) projection.**
- **Not supported, and not claimed: what the paper's projection is.** No single point row
  matches all four anchors inside the class — the centre line is the only projection inside
  at every anchor (worst |Δ| 3.81 dB), while y ≈ H/4 matches max/25/29 GHz to within 0.4 dB
  and is 6.09 dB low at 10 GHz.
- Physical self-consistency check of the modal reading, unplanned and worth recording: at
  10 GHz only n = 0 and n = 2 propagate (cutoffs n·c/2H = n·3.75 GHz) and, by the fixture's
  symmetry about y = H/2, only even n are excited; cos(2πy/H) has a NODE at y = H/4. The
  y ≈ H/4 probe therefore degenerates to the TEM projection at 10 GHz — measured −24.51 vs
  the y-mean's −24.50, agreeing to 0.01 dB. The n = 2 mode is exactly what the y-average
  throws away.

**Verdict authority of this arm: NONE in this lane.** Per Correction R3(b) it is BURNED — the
review reported its outcome before its window was written. It is implemented, committed and
re-runnable so that M2 and any re-run inherit an unburned fixture-fidelity gate, and so the
check lives in the repository instead of in a review comment.

### 4.4 F-M1b-rod window: the transplanted absolute number (Correction R3(c))

`m1b_retry.py:62 WIN_ROD_LINEAR = 0.0941` was formed as `0.0529 × 10^(5/20)`, i.e. the
paper's ABSOLUTE linear mismatch plus 5 dB. The "+5 dB rule" is a RELATIVE class allowance,
and the paper's curves peak at 0.3968 while ours (y-mean) peak at 0.2073, so the frozen
window delivers `20 log10(0.3968/0.2073)` = **5.64 dB more allowance than the rule intends**
— a factor 1.91 looser. The rescaling rule was declared in Correction R3(c) before the
rescaled value was computed:

| quantity | value |
|---|---|
| frozen window (keeps verdict authority) | 0.0941 |
| looseness vs the +5 dB rule | 5.64 dB (×1.91) |
| rescaled window, R3(c) rule | 0.0491 |
| measured r = 6 mismatch (UNCHANGED, read from `portgrid_m1b_retry_rods.json`) | 0.0299 |

The measurement is untouched; only the window it is compared against is re-expressed, and it
meets the tighter one too (by 4.3 dB). Both numbers are now written into `m1b_retry.py` with
the caveat in the source, and `rescaled_rod_window()` computes the tighter one from committed
provenance rather than a hardcoded constant.

### 4.5 Open limitation carried to M2 (named, not repaired here)

F-M1b-rod's verdict, like F-M1b-r2's, was taken under the y-mean observable, and this lane
did NOT re-take it under a point probe — that measurement was not pre-declared in R3 and is
not run here. Any M2 or re-run that quotes the rod arm must either re-declare and re-take it
under an explicitly chosen projection, or repeat this qualification.

### 4.6 Remaining review items

- **F-M0-c was a one-sided gate.** Every prior F-M0-c assertion reads `dt_cert ≥ <classical
  CFL>` ("the certificate can only be looser"), so a mutation that merely GROWS `dt_cert`
  fires nothing: `certificate.py:77 d_lpy[:, 0] = dy/2 → dy` leaves the M0 battery **51
  passed, 0 failed**. Repaired two-sided by
  `test_portgrid_m0.py::test_region_dual_cells_tile_the_area_exactly`, a geometric identity
  (Yee dual cells tile the region exactly once, so Σ primary·dual length per E-family = the
  region area exactly, which forces the boundary dual edge to be exactly half the interior
  one). Verified to fire on all three parametrizations for `dy/2 → dy` AND for `dy/2 → dy/4`;
  clean tree 54 passed.
- **"reverse block = P-adjoint replication" was over-stated.** The results line above
  ("…its reverse counterpart is the cb/r replication — the P-adjoint of the averaging, as
  predicted by T_c2f = P_f⁻¹T_f2cᵀP_c") is witnessed at that strength only for the STANDALONE
  operator, by `test_portgrid_m0.py::test_jax_vjp_matches_p_adjoint`, which checks
  `vjp(T_c2f·x) ≡ P_c·T_f2c·P_f⁻¹·w` on the matrix pair. What
  `test_portgrid_m1.py::test_vjp_p_adjoint_structure_of_step` witnesses for the STEPPER is
  narrower: `jacrev(step) ≡ jacfwd(step)` to ≤ 1e-12·scale, plus the FORWARD interface
  coefficient d(coarse Ex)/d(fine boundary Hz) = cb/r. Narrowed claim: *the stepper's reverse
  pass agrees with its forward pass to 1e-12 and the forward interface block carries the
  analytic cb/r weight; the exact P-adjoint identity is witnessed on the standalone operator.*
- **No CI workflow runs this battery.** `pyproject.toml:77 testpaths = ["tests"]`, and both
  workflows that run pytest either pass an explicit `tests/…` file list
  (`.github/workflows/pr-tests.yml`) or invoke bare `pytest` with the same testpaths
  (`pr-tests.yml` fast-suite job, `.github/workflows/validation.yml`). `validation/research/`
  is collected by nobody. Consequence, recorded plainly: the `-40 dB` battery-level PML floor
  gate added at `8281c79`, and the new two-sided F-M0-c gate, guard nothing automatically —
  they fire only when someone runs `pytest validation/research/portgrid -o addopts=""` by
  hand. Wiring this directory into CI is out of this lane's scope (it changes a shared
  workflow); it is named here so no one reads "gate added" as "gate enforced".
- **The 2022 Corrections could not be verified from this host.** §6 of the retry
  pre-declaration pre-seeds an M2 checklist from IEEE TAP 70(4):3132 (2022) Corrections. That
  document was NOT re-obtained in this lane — the preprint URL refused connection from here,
  and no PDF of it is in the worktree. The checklist items C1–C6 are therefore carried as
  UNVERIFIED-FROM-SOURCE and bind M2 only; C6 in particular ("the 2-D pieces are not affected
  by the Corrections") is an inference this lane could not check against the Corrections text
  and must be re-derived by M2 from the document itself before any 2-D claim leans on it.
  Nothing in M0/M1's measured record depends on it.

### 4.7 Discipline record

No frozen window was widened, narrowed or moved. The rod-window rescaling is a strictly
tighter re-expression declared before evaluation, and the frozen window keeps verdict
authority. Every new measurement in this correction was declared in `Correction R3` at
commit `5c50017`, before the run. Design notes remain append-only: this section adds; it
edits nothing above it. The interface arm was re-run after the `probe_full` addition and
reproduces the committed JSON bit-identically.
