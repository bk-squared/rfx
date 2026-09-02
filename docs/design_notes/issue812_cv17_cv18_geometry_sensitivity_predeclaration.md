# issue #812 — cv17 / cv18 re-gate: gate windows derived from geometry sensitivity

Status: **PRE-DECLARATION.** Every numeric window below is fixed in this commit,
which PRECEDES the measurements that judge it. Branch `agent/regate-mie-iris`.
Date: 2026-09-01. Lane: `geometry-sensitivity` (issue #812, post-Phase-1).

Scope: `validation/crossval/17_dielectric_sphere_mie.py` +
`tests/crossval/test_rcs_dielectric_sphere_mie_gates.py`, and
`validation/crossval/18_wr90_iris_modematch.py` +
`tests/crossval/test_wr90_iris_modematch_gates.py`. **Nothing in cv05 / cv09 / cv10 /
cv02 / cv14 (Phase 0/1, PRs #814-#818) is touched.** cv16's
translation-invariance finding is physics and is filed as #820; it is not
touched here.

This is INSTRUMENT work. No physics verdict of either case is challenged.

---

## 0. The two audit findings, restated as the defect each gate must catch

* **cv17** — the case exists to make the first cross-method record of rfx's
  *binary dielectric rasterize* path at `eps_r = 2.56`, yet **no gate anywhere
  in the case looks at the permittivity that was actually rasterized**. The
  audit measured the dB gate passing for a permittivity wrong by a factor.
* **cv18** — the case is declared the calibrated prerequisite for every
  downstream multi-iris filter, yet a **one-cell aperture error** is inside
  both gated windows.

Both are *resolution* findings: the question is what error in the geometry /
material parameter the gated observable can still resolve.

---

## 1. cv17 — sensitivity of the observable to the permittivity

### 1.1 Measured sensitivity (oracle only, no FDTD)

Bohren-Huffman dielectric-sphere backscatter, `m = sqrt(eps_r)`, evaluated at
the four gated coarse bins, differentiated at `eps_r = 2.56`:

| ka | sigma/(pi a^2) | d(sigma_dB) / d(eps/eps) |
|---:|---:|---:|
| 0.50 | 0.026584 | 9.816 dB |
| 0.75 | 0.11468  | **10.202 dB** |
| 1.00 | 0.25819  | 9.978 dB |
| 1.25 | 0.30542  | 7.134 dB |

(computed with the frozen gate test's own independent Mie re-implementation,
`tests/test_rcs_dielectric_sphere_mie_gates._mie_backscatter_over_pi_a2`.)

### 1.2 What the present 6.3 dB gate therefore polices

Holding the committed `rfx_sigma_over_pi_a2` rows fixed and moving the declared
permittivity (so the oracle moves and the run does not), the widest `eps_r`
that still satisfies `|delta| <= 6.3 dB` on **every** row of each gate's own
row population:

* script gate (4 `gated_coarse` rows): `eps_r` in **[1.601, 3.867]**
* frozen gate (40 rows: `gated_coarse` + `domain_realizations` 30/40 +
  `clearance_scan`): `eps_r` in **[1.684, 3.787]**

i.e. the dB channel resolves the permittivity only to about a **factor of
1.5 either way**. The audit reported [1.988, 5.622] and [2.054, 4.212] for the
same two legs; those exact edges do **not** reproduce under this
oracle-side-only model. The qualitative finding (a factor-wide window)
reproduces; the edges are re-measured here and the audit's edges are not
carried forward. Neither number is in a committed file, so no committed
document is corrected by this.

### 1.3 The dB channel cannot be tightened, and cannot police eps

`GATE_COARSE_DB = 6.3` is already `round-up(measured envelope 4.181 x 1.5)` at
the case's 0.1 dB quantum, so the repo's own gate rule pins it; the only
tightening available is a quantum change (6.3 -> 6.28 at quantum=100), which
moves the eps window by <0.5% and is **declined** as churn. **The dB observable
genuinely cannot resolve a permittivity error smaller than tens of percent.
This is stated in the claim, not papered over.**

### 1.4 Pre-declared new gates (cv17)

The permittivity gets a channel of its own, the material twin of the existing
per-row geometric gate `A_EFF_TOL_COARSE` (which already checks the *geometry*
claim separately from the dB tolerance "so a_eff can never silently absorb a
real rasterization regression").

* **G17-A `EPS_REALIZED_TOL = 0.005`** (relative), asserted on the permittivity
  read back out of the rasterized grid on every gated row.
  *Derivation (sensitivity):* half of the gate's own reporting quantum
  (0.1 dB, the PR #475 round-up convention) divided by the worst gated-ka
  sensitivity from 1.1: `0.05 / 10.202 = 0.0049` -> **0.005**. Any permittivity
  error above this can move the recorded dB envelope by more than the case can
  report; anything below it is beneath the case's own resolution.
* **G17-B binary-material structure**, no tolerance: the rasterized `eps_r`
  array must take **exactly two** distinct values, background `1.0` and the
  declared `EPS_R`. *Derivation:* structural, from the case's headline scope
  claim ("the BINARY rasterize dielectric interface — no sub-cell averaging
  exists"). Sub-cell averaging, a wrong material value, or a partially filled
  sphere each introduce a third value. This claim is presently prose only and
  is checked by nothing.

**Expected (A):** on today's code both gates pass on the declared material by
construction, with the realized value exact to float32 round-trip
(~1e-7 relative, i.e. ~4 orders inside G17-A).

**Expected (B):** a run whose rasterized permittivity is set to the upper edge
of the window in 1.2 must **fail G17-A** while the 6.3 dB gate still **passes**
— that is the whole finding, demonstrated in one run. The edge is re-measured
live rather than assumed, because 1.2 holds the FDTD side fixed and a real run
at the wrong material moves it.

---

## 2. cv18 — sensitivity of the observable to the aperture

### 2.1 Measured sensitivity (mode-matching oracle, no FDTD)

`max` over the committed 29-point band of `| |S11|(d+step) - |S11|(d) |`,
WR-90, t = 1.524 mm, from the frozen gate test's independent oracle:

| d (mm) | \|S11\| range | +1 fine (0.381) | -1 fine | +1 coarse (0.762) | -1 coarse |
|---:|---|---:|---:|---:|---:|
| 18.288 | 0.095-0.233 | 0.0325 | 0.0347 | 0.0622 | 0.0714 |
| 12.192 | 0.526-0.856 | 0.0436 | 0.0439 | 0.0865 | 0.0891 |
| 7.620  | 0.950-0.992 | **0.0168** | **0.0132** | 0.0376 | 0.0236 |

The strong aperture is the *least* sensitive because `|S11| -> 1` saturates:
the magnitude-only lane posture is what costs the resolution there.

### 2.2 The audit's defect reproduces exactly, and its mechanism is now named

Model the defect on the committed rows by shifting the oracle by the realized
aperture error (`rfx_defect(rung) = rfx_committed(rung) + [orc(d_realized) -
orc(d)]`). Two classes:

* **M1, rung-local**: aperture one *fine* cell off at the fine rung only.
* **M2, dx-proportional**: aperture one cell off **at each rung** (fine
  `d + 0.381`, coarse `d + 0.762`) — the recurrence of the campaign's own
  setup defect (3) ("the fin footprint made the ELECTRICAL aperture d + 2*dx")
  at half its former size.

At `d = 7.620`, `+1` cell, M2 gives **fine 0.0097 -> 0.0265** (gate 0.04) and
**Richardson 0.0010 -> 0.0030** (gate 0.01) — **bit-for-bit the audit's two
numbers.** The audit's defect is M2, and its mechanism is now stated:

> **A dx-proportional geometry error is first-order in dx, which is exactly
> what `2*S(a/60) - S(a/30)` is built to remove. The Richardson witness
> cancels it by construction.** M2 Richardson deviations across all 8 configs
> land at 0.0018-0.0069, all inside the 0.01 gate, whose own envelope is
> 0.0051 — **no tightening of the Richardson gate can ever catch this class**,
> and it is not attempted. M1, which does not scale with dx, moves Richardson
> to 0.0255-0.0922 and is caught today with >=2.5x margin.

### 2.3 What the fine gate can be tightened to

The pooled gate `GATE_FINE_ABS = 0.04 = round-up(pooled envelope 0.0232 x 1.5)`
is set by the *worst* of 8 configurations and spent at all 8. Applying the
repo's own `gate = round-up(envelope x 1.5)` rule **per configuration**, at
quantum 1000 (0.001; precedent: `tests/unit/sparams/test_msl_port_integration.py`), using
each configuration's own committed `max_gap_abs` as its envelope:

| d (mm) | glen | frac | committed gap | pre-declared gate |
|---:|---:|---:|---:|---:|
| 18.288 | 0.20 | 0.50 | 0.0122 | **0.019** |
| 12.192 | 0.20 | 0.50 | 0.0223 | **0.034** |
| 7.620  | 0.20 | 0.50 | 0.0097 | **0.015** |
| 18.288 | 0.20 | 0.42 | 0.0145 | **0.022** |
| 12.192 | 0.20 | 0.42 | 0.0232 | **0.035** |
| 7.620  | 0.20 | 0.42 | 0.0097 | **0.015** |
| 12.192 | 0.16 | 0.50 | 0.0222 | **0.034** |
| 12.192 | 0.24 | 0.50 | 0.0222 | **0.034** |

The committed rows are prior provenance (they are already the source of the
pooled gate); the gates judge future runs, not those rows. `GATE_FINE_ABS =
0.04` is **retained unchanged** — no existing gate is widened; the per-config
gate is strictly tighter and becomes the binding one.

### 2.4 Pre-declared detection table (this is the claim, and it is gated)

M2 fine gap vs the per-config gate, modeled as in 2.2:

| d (mm) | frac/glen | gate | M2 `+1` | detect | M2 `-1` | detect |
|---:|---|---:|---:|:--|---:|:--|
| 18.288 | 0.50 | 0.019 | 0.0446 | yes 2.35x | 0.0225 | yes 1.18x |
| 12.192 | 0.50 | 0.034 | 0.0648 | yes 1.91x | 0.0269 | **no** |
| 7.620  | 0.50 | 0.015 | 0.0265 | yes 1.77x | 0.0035 | **no** |
| 18.288 | 0.42 | 0.022 | 0.0458 | yes 2.08x | 0.0230 | yes 1.05x |
| 12.192 | 0.42 | 0.035 | 0.0649 | yes 1.85x | 0.0270 | **no** |
| 7.620  | 0.42 | 0.015 | 0.0265 | yes 1.77x | 0.0036 | **no** |
| 12.192 | 0.16 | 0.034 | 0.0647 | yes 1.90x | 0.0269 | **no** |
| 12.192 | 0.24 | 0.034 | 0.0647 | yes 1.90x | 0.0268 | **no** |

**Declared resolution, and the re-scope:** the fine |S11| gate detects a
one-cell **over**-aperture at 8/8 configurations with margin >= 1.77x, and a
one-cell **under**-aperture at only 2/8 (both at the weak aperture, margins
1.18x / 1.05x). At d = 12.192 and d = 7.620 a one-cell under-aperture is
**below the fine rung's own first-order discretization error** — at d = 7.620
the committed fine trace is *closer* to the oracle at `d - 1 fine cell`
(0.0035) than to the oracle at the declared d (0.0097), i.e. the rung's
staircase error is itself worth about -0.6 to -1 cell of effective aperture,
exactly the "half-cell effective-aperture ambiguity" the script's own raster
comment names. **No legal gate can resolve it, and the claim will say so
instead of implying the calibration is aperture-exact.**

> **CORRECTION (issue #812 round 2, 2026-09-01), scoped to the sentence above
> and to nothing else in this section.** The table, the declared resolution
> (8/8 over, 2/8 under) and the conclusion are re-derived and unchanged — they
> are now committed in
> `validation/crossval/_18_wr90_iris_results/aperture_resolution.json` and
> re-derived from the committed traces by an independent oracle in the gate
> test. What was wrong is the explanation: the trace is *farther* from the
> oracle at `d - 1 fine cell` than at the declared `d` at all eight
> configurations, and its nearest oracle on the declared offset grid is at
> `d + 0.5` fine cells everywhere. The 0.0035 quoted above is the M2 `-1`
> DEFECT metric from this table's own column, not a distance; the
> half-cell effective-aperture ambiguity the raster comment names is a WIDE
> half cell, which is precisely why a one-cell narrowing cancels against it.
> See §5.2's correction block.

### 2.5 Pre-declared new gates (cv18)

* **G18-A** per-configuration fine gates, table in 2.3, enforced live in the
  script and on every committed row in the frozen test; the `--write-fixture`
  self-check extends its EXACT `round-up(x1.5)` equality demand to them.
* **G18-B** the detection table of 2.4 is itself gated: the frozen test
  re-derives each M2 entry from the committed rows and the independent oracle
  and asserts detection exactly where 2.4 declares it — so the resolution
  claim cannot silently degrade, and a future regeneration that loses
  detection goes red instead of quietly re-scoping.
* **G18-C** the declared aperture set is pinned: every committed row's `d_mm`
  must lie in `{18.288, 12.192, 7.620}` **and** be an exact integer number of
  cells at *both* rungs (`d / (a/30)` and `d / (a/60)` integral, and even, the
  symmetric-fin parity condition). *Derivation:* geometric, from
  `a = 22.86 mm` and the two declared rungs. This closes the one class no
  numeric gate can see — a silent one-cell **relabel** of the aperture, where
  the oracle follows the wrong d and every residual stays nominal. A one-fine-
  cell relabel (7.620 -> 8.001) is not an integer number of coarse cells and
  fails with zero tolerance.

**Expected (A):** all 8 committed configurations pass their per-config gate
with >= 33% headroom (that is the x1.5 rule), and the live re-run on today's
code reproduces the committed gaps.

**Expected (B):** the audit's own defect — aperture one cell wider at each
rung, d = 7.620, oracle and record at the declared 7.620 — must FAIL G18-A at
0.0265 > 0.015 and must be shown to have PASSED the 0.04 pooled gate. To be
demonstrated with a real FDTD pair (fine aperture 8.001 mm, coarse aperture
8.382 mm, one fin displaced by one cell at each rung — the only way the
symmetric-fin lattice can express a one-cell aperture error), not only with
the model of 2.2. The Richardson leg must be shown still passing, for the
reason named in 2.2.

### 2.6 Recorded for the review: what already catches M2

`run_point`'s raster assert `len(open_y) == d_c - 1` (and its frozen twin
`test_operating_point_is_grid_exact_on_every_row`) **does** fire on M2 arising
as a fin-footprint bug, because the realized aperture cell count then stops
matching `round(d_phys/dx)`. The audit's claim is about the two numeric gates
and is accurate as stated; the case is not defenceless against that defect
class. What the asserts cannot see is 2.5's G18-C class (`d_phys` itself
wrong), and what no gate could see is the resolution limit of 2.4.

---

## 3. Discipline

Windows above come from: the two analytic oracles (first principles), the
guide/grid geometry, and the repo's own committed `round-up(envelope x 1.5)`
provenance. None is fitted to a number measured after this commit. No existing
gate is widened. This note is append-only; a later correction states the old
value and why it was wrong.

---

## 4. CORRECTION to section 1.2 (2026-09-01, written after the measurement)

**What section 1.2 said, and it is wrong.** It published, as the `eps_r`
window the present 6.3 dB gate polices, **[1.601, 3.867]** (script leg) and
**[1.684, 3.787]** (frozen leg), and said the audit's **[1.988, 5.622]** /
[2.054, 4.212] "do not reproduce". **My two numbers are wrong and the audit's
script-leg window is right.**

**Why it was wrong.** The model behind 1.2 held the FDTD side fixed and moved
only the oracle. That is not what the defect does: a run at a different
permittivity changes the FDTD result too, and above 2.56 the staircase error
moves in the direction that *cancels* part of the oracle shift, so the real
window is much wider than the oracle-only estimate on the high side.

**Re-measured live** (four gated coarse bins per point, `rasterize` handed the
wrong material while the declared 2.56 drives the oracle and the operating
point; `max |delta_db|` over the four bins against the 6.3 dB gate):

| eps_r | 1.2 | 1.4 | 1.6 | 1.8 | **2.0** | 2.2 | **2.56** | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 | **5.5** | 6.0 | 7.0 | 8.0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| max\|delta\| dB | 18.87 | 13.12 | 9.90 | 7.70 | **6.07** | 4.80 | **3.08** | 1.63 | 2.68 | 3.66 | 4.41 | 5.52 | **5.50** | 8.55 | 11.64 | 11.90 |
| verdict | FAIL | FAIL | FAIL | FAIL | **PASS** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | FAIL | FAIL | FAIL |

so the blind window is bracketed at **(1.8, 6.0)**, passing at 2.0 and 5.5 —
the audit's [1.988, 5.622] sits inside that bracket and is corroborated, not
refuted. The finding is *worse* than 1.2 made it look: the case's dB gate
accepts a permittivity more than double the declared one.

**Consequence for the pre-declared gates: none.** `EPS_REALIZED_TOL = 0.005`
was derived in 1.4 from the oracle sensitivity and the gate's own reporting
quantum, not from this window. The window only fixes where criterion (B)'s
"edge of the current tolerance" sits, which is now **measured** at
`eps_r = 5.5` (upper) and `2.0` (lower) instead of assumed.

---

## 5. Measured results (2026-09-01, on `agent/regate-mie-iris`)

### 5.1 cv17

**(A) — today's code, declared material.** `17_dielectric_sphere_mie.py`
gated set, live: `delta_db` = -1.49 / -0.16 / -3.08 / +0.29 dB at
ka = 0.50/0.75/1.00/1.25 against the 6.3 dB gate (worst 3.08, margin 2.04x);
`a_eff/a` = 0.9880/0.9880/0.9893/0.9954 against `A_EFF_TOL_COARSE` 1.5%;
realized `eps_r` = float32(2.56) = 2.5599999428, **2.2e-8 relative** against
`EPS_REALIZED_TOL` 0.005 (a factor 2.2e5 of margin), **2 distinct values** in
the rasterized array at every bin. `RESULT: ALL CHECKS PASSED`, exit 0.

**(B) — the defect, at both measured edges of the blind window.** `rasterize`
made to deliver `eps_r = 5.5` (and, separately, `2.0`) where 2.56 was
declared:

* the **dB gate PASSES on every one of the four gated bins** (max \|delta\|
  5.50 dB at 5.5, 6.07 dB at 2.0, both inside 6.3) — the audit's finding,
  reproduced live on today's code;
* the **material gate FAILS on every bin, for the right reason**, printing
  `MATERIAL FAIL (realized eps_r 5.5, 2 distinct values; declared 2.56, tol
  0.005 rel, 2 values)`; script prints `SOME CHECKS FAILED` and exits **1**.

The structural half (G17-B) is falsified separately in
`tests/crossval/test_rcs_dielectric_sphere_mie_gates.py`: an array carrying one
averaged interface value passes G17-A (its max is still 2.56) and is rejected
by G17-B.

> **CORRECTION (issue #812 round 2, 2026-09-01) — provenance, not physics.**
> The four dB magnitudes above (`5.50` / `6.07` / `7.70` / `8.55`) are live
> FDTD measurements that no committed artifact carries, so they have been
> withdrawn from cv17's `claim_scope` and from the script and test comments.
> The window itself is now committed in
> `validation/crossval/_17_dielectric_results/material_blind_window.json`,
> built with no FDTD from the committed `gated_coarse` deltas plus the Mie
> oracle (`scripts/diagnostics/build_cv17_material_blind_window.py --check`)
> and re-derived against an independent Mie leg in
> `test_material_blind_window_is_rederived_from_the_committed_deltas` in `tests/crossval/test_rcs_dielectric_sphere_mie_gates.py`.
> The model returns the SAME pass/fail verdict as the live runs at all four
> permittivities (`summary.blind_window_bracket_eps`,
> `summary.first_failing_eps_below`, `summary.first_failing_eps_above`), so
> the conclusion is unchanged and now has a source. The live magnitudes stay
> here, in this note, as dated corroboration.

### 5.2 cv18

**(A) — today's code, through the live gates.** `18_wr90_iris_modematch.py`
(no flags), full gated set, 16 FDTD runs:

| d (mm) | glen | frac | fine gap | per-config gate | coarse gap | Richardson |
|---:|---:|---:|---:|---:|---:|---:|
| 18.288 | 0.20 | 0.50 | 0.0122 | 0.019 | 0.020 | 0.004 |
| 12.192 | 0.20 | 0.50 | 0.0223 | 0.034 | 0.041 | 0.005 |
| 7.620  | 0.20 | 0.50 | 0.0097 | 0.015 | 0.018 | 0.001 |
| 18.288 | 0.20 | 0.42 | 0.0145 | 0.022 | 0.026 | 0.004 |
| 12.192 | 0.20 | 0.42 | 0.0232 | 0.035 | 0.043 | 0.005 |
| 7.620  | 0.20 | 0.42 | 0.0097 | 0.015 | 0.018 | 0.001 |
| 12.192 | 0.16 | 0.50 | 0.0222 | 0.034 | 0.041 | 0.005 |
| 12.192 | 0.24 | 0.50 | 0.0222 | 0.034 | 0.040 | 0.005 |

`RESULT: ALL CHECKS PASSED`, exit 0. **Every one of the sixteen gaps
reproduces the committed 2026-07-28 fixture value exactly** (an independent
rerun harness measured the same sixteen before the script run, with the same
numbers), so the fixture is still the record of today's code and the
tightened gates are satisfied by the same physics that set them, with the
x1.5 headroom the rule guarantees.

**(B) — the audit's defect, run for real.** `d = 7.620 mm`, upper fin drawn
one cell short at each rung (the symmetric two-fin lattice can only realise
an EVEN aperture cell count, so a one-cell error is necessarily asymmetric),
with `d_phys`, the record and the oracle all at the declared 7.620 mm:

* the defect is realised as intended — rasterized aperture **20 cells at the
  fine rung (nominal 19) and 10 at the coarse (nominal 9)**, i.e. one cell too
  wide at each rung, dx-proportional;
* measured **fine gap 0.02842**: **PASSES the pre-#812 pooled gate 0.04** —
  the audit's finding, reproduced live on today's code — and **FAILS the new
  per-config gate 0.015** by a factor 1.89;
* measured **Richardson deviation 0.00588: PASSES the 0.01 gate**, exactly as
  section 2.2 predicts and for the stated reason. This is left standing and
  documented, not papered over.
* Through the live gate path (the eight configurations replayed from the
  fixture with this measured pair substituted at d = 7.620), the script
  prints `d= 7.62 glen=0.20 frac=0.50: max|dS11|=0.0284 gate 0.015 ... FAIL`
  and **only** that row, then `RESULT: SOME CHECKS FAILED`, exit **1**.

The first-order model of section 2.2 predicted 0.0265 / 0.0030 for this
defect against the measured 0.02842 / 0.00588 — same class, 7% apart on the
gated leg, which is the accuracy that model is claimed at and the reason the
committed detection table in `test_one_cell_aperture_resolution_is_declared_
and_pinned` is a *model* pinned by a real measurement rather than the reverse.

**What is NOT closed, and is stated in the claim instead:** a one-cell
UNDER-aperture. Detection margins are 1.18x and 1.05x at d = 18.288 (below
the repo's own 1.5x) and there is no detection at all at d = 12.192 and
d = 7.620 -- at the strong aperture the fine rung's own first-order staircase
error is already worth about -0.6 to -1 cell of effective aperture, so the
committed trace sits CLOSER to the oracle one cell narrow (0.0035) than to
the declared one (0.0097). No legal gate can resolve it and none pretends to.

> **CORRECTION (issue #812 round 2, 2026-09-01) — the paragraph immediately
> above is SIGN-INVERTED and MIS-SOURCED; it is left in place per the
> append-only rule, and this block is the record.** The 0.0035 it quotes is
> not a distance from the committed trace to the narrow oracle; it is the
> one-cell UNDER-aperture DEFECT metric
> (`max_f |(fine + [oracle(d-dx) - oracle(d)]) - oracle(d)|`), which carries
> the oracle shift with the opposite sign. Re-derived from the committed
> traces, the fine rung's effective aperture is WIDER than nominal, not
> narrower: over the declared offset grid the trace's nearest oracle sits at
> `d + 0.5` fine cells at all eight configurations, and the distance to the
> oracle one fine cell NARROW is larger than the distance at the declared `d`
> at every configuration. That is why the under-aperture defect is invisible —
> narrowing the geometry by a cell moves it toward the measured trace. Every
> number in this class now lives in
> `validation/crossval/_18_wr90_iris_results/aperture_resolution.json`
> (`pairs[*].oracle_distance_abs`, `pairs[*].nearest_offset_fine_cells`,
> `pairs[*].one_cell_defect.under.*`, `summary.*`), is rebuilt by
> `scripts/diagnostics/build_cv18_aperture_resolution.py --check`, and is
> re-derived from the committed traces by an independent oracle in
> `test_aperture_resolution_artifact_is_rederived_from_committed_traces` in `tests/crossval/test_wr90_iris_modematch_gates.py`.
> The detection COUNTS and the conclusion the section draws (an over-aperture
> is resolved everywhere, an under-aperture nowhere with margin) are unchanged
> and re-verified; only the explanation's sign and its source were wrong.
>
> The live-FDTD (B) numbers in section 5.2 above (`0.02842` / `0.00588`) are
> likewise prose-only: no artifact carries them. They are NOT relied on by any
> gate — criterion (B) is carried by the model table, which is committed and
> mechanically checked — and a VESSL job to re-emit them as a committed JSON
> is reported with this round's hand-off. Until it lands, treat those two
> digits as unprovenanced corroboration.

### 5.3 What was NOT done, and why

* **No fixture regeneration.** Both fixtures are frozen 2026-07-27/28 evidence
  and cv18's reproduces value-for-value on today's code; regenerating would
  re-freeze the evidence record (a much larger claim than a re-gate) and, for
  cv17, would land the deferred issue-#725 `a_eff` convention change in the
  same commit. The cv17 material fields are therefore gated live in the script
  and guarded forward on the frozen leg.
* **The cv18 Richardson gate was not tightened.** 0.01 is already the smallest
  legal value at its quantum (envelope 0.0051), it already detects a
  rung-LOCAL one-cell error with >= 2.5x margin, and it is blind to the
  dx-proportional class by construction — a quantum change would buy margin on
  a channel that has it and nothing on the class that matters.
* **The cv17 dB gate was not tightened.** 6.3 dB is round-up(4.181 x 1.5); the
  only move available is the quantum (6.3 -> 6.28), which shifts the blind eps
  window by under 0.5%.
* **cv16 was not touched.** Its translation-invariance finding is physics
  (#820). This lane's cv17 changes are in the same file family (`rfx/rcs.py`
  is read by both cases) but touch no extraction path: the material gate reads
  the rasterized `eps_r` array before the solve and changes no computation.

## 6. Round-2 numeric-provenance ledger (2026-09-01)

Every quantity this lane asserts in a durable document, and where it now lives:

| claim | source of record |
|---|---|
| cv18 per-configuration fine gates | `validation/crossval/_18_wr90_iris_results/aperture_resolution.json::pairs[0].fine_gate_abs = 0.019` … `validation/crossval/_18_wr90_iris_results/aperture_resolution.json::pairs[2].fine_gate_abs = 0.015` (eight entries, identical to the per-configuration dict in `validation/crossval/_18_wr90_iris_results/rfx.json` under gates), bound to the script constant and to `round-up(env x 1.5)` in the gate test |
| cv18 one-cell detection counts, margins, Richardson blindness | `validation/crossval/_18_wr90_iris_results/aperture_resolution.json::summary.over_aperture_detected = 8` and the sibling `summary` keys, and each pair's `one_cell_defect` block |
| cv18 effective aperture of the fine rung | `validation/crossval/_18_wr90_iris_results/aperture_resolution.json::summary.nearest_offset_fine_cells_values[0] = 0.5` and each pair's `oracle_distance_abs` (this is the quantity §2.4 and §5.2 got sign-inverted) |
| cv17 permittivity sensitivity (dB per unit relative eps) | re-derived in `test_the_declared_permittivity_sensitivity_is_the_one_recorded` in `tests/crossval/test_rcs_dielectric_sphere_mie_gates.py` and bound to `claim_scope` |
| cv17 dB-gate permittivity blind window | `validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[0][0] = 2.0`–`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[0][1] = 4.5` and `validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[1][0] = 5.0`–`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[1][1] = 5.6`, the island `validation/crossval/_17_dielectric_results/material_blind_window.json::summary.fail_islands_eps[0].eps_r[0] = 4.6`–`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.fail_islands_eps[0].eps_r[3] = 4.9` (worst bin `validation/crossval/_17_dielectric_results/material_blind_window.json::summary.fail_islands_eps[0].max_abs_delta_db = 9.787` dB), and the `scan` rows |
| cv18 live-FDTD (B) run (`0.02842` / `0.00588`) | **prose only, §5.2** — not gate-bearing; criterion (B) is carried by the committed model table. `scripts/diagnostics/probe_cv18_one_cell_aperture_defect.py` re-runs that pair and writes `one_cell_defect_live.json` (it exits 1 unless the defect passes the pooled 0.04 gate and fails the per-config 0.015 one); it is submitted by a VESSL yaml reported to the orchestrator (`scripts/vessl_issue812_r2_cv17_cv18.yaml` in the worktree; `**/vessl*.yaml` is gitignored by repo convention, so the yaml is a hand-off artifact, not a commit). Its geometry half is verified locally with `--geometry-only`: aperture 20 nodes at the fine rung and 10 at the coarse, against nominal 19 / 9. |
| cv17 live-FDTD defect runs (four dB magnitudes) | **prose only, §5.1** — not gate-bearing; the window is carried by `material_blind_window.json`. |

Both builders are deterministic, read only committed artifacts, run no FDTD,
and support `--check` (rebuild and diff, exit 1 on any drift).
The same VESSL yaml re-runs criterion (A) for both cases through their live
gates and re-checks both builders on the staged tree, so a green (A) cannot sit on
a drifted artifact. No leg passes `--write-fixture`: the frozen evidence record is
not regenerated by this round (§5.3 still holds).

## 7. Round-2 review (2026-09-02). Four blockers; no gate, window or measurement moved.

An independent reviewer re-derived every number in §§0–6 against the artifacts (the 40
cv18 oracle distances, the 16-entry detection table, the per-configuration gates, the
cv17 sensitivities and the blind-window grid all agreed to the printed digits),
confirmed with git that every window was fixed in `a2a14d1` before the commit that
measured it and that no committed measurement row was regenerated, and returned
FIX-THEN-SHIP. Append-only; the one in-place edit is named in 7.5.

**7.1 (B1) The cv17 blind window was reported as one contiguous bracket the model does
not have.** The declared grid (…, 4.0, 5.0, 5.5, …) stepped over a FAIL island: at eps
4.6–4.9 the ka = 1.25 bin sits on a Mie resonance and the model's worst bin reaches
`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.fail_islands_eps[0].max_abs_delta_db = 9.787` dB, so "[2.0, 5.5]" was
self-consistent on the grid and wrong as described (the #829 note's class 3). The grid
now carries 0.1 steps from 4.0 to 5.6 and the artifact reports the PASS SET:
`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[0][0] = 2.0`–`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[0][1] = 4.5` (the
run containing the declared 2.56) and
`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[1][0] = 5.0`–`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.pass_runs_eps[1][1] = 5.6`, with
the island `validation/crossval/_17_dielectric_results/material_blind_window.json::summary.fail_islands_eps[0].eps_r[0] = 4.6`–`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.fail_islands_eps[0].eps_r[3] = 4.9`
between them. The physical conclusion survives — the dB channel is blind to a
factor-wide permittivity error, `validation/crossval/_17_dielectric_results/material_blind_window.json::summary.blind_window_over_material_gate_x = 237.5`
over the whole pass set — but the word "bracket" is withdrawn wherever it stood (script
docstring and gate comment, the shared claim_scope, manifest, README, public row). No
live defect point sits in the island: the live runs of §5.1 were at 1.8 / 2.0 / 5.5 /
6.0 and agree with the model's verdicts there; the model–live magnitude gap already
seen at 4.5 (live 4.41 dB vs model 5.31 dB) says the island's *edges* are model-only.
The run that settles them is the cv17 defect harness at eps 4.6 / 4.7 / 4.8 / 4.9 on
the gated bins; it is owed, not run.

**7.2 (B2) §4's diagnosis of §1.2 was itself wrong.** §4 said §1.2's [1.601, 3.867] /
[1.684, 3.787] came from "holding the FDTD side fixed and moving only the oracle". The
committed builder makes exactly that assumption and gives [1.988, 4.556] (outer edge 5.62
through the island) for the 4-row leg and [2.054, 4.212] for the 40-row leg — which are
also the audit's own numbers. Flipping the sign of the Mie shift reproduces §1.2
exactly. So §1.2 was a **sign-inverted model** — the same failure class this round exists
to close — and §4 misnamed the cause. Correct statement: §1.2 subtracted the oracle shift
where it should have added it; the artifact model is the sign-corrected §1.2; the live
runs corroborate it at the four probed permittivities.

**7.3 (B3, B4) Prose-only live digits and an untraceable edge.** The cv18 manifest
`claim_scope` carried the live fine gap 0.02842 with no artifact behind it (§6 already
said so); it stays until `one_cell_defect_live.json` from the VESSL job lands and is
cited by key (7.6). The cv17 manifest and public row said "measured live … (1.8, 6.0)"
for what is the model's first-failing grid neighbours; both now cite
`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.first_failing_eps_below = 1.8` and
`validation/crossval/_17_dielectric_results/material_blind_window.json::summary.first_failing_eps_above = 4.6` (the latter moved from 6.0 because the run
containing 2.56 now ends at 4.5) and name the live points separately. The script's
"roughly [2.0, 5.6]" (two sites) is replaced by the pass set; three different windows
were being printed across sites.

**7.4 Recorded, not changed.** The four cv17 sensitivities (9.816 / 10.202 / 9.978 /
7.134 dB per unit relative eps) are forward differences at h = 1e-3, not derivatives;
central differences read 9.827 / 10.213 / 9.990 / 7.150, and the tolerance derivation is
unaffected (0.05 / 10.213 = 0.0049 → 0.005). The live cv17 (A) deltas (−1.49 / −0.16 /
−3.08 / +0.29 dB) differ from the committed rows (−1.794 / −0.442 / −3.276 / +0.272) by
up to 0.35 dB because #731 moved the Mie evaluation to `ka_eff` without regenerating
the fixture; the blind-window model is built on the old-convention deltas while the live
defect runs use the new one — a convention mismatch, stated. Per-configuration cv18
gates are ×1.5 of a population of one row each (the pooled gate's population was eight);
they are accepted on the value-for-value fixture reproduction of §5.2 and named as such.
G17-A/B read the array `rasterize` returns, before `MaterialArrays`; a defect between
the array and the update coefficients stays visible only to the factor-wide dB channel,
which is adequate for the binary-rasterize claim as scoped.

**7.5 Citations, form only.** The note's ten backtick spans with a double colon were
`.py` test pointers, wildcard key paths, or artifact paths missing their
`validation/crossval/` prefix; none parsed under the #829 gate. They are rewritten in
place to scalar keys with values (this is the in-place edit; no sentence changed
meaning). `scripts/vessl_issue812_r2_cv17_cv18.yaml` was gitignored while §6 described
it as the hand-off; it is force-added, matching the other tracked job files, and staged
to local disk before running (a script writing beside itself on the read-only NFS
checkout cost another lane a finished solve). Its first attempt (VESSL 369367257706)
segfaulted at `import jax`: pinning `jax[cpu]==0.6.2` over the NVIDIA image leaves an
incompatible CUDA plugin installed; the plugin is now removed first.

### 7.6 The island, run live: the model over-predicts, the live window is wider

`scripts/diagnostics/probe_cv17_permittivity_island.py`, VESSL 369367257712 (remilab-c0,
CPU; log harvested, run deleted): the rasterizer delivers eps 4.5 … 5.0 while the oracle
stays at the declared 2.56, on the four gated coarse bins — the audit's defect, exactly as
round 1 built it, now inside the region 7.1 called model-only. Committed as
`validation/crossval/_17_dielectric_results/cv17_permittivity_island.json` and pinned by
`test_island_probe_records_that_the_live_window_is_wider_than_the_model`.

| eps | live worst bin (dB) | live | model worst bin (dB) | model |
|---:|---:|---|---:|---|
| 4.5 | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[0].max_abs_delta_db = 4.413` | PASS | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[0].model_max_abs_delta_db = 5.313` | PASS |
| 4.6 | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[1].max_abs_delta_db = 4.545` | PASS | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[1].model_max_abs_delta_db = 7.134` | FAIL |
| 4.7 | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[2].max_abs_delta_db = 5.152` | PASS | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[2].model_max_abs_delta_db = 9.039` | FAIL |
| 4.8 | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[3].max_abs_delta_db = 6.138` | PASS | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[3].model_max_abs_delta_db = 9.787` | FAIL |
| 4.9 | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[4].max_abs_delta_db = 6.405` | FAIL | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[4].model_max_abs_delta_db = 8.116` | FAIL |
| 5.0 | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[5].max_abs_delta_db = 5.523` | PASS | `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::runs[5].model_max_abs_delta_db = 5.299` | PASS |

`validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::summary.n_verdicts_agree_with_model = 3` of `validation/crossval/_17_dielectric_results/cv17_permittivity_island.json::summary.n_runs = 6`. The worst
live bin is the ka = 1.25 resonance bin in every island row, as the model said; its
*magnitude* is not what the model said. The model holds the solver's discretization
error fixed at the committed 2.56 value and moves only the oracle; on the resonance the
FDTD sphere's own response moves with the permittivity too, so the two partly cancel and
the live curve is shallower. First-order means first-order; 7.1's "the island's edges are
model-only" is now measured: the live island on this grid is the single point 4.9, and
4.8 passes 0.16 dB inside the gate.

What this changes: nothing in the gates, and the direction of the conclusion is
unchanged — the dB channel is *blinder* to a permittivity error than the model claimed,
which is the whole reason G17-A/B exist. What it withdraws: every sentence that said the
model "reproduces the live verdict at every probed permittivity" — it did so at round 1's
four points and does not inside the island; the shared `claim_scope`, the builder's
`method` string, the script docstring and gate comment, the manifest and the public row
now say so with the keys above. The model artifact is kept as the declared-grid model it
is, with its island; the live probe sits beside it as the measurement.

### 7.7 The re-verification job ran: (A) on both cases, (B) live on cv18, keyed

`scripts/vessl_issue812_r2_cv17_cv18.yaml`, VESSL 369367257708 (remilab-c0, 32 CPU;
log harvested, run deleted; try-1 369367257706 is 7.5). Both no-FDTD builders re-checked
green on the staged tree first. Logs and the live artifact are committed beside the
results they belong to.

* **cv17 (A)**: `17_dielectric_sphere_mie.py` through its live gates, exit 0
  (`validation/crossval/_17_dielectric_results/cv17_gated_A.log`).
* **cv18 (A)**: `18_wr90_iris_modematch.py`, all sixteen fine/coarse checks PASS,
  `RESULT: ALL CHECKS PASSED`, exit 0, every fine gap equal to the §5.2 table
  (`validation/crossval/_18_wr90_iris_results/cv18_gated_A.log`).
* **cv18 (B), live** — the number §6 called "prose only" now has a key:
  `validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json::measured.fine_gap_abs = 0.02842` PASSES the pre-#812 pooled gate
  (`validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json::config.pooled_fine_gate_abs = 0.04`) and the Richardson gate
  (`validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json::measured.richardson_dev_abs = 0.00588` against
  `validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json::config.richardson_gate_abs = 0.01`) — the measured blindness — and FAILS the
  per-configuration gate `validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json::config.fine_gate_abs_per_config = 0.015` by
  `validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json::measured.per_config_margin_x = 1.895`. The first-order model row beside it,
  `validation/crossval/_18_wr90_iris_results/aperture_resolution.json::pairs[2].one_cell_defect.over.fine_gap_abs = 0.0265`, is 7 % low on the gated
  leg and predicts the same three verdicts. The probe exits 1 unless exactly that
  pattern holds; it exited 0. Pinned by
  `test_live_one_cell_defect_is_caught_by_the_per_config_gate_and_not_the_old_ones`.
* **Gate-test leg**: exited 1 with "No module named pytest" — the image does not ship
  pytest and the job did not install it. A job-file defect, not evidence: the same two
  test files ran green on the branch locally and on CI. `pytest` is added to the job's
  install line for the record; the 2.5 h job is not rerun for a leg CI already covers.

The manifest's cv18 `claim_scope` cites the live keys instead of restating 0.02842.
Nothing in §5.2 changes; it acquires a source.
