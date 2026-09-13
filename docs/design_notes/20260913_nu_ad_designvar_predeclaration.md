# NU AD-Q second design: withdrawal and re-declaration

## Withdrawal before measurement

This note supersedes the AD6 declaration committed as `73316579`, which
remains in history. **The withdrawal happens before any measurement against
that declaration.** At withdrawal, HEAD is `733165791f417cd57797b767258a3ae0c28eb6d5`;
`git log 73316579..HEAD --oneline` is empty and a repository-wide search for
`"ad6"` JSON keys finds none. Uncommitted AD6 replay-test drafts found in the worktree were
discarded unrun: they tested the withdrawn design.

The PI rejected the following, quoted from the replacement request:

> “a flat 15 % gate is not a quantitative claim, and neither is the existing convention”

> “the directions must be the DESIGN VARIABLES, not abstract vectors”

Accordingly, tied indicators, all-ones and seeded random directions are
withdrawn. The replacement uses layer thickness and permittivity directions
supplied by a generalized E4 fixed-topology stackup map. The primary claim
will be Taylor-remainder order (R0 slope 0.9–1.1; R1 slope 1.8–2.2), with
central FD assessed against three times its own Richardson/float32 error
estimate. A reference bar exceeding 15 % is inconclusive. The old 5 %
dominance / 15 % agreement rule remains a labelled legacy smoke figure;
its constants elsewhere in the repository will not change.

This first commit records only the withdrawal and replacement intent.
The complete map, ladders, data-dependent window algorithm, error-bar
arithmetic, and one-attempt arms will be frozen in a subsequent commit
before any new measurement. No production `rfx/` behavior change is authorized.

## Results

No replacement measurement has run at the time of this withdrawal commit.

## Frozen replacement instrument (before measurement)

Arms are `stack_l1`, `stack_l2`, `resolution` and `revert_l1_nodt`, each once, separate JSONs
with import path and git SHA. No adaptive reruns. Reuse E4's 20/4/20/20
cells, 14/2/14/14 mm layers, eps 4.3/3/4.3/1, transverse box 30 x 3 mm,
0.5 mm transverse cells, sources/probes and waveform. L1 is E4's 120-step
transient probe energy; L2 is its 8000-step DFT power at the fixed analytic
nominal resonance (checkpoint 100). No flank observable or re-centering.

### Physical controls and metric

Four thickness controls, named h_core_left, h_thin, h_core_right, h_air,
are one-dimensional constrained physical layer edits at fixed 44 mm total
length. Increasing a core takes the same length from the other core;
increasing thin or air takes half from each core. Thus the thin control
is exactly E4's map, and each selected layer increases by its requested
physical displacement. With thickness deltas ordered left/thin/right/air,
the realized layer delta is A delta_h, with columns
(1,0,-1,0), (-1/2,1,-1/2,0), (-1,0,1,0), (-1/2,0,-1/2,1).
These are overlapping controls, not four independent actual thicknesses
at fixed total length (which is impossible); the thickness rank is three.
All four layer permittivities vary independently, including the nominal
air value. Each cell run is uniform; interfaces remain nodes. Dual-cell
node eps is recomputed inside JAX exactly as in E4.

Cell space is `(dz[64], eps_cell[64])`; the loss constructs node eps from
these via E4's dual average. This avoids confusing a material interface's
weight derivative with a geometric cell direction. Report each Jacobian
column in physical units, all nominal cell/node arrays, AD cell gradient,
and the design gradient both directly and as J-transpose-g. The geometric
part is constant on all four z minimum cells, and x/y directions are zero
on their uniform tied sets. Check spread exactly zero on all three sets;
this makes dt differentiable along the control despite its per-cell min
subgradient. Interface-node eps Jacobians are also saved.

Report ||projection_span(J) g|| / ||g|| with normalized columns and SVD
rank cutoff 1e-12, both all controls and controls passing BOTH new gates.
The requested raw cell norm mixes metres and relative eps: explicitly
label its coordinate dependence. Also report separate dz and eps-cell
block coverage and a dimensionless metric using nominal cell widths and
cell eps as coordinate scales. No minimum coverage is assumed.

### Taylor ladder and selector

Every control uses relative steps h = 2**(-k), k=3,...,17, evaluated in
ascending order. Absolute parameter displacement = h * nominal parameter.
R0 = abs(Lplus-L0), R1 = abs(Lplus-L0-h*g_relative). Save both signed
loss differences too. Use host float64 arithmetic on float32 loss outputs.
N = 32 loss quanta: a point is eligible if R1 > 32 times the maximum
float32 ulp of L0,Lplus,Lminus and the symmetric curvature increment
abs(Lplus+Lminus-2L0) <= 0.25*abs(Lplus-Lminus). This latter guard keeps
curvature small compared with the leading change without selecting on a
fitted slope. Choose the longest contiguous eligible run with strictly
increasing R1 as h increases, at least four points; ties choose the run
with the smaller starting h. Never search subwindows for a passing slope.
Fit log(R0), log(R1) against log(h), ordinary least squares with intercept;
save slopes, log residual vectors, RMS log residual, slope standard errors,
window, indices and count. Gate both slopes: R0 in [0.9,1.1], R1 in
[1.8,2.2]. No eligible run => INCONCLUSIVE; a fitted miss => FIRED.
The same rule is used unchanged for the resolution arm's Taylor checks.

### FD error estimate

For each h, D(h)=(Lplus-Lminus)/(2h) in relative-control units. Use the
adjacent finer step h/2: T(h)=4*abs(D(h)-D(h/2))/3, from D(h)=g+c*h^2.
At the finest endpoint use abs(D(2h)-D(h))/3. Roundoff allowance
Q(h)=(ulp(Lplus)+ulp(Lminus))/(2h), one ulp per computed loss; bar=B=Q+T.
Save Richardson derivative, Q,T,B in relative and physical derivative
units, B/abs(g), abs(g-D)/B and abs(g-D)/(3B) at EVERY step. These are
estimated FD errors, not rigorous solver forward-error bounds; accumulated
float32 arithmetic can exceed a single output ulp. A zero derivative or
3B/abs(g)>0.15 is INCONCLUSIVE (the factor-three acceptance band itself
must beat the legacy 15%). Otherwise abs(g-D)<=3B is HELD, else FIRED.
A direction's FD gate is FIRED if ANY informative step fires, HELD if
at least one is informative and none fires, otherwise INCONCLUSIVE.
Report the narrowest informative band as the compact table row, retaining
all steps. No selecting a favorable step to erase a fired one. Legacy
relative errors and 15%/sign figures remain labelled smoke only.

### Attempt-1 resolution arm and legacy tables

Recompute W7 A3 at 120 steps on all x/y/z cells for relative steps
(0.00025,0.0005,0.001,0.002,0.004,0.01,0.02,0.04). Report per-cell
AD, FD+, FD-, central FD, quantum=ulp(L0)/(2*h*d_k), quanta, dominant
non-tied indices (>0.05*max non-tied abs(FD)), worst error, its cell,
error in quanta, and worst resolved error (>=50 quanta). Both y and z
are adjudicated; x supplies legacy context. Historical attempt-1 and
second-attempt rows remain immutable. Reproduction comparisons at 0.001
are reported, not silently forced to match across compiler environments.
Taylor cross-checks use physical dilation of the complete y or z axis
(v_axis=nominal profile, other axes zero), on the same power-of-two ladder
and selector above. These are physical box-width/length controls,
constant on each axis's eight tied minima, not random vectors. Also fit
every non-tied y/z cell separately from its own recorded resolution
ladder (same eligibility rule; non-geometric ladder spacing is allowed).
A slope is a window property, not a pass at an individual h.

The FD-resolution explanation is supported if y/z attempt-1 worst cells
have <50 quanta, errors decrease at 0.01 and 0.04 compared with 0.001,
and both axis Taylor fits hold. If these conditions fail, report which;
call it unconfirmed rather than attributing all disagreement to AD.
Report per-cell Taylor counts too; axis dilation alone does not validate
every non-tied component. For tied cells label the limiting relation
`g_ad = FD+ + (FD- - FD+)/n_tied`; finite-h curvature/roundoff can perturb
it. A per-cell optimizer breaking a tie cannot treat the equal split as
its unique directional derivative. Historical n_tied=8 each axis and
(gmax_all,gmax_nontied)=(991.528,10.0445), (585.090,0.779206),
(946.350,0.312869) are retained with the newly measured tables.

### Replay and tree checks

Replay reconstructs arithmetic, fits, directions and coverage without
rerunning FDTD; pins every fired/inconclusive outcome as a recorded result,
not a newly passing physics claim. Synthetic linear wrong-gradient and
quadratic true-gradient cases verify the selector/order check before the
arms. Existing attempt-1 strict xfail is unchanged. Before/after full and explicit
suites use the requested CPU command and markers; ruff uses the requested
paths and selectors. Failure sets must match; new replay asserts recorded
verdicts without hiding fired results. No `rfx/` changes, push, or PR.


## Lead additions before measurement (2026-09-13)

The lane was taken over from the delegated agent by the lead at this point,
on the PI's instruction that implementation and experiments are done by the
lead and the delegated model is used for review only. Nothing had been
measured. Three additions, all to the judges or to their power, none to a
window above:

**A. Richardson validity check.** The FD error bar assumes `D(h) = g + c h^2`.
At the coarse end of the ladder (relative steps up to 1/8, i.e. a 12.5 %
thickness change on a resonant cavity) higher-order terms can dominate, and
then Richardson UNDER-estimates the truncation error; with a bar that is too
narrow, `|g - D| <= 3B` would fire on a failure of the FD model, not of AD.
So a step is informative only if the centered ratio
`rho = (D(2h) - D(h)) / (D(h) - D(h/2))` lies in `[2, 8]` — a factor-two band
around its `h^2` value 4. The two ladder endpoints cannot be checked and are
INCONCLUSIVE. At the fine end roundoff makes `rho` random, which the same
check turns into INCONCLUSIVE rather than a verdict. This is a validity
condition on the reference, not a tolerance on AD.

**B. Judge self-test, run before every arm.** Losses rounded to float32 as the
solver's are: `L = L0 + a h + b h^2 + h^3/2` (a = 0.37, b = 2.9, L0 = 0.1306).
With the TRUE slope the order judge must return HELD and the FD judge HELD;
with the slope 10 % wrong the order judge must return FIRED. Any arm refuses
to run otherwise. Measured on this tree before any arm: true slope ->
R0 1.047, **R1 2.003**, order HELD, FD HELD; 10 % wrong -> R0 1.020,
**R1 0.909**, order FIRED, FD FIRED.

**C. Revert-proof on the real loss — does the gate have power against the dt
path?** The whole question of this lane is the tied-minimum cells, which carry
the CFL time step. Arm `revert_l1_nodt` evaluates the TRUE L1 losses along
each thickness control, but judges them against a gradient computed with
`dt` passed through `stop_gradient` — same forward value (checked and
recorded), gradient missing exactly the dt path. `dt` depends only on the
minimum z cell, which is the thin layer, so:
- **`h_thin` must FIRE** (its gradient loses the dt path);
- `h_core_left`, `h_core_right`, `h_air` must return the SAME verdicts as the
  true-gradient `stack_l1` arm (their dt-path contribution is exactly zero —
  they do not move the minimum cell within the ladder).
The arm also records each control's dt-path share
`(g_true - g_nodt) / g_true`. If `h_thin` HOLDS here, the Taylor gate has no
power against the dt path on this fixture at the declared ladder, and the
`stack_l1` h_thin verdict may not be read as verifying the dt path — the lane
records that plainly instead of reporting coverage as verification.

Ordering: `revert_l1_nodt` and `stack_l1` first (seconds), then `resolution`,
then `stack_l2` (8000-step loss, the long one).

## Results — first attempt (all four arms; no window above changed)

Measured 2026-09-13 on this tree (`c3d0f216`), CPU, `rfx.__file__` under
`rfx-ad-quant`. Judge self-test passed before every arm. Raw JSONs:
`validation/research/multiband_nu/results/adq_{revert_l1_nodt,stack_l1,resolution,stack_l2}.json`.
Replay: `tests/unit/nonuniform/test_adq_designvar_replay.py`, 12 passed.

### Revert-proof (`revert_l1_nodt`) — the gate has power against the dt path

Forward values with and without `stop_gradient(dt)`: identical. dt-path share
of each thickness gradient: **h_thin 0.1852**, h_core_left / h_core_right /
h_air **exactly 0** (as declared: they do not move the minimum cell).

| control | order (no-dt gradient) | R1 slope | FD |
|---|---|---|---|
| **h_thin** | **FIRED** | **0.947** (13 pts, 7.6e-6..3.1e-2) | **FIRED** |
| h_core_left | HELD | 1.979 | HELD |
| h_core_right | HELD | 1.994 | HELD |
| h_air | FIRED | 2.431 (4 pts, 1.6e-2..1.25e-1) | HELD |

Declared requirement met: h_thin FIRES with the dt path removed, and the core
and air controls return the same verdicts as the true-gradient arm (h_air
FIRED in both — see below).

### L1 transient probe energy (`stack_l1`)

| control | R0 | R1 | pts | window | order | FD | narrowest 3B/\|g\| |
|---|---|---|---|---|---|---|---|
| h_core_left | 1.045 | 1.979 | 5 | 4.9e-4..7.8e-3 | HELD | HELD | 5.4e-4 |
| **h_thin** | 0.958 | **2.047** | 8 | 4.9e-4..6.2e-2 | **HELD** | **HELD** | **2.8e-4** |
| h_core_right | 0.946 | 1.994 | 5 | 4.9e-4..7.8e-3 | HELD | HELD | 3.8e-4 |
| h_air | 0.999 | 2.431 | 4 | 1.6e-2..1.25e-1 | FIRED | HELD | 6.9e-4 |
| eps_core_left | — | — | 2 | — | INCONCLUSIVE | HELD | 3.6e-3 |
| eps_thin | — | — | 0 | — | INCONCLUSIVE | HELD | 2.6e-2 |
| eps_core_right | 0.951 | 1.961 | 7 | 2.0e-3..1.25e-1 | HELD | HELD | 8.6e-4 |
| eps_air | — | — | 0 | — | INCONCLUSIVE | INCONCLUSIVE | — |

`J^T g` vs the direct parameter gradient: max abs difference 2.2e-6.
Coverage of the cell-space gradient norm: all design directions **0.8875**
(rank 7); **the four controls passing BOTH gates 0.8830**; dz block alone
0.8875 (rank 3); eps-cell block 0.2475 (rank 4 — a per-layer eps reaches
only the layer mean of a gradient that varies cell to cell within a layer);
dimensionless 0.7897.

The dt-carrying control is verified: h_thin is second order (R1 2.047) and
agrees with FD inside a 3B band of **2.8e-4 relative** — about 500x tighter
than the legacy 15 % — and the same test FIRES when the dt path is removed.

eps_air is INCONCLUSIVE because its gradient is below every resolvable
step: the source and probe sit at the core/thin interfaces and 120 steps do
not reach the air region.

**h_air, recorded FIRED, and why that fire is not a gradient-error
signature.** R1 slope 2.431 is ABOVE the band. Taylor theory: with gradient
error e, `R1(h) = |h e + h^2 c/2 + h^3 d/6 + ...|`, so an error drives the
slope DOWN toward 1; a correct gradient gives 2 (or 3 when `c` is small).
h_air's gradient is small, so R1 clears the 32-quantum floor only at the
coarse end, where the cubic term contaminates a 4-point window. Its dt-path
share is exactly zero and its gradient is bit-identical with and without
the dt path. **Reviewer note (lead):** the band's upper edge 2.2 carries no
gradient-correctness meaning; I adopted it from the delegated draft without
catching that. It stays frozen here. A lower-side criterion (R1 slope >= 1.8)
is reported as a separate, labelled diagnostic in the second attempt below,
not as a replacement gate.

### Resolution arm (A3, attempt 1 adjudicated) — SUPPORTED

Attempt-1 values reproduced bit for bit: y **0.5224044347795822**, z
**0.9923850266377611**. Worst dominant relative error vs FD quanta:

| relative h | y error | y quanta | z error | z quanta |
|---|---|---|---|---|
| 0.00025 | 1.139 | 4 | 1.254 | 3 |
| 0.0005 | 0.555 | 9 | 1.182 | 1 |
| **0.001** | **0.522** | **7** | **0.992** | **3** |
| 0.002 | 0.133 | 16 | 0.515 | 6 |
| 0.004 | 0.061 | 95 | 0.241 | 20 |
| 0.01 | 0.036 | 83 | 0.271 | 24 |
| 0.02 | 0.075 | 129 | 0.049 | 45 |
| 0.04 | 0.180 | 235 | 0.031 | 259 |

The FD error is the classic U: roundoff-limited at small h, truncation-
limited at large h. Attempt 1 sat at 7 and 3 quanta. **Whole-axis dilation
Taylor (moves all eight tied cells together, so it exercises the dt path):
y R1 1.990 (7 pts), z R1 1.882 (6 pts), both HELD.** All four declared
conditions hold on both axes -> **SUPPORTED: attempt 1 fired on FD
resolution, not on the gradient.** Every non-tied y and z cell (38 each) is
INCONCLUSIVE on its own Taylor ladder — those gradients (max 0.78 on y, 0.31
on z) are too small to clear the floor, and even at the U's minimum the FD
reference itself cannot resolve y's non-tied cells better than ~3.6 %. The
per-cell view could never have been the quantitative claim on this fixture.

### L2 resonant power (`stack_l2`, 8000 steps)

| control | R0 | R1 | pts | order | FD |
|---|---|---|---|---|---|
| h_core_left | 0.840 | 3.455 | 5 | FIRED (upper) | HELD |
| **h_thin** | 0.953 | **1.753** | 5 | **FIRED (lower)** | **FIRED** |
| h_core_right | 0.939 | 2.636 | 5 | FIRED (upper) | HELD |
| h_air | 1.026 | 1.497 | 7 | FIRED (lower) | HELD |
| eps_core_left | 1.045 | 1.984 | — | HELD | HELD |
| eps_thin | — | — | 0 | INCONCLUSIVE | HELD |
| eps_core_right | 1.024 | 1.929 | — | HELD | HELD |
| eps_air | 1.019 | 1.881 | — | HELD | HELD |

All four L2 thickness controls FIRED on order; h_thin also FIRED on FD.
Recorded as fired. Evidence that the reference, not the gradient, is what
failed — stated as diagnosis, to be tested below, not as a verdict:

1. **The L2 loss's float32 noise is ~150 ulp, not 1.** At h_thin's finest
   step the central FD is 3.26e-21 against a gradient of 7.41e-21 — off by
   56 % — while its 1-ulp bar claims 0.4 %; the implied loss difference
   error is ~157 ulp of `loss0 = 4.92e-21`. An 8000-step DFT accumulation in
   float32 carries far more rounding than one output ulp; the judges assumed
   one ulp.
2. h_thin's order window starts at 34 one-ulp quanta; h_air's includes three
   points at 33-106 quanta whose R1 is flat. Above them h_air's R1 ratios per
   doubling are 3.5 / 3.8 / 3.95, i.e. slope 1.8 -> 2.0.
3. h_thin's single FD fire is at relative h = 3.1e-2 (a 3 % thickness change
   on a resonant observable, where FD samples across the line shape), with
   an in-band `rho = 4.1` isolated between out-of-band neighbours (0.78,
   -0.78) — a coincidental pass of the Richardson check, not an asymptotic
   regime.
4. h_core_left: at h = 4.9e-4 and 9.8e-4, R1 is 4.4e-28 and 4.7e-28 — about
   one ulp — so the linear prediction with the AD gradient is exact to the
   last bit there: an empirical bound of ~6.6e-6 relative on that gradient.
5. E4's independent central FD on h_thin at h = 1e-3 agreed to 2.8e-3 (PR #962).

## Second attempt, L2 thickness only — declared BEFORE it runs

**Reason (instrument defect in the reference, not a tolerance edit):** the
noise model. Both judges take one float32 ulp of the loss as the noise
floor. For the 8000-step L2 loss the measured jitter is ~two orders of
magnitude larger (item 1 above), so points at "32 quanta" are still noise-
dominated and the FD roundoff allowance is ~100x too narrow.

**What changes:** only the noise floor, from assumed to MEASURED. For each of
the four thickness controls, evaluate L2 on a fresh fine grid of 64 relative
steps uniformly spaced in [1e-5, 1e-4] along the control; fit
`L0 + a h + b h^2` by least squares; the floor `sigma` is the RMS residual
(scaled by sqrt(64/61)). Validity check, declared: a cubic fit must reduce
the RMS residual by less than 10 %, else the window is not smooth-quadratic
and sigma for that control is marked unreliable. Then re-judge the STORED
first-attempt ladders with `sigma` in place of the ulp, same form:
eligibility `R1 > 32 sigma`, FD roundoff `Q = sigma / h` (the first attempt
used `2 ulp / 2h`). Bands unchanged: R0 [0.9, 1.1], R1 [1.8, 2.2],
`|g - D| <= 3B`, INCONCLUSIVE above 15 %, Richardson `rho` in [2, 8].

**Applied uniformly** to all four L2 thickness controls — not only the ones
that fired. The eps controls are not re-judged (they held under the stricter
1-ulp floor).

**Reported alongside, not gates:** (a) the lower-side criterion, R1 slope
>= 1.8 alone, which is the part of the band a gradient error can violate;
(b) the empirical relative gradient-error bound `min_h R1(h) / (h |g.v|)`
over eligible points.

**Declared outcome rule:** if h_thin still FIRES with the measured floor,
L2 thickness gradients are not verified by this lane, and the note says so;
the first-attempt record stays FIRED either way.
