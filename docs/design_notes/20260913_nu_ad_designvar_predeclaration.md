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
