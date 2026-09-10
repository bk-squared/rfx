# G5 — CPML localization, replacement gate pre-declaration

2026-09-11. PI-authorized replacement of G4's fired gate. Baseline: `29a92108`,
branch `feat/nu-cost-reduction`; work only in `/Users/byungkwankim/Documents/rfx-nu-cost`.
The G4 note is immutable. No candidate measurement or production edit precedes
this declaration. Results and baseline-only calibration will be appended.

## Frozen experiment and decision

G5-1 EXPRESSION IDENTITY. Reuse `887f8ff7:rfx/boundaries/cpml.py` exactly.
Baseline quotes: `_ch_full = dt / (materials.mu_r * MU_0)` followed by
`ch_xlo = _ch_full[:n_x, :, :]`; candidate:
`ch_xlo = dt / (materials.mu_r[:n_x, :, :] * MU_0)`.
Restriction commutes with pointwise multiplication/division, so each selected
cell evaluates `dt / (mu_r[cell] * MU_0)` in both. The six E coefficients obey
the same argument with eps_r and EPS_0.
Baseline: `hz_shifted_xlo = _shift_bwd(state.hz, 0)[:n_x, :, :]`;
candidate: `hz_shifted_xlo = _shift_slab(state.hz, 0, n_x, False, False)`.
At index i both read H[i-1], or zero for i=0. Forward shifts similarly read
E[i+1], or zero at the last cell. This applies independently to all 24 slab
derivatives, including transverse permutations.
Both retain verbatim:
`new_psi_ey_xlo = b_x_lo * cpml_state.psi_ey_xlo + c_x_lo * curl_hz_dx_xlo`.
Baseline: `ey = ey.at[:n_x, :, :].add(-ce_xlo * new_psi_ey_xlo)`;
candidate: `ey = _slab_add(ey, -ce_xlo * new_psi_ey_xlo, 0, False)`;
helper: `slab = lax.dynamic_slice(field, starts, correction.shape)` then
`return lax.dynamic_update_slice(field, slab + correction, starts)`.
Each slab index is unique, so both evaluate old_cell + correction exactly
once. Each subsequent kappa add stays separate and in the same order.
All psi recurrences, b/c/kappa selections, signs, transpose operations,
axis/face/component order and H -> CPML-H -> PMC -> E -> CPML-E -> PEC ->
source order remain unchanged. No public signatures or Yee edits.

Algebraic zero statement, for finite fields and initialized zero psi:
for c outside union of absorbing slabs, either no update addresses c, or
its padded/no-op profile has b=1, c_profile=0, kappa=1. Inductively
psi[t+1]=1*0+0*D=0 and correction = signed_coeff*0 +
signed_coeff*(1/1-1)*D = 0. Thus delta_E(c)=delta_H(c)=0 in both versions.
This is a source/algebra argument, not an empirical zero claim for arbitrary
nonzero externally supplied psi.

G5-2 CONTRACTION-NEUTRAL BIT-IDENTITY. All seven G4 fixtures, unchanged from
`tests/unit/boundaries/test_cpml_localization.py`: uniform8, graded8, mixed8,
uniform4, uniform16, periodic8, kappa8. Exactly 200 steps, all six fields
and all 24 psi, float32: require `np.array_equal`, zero differing elements.
Fresh-process baseline controls try `XLA_FLAGS=--xla_allow_excess_precision=false`,
`jax_default_matmul_precision=highest`, inspect installed per-op precision
signatures, and installed XLA CPU fast-math/fusion controls. No mechanism is
accepted merely because the option parses. It must change at least one
baseline output from its unflagged self AND codegen must demonstrate no FMA
contraction in the relevant field arithmetic. If a control does not change
baseline, explicitly call it vacuous. If no proven mechanism is found, only
G5-3 can decide numerical acceptance. A proven mechanism with a candidate
inequality fires G5-2; fallback does not excuse that failure.

G5-3 FLOAT64 REFERENCE DISTANCE. Additionally reported regardless of G5-2;
decisive fallback iff G5-2 is vacuous. Independent NumPy float64 full-field
Yee/CPML recurrence, not a float64 recompile of candidate or baseline. Use
identical initialized float32 coefficients/metrics promoted to float64,
float64 physical scalar constants and grid scalars, same zero initial state,
same source and boundary ordering. Evaluate the seven named fixtures at
200 completed steps. For each of six fields and 24 psi require BOTH
`abs(candidate-ref64) <= 1.05*abs(baseline-ref64)` at EVERY element and
`max(abs(candidate-ref64)) <= 1.05*max(abs(baseline-ref64))`.
No additive floor: an exactly correct baseline element requires exact
candidate agreement. Report violating counts and max errors separately.
Float64 is a numerical reference, not a claim of exact real arithmetic.

G5-4 PHYSICS INVARIANTS.
(a) Committed measurement class: `tests/oracle/test_pml_reflectivity_upml.py`
module docstring and `test_upml_reflection_floor_regression` explicitly give
CPML **-68.3 dB**, f0=2e9, freq_max=5e9, 8 layers, 250 steps. Use exactly
`tests/unit/boundaries/test_cpml.py::_reflection_db_vs_clean_reference`,
cpml_domain=0.06, GaussianPulse bandwidth=0.5; reference extent is
1.15*250*dt*C0. Baseline-only calibration: THREE fresh-process repetitions,
spread S=max(dB)-min(dB); freeze window W=2*S dB about baseline median.
If S=0, W=0; no rounding allowance or minimum epsilon. Append numerical
calibration and commit it BEFORE candidate application or comparison.
Candidate uses the same oracle and must satisfy |candidate_dB-median|<=W.
(b) PEC-closed null control: same 12-cell uniform domain, dx=1mm,
cpml_layers=0, 200 steps, same center pulse; all six fields at every step
must be bit-identical, as must the energy history computed from them.
No absorber is expected to produce decay; this is an equality control.
(c) mixed8: x_lo PEC, y_hi PMC; x_hi=4,y_lo=6,z_lo=4,z_hi=8.
All eight psi arrays for x_lo/y_hi must be exactly zero in both at 200
steps; absorbing-face psi must meet the applicable G5-2 or G5-3 rule.

G5-5 DIVERGENCE-GROWTH WITNESS. All seven fixtures, steps 1..200,
unflagged float32. For this sensitivity experiment only, initialize center
ez to 1.0 in both baseline and candidate (all other fields/psi zero), so
one ulp has a normal, nonzero scale. Perturbed baseline changes that one
cell to nextafter(float32(1),+inf)=1+2^-23; delta=1.1920928955078125e-7.
Use the same Gaussian pulse thereafter. For every step and each of six
fields, max|candidate-baseline| <= max|perturbed_baseline-baseline|, with
NO tolerance or floor. Archive all 200-point curves; any excess fires STOP.
This seeded witness is separate from unchanged zero-initialized G4 fixtures.

A fired gate stops candidate development and GPU timing; record failure,
never tune a candidate/window away. Complete requested reporting where
possible, labeling unexecuted gates. Restore production after rejection.

## Performance falsifier and regression

Only after numerical acceptance and G5-4/G5-5 pass: ONE GPU ladder (one
resubmission only for run failure), same G1b YAML/harness/card: bare-slow and
nu-uniform, 300^3 L=0/4/8/16 and 400^3 L=8. Three timing windows, median,
max-minus-min spread. Before: `results/w8b_nu_kernel_ablation_4090.json`.
At 300^3 L=8 BOTH lanes must improve rate by MORE THAN twice the larger
before/after spread; report same predicate for L=4/16. L=0 rate changes
must be <= twice the larger spread. This preserves G4's rate-based test
(the G4 prose's subtraction sign conflicts with its speedup intent; higher
Mcells/s is faster). Failure stops the lane and records whole-array
structure was not demonstrated to be the cost. No missing 400^3 baseline
substitution. Authorized rsync, SHA staging, VESSL submission from home,
polling, mandatory harvest/deletion and JSON destination follow user command.
No push, no new PR. All commits end with the supplied co-author trailer.

Before and after: requested full unit/contracts selection (not gpu, slow,
slow_physics), no cacheprovider, pinned PYTHONPATH/Python; ruff E,F,W with
E501,F401,E741,E731,E701,E702,E402 ignored. PYTHONDONTWRITEBYTECODE=1 and
ruff --no-cache avoid writes in the externally located tool environment.
Do not chase the three named pre-existing slow_physics oracle failures.

## Baseline-only reflection calibration (before candidate application)

Three fresh Python processes, exact committed helper/config declared above:
`-68.26476397028848`, `-68.26476397028848`, `-68.26476397028848` dB.
Median **-68.26476397028848 dB**, max-minus-min spread **0 dB**.
Frozen W=2*S = **0 dB**; candidate must reproduce that numerical dB value
exactly. This deterministic CPU calibration does not estimate GPU variability.
Log: `validation/research/nu_cost/g5/baseline_reflection.log`.

## Baseline-only contraction control

Installed JAX/jaxlib 0.10.2, CPU arm64. Seven fixtures x 30 arrays at step
200 (210 arrays), fresh process for each control. `controls.json` records
per-array differing counts against unflagged baseline.

| mechanism | changed elements | changed arrays | effective suppression? |
|---|---:|---:|---|
| --xla_allow_excess_precision=false | 0 | 0 | vacuous |
| jax_default_matmul_precision=highest | 0 | 0 | vacuous |
| --xla_cpu_enable_fast_math=false | 0 | 0 | vacuous |
| --xla_cpu_use_fusion_emitters=false --xla_cpu_enable_fast_math=false | 2129538 | 194 | no: 9583 FMA instructions remain |
| --xla_disable_hlo_passes=fusion --xla_cpu_enable_fast_math=false | 2129643 | 194 | YES for field/psi arithmetic |

Selected exact mechanism: `XLA_FLAGS="--xla_disable_hlo_passes=fusion
--xla_cpu_enable_fast_math=false"`. Disabling fusion places multiplication
and addition in separate compiled kernels, suppressing their contraction.
`otool -tvV` audit of all 961 generated objects for seven baseline fixtures:
**0 FMA instructions outside exponential kernels**; 81 instructions remain
inside the implementation of the atomic `exp` operation (source Gaussian
and traced graded profile initialization). These are not contractions across
field/psi arithmetic operators. Unflagged control has **8655 FMA instructions**
in 668 objects. LLVM text has zero explicit `contract`/`llvm.fma` markers even
in unflagged output: checking LLVM markers alone would have been insufficient.
The effectiveness condition is met by **2129643 changed elements**; this is
not a claim that disabling fast-math alone disables contraction.
Per-op inspection: lax.mul exposes out_dtype but no precision argument;
lax.add and jnp.multiply/add expose no precision argument. dot_general has
precision but no dot appears in these elementwise Yee/CPML updates.
Full counts, FMA locations and logs live under `validation/research/nu_cost/g5/`.
Candidate must now pass G5-2 with this selected mechanism; G5-3 remains a
required additional report, not a fallback excuse for an effective G5-2 failure.

## Results — STOP, G5-5 fired

No production localization is accepted. Candidate `b7f14a57` exactly reuses
`887f8ff7`; restoration `e1a32478` returns all of `rfx/` to starting HEAD
`29a92108`. No production changes remain. No candidate tuning followed the
failure. G4 note is byte-unchanged. No push or new PR.

### G5-1 EXPRESSION IDENTITY — PASS

The quoted restriction/neighbor/read-add-write equivalences above apply to
every face. AST audit additionally verifies all **48 psi assignments**
(24 recurrences plus 24 unchanged-carry branches) and all **48 correction
operands and their order** match. Candidate SHA256
`01e55ec9c9ce28a7e11da39e349ceca73583775f0ae2e20c6858c1a7a2da4959`;
baseline SHA256
`4979e28793d9530dd362f42d5aac913dd07360ffd757fdb7c0d3a9d3c7c3c3fd`.
Algebraic zero outside every absorbing slab is established in the declaration;
no measured epsilon is substituted for it. Audit: `g5/g5_1.json`.

### G5-2 CONTRACTION-NEUTRAL BIT-IDENTITY — PASS, decisive numerical gate

Selected mechanism and effectiveness control are recorded above. BOTH
baseline and candidate were freshly recompiled with
`XLA_FLAGS="--xla_disable_hlo_passes=fusion --xla_cpu_enable_fast_math=false"`.
Every one of **7 fixtures x 30 arrays = 210 comparisons** passes
`np.array_equal` at **200 steps**: **0 differing elements, max_abs=0**.
This includes all six fields and all 24 psi per fixture, not just a norm.
Candidate-comparison object audit (both versions in that fresh process):
**1961 objects, 0 FMA outside exponential kernels, 162 inside exp**.
The residual exp implementation uses FMA internally, but no field/psi
multiply-add operators contract. Control baseline differs from unflagged
baseline in **2129643 elements across 194 arrays**, so this is not vacuous.
`g5_2.json`, `g5_2.log`, `candidate_codegen.json` archive the evidence.

### G5-3 FLOAT64 REFERENCE DISTANCE — FAIL as additional report

This is NOT the selected acceptance fallback: effective G5-2 passed.
All seven fixtures and all 30 arrays were evaluated at 200 steps against
independent NumPy float64 updates (`scripts/diagnostics/cpml_g5_reference.py`).
The reference shares initialized profile/metric data and grid geometry only;
it does not call production stepping or boundary operators. Profiles are
captured from initialization inside the G4 compiled runner so traced graded
initialization uses the same float32 coefficient data; spacing/physical scalar
values retain their source values. The scalar Gaussian is evaluated in float64.

A separate audit compared this NumPy reference with baseline stepping in
float64, promoting the SAME prepared coefficients/metrics/materials, without
using that audit to alter any G5-3 decision. Across seven fixtures, electric
field discrepancies are <=2.6645352591003757e-15 and magnetic discrepancies
<=5.871152625611529e-18. Relative discrepancies for near-zero hz can be order
one because both values are around 1e-18; absolute audit records for all 30
arrays are in `reference_audit.json`. Float64 is not exact real arithmetic.

No floor was added to the 1.05 elementwise or max-norm predicate. Counts
below include fields AND psi; full per-array counts/errors are in `g5_3.json`.

| fixture | violating elements | arrays failing max-norm (of 30) |
|---|---:|---:|
| uniform8 | 115889 | 15 |
| graded8 | 0 | 0 |
| mixed8 | 0 | 0 |
| uniform4 | 0 | 0 |
| uniform16 | 0 | 0 |
| periodic8 | 47242 | 4 |
| kappa8 | 133051 | 17 |

Per-field max errors and elementwise violations (C/B is the ratio of max
errors, with limit 1.05; E and H carry their native field units):

| fixture | field | baseline max error | candidate max error | C/B | violating elements |
|---|---|---:|---:|---:|---:|
| uniform8 | ex | 6.73359823233e-07 | 4.11913499754e-07 | 0.611728656 | 9836 |
| uniform8 | ey | 6.73359823233e-07 | 4.11913499532e-07 | 0.611728656 | 10022 |
| uniform8 | ez | 7.94661410009e-07 | 2.21553029145e-07 | 0.278801797 | 9682 |
| uniform8 | hx | 7.47406719225e-09 | 7.94575565832e-09 | 1.063110011 | 9586 |
| uniform8 | hy | 6.48911741699e-09 | 6.42525616737e-09 | 0.990158716 | 9103 |
| uniform8 | hz | 5.56533441312e-09 | 4.9764374819e-09 | 0.894184808 | 10277 |
| graded8 | ex | 3.89552225588e-07 | 3.89552225588e-07 | 1.000000000 | 0 |
| graded8 | ey | 6.43727135152e-07 | 6.43727135152e-07 | 1.000000000 | 0 |
| graded8 | ez | 1.31019233063e-06 | 1.31019233063e-06 | 1.000000000 | 0 |
| graded8 | hx | 2.19934494154e-09 | 2.19934494154e-09 | 1.000000000 | 0 |
| graded8 | hy | 1.78396118906e-09 | 1.78396118906e-09 | 1.000000000 | 0 |
| graded8 | hz | 2.56775090153e-09 | 2.56775090153e-09 | 1.000000000 | 0 |
| mixed8 | ex | 7.16402565049e-07 | 7.16402565049e-07 | 1.000000000 | 0 |
| mixed8 | ey | 5.27167947739e-07 | 5.27167947739e-07 | 1.000000000 | 0 |
| mixed8 | ez | 4.28823101473e-07 | 4.28823101473e-07 | 1.000000000 | 0 |
| mixed8 | hx | 2.681841832e-09 | 2.681841832e-09 | 1.000000000 | 0 |
| mixed8 | hy | 5.40199966628e-09 | 5.40199966628e-09 | 1.000000000 | 0 |
| mixed8 | hz | 2.67796451737e-09 | 2.67796451737e-09 | 1.000000000 | 0 |
| uniform4 | ex | 2.94607510831e-07 | 2.94607510831e-07 | 1.000000000 | 0 |
| uniform4 | ey | 5.33026089489e-07 | 5.33026089489e-07 | 1.000000000 | 0 |
| uniform4 | ez | 5.5118638187e-07 | 5.5118638187e-07 | 1.000000000 | 0 |
| uniform4 | hx | 2.98697427074e-09 | 2.98697427074e-09 | 1.000000000 | 0 |
| uniform4 | hy | 5.17297060302e-09 | 5.17297060302e-09 | 1.000000000 | 0 |
| uniform4 | hz | 2.31589880675e-09 | 2.31589880675e-09 | 1.000000000 | 0 |
| uniform16 | ex | 5.05769397696e-07 | 5.05769397696e-07 | 1.000000000 | 0 |
| uniform16 | ey | 5.5425762846e-07 | 5.5425762846e-07 | 1.000000000 | 0 |
| uniform16 | ez | 4.38779192757e-07 | 4.38779192757e-07 | 1.000000000 | 0 |
| uniform16 | hx | 6.18868368319e-09 | 6.18868368319e-09 | 1.000000000 | 0 |
| uniform16 | hy | 6.29359497508e-09 | 6.29359497508e-09 | 1.000000000 | 0 |
| uniform16 | hz | 4.15217593512e-09 | 4.15217593512e-09 | 1.000000000 | 0 |
| periodic8 | ex | 4.14582905117e-07 | 3.68837621778e-07 | 0.889659504 | 4107 |
| periodic8 | ey | 2.21450575433e-07 | 4.07039398187e-07 | 1.838059790 | 5024 |
| periodic8 | ez | 5.3596994154e-07 | 3.18589794324e-07 | 0.594417279 | 4791 |
| periodic8 | hx | 3.64638347075e-09 | 1.73039787834e-09 | 0.474551811 | 5463 |
| periodic8 | hy | 2.12923124733e-09 | 1.6663694984e-09 | 0.782615557 | 4471 |
| periodic8 | hz | 1.86411819262e-09 | 1.47489587292e-09 | 0.791202982 | 4841 |
| kappa8 | ex | 4.48091965755e-07 | 4.37584468216e-07 | 0.976550578 | 11455 |
| kappa8 | ey | 4.48091965755e-07 | 3.18375178221e-07 | 0.710513025 | 10177 |
| kappa8 | ez | 9.22738131237e-07 | 4.45900973034e-07 | 0.483236747 | 11396 |
| kappa8 | hx | 3.4240256684e-09 | 3.39520519461e-09 | 0.991582869 | 10768 |
| kappa8 | hy | 5.44535542021e-09 | 3.87425394907e-09 | 0.711478618 | 11725 |
| kappa8 | hz | 3.55557849795e-09 | 4.14881240439e-09 | 1.166845960 | 11182 |

### G5-4 PHYSICS INVARIANTS — PASS

(a) Exact committed reflection oracle: baseline repetitions all
**-68.26476397028848 dB**, spread **0 dB**; candidate
**-68.26476397028848 dB**, absolute difference **0 dB**, frozen window
**0 dB**. Same 2 GHz / 8-layer / 250-step fixture; no window widening.

(b) PEC-closed, cpml_layers=0: all six field histories at steps 1..200 are
bit-identical, **0 differing elements**. All 200 energy samples are identical;
step-200 energy sum is **1.674324749477396e-10** in both. This is the
cellwise energy-density sum (no cell-volume factor), not a Joule integral.
The production no-absorber branch bypasses CPML initialization and updates.
Initial harness erroneously called init_cpml at L=0, which raised
ZeroDivisionError before any field comparison; corrected to that bypass.
`g5_4_initial_harness_error.log` preserves the error. No physics predicate
or fixture was changed; `g5_4.log` records the completed rerun.

(c) mixed8: all **8 non-absorbing-face psi arrays** for PEC x_lo and PMC
y_hi have **0 nonzero elements in each version**. All 16 absorbing-face
psi arrays pass G5-2 with **0 differing elements** (as do all 24 mixed8 psi).
`g5_4.json` lists the individual non-absorbing arrays.

### G5-5 DIVERGENCE-GROWTH WITNESS — FAIL, STOP

Frozen seed is ez[center]=1; perturbed baseline seed is 1+2^-23, delta
**1.1920928955078125e-7**, single cell. Same unflagged Gaussian-driven scan
thereafter. The entire 200-step uniform8 curve was captured in each run;
first violation in chronological order is **hy at step 35**.
The comparisons use float64 subtraction of stored float32 field values to
measure their distance without additional subtraction rounding.

| field | first violating step | candidate distance | baseline sensitivity | violating steps (of 200) | maximum excess |
|---|---:|---:|---:|---:|---:|
| ex | 39 | 2.384185791015625e-07 | 1.3411045074462891e-07 | 129 | 4.76837158203125e-07 |
| ey | 41 | 1.4901161193847656e-07 | 1.3411045074462891e-07 | 64 | 3.5762786865234375e-07 |
| ez | 39 | 4.76837158203125e-07 | 2.384185791015625e-07 | 87 | 4.76837158203125e-07 |
| hx | 38 | 5.5581494962098077e-10 | 4.9760728870751336e-10 | 28 | 1.2942109606228769e-09 |
| hy | 35 | 3.7016434362158179e-10 | 3.3651303965598345e-10 | 50 | 1.216903910972178e-09 |
| hz | 44 | 9.4339094869333451e-10 | 5.8808229119744482e-10 | 48 | 5.8916886647164546e-10 |

All six fields exceed their declared sensitivity bounds. The remaining six
witness fixtures were **not run after STOP**; no seed or window was retuned.
All 200-point curves and violating step indices are archived in `g5_5.json`;
`g5_5_witness.svg` plots them. Passing contraction-neutral identity does not
waive this independently required gate. The candidate is rejected.

### Before/after ladder — not run after correctness STOP

Mcells/s; spread=max-min across three windows. Before rows are taken exactly
from `results/w8b_nu_kernel_ablation_4090.json`. Missing 400^3 rows stay missing.

| n | lane | L | before median | before spread | after median | after spread | speedup |
|---:|---|---:|---:|---:|---|---|---|
| 300 | bare-slow | 0 | 10015.873987 | 82.342573 | not run | — | — |
| 300 | bare-slow | 4 | 2030.470869 | 124.760412 | not run | — | — |
| 300 | bare-slow | 8 | 2120.835565 | 0.032547 | not run | — | — |
| 300 | bare-slow | 16 | 2037.088057 | 81.397819 | not run | — | — |
| 300 | nu-uniform | 0 | 10037.886593 | 57.932029 | not run | — | — |
| 300 | nu-uniform | 4 | 1795.295944 | 76.819075 | not run | — | — |
| 300 | nu-uniform | 8 | 1780.745256 | 3.281352 | not run | — | — |
| 300 | nu-uniform | 16 | 1549.203037 | 52.644073 | not run | — | — |
| 400 | bare-slow | 8 | unavailable | — | not run | — | — |
| 400 | nu-uniform | 8 | unavailable | — | not run | — | — |

No after speedups, spreads or 0-layer throughput controls were measured.
The timing falsifier is **not evaluated**: this rejection does not establish
whether whole-array structure was the performance cost. No GPU YAML,
rsync staging, .staged-commit, VESSL submission or w9 JSON was produced.
VESSL submissions **0**, run ID **none**, harvest/deletion **not applicable**.
No run exists to harvest/delete; no resubmission.

### Reproduction and regression status

All commands run in this worktree, with PYTHONDONTWRITEBYTECODE=1,
PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-cost and the user-authorized
/Users/byungkwankim/Documents/rfx/.venv/bin/python executable. No other
worktree was edited. Original G4 artifacts are read-only.

Fresh-process baseline controls: `scripts/diagnostics/cpml_g5_controls.py`.
For rejected-candidate gates from the restored tree, set
`RFX_G4_REJECTED_CANDIDATE=1` and run
`scripts/diagnostics/cpml_g5_gates.py identity` with the selected XLA_FLAGS.
Run `reference`, `physics`, or `witness` with no XLA_FLAGS for the other gates.
The archived candidate is byte-identical to G5's applied candidate.
For physics reproduction the oracle's own globals must select that archived
candidate too (the harness supplies this binding). To audit generated objects,
use `scripts/diagnostics/cpml_g5_controls.py audit <dump-label>`.
G5-3 reads `g5/unflagged.npz`, regenerated by baseline controls; generated NPZ,
XLA dumps and Matplotlib cache are ignored, while numerical reports are tracked.

The full BEFORE command began with baseline CPML imported by test collection
before candidate application. It continues against that already-loaded module;
Python source replacement does not replace imported function objects. The
full AFTER command began after production restoration. Both use the exact
requested unit/contracts marker selection, addopts override and no cacheprovider.
Results will be appended after completion; no running suite is claimed passed.
Ruff for the requested rfx/ and tests/ selection: **0 findings before,
0 on candidate, 0 after restoration**. Research harness ruff also passes.

### Completed before/after regression

Both requested full unit/contracts runs completed with **exit 0**. AFTER
means the delivered, restored production tree; it is not an unflagged
full-suite acceptance claim for the rejected candidate.

| run | passed | skipped | deselected | xfailed | warnings | seconds | exit |
|---|---:|---:|---:|---:|---:|---:|---:|
| before (baseline imported before application) | 4563 | 34 | 269 | 6 | 2153 | 3263.05 | 0 |
| after production restoration | 4563 | 34 | 269 | 6 | 2153 | 3376.59 | 0 |

Exact command for each, with output redirected to `g5/before_regression.log`
and `g5/after_regression.log` respectively:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-cost \
/Users/byungkwankim/Documents/rfx/.venv/bin/python -m pytest \
  tests/unit tests/contracts -q -o addopts= \
  -m "not gpu and not slow and not slow_physics" -p no:cacheprovider
```

No failures were chased or hidden, and no regression rerun was needed.
The named pre-existing slow_physics oracles were excluded and not modified.
Requested ruff selection: **0 findings** before, on candidate, and after
restoration (`before_ruff.log`, `candidate_ruff.log`, `after_ruff.log`).
The independent harness files also pass the same ruff rule selection.

Final integrity: `git diff 29a92108 -- rfx/` is empty; the G4 note is
byte-identical to `29a92108`; the original G5 declaration from `18693717`
is an unchanged byte prefix of this file. The witness SVG was visually
checked against its underlying 200-point JSON curves. No GPU run, push,
new PR, external-worktree edit, or frozen-window change occurred.

Commit sequence before this final regression record: declaration `18693717`,
reflection-window calibration `96c33db3`, baseline controls/reference
`b069fc49`, candidate `b7f14a57`, restoration `e1a32478`, gate evidence
`28e7cbae`. **G5 remains STOPPED by G5-5.**
