# G4 — CPML localization — pre-declaration

**Status:** pre-declaration, before code changes or validation runs. Gates
below are frozen by this commit. Results will be appended, never used to
rewrite a gate.
**Tree:** `/Users/byungkwankim/Documents/rfx-nu-cost`,
`feat/nu-cost-reduction`, baseline `c30d30208ff3fc3b73cf7d24a2f0fcf6fd08025b`.
Date: 2026-09-10 (KST).

## Mechanism found by reading

Read the G1b measured section and its committed JSON first. At 300^3,
CPML-8 costs 4.722608 plain steps on bare-slow and 5.636902 on nu-uniform.
The L=4/16 costs are 0.492497/0.490897 ns and 0.557011/0.645493 ns,
while absorbing fraction grows 0.075676 -> 0.261472. This supports a
whole-array cost in the CPML step, but does not identify its operator.

The proposed profile-expansion mechanism is **not what this HEAD does**.
Line references below are to baseline `rfx/boundaries/cpml.py`:

- 570: `return arr if depth >= n_alloc else arr[:depth]`;
  580: `return arr if depth >= n_alloc else arr[n_alloc - depth:]`.
  Neither helper expands a profile to the domain.
- 668: `b_x_lo = _clip_lo(px_lo.b, n_x, n)[:, None, None]` is
  `(n_x, 1, 1)`, broadcasting over a slab, not over the whole domain.
- 731: `new_psi_ey_xlo = b_x_lo * cpml_state.psi_ey_xlo + c_x_lo * curl_hz_dx_xlo`;
  732: `ey = ey.at[:n_x, :, :].add(-ce_xlo * new_psi_ey_xlo)`.
  Psi arithmetic and correction operands are already slab-sized. Psi
  allocation at 541–554 uses `(axis_depth, perpendicular_1, perpendicular_2)`.
- 637: `_ce_full = dt / (materials.eps_r * EPS_0)`;
  911: `_ch_full = dt / (materials.mu_r * MU_0)` compute full coefficients
  before slicing. 728: `hz_shifted_xlo = _shift_bwd(state.hz, 0)[:n_x, :, :]`
  shifts a full field before slicing. `_shift_bwd/fwd` in `core/yee.py`
  pad and slice full arrays. Whether XLA eliminates this excess is unknown.
- Each `.at[slab].add` returns a full field array. There are two ordered
  adds per face/component (psi then kappa), including at intersecting faces.
  This is a candidate source of full-volume fusion/scatter traffic.

Uniform setup stores psi in `carry_init["cpml"]` (simulation.py:844–845).
Its core step passes carry through H CPML (1302), then E CPML (1399),
and into the next scan iteration. NU does the same (nonuniform.py:2044,
2087), using `cpml_axes_eff` selected from allocated pads (1874).
Both call the same CPML functions with materials. NU initialization passes
PEC/PMC sets explicitly and retains independent z-lo/z-hi cell sizes.
The existing active-depth clamp has legacy-grid and excluded-axis guards;
these semantics must not be inferred solely from zero padding.

## Intended change

Localize remaining coefficient and shifted-neighbor expressions before
arithmetic. Replace scatter-add field corrections with slab read/add/write
using a contiguous dynamic-update-slice, keeping each addition separate and
in the existing order. Further restrict a face to its own active region
only where existing profile metadata proves the omitted region is no-op;
retain all psi carry shapes and untouched padding. No profile, cell size,
boundary token, or caller policy change. Both lanes share this implementation.
Compiler fusion may still reorder arithmetic despite identical Python
parenthesization: that is a possible gate failure, not permission to relax it.

## Frozen gates

1. **BIT-IDENTITY:** after 200 steps, `np.array_equal` for each float32
   ex/ey/ez/hx/hy/hz and all 24 psi arrays, new vs baseline. Fixtures:
   uniform CPML-8 open box with a soft source; graded-z CPML-8 box;
   per-face PEC/PMC/CPML box with differing face counts; CPML-4; CPML-16.
   Use 12 interior cells per axis, dx=1 mm, graded z spanning 0.5–1 mm,
   a center soft ez Gaussian pulse. Include a periodic-axis fixture and
   kappa>1 coverage. If any bit moves, **STOP**, report the first differing
   operation and its arithmetic/codegen change; no epsilon-close acceptance
   and no further candidate tuning or GPU timing.
2. **AD parity:** `jax.grad` with respect to dz_profile and eps_r on the
   12-cell graded-z CPML-8 fixture, 200 steps, objective sum of squared
   final E fields. Compare maximum absolute gradient difference divided by
   maximum absolute reference gradient (floor 1e-30), <=1e-6 for each.
   This fixture exercises traced boundary spacing, dt, curl metrics,
   material coefficients and recursive psi; require finite nonzero gradients.
3. No public signature change. Uniform fused fast path and stencil_order=4
   remain untouched. No changes to simulation.py/nonuniform.py step order.
4. Run focused identity/AD gates, then all unit/contracts excluding gpu,
   slow, slow_physics, with the supplied Python/PYTHONPATH and no cache
   provider; run ruff E,F,W ignoring E501,F401,E741,E731,E701,E702,E402.
   Report counts and failures; do not chase the named pre-existing oracles.
5. One RTX 4090 VESSL run after correctness passes: same G1b marginal-cost
   harness (64 -> 1088, three windows, median and max-minus-min spread),
   bare-slow and nu-uniform at 300^3 L=0/4/8/16, plus both at 400^3 L=8.
   Resubmit once only for a failed run. Harvest logs and delete with the
   prescribed harvest script. No push or PR.

## Expected gain and falsifier (before any run)

CPML-8 ideal shell localization predicts
`c_new = c0 + f8*(c8-c0)`: 0.153327 ns bare-slow and 0.166100 ns NU,
about 3.08x and 3.38x speedups. Retaining half the removable full-array
excess gives about 1.51x and 1.54x. Pre-declared expected CPML-8/300 gain:
**1.5–3.4x**. This is a model range, not a measured claim; slab scatter
writes may retain volume traffic. L=4 should benefit at least as much as
L=16 in the ideal model; report all rows without dropping noisy windows.

Timing gate: at 300^3 CPML-8 on **both** lanes, new median minus old
median must exceed **twice the larger of the two measured spreads**.
If either fails, STOP and record that this localized whole-array structure
was not demonstrated to be the cost. Also report this predicate for L=4/16.
L=0 is the no-absorber control: code path unchanged and median difference
must be within twice the larger spread; otherwise comparison is confounded
and no gain is accepted. Expectation held means both CPML-8/300 ratios are
within 1.5–3.4 and timing/control gates pass. Record any fired gate plainly.

Before rows come from `w8b_nu_kernel_ablation_4090.json`. That JSON has
**no bare-slow or nu-uniform 400^3 row**: report those missing before cells
as unavailable, with any older G1 comparison explicitly labeled separately.
Do not substitute graded nu-z or scalar-inv for the requested lane.

## Results

**Decision: STOP — BIT-IDENTITY FIRED. No production localization accepted.**
The frozen expectation is **untested**, not met: correctness stopped this
lane before AD parity or GPU timing. The timing falsifier was not evaluated;
this result does not establish that whole-array traffic was or was not the
measured cost. The narrower claim that `_clip_lo/_clip_hi` broadcasts span
the domain was contradicted by the initial code reading.

Commits: pre-declaration `a1e8fcec`; initial candidate `887f8ff7`;
rejection evidence/tests `4629b6f9`; production restoration `b1e49a01`.
`git diff c30d3020 -- rfx/` is empty after restoration. The candidate is
archived at `validation/research/nu_cost/g4/cpml_candidate.py`; the reference
is an exact copy of baseline cpml.py in `cpml_baseline.py`.
The initial candidate localized coefficients, shifted neighbors and
read/add/write field corrections for both lanes. Further trimming unequal
faces within their existing axis-depth psi allocation was **not completed**:
the first correctness gate stopped development before that work. No new
boundary policy or count was shipped.

### Frozen bit-identity gate

Local environment: JAX 0.10.2, CPU/arm64, Python from the user-specified
venv, PYTHONPATH pinned to this worktree. This is a CPU rejection; no RTX
4090 correctness or performance claim follows from it. Fixture shape
29x29x29, 12 interior cells per axis plus Yee node and CPML-8 padding;
zero initial fields/psi, center soft ez Gaussian pulse, float32, 200 steps.
The unit fixture uses a `lax.scan` with the same H -> CPML-H -> E -> CPML-E
carry order as the two production runners; it does not claim an end-to-end
Simulation API test. Tests inspect all six fields and all 24 psi carries.

| candidate fixture | verdict |
|---|---|
| uniform CPML-8 soft source | FAIL: six fields and all 24 psi differ |
| graded-z CPML-8 | not run: STOP on first failed fixture |
| mixed PEC/PMC/CPML, unequal face counts | not run: STOP |
| CPML-4 | not run: STOP |
| CPML-16 | not run: STOP |
| periodic-axis CPML-8 (additional) | not run: STOP |
| kappa=3 CPML-8 (additional) | not run: STOP |

Focused candidate command stopped with **1 failed in 2.56s** (`-x`);
no failing value was accepted as epsilon-close. Verbatim comparison output:

```
uniform8
ex: equal=False, differing=20382, max_abs=4.76837158203125e-07
ey: equal=False, differing=20386, max_abs=3.5762786865234375e-07
ez: equal=False, differing=20392, max_abs=7.152557373046875e-07
hx: equal=False, differing=21168, max_abs=1.0577663189792474e-09
hy: equal=False, differing=21168, max_abs=1.2921370640128771e-09
hz: equal=False, differing=21168, max_abs=9.9403774100892406e-10
psi_ex_ylo: equal=False, differing=5292, max_abs=9.892483276985331e-09
psi_ex_yhi: equal=False, differing=5292, max_abs=1.5531984587369152e-08
psi_ex_zlo: equal=False, differing=5292, max_abs=1.2664713722188026e-08
psi_ex_zhi: equal=False, differing=5292, max_abs=2.7835767468786798e-08
psi_ey_xlo: equal=False, differing=5292, max_abs=9.8453094565797983e-09
psi_ey_xhi: equal=False, differing=5292, max_abs=1.8486556285779443e-08
psi_ey_zlo: equal=False, differing=5291, max_abs=1.1521933629410341e-08
psi_ey_zhi: equal=False, differing=5292, max_abs=2.9317561711650342e-08
psi_ez_xlo: equal=False, differing=5292, max_abs=1.1852250736410497e-08
psi_ez_xhi: equal=False, differing=5291, max_abs=2.463639248162508e-08
psi_ez_ylo: equal=False, differing=5292, max_abs=1.1981683201156557e-08
psi_ez_yhi: equal=False, differing=5292, max_abs=2.3092979972716421e-08
psi_hx_ylo: equal=False, differing=5284, max_abs=7.674098014831543e-06
psi_hx_yhi: equal=False, differing=4525, max_abs=4.7124922275543213e-06
psi_hx_zlo: equal=False, differing=5281, max_abs=6.7427754402160645e-06
psi_hx_zhi: equal=False, differing=4523, max_abs=6.1658211052417755e-06
psi_hy_xlo: equal=False, differing=5278, max_abs=7.8268349170684814e-06
psi_hy_xhi: equal=False, differing=4523, max_abs=5.0575472414493561e-06
psi_hy_zlo: equal=False, differing=5280, max_abs=5.9232115745544434e-06
psi_hy_zhi: equal=False, differing=4527, max_abs=4.0372833609580994e-06
psi_hz_xlo: equal=False, differing=5288, max_abs=7.3667615652084351e-06
psi_hz_xhi: equal=False, differing=4527, max_abs=5.3565017879009247e-06
psi_hz_ylo: equal=False, differing=5285, max_abs=6.970483809709549e-06
psi_hz_yhi: equal=False, differing=4532, max_abs=4.1320454329252243e-06
```

### Exact arithmetic divergence

The first divergent completed step is **13**, at ex[13,13,24] and
ey[12,13,24]. At that step all H and psi carries still agree. Both a
200-step scan history and a separate 13-step final-only scan reproduce:

```
13-step final without history baseline ex 1.2256234e-10 788972091
13-step final without history baseline ey 6.375286e-10 808402454
13-step final without history candidate ex 1.2256235e-10 788972092
13-step final without history candidate ey 6.375285e-10 808402452
```

The last columns are uint32 bit patterns (ex moves 1 ulp, ey 2 ulp).
The reordered operation is **the two reciprocal-spacing products and
subtraction in `rfx/core/yee.py::curl_h`**, lines 307–308 (ex) and
312–313 (ey), although that source file was never edited. For
`d1*inv - d2*inv`, inv=999.9999389648438:

- Baseline at z=24 is in the scalar tail (z=24..28):
  `fma(d1, inv, -round32(d2*inv))`.
- Candidate at z=24 is in vector code (its interior vectors cover z=1..28):
  `fma(-d2, inv, round32(d1*inv))`.

The baseline assembly has `fmul` then scalar `fnmsub` at offsets
0x1a0/0x1a4; the candidate uses vector `fmul.4s` then `fmls.4s`.
`baseline_ex.asm.txt` and `candidate_ex.asm.txt` archive the generated
arm64 disassembly. Removing the full-array padded shifts changes the
Yee fusion inputs: baseline has five inputs including pre-padded arrays
and outer-dimension partitioning; candidate has three inputs with neighbor
reads inside the fusion. Its vectorization boundary changes accordingly.
This is compiler contraction, not a reordered Python face loop.

`diagnosis.log` includes a scalar float64-emulated FMA replay (rounded once
to float32). It reproduces both exact outputs from identical input fields,
identical psi and identical coefficients. Running CPML-E alone on a
separately materialized Yee-E output gives identical baseline/candidate
outputs, further localizing the first discrepancy to the full-scan Yee
curl code generation. No alternate candidate was tuned or timed.

### AD parity

Verbatim status (there are no candidate AD measurements):

```
dz_profile: NOT RUN — frozen BIT-IDENTITY STOP fired first.
eps_r: NOT RUN — frozen BIT-IDENTITY STOP fired first.
```

Both requested gradient tests are provided, with a 200-step graded CPML-8
fixture and max-relative <=1e-6 predicate. Tests on restored production
compare baseline with baseline; they cannot establish parity for the
rejected candidate and must not be presented as candidate AD numbers.

### Before/after ladder

Rates and spreads are Mcells/s. Spread is max minus min of three windows.
All before rows below are from the frozen G1b JSON, without substituting a
different mesh or arm. No after rows or speedups exist because the
correctness gate fired before a GPU submission.

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
| 400 | bare-slow | 8 | unavailable in G1b | — | not run | — | — |
| 400 | nu-uniform | 8 | unavailable in G1b | — | not run | — | — |

The 0-layer code is unchanged (indeed all production code was restored),
but **unchanged throughput was not measured**; the timing control cannot be
claimed to have passed. The 1.5–3.4x expected gain remains untested.

VESSL submissions: **0**. Run ID: none. Harvest/deletion: not applicable.
No rsync staging, YAML submission, resubmission, or
`w9_cpml_localization_4090.json` was produced: a fabricated after artifact
would imply a measurement that never happened. No push or PR.

### Reproduction and delivered-tree validation

To reproduce the rejected gate from the final restored tree:

```
PYTHONDONTWRITEBYTECODE=1 RFX_G4_REJECTED_CANDIDATE=1 \
PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-cost \
/Users/byungkwankim/Documents/rfx/.venv/bin/python -m pytest \
  tests/unit/boundaries/test_cpml_localization.py -x -q -s \
  -o addopts= -p no:cacheprovider
```

The opt-in selects the archived candidate only; ordinary unit tests use
production CPML. `validation/research/nu_cost/g4/diagnose.py` automatically
selects the archived candidate and regenerates the history, scalar replay
and optimized HLO. Set `XLA_FLAGS=--xla_dump_to=<local-dir>` with
`--xla_dump_hlo_as_text` to additionally generate LLVM/object artifacts;
`otool -tvV` disassembles the `multiply_add_fusion.4` object on this Mac.
Generated HLO/NPZ/XLA dumps are ignored; the decisive logs and assembly
are committed. All files and command working directories stayed in this
worktree, except executing the specifically authorized Python/ruff binaries.

The requested full unit/contracts regression was started **after production
restoration**, as a check on the delivered artifacts, not continued candidate
development. Its counts are recorded below when complete.
Ruff, supplied E/F/W selection and exclusions, `--no-cache`:
**All checks passed!** (zero findings).

The first full regression encountered one new test-harness error in
`test_cpml_localization_ad[eps_r]`: a fixed JAX dz array was captured inside
`jit`, then the host grid builder indexed it and attempted `float(tracer)`.
This was a `ConcretizationTypeError` before a gradient comparison, not a
production failure. The fixed mesh now uses a host NumPy array of the same
float32 values when only eps is differentiated; the dz differentiation
still takes the traced design variable. No frozen fixture values or gate
thresholds changed, and the archived candidate was not changed or rerun.

Focused validation of **restored production**, after that harness correction:
**9 passed in 15.28s**. All seven identity fixtures have zero differing
elements in every field and psi. The following AD values are explicitly
**restored baseline vs baseline**, NOT rejected-candidate AD parity:

```
dz_profile: relative_max=0, ref_max=23997, max_abs=0
eps_r: relative_max=0, ref_max=30.753847122192383, max_abs=0
```

Ruff was repeated after the test correction: **All checks passed!**.
The original full-regression process had already imported the old test;
its final summary will retain that one harness failure. The focused rerun
above validates its correction without re-running unrelated tests.

Completed full-regression command (the exact requested selection, with
PYTHONDONTWRITEBYTECODE=1 added to avoid bytecode writes): exit **1**.
Verbatim summary:

```
2 failed, 4558 passed, 34 skipped, 269 deselected, 6 xfailed, 2153 warnings in 3318.33s (0:55:18)
```

Failures were BOTH introduced artifact-integration issues, now corrected:

1. `tests/unit/boundaries/test_cpml_localization.py::test_cpml_localization_ad[eps_r]`
   — fixed-mesh tracing error described above, corrected in `8e0eafed`.
   Focused rerun of all seven identity and two AD tests: **9 passed**.
2. `tests/contracts/test_example_fidelity_contract.py::test_discovery_matches_classification_table`
   — the three new research Python files lacked explicit classification
   entries. `9e37ebdf` registers the two archived operator modules and the
   low-level diagnosis script as `no_simulation` (none constructs an rfx
   `Simulation`; the contract machine-checks this against their ASTs).
   No contract predicate or snapshot was weakened or changed. Rerun of the
   ENTIRE fidelity-contract file, including the three new classification
   cases: **179 passed, 9 warnings in 35.58s** (exit 0).

There were no other failures. The named slow_physics oracles were not run
or modified. The full 55-minute command was not repeated after these two
artifact fixes; each affected test scope was rerun and passed as recorded.
This is a transparent record of the full command's two reds and their
successful targeted validation, not a claim that the original full command
exited green. Final ruff: **All checks passed!**, zero findings.

Final integrity checks: the original pre-declaration is an unchanged byte
prefix of this note; `git diff c30d3020 -- rfx/` is empty. Archived baseline
SHA256 `4979e28793d9530dd362f42d5aac913dd07360ffd757fdb7c0d3a9d3c7c3c3fd`
matches `c30d3020:rfx/boundaries/cpml.py`; candidate SHA256
`01e55ec9c9ce28a7e11da39e349ceca73583775f0ae2e20c6858c1a7a2da4959`
matches `887f8ff7:rfx/boundaries/cpml.py`.

**G4 remains STOPPED by its frozen identity gate.** No GPU run, no candidate
AD numbers, no speedup claim, no public API or production source change.
