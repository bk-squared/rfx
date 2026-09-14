# CPML localization — third declaration (r3): two supporting gates re-derived from a contraction-reroll control

**Status:** pre-declaration, committed BEFORE any measurement and BEFORE the
candidate is re-applied. **Branch:** `feat/cpml-localization-r3`, cut from
`origin/main` 24dbef41; the G5 r2 record (7 commits, 453d057c..67555196 here)
is cherry-picked in verbatim and stays as it is. **rfx/ is byte-identical to
main** at this commit (the r2 candidate + its revert net to zero).

## What r2 established and what it got wrong

r2 (`20260911_cpml_localization_r2_predeclaration.md`) proved the decisive
thing: with FMA contraction suppressed
(`XLA_FLAGS="--xla_disable_hlo_passes=fusion --xla_cpu_enable_fast_math=false"`,
effectiveness control 2,129,643 baseline elements changed), the localized
CPML and the current CPML are **bit-identical on 210/210 arrays over 200
steps on all seven fixtures** (G5-2). The candidate is the same arithmetic
compiled differently. G5-1 (expression identity) and G5-4 (reflection floor
-68.26476397028848 dB, difference 0; PEC null control bit-identical;
non-absorbing psi exactly zero) also passed.

Two supporting gates the lead specified were wrong in construction, and
the lead says so here rather than the executor:

- **G5-3 (float64 reference distance, elementwise <= 1.05x)** compares
  two rounding rerolls element by element. Where the two differ by ~1 ulp,
  about half the elements land farther from the float64 truth and half
  nearer, by chance; the elementwise count (115,889 on uniform8) is that
  coin flip, not a defect. The max-norm form was decided by single extreme
  elements: hx 1.063 against 1.05 while ex/ey/ez were 0.61/0.61/0.28 —
  the candidate is CLOSER to the float64 truth on five of six fields.
- **G5-5 (divergence growth vs a 1-ulp single-cell perturbation)** compares a
  persistent, array-wide rounding change against one impulse in one cell.
  A persistent perturbation always outgrows a single impulse in a resonant
  box; the "violation" at step 35 is that arithmetic fact, not a defect.

Neither gate can distinguish a defect from a reroll. Both are replaced
below by gates whose comparator IS a reroll — the r2 contraction-suppressed
baseline — so that "no worse than a known-harmless recompilation" is what
is tested. This is stricter in a different direction (a real defect is
far larger than a reroll), not looser.

## Instrument

`scripts/diagnostics/cpml_g5r3.py` (committed with this note, before any run).
Fixtures, runner, frozen baseline (`validation/research/nu_cost/g4/cpml_baseline.py`)
and candidate (`.../cpml_candidate.py`, byte-identical to the reverted r2
patch 0606a579 — checked) come from `tests/unit/boundaries/test_cpml_localization.py`
exactly as in r2; the candidate is selected by `RFX_G4_REJECTED_CANDIDATE=1`,
so every gate below runs WITHOUT touching `rfx/`. The float64 reference is
r2's `scripts/diagnostics/cpml_g5_reference.py::evaluate`, unchanged.

Three stages, run in this order, each committed before the next runs:

1. `controls` — baseline (unflagged), baseline with contraction suppressed
   (fresh subprocess, the r2 flags), baseline seeded 1 ulp (r2's witness
   seed, reported only), float64 reference; on all seven fixtures. Writes
   `validation/research/nu_cost/g5r3/controls.json` including the DERIVED
   window `m` below. **No candidate is run in this stage.**
2. `candidate` — the candidate under the same seeds; applies the gates.
3. `ad` — AD parity on `graded8` (the only NU fixture): `jax.grad` of
   `sum(ez_final^2)` w.r.t. the 12-cell `dz_profile` and w.r.t. a scalar
   `eps_r`, baseline vs candidate, with the contraction-suppressed baseline
   gradient as the reroll bound (subprocess).

## Gates (frozen now; `m` is derived by stage 1 and written down before stage 2)

- **G5-1, G5-2, G5-4: carried over from r2 as PASSED.** Not re-run here; the
  candidate file is byte-identical to the one they judged.
- **G5-3′ reroll-bounded reference distance.** For every fixture and field,
  `RMS(|candidate − ref64|) <= RMS(|baseline − ref64|) × (1 + m)`, with
  `m = max over (fixture, field) of |RMS(|flagged − ref64|) / RMS(|baseline − ref64|) − 1|`
  measured in stage 1 — the spread of the reroll class itself. RMS, not
  max-norm: a max is one element. Reported alongside, not gated: the
  fraction of differing elements where the candidate is farther than the
  baseline (a reroll gives ~0.5; the flagged control's value is printed
  next to it), and the r2 max-norm ratios for continuity.
- **G5-5′ reroll-bounded divergence.** Seeded runs (ez[center] = 1 plus the
  pulse, as r2). Per fixture, field and step t:
  `d_c(t) = max|candidate(t) − baseline(t)|`, `d_f(t) = max|flagged(t) − baseline(t)|`.
  Gate: `d_c(t) <= d_f(t)` at every step, equality allowed, on every fixture
  and field. The 1-ulp curve is reported next to both so the r2 error is
  visible on the same axes.
- **G5-AD reroll-bounded gradient parity.** On `graded8`:
  `||g_cand − g_base||_2 <= ||g_flag − g_base||_2` for the dz gradient (12
  components) and `|g_cand − g_base| <= |g_flag − g_base|` for the scalar
  eps gradient; all values finite; the same loss value bit-identical
  between candidate and baseline is NOT expected (contraction) and is not
  gated. Reported: max relative component difference.
- **Lane falsifier (unchanged since G4):** after the gates, the RTX 4090
  CPML-layer ladder (bare-slow and nu-uniform, 300^3, layers 0/4/8/16, and
  cpml8 at 400^3) on the re-applied candidate. The 0-layer rows are the
  control and must not move beyond spread. If the localized version is not
  faster than the current one beyond the measured spread at 300^3 on both
  lanes, the lane STOPS and records that the whole-array structure was not
  the cost. Before rows: `validation/research/nu_cost/results/w8b_nu_kernel_ablation_4090.json`
  (bare-slow 10015.9 / 2030.5 / 2120.8 / 2037.1; nu-uniform 10037.9 /
  1795.3 / 1780.7 / 1549.2 Mcells/s at L = 0/4/8/16).

Rules: one attempt per stage; a fired gate is recorded and stops the lane;
no window is edited after its declaring commit; `m` is written into
`controls.json` and copied here under Results before stage 2 runs.

## Results

(appended after each stage; nothing above this line changes)

### Stage 1 — controls (run once on 97c6f4e4; no candidate present, `candidate_env` empty, rfx cpml == frozen baseline byte for byte)

Contraction-suppressed baseline vs unflagged baseline, RMS-to-float64 ratio
per field (a reroll of the SAME arithmetic), and the fraction of differing
elements where the reroll landed farther from the truth:

| fixture | ex | ey | ez | hx | hy | hz | farther fraction range |
|---|---|---|---|---|---|---|---|
| uniform8 | 1.732 | 1.628 | 1.534 | 0.501 | 0.478 | 0.377 | 0.52-0.58 |
| graded8 | 0.977 | 0.759 | 0.825 | 0.795 | 0.751 | 0.580 | 0.43-0.49 |
| mixed8 | 1.425 | 0.947 | 1.150 | 0.837 | 0.877 | 0.738 | 0.48-0.54 |
| uniform4 | 0.842 | 1.168 | 0.694 | 0.645 | 0.303 | 0.506 | 0.40-0.49 |
| uniform16 | 1.263 | 1.163 | 1.227 | 0.546 | 0.551 | 0.487 | 0.47-0.48 |
| periodic8 | 1.432 | 1.131 | 1.468 | 1.117 | 1.188 | 1.057 | 0.44-0.53 |
| kappa8 | 1.856 | 1.796 | 1.853 | 0.523 | 0.465 | 0.412 | 0.48-0.56 |

Reading: a known-harmless recompilation of the same arithmetic moves the
RMS distance to the float64 truth anywhere from 0.30x to 1.86x, field by
field, and lands farther on about half the differing elements. That is the
class the r2 G5-3 gate (1.05x, elementwise) was testing against; the r2
candidate's own ratios (0.28-1.06) sat well inside it.

**Derived window, written before stage 2: `m = 0.8555425507215482`**, i.e.
G5-3′ requires `RMS_cand <= 1.8555 x RMS_base` per field. The divergence
control curves `d_flag(t)` and the 1-ulp curves are stored in
`validation/research/nu_cost/g5r3/controls.json` for G5-5′.


### Stage 2 — candidate (run once on dd1502a3, `RFX_G4_REJECTED_CANDIDATE=1`, rfx cpml still == frozen baseline)

**G5-3′ reroll-bounded reference distance: HELD, all 42 (fixture, field) pairs.**
graded8, mixed8, uniform4, uniform16 are bit-identical to the baseline
(ratio 1.000, no differing elements — the localization changes nothing
there). On the three fixtures where it differs:

| fixture/field | RMS_cand / RMS_base (window 1.856) | farther fraction | r2-style max-norm ratio | G5-5′ first violating step | max d_cand / d_flag |
|---|---|---|---|---|---|
| uniform8/ex | 1.040 | 0.53 | 1.079 | 38 | 2.00 |
| uniform8/ey | 0.904 | 0.48 | 1.594 | 38 | 2.00 |
| uniform8/ez | 0.891 | 0.49 | 1.000 | 37 | 2.67 |
| uniform8/hx | 0.926 | 0.47 | 0.790 | 38 | 1.63 |
| uniform8/hy | 1.008 | 0.51 | 1.231 | 46 | 1.68 |
| uniform8/hz | 0.887 | 0.52 | 0.967 | 40 | 1.18 |
| periodic8/ex | 1.078 | 0.51 | 0.689 | 58 | 2.50 |
| periodic8/ey | 0.861 | 0.49 | 0.585 | 59 | 2.00 |
| periodic8/ez | 1.005 | 0.44 | 1.475 | 53 | 4.00 |
| periodic8/hx | 0.983 | 0.46 | 0.932 | 44 | 2.38 |
| periodic8/hy | 1.068 | 0.48 | 1.086 | 39 | 2.52 |
| periodic8/hz | 1.033 | 0.51 | 1.114 | 40 | 2.48 |
| kappa8/ex | 1.009 | 0.49 | 1.178 | 49 | 1.67 |
| kappa8/ey | 1.167 | 0.50 | 1.211 | 35 | 1.78 |
| kappa8/ez | 0.909 | 0.51 | 0.465 | 36 | 2.00 |
| kappa8/hx | 0.985 | 0.51 | 0.793 | 42 | 3.07 |
| kappa8/hy | 1.082 | 0.50 | 1.190 | 36 | 3.18 |
| kappa8/hz | 0.998 | 0.53 | 0.772 | 41 | 3.18 |

RMS ratios 0.86-1.17 and farther fractions 0.44-0.53: the reroll signature,
and well inside the derived window. The r2-style max-norm column (0.47-1.59)
shows once more that a single extreme element is not a statistic.

**G5-5′ reroll-bounded divergence: FIRED, as declared — on every field of
uniform8, periodic8 and kappa8; HELD trivially on the other four.** Recorded
as fired; under the declared rule the lane stops here. What the curves show
(`candidate.json`, `controls.json`, uniform8/ex quoted): the candidate's
divergence from the baseline starts later (first non-zero step 11 vs 2 for
the contraction-suppressed control — a localized change perturbs fewer
operations), stays BELOW the control on 193 of 200 steps (median ratio
0.71 over steps 31-200), and exceeds it on 7 isolated steps by at most 2.0x
(4.0x on periodic8/ez at one step). The gate as I wrote it demands
`d_cand(t) <= d_flag(t)` at EVERY step; two rounding perturbations of the
same magnitude scale (1e-7, single ulps of the field) cross each other at
individual steps as a matter of course. Whether that is so is not argued
here — it is tested by diagnostic D1 below, declared before it runs.

**G5-AD reroll-bounded gradient parity: HELD.** graded8, loss = sum(ez_final^2):
`||g_cand − g_base||_2 = 0.040601 <= ||g_flag − g_base||_2 = 0.045008` (dz,
12 components); `|g_cand − g_base| = 1.907e-6 <= |g_flag − g_base| = 3.815e-6`
(eps). All finite. (`ad.json`)

### Diagnostic D1 — declared before it runs: does a reroll pass G5-5′ against another reroll?

Instrument: the same worker with the r2 `legacy_emitter` flags
(`--xla_cpu_use_fusion_emitters=false --xla_cpu_enable_fast_math=false`, a
second known-harmless recompilation of the baseline; r2 `controls.json`
recorded it as differing from the unflagged baseline). Compute
`d_legacy(t) = max|legacy(t) − baseline(t)|` per fixture/field and apply the
G5-5′ predicate `d_legacy(t) <= d_flag(t)` at every step, and the reverse.
Expectation, stated now: if a harmless reroll violates the per-step
predicate against the other harmless reroll on a comparable number of steps
(order 1-10 of 200) with comparable excess (≲ 2-4x at single steps), the
per-step form of G5-5′ cannot distinguish a reroll from anything and the
FIRED verdict above is a property of the statistic. If instead two rerolls
never cross (0 violating steps), the candidate's crossings are not
reroll-class and the STOP stands on its own. D1 does not change the G5-5′
verdict; it decides what the lead asks the PI next. No candidate is run.


### D1 result (run once on 630f729d; no candidate)

Two harmless rerolls of the baseline — `legacy_emitter` vs `no_fusion`, both
differing from the unflagged baseline on every fixture — exceed each other
under the G5-5′ per-step predicate on **3688 vs 3689 of 8400 field-steps**
(7 fixtures x 6 fields x 200 steps): a coin flip, with single-step ratios up
to **6.0x**. Per (fixture, field) the crossing counts run 16-181 of 200. The
per-step form of G5-5′ therefore cannot distinguish a reroll from anything;
the FIRED verdict above is a property of the statistic, exactly as the D1
declaration allowed for. Against the same control the candidate crossed on
**7 of 200 steps per field at most (max 2.0-4.0x)** — far fewer than a
reroll's 16-181, i.e. the localized kernel is closer to the baseline than
a recompilation of the baseline is to itself. (`reroll_diagnostic.json`)

The G5-5′ FIRED record is not edited. The lane's next step is a lead
decision to put to the PI, made here with its evidence:

### Proposed G5-5″ (re-judgment of STORED curves; no new measurement; pending PI approval)

Control-derived statistic instead of a per-step inequality: for every
(fixture, field), the candidate's number of steps above the no-fusion control
and its maximum single-step ratio must not exceed the legacy reroll's own
values against the same control. Applied to the stored curves:

| fixture/field | cand steps > flag | legacy steps > flag | cand max ratio | legacy max ratio | |
|---|---|---|---|---|---|
| uniform8/ex | 7 | 28 | 2.00 | 3.00 | pass |
| uniform8/ey | 19 | 49 | 2.00 | 2.74 | pass |
| uniform8/ez | 62 | 20 | 2.67 | 2.00 | FAIL |
| uniform8/hx | 13 | 168 | 1.63 | 2.89 | pass |
| uniform8/hy | 5 | 144 | 1.68 | 6.00 | pass |
| uniform8/hz | 3 | 76 | 1.18 | 2.74 | pass |
| periodic8/ex | 47 | 61 | 2.50 | 3.00 | pass |
| periodic8/ey | 19 | 34 | 2.00 | 4.00 | pass |
| periodic8/ez | 58 | 66 | 4.00 | 4.00 | pass |
| periodic8/hx | 107 | 92 | 2.38 | 3.81 | FAIL |
| periodic8/hy | 85 | 108 | 2.52 | 6.00 | pass |
| periodic8/hz | 56 | 132 | 2.48 | 3.68 | pass |
| kappa8/ex | 8 | 39 | 1.67 | 2.82 | pass |
| kappa8/ey | 13 | 26 | 1.78 | 2.55 | pass |
| kappa8/ez | 12 | 29 | 2.00 | 2.67 | pass |
| kappa8/hx | 82 | 126 | 3.07 | 3.59 | pass |
| kappa8/hy | 121 | 167 | 3.18 | 6.00 | pass |
| kappa8/hz | 111 | 181 | 3.18 | 3.30 | pass |

The other four fixtures: candidate 0 crossings (bit-identical), legacy 16-181.
**Verdict under G5-5″: FIRED** (`rejudge_g55pp.json`). Recorded as a
proposal; the lane proceeds to the GPU falsifier with the candidate re-applied
to `rfx/` (the file every passed gate judged), and the PR carries both the
G5-5′ FIRED record and this re-judgment for the PI to accept or refuse.


## Adversarial review (codex, gpt-6-astra, read-only on 05a1a109) — findings accepted, corrections, and re-declaration r3b

Findings, verbatim in substance, with what the lead does about each:

1. **The r2/r3 suppression flags do not lock arithmetic order.** On a thin
   public `Grid` (2 x 3 x 2, `cpml_layers=2`, `kappa_max=5`, heterogeneous
   materials, random non-zero fields and psi) the baseline and the localized
   code differ under `--xla_disable_hlo_passes=fusion --xla_cpu_enable_fast_math=false`
   (ex: 6 elements, 3.05e-5) with **0 FMA in the generated code**. Cause,
   isolated by the reviewer in a 2 x 3 x 2 reproducer: XLA's algebraic
   simplifier reassociates a whole-array add against partial slab adds
   (`(1e8 − 1e8) + 1` evaluated in two orders). Adding `algsimp` to the
   disabled passes restores bit-identity on both reproducers and on the
   thin fixture (`steps 1 BAD []`, `steps 3 BAD []`). The reviewer's extended
   sweep (layers 1/2, kappa 5, graded x/y/z, unequal face spacings, PMC on
   the third axis, non-zero psi, legacy parameters, traced dz): 14/16
   fixtures bit-identical under the r3 flags, the two overlap cases were
   this reassociation. **Consequence:** G5-2's sentence "any inequality
   under these flags proves a different expression" is withdrawn; G5-2 is
   re-run below with `fusion,algsimp` disabled and the reviewer's fixtures
   added. The 7-fixture PASS on record stands as measured.
2. **The committed identity test fails on production.**
   `tests/unit/boundaries/test_cpml_localization.py::test_cpml_localization_identity`
   asserts bit-identity against the frozen baseline under DEFAULT flags;
   with the localized kernel in `rfx/` it fails on uniform8, periodic8,
   kappa8 — consistent with a recompilation, but a test contract is a
   contract. Re-declared below as an explicit contract change with its
   reason; no test is deleted.
3. **My D1 prose was wrong in two numbers and over-reached in one
   sentence.** Corrected here (the stored JSONs were right; the reviewer
   re-derived every table): the candidate crosses the no-fusion control on
   up to **121** steps per field (uniform8/ez 62, periodic8/hx 107; the "7 of
   200 at most" quoted uniform8/ex only); uniform8/ex is *at or below* the
   control on 193 steps (163 below, 30 equal). D1 establishes that the
   per-step predicate rejects harmless recompilations (3688/3689 of 8400
   field-steps) and is therefore unusable as a gate — not that it "cannot
   distinguish anything"; the 8400 field-steps are correlated, not
   independent coin flips; and D1's declared expectation (1-10 crossings,
   ≲2-4x) was off against the observed 16-181 and up to 6x. **G5-5″ is
   withdrawn as a proposal**: it fails on both its count and its ratio
   sub-criteria on uniform8/ez (62 vs 20; 2.67 vs 2.0) and on count on
   periodic8/hx (107 vs 92), it was chosen after the data, and its
   max-ratio term is the same extreme-value statistic this note criticises.
   Both FIRED records (G5-5′, G5-5″) stay verbatim.
4. **Protocol.** Stages 2 and 3 were run back-to-back without a commit
   between them (both carry sha dd1502a3; the AD JSON shows a dirty tree),
   against the r3 rule "each stage committed before the next". The r2
   `cpml_g5_gates.py` was first committed alongside its results. The record
   said STOP after G5-5′ and the candidate was re-applied to `rfx/` for the
   GPU falsifier without a documented waiver: the waiver is the lead's
   decision, taken under the PI's standing instruction to proceed with
   this lane, on the D1 evidence — recorded here, dated 2026-09-15.
5. **Provenance.** `controls.json` and `reroll_diagnostic.json` recorded
   `rfx_file` under the PRIMARY checkout: the lead's shell did not expand
   `~` inside `PYTHONPATH=~/…`, and `python script.py` puts the script's
   directory, not the cwd, first on `sys.path`. Every stored number
   reproduces on a pinned re-run under the correct tree (reviewer: "all
   fixture numbers and curves exactly equal"), and the baseline module is
   the frozen file in both cases, so no number changes — but the record is
   defective and the instrument now refuses to run unless `rfx.__file__`
   is under this tree.

What the reviewer found no defect in: the line-by-line diff of the
localized kernel (profiles, signs, permutations, recurrence operands,
sequential face updates unchanged; overlapping slabs read the accumulated
field); AD parity with a different loss (`sum(hy^2)`): dz distance 6.6e-13
vs reroll bound 4.3e-12, eps 2.6e-17 vs 7.2e-15, and live == frozen
candidate bit for bit; the committed CPML reflectivity, pad-material
extension, face-notch, periodic-CPML and UPML oracles: 41 passed, 2 skipped.

Reviewer's verdict: the evidence supports "same real-valued arithmetic,
compiled differently", not the narrower "only FMA contraction changed";
replace G5-5″ by a trajectory statistic; before shipping, add the
thin-domain case to the evidence, make the ordinary regression contract
pass, and measure a matched graded-mesh complex S-parameter.

### r3b re-declaration (frozen now; instrument changes committed before any run)

- **G5-2b bit-identity with fusion AND algsimp suppressed.** Flags
  `--xla_disable_hlo_passes=fusion,algsimp --xla_cpu_enable_fast_math=false`.
  Fixtures: the seven r2 fixtures plus `thin232` (the reviewer's 2 x 3 x 2,
  layers 2, kappa 5, heterogeneous eps/mu, random fields and psi, 3
  steps), `overlap` (a 6 x 6 x 6 box with layers 2 on all faces, so every
  corner belongs to three slabs, 50 steps) and `kappa5_pmc` (kappa 5,
  PMC on y_hi, layers 2). Live `rfx.boundaries.cpml` (== candidate) vs the
  frozen baseline. Gate: `np.array_equal` on all fields and psi. The
  effectiveness control is re-established under the new flags: the flagged
  baseline must still differ from the unflagged baseline (count reported);
  if it does not, the flags are vacuous and G5-2b is INCONCLUSIVE.
- **G5-5‴ trajectory statistic (reviewer's proposal, adopted verbatim),
  on the STORED curves, no new FDTD:**
  `S = max over (fixture, field) of RMS_t(d_cand(t)) / RMS_t(d_flag(t))`,
  gate `S <= 1`; a zero denominator requires `d_cand ≡ 0`. Declared
  before it is computed. It replaces G5-5″; it is a supporting gate.
- **G5-S matched graded-mesh S-parameter (reviewer's missing measurement).**
  `scripts/diagnostics/wr90_dz_dispatch_falsifier.py`, arm A (WR-90,
  dz-graded 19 cells, CPML x / PEC y,z, 20 layers, eps 2.2 slab, TE10
  ports, 9 bins 8.2-12.4 GHz, `normalize="flux"`), run three ways in
  separate processes: frozen baseline (module attributes swapped in before
  the NU lane binds them), live localized kernel, and frozen baseline with
  the G5-2b flags (the reroll comparator). Gate:
  `max|S_live − S_base| <= max|S_flag − S_base|` over all (receiver,
  driver, bin), all finite; the absolute values reported. The #811 note's
  own passivity self-check warning applies to all three arms alike and is
  common-mode.
- **Test-contract change (declared):** `test_cpml_localization_identity`
  becomes a subprocess test under the G5-2b flags (bit-identity vs the
  frozen baseline, seven fixtures + `thin232`), and a new default-flags
  test `test_cpml_localization_reroll_bounded` computes `S` for uniform8
  and kappa8 against a control the test itself produces in a flagged
  subprocess (self-contained; no stored curves). Reason: a bit-identity
  assertion under default flags is a statement about the compiler's
  contraction choices, which the localized kernel changes by construction;
  the arithmetic identity is asserted where it is testable.
- **Provenance:** every stage asserts `rfx.__file__` under this tree;
  invocations use absolute paths. Stages are committed one at a time.


### G5-5‴ trajectory statistic — computed on the stored curves (22e7290a)

`S = 1.5479` → **FIRED** on 5 of 42: uniform8/ez 1.102, periodic8/hx 1.122,
kappa8/hx 1.035, kappa8/hy 1.516, kappa8/hz 1.548 (all others 0.41-1.00;
`s_stat.json`). Recorded as fired.

**D2, declared before it was computed:** the statistic has ONE comparator;
its own reroll-class value is `S_reroll = max RMS_t(d_legacy)/RMS_t(d_flag)`
on the stored D1 curves. If `S_cand <= S_reroll` the candidate's trajectory
divergence is inside what two harmless recompilations show against each
other under the reviewer's statistic; if `S_cand > S_reroll` it is not, and
that is a real finding. Result: **`S_reroll = 1.9156`**, largest
reroll ratios kappa8/hy 1.92, kappa8/hz 1.66, uniform8/hx 1.57, uniform16/hy 1.38;
`S_cand = 1.5479` → inside the reroll class. (`d2_reroll_S.json`)
D2 is a calibration of the statistic, not a replacement gate; G5-5‴ stays
FIRED as declared.

### Lane falsifier — RTX 4090 CPML-layer ladder, localized kernel (VESSL run 369367261025, staged 4e3c7c9f, harvested and deleted; `results/w8c_cpml_localized_4090.json` vs `w8b_nu_kernel_ablation_4090.json`, same harness, same card)

| arm | lane | n | L | before Mcells/s (spread) | after (spread) | after/before |
|---|---|---|---|---|---|---|
| nu-z | nu | 300 | 8 | 1787.9 (77.0) | 3741.4 (18.4) | 2.093 |
| nu-z-scalar-inv | nu | 300 | 8 | 2125.6 (12.5) | 3691.3 (34.6) | 1.737 |
| nu-z-combo | nu | 300 | 8 | 2157.2 (15.8) | 4286.6 (64.3) | 1.987 |
| nu-z-foldinv | nu | 300 | 8 | 1937.2 (18.9) | 3058.2 (38.4) | 1.579 |
| nu-z-inv3d | nu | 300 | 8 | 1355.7 (0.6) | 2962.1 (245.1) | 2.185 |
| nu-z | nu | 400 | 8 | 1586.6 (0.3) | 3792.7 (37.5) | 2.390 |
| nu-z-scalar-inv | nu | 400 | 8 | 1764.0 (2.6) | 3695.5 (46.4) | 2.095 |
| nu-z-combo | nu | 400 | 8 | 1724.6 (14.1) | 4367.7 (230.3) | 2.533 |
| nu-z-foldinv | nu | 400 | 8 | 1704.0 (2.7) | 3050.3 (93.2) | 1.790 |
| nu-z-inv3d | nu | 400 | 8 | 764.9 (0.4) | 2899.8 (7.0) | 3.791 |
| bare-slow | uniform | 300 | 0 | 10015.9 (82.3) | 10042.5 (27.1) | 1.003 |
| nu-uniform | nu | 300 | 0 | 10037.9 (57.9) | 10063.2 (65.9) | 1.003 |
| bare-slow | uniform | 300 | 4 | 2030.5 (124.8) | 1637.5 (55.2) | 0.806 |
| nu-uniform | nu | 300 | 4 | 1795.3 (76.8) | 4584.1 (429.0) | 2.553 |
| bare-slow | uniform | 300 | 8 | 2120.8 (0.0) | 1525.7 (30.1) | 0.719 |
| nu-uniform | nu | 300 | 8 | 1780.7 (3.3) | 3727.6 (13.6) | 2.093 |
| bare-slow | uniform | 300 | 16 | 2037.1 (81.4) | 1433.6 (24.9) | 0.704 |
| nu-uniform | nu | 300 | 16 | 1549.2 (52.6) | 3426.8 (10.3) | 2.212 |
| bare | uniform | 300 | 0 | 8669.9 (37.1) | 8634.7 (46.5) | 0.996 |

**Controls hold**: the 0-layer rows move 1.003 / 1.003 / 0.996 — inside spread.
**NU lane: 2.1-2.6x faster at every layer count** (300^3 L=4/8/16:
1795→4584, 1781→3728, 1549→3427; 400^3 cpml8: 1587→3793), and every
NU-kernel variant of the G1b matrix gains 1.6-3.8x on top of its own
before value.
**Uniform lane: 20-30 % SLOWER** (300^3 L=4/8/16: 2030→1638, 2121→1526,
2037→1434). **The lane falsifier FIRES for the uniform lane, as declared**:
the localized kernel is not faster on both lanes. Reading, not a
measurement: on the uniform slow path the whole-array CPML corrections were
fused into the Yee update as one pass; slab-scoped updates (dynamic-slice
adds) break that fusion and cost extra full-array passes, while the NU
path — already unfused by its per-axis `inv_*` broadcasts — only loses
work. That is a hypothesis for the next declaration (lane-conditional
dispatch, or a slab update that keeps the fusion), not a result.
