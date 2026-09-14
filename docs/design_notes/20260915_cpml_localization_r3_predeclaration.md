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
