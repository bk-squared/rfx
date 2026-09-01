# Issue #812 — cv03 re-gate: measured results

Companion to `issue812_cv03_dispersion_regate_predeclaration.md`, which froze
every threshold below in commits **preceding** these measurements. Nothing here
widened anything. Host: CPU, no Meep, no openEMS — so the E4 leg exits 2 and
the E2 leg added by this lane is what carries the case on this machine.

Reproduce:

```
PYTHONPATH=<worktree> .venv/bin/python validation/crossval/03_straight_waveguide_flux.py
PYTHONPATH=<worktree> .venv/bin/python scripts/diagnostics/cv03_flux/regate_falsifiers.py
PYTHONPATH=<worktree> .venv/bin/python -m pytest tests/test_cv03_slab_dispersion_oracle.py \
    -o addopts="" -m "not gpu"
```

## Criterion (A) — the case still passes on correct code, with margin

Unmodified case, exit 2 (Meep absent, as before this lane):

| gate | statistic | measured | threshold | margin |
|---|---|---:|---:|---:|
| G1 estimator premise | max two-wave rel. residual over the gated band | **0.0077** | ≤ 0.05 | 6.5x |
| G1 (E2) | max \|n_eff_rfx/n_eff_analytic − 1\| over the gated band | **0.262 %** | ≤ 2.0 % | **7.6x** |
| G2 (E1) | band-mean T | **0.9657** | 1.0 ± 0.05 | unchanged |
| G3 (E4) | Meep | SKIP | — | — |

Supporting numbers from the same run: band-mean `n_eff` rfx **2.84634** vs
analytic **2.84203**; the max-deviation bin is `f = 0.1592 c/a` (rfx 2.89607 vs
analytic 2.88849); per-bin deviations across the gated band are
`+0.013 +0.177 +0.160 +0.052 +0.128 +0.172 +0.101 +0.132 +0.199 +0.150 +0.160
+0.262 +0.229 +0.180` %.

**The case's physics output did not move.** Not asserted — measured: the
pre-change script, taken straight out of `main` with `git show
main:validation/crossval/03_straight_waveguide_flux.py`, and run on the same
host, prints `T_rfx(f_peak) = 0.9736` and `rfx band-mean T [0.135,0.165]:
0.9657`; the re-gated script prints the same two values to every digit. The
only addition to the run is a passive DFT plane probe.

The deviation is small and one-signed across the band (+0.01 % … +0.26 %),
which is the sign and order the §3 derivation predicts: the discrete Yee scheme
carries a slightly large `beta` in the core (+0.35 % … +0.52 % on its own at
these frequencies), partly cancelled by the transverse-eigenvalue term. The gate
is 7.6x wider than the residual it must tolerate and 2.7x tighter than the
smallest defect it must catch. **This margin is recorded so a future lane can
tighten with provenance; it must never be used to widen.**

## Criterion (B) — the new gate fails on the defect the audit measured, and the old one does not

`scripts/diagnostics/cv03_flux/regate_falsifiers.py`, one textual edit per row
applied to a copy of the case:

| case | exit | G1 dev | G1 | resid | G2 ⟨T⟩ | G2 | what it models |
|---|---:|---:|:--|---:|---:|:--|---|
| baseline | 2 | 0.262 % | PASS | 0.0077 | 0.9657 | PASS | unmodified (criterion A) |
| F1 `eps=11` | 1 | **5.361 %** | **FAIL** | 0.0135 | 0.9700 | PASS | the audit's own `eps_wg` edit |
| F1 `eps=10` | 1 | **10.968 %** | **FAIL** | 0.0118 | 0.9882 | PASS | " |
| F1 `eps=8` | 1 | **22.863 %** | **FAIL** | 0.0226 | 1.0257 | PASS | " (audit endpoint) |
| F2 `d=0.9a` | 1 | **3.155 %** | **FAIL** | 0.0157 | 1.0103 | PASS | 9-cell guide, **no declared constant changed** |
| F2b 1-edit | 1 | 3.192 % | FAIL | 0.0356 | 0.9428 | FAIL | same width error, careless single edit |
| F3 no-subpixel | 2 | 0.580 % | PASS | 0.0137 | 0.9943 | PASS | solver-flag regression (reported, not required) |

**The F1 row reproduces the audit exactly.** Its band-mean T values —
0.9657 / 0.9700 / 0.9882 / 1.0257 — are the audit's four published numbers to
every printed digit, so this is the same measurement, not a similar one. What is
new is the G1 column beside them.

G1's failure message on every failing row is the intended one:

> reason: the simulated guide's dispersion is not that of the declared recipe
> (eps=12, d=1a)

and the printed line names both operands, e.g. at `eps=8`:
`max |n_eff_rfx/n_eff_analytic - 1| over band: 22.863% at f=0.1367 (rfx 2.13720
vs analytic 2.77066)`, with `n_eff band mean: rfx 2.20159 analytic 2.84203`.
Note what that says: 2.20159 is the *correct* index for an `eps = 8` guide (the
closed form gives 2.20300), so G1 is not reporting a solver error — it is
reporting that the structure being solved is not the structure the case
declares. The measured shifts track the closed-form predictions of
§2 (−5.30 / −10.82 / −22.54 %) to within 0.32 percentage points, so the gate is
firing on the physics, not on an artefact of the estimator.

**F2 is the row that answers "you only compared two literals".** It edits the
geometry-construction line and the source span so a 9-cell core is built
consistently; the script still declares `eps_wg = 12.0` and `wg_width = 1.0`,
and `RECIPE_EPS_WG` is untouched. G1 fires at 3.155 % — 1.6x the gate, against
the closed form's −2.95 % for `d = 0.9a` — while G2 reads 1.0103 and passes.
The gate measures the guide.

F2b is the same physical width error introduced by one careless edit that also
pushes the topmost source out of the core; there G2 fails too, but on the stray
source's radiation, not on the width. It is reported so the two mechanisms stay
distinguishable.

F3 is reported and **not** claimed: turning subpixel smoothing off moves the
deviation 0.262 % → 0.580 %, a real 2.2x signal that stays inside the budget.
This guide's faces are grid-aligned at `dx = a/10`, so there is little for
smoothing to do; G1 does not claim to catch this defect class and does not.

## Estimator self-checks

`tests/test_cv03_slab_dispersion_oracle.py`, 12 tests, all passing:

- **S2**: the closed-form oracle vs an independent 1-D FD Helmholtz eigensolve
  that shares no code with it — agreement 1.35e-4 … 1.57e-4 relative across
  `eps = 8, 10, 11, 12`, consistent with that grid's own O(h²).
- **S1'**: the two-wave estimator on synthetic lines with `|B/A| = 0, 0.5, 0.9`
  — `n_eff` recovered to < 1e-9 relative, `|B/A|` to < 1e-9.
- The replaced estimator's failure is pinned as a test: on a noise-free
  synthetic line with `|B/A| = 0.53`, a plain unwrapped-phase slope is wrong by
  more than 0.5 %, a quarter of the whole G1 budget, while the two-wave fit is
  exact.
- The arithmetic of the eps sweep firing G1 is pinned without any FDTD.

## What did NOT change

- No physics verdict of cv03. `T(f_peak) = 0.9736` and band-mean `T = 0.9657`
  are the same numbers the case produced before this lane.
- G2's value and statistic are byte-identical; only its label and its standing
  in the exit logic changed.
- G3 (Meep) is untouched and still exits 2 without Meep.
- The known cv03 caveats stay as written: the 11.5 cells/λ_eff preflight
  advisory, the fixed 400 a/c₀ integration window, and the #160 flux-region
  congruence note.

## Open, and NOT fixed here

§8 and §8.1 of the pre-declaration: this guide carries a standing wave with
`|B/A| = 0.393 … 0.585` over the gated band (0.53 at the carrier). A
time-of-flight test settles what it is — a reflection of the guided mode off the
domain termination, `|B/A| = 0.0002` before the round trip can complete and
0.4979 after, i.e. about **−6 dB in amplitude, 25 % in power** — and does not
settle why it is four times worse than the ledger's resolved hollow-WR-90
entry or why it worsens with absorber depth. That is a physics/solver finding
and needs its own issue and owner. This lane only measured it, reported it, and
built an estimator that is correct in its presence.

One number from that test belongs here, because it is the cleanest statement of
criterion (A) available: in the reflection-free window the measured `n_eff` is
**2.84338** against the closed form's **2.84411**, a deviation of **−0.026 %**,
77x inside G1's 2.0 %. The 0.262 % of the baseline row is what the recipe's own
16a domain and 400 a/c₀ window cost, not what the solver's dispersion costs.

## Corrections to this file

- **2026-09-01, before publication.** The first draft of the "G1 message"
  paragraph above quoted the `eps = 8` operands as `rfx 2.13383 vs analytic
  2.76610`. Those digits were written from the line's *format* rather than read
  off the run. The verified values, from re-running the committed case with the
  single edit `eps_wg = 12.0` -> `eps_wg = 8.0`, are **`rfx 2.13720 vs analytic
  2.77066`**. The deviation, 22.863 %, and every other number in the tables were
  read from the driver's own output and are unaffected. Recorded rather than
  silently fixed, because this issue has shipped unverified digits into durable
  documents twice already.
