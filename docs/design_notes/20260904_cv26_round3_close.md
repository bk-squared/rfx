# cv26 oblique slab Fresnel — round 3 closing report

Lane: `agent/gap4-oblique-fresnel`, rebased onto `agent/issue-888-aux-absorber`.
Answers the diagnosis in `docs/design_notes/20260903_cv26_oblique_defect_diagnosis.md`
(issue #888, that file lives on `agent/issue-888-oblique-diagnosis` @ 831ea3c1 and is
NOT in this worktree). Pre-declaration: `docs/design_notes/20260902_cv26_oblique_fresnel_predeclaration.md`.

Append-only. Corrections go in a new section.

---

## 0. Verdict in one paragraph

**This lane is NOT landable as it stands.** The repair itself is sound and I could
reproduce it: the comparator now models the auxiliary absorber that actually ships,
`e_absorber` now sees the auxiliary echo it was structurally blind to, and all 18
`RECORD_DECLARED` entries were re-derived on the post-#888 code. But the honest
consequence of that repair is that **four committed gates are RED, three of the
pre-declaration's own numeric claims are refuted, and six blocker findings stand** —
including one that pre-dates this lane and that #888 did not fix: at the grazing band
the redesigned auxiliary absorber is only 1.8x better than the one it replaced, and no
gate anywhere in the case can see that. Nothing was widened, xfailed or dropped to make
this read greener, and the working tree is left uncommitted for a human to decide.

---

## 1. Tree as observed (not as reported)

Verified 2026-09-05 in `/Users/byungkwankim/Documents/rfx-worktrees/gap-oblique`.

```
$ git status --short
 M tests/_example_fidelity_lib.py
 M tests/crossval/test_cv26_oblique_fresnel_comparator.py
 M validation/crossval/comparators/oblique_fresnel.py
$ git branch --show-current
agent/gap4-oblique-fresnel
$ git log --oneline -1
97e845ef docs(cv26): round-2 section of the pre-declaration, the round-2 lane, and the gates it must meet
```

Diffstat: 3 files, +473 / −86. **Nothing is committed.** No `rfx/` source file is
modified by this lane; the #888 constants arrive through the rebase.

Environment check (the brief's trap): `rfx.sources.tfsf_2d` resolves to
`/Users/byungkwankim/Documents/rfx-worktrees/gap-oblique/rfx/sources/tfsf_2d.py` with
`AUX_N_CPML == 200`. **But see finding B-6: the brief's PYTHONPATH premise is false for
this lane's entry points and its prescribed check cannot detect the trap it names.**

### 1.1 Measured test counts — these supersede every earlier phase report

```
$ PYTHONPATH=<worktree> <venv>/bin/python -m pytest tests/crossval tests/contracts \
      -q --no-header -p no:randomly
3 failed, 1772 passed, 24 skipped, 31 deselected, 1 xfailed  (181.77s)
```

```
$ ... -m pytest tests/crossval/test_cv26_oblique_fresnel_comparator.py -q --no-header \
      -p no:randomly -m ""
4 failed, 36 passed  (81.48s)
```

**There are FOUR red tests, not three.** The fourth is `slow`-marked and is deselected
from the default lane, which is why earlier phase tallies read three. Every earlier
report that says "three reds" is wrong on the count; the four are enumerated in §4.

`tests/crossval/test_cv26_oblique_fresnel_gates.py` reports **17 skipped** — it needs
committed run artifacts, and `validation/crossval/_26_oblique_results/` does not exist
in this repo. The gate-level test file is therefore inert today.

---

## 2. What was broken

1. **The comparator carried a private copy of the auxiliary-grid constants.** Six
   literals (`n_cpml 30`, `margin 25`, `src_offset 3`, `order 4`, `kappa_max 7.0`,
   `sigma_factor 0.8`) restated the pre-#888 `tfsf_2d`, plus a third stale copy in
   `rig_cells` written as the literal `55 - 33`. After #888 landed, the comparator was
   modelling an auxiliary absorber that no longer exists — 30 cells against the shipped
   200, `sigma_max` 74.3 S/m against the shipped 0.8558 S/m (87x).

2. **`predict_settling`'s `e_absorber` was structurally blind to the auxiliary echo.**
   It differenced two solutions that both called `aux_lattice_field` with the rig's own
   auxiliary CPML, so `Einc` was identical on both sides and the auxiliary echo cancelled
   exactly. Measured: driving the auxiliary absorber from |B/A| = 9.6e-07 to 4.0e-01 — a
   factor 4e+05 — moved the old `e_absorber` at te_00 only from 9.55e-07 to 5.15e-06,
   and at te_45 not at all (0.066736 → 0.063682).

3. **`RECORD_DECLARED` was derived on the pre-#888 rig** and was stale for both of the
   above reasons.

---

## 3. What was changed, with before/after

### 3.1 Constants — one source of truth

`validation/crossval/comparators/oblique_fresnel.py` now does a module-level
`from rfx.sources.tfsf_2d import ...`. Read back after the change:

| constant | before | after |
|---|---|---|
| `AUX_N_CPML` | 30 | 200 |
| `AUX_N_MARGIN` | 25 | 25 |
| `AUX_SRC_OFFSET` | 3 | 3 |
| CPML order | 4 | 3 |
| `kappa_max` | 7.0 | 1.0 |
| `R_asymptotic` | (none — `sigma_factor 0.8`) | 1e-14 |
| derived `sigma_max` | 74.3 S/m | 0.8558 S/m |
| `aux_src_to_x_lo` | literal `55 - 33` | `AUX_N_MARGIN_X - AUX_SRC_OFFSET` (= 22, unchanged) |

Profile equivalence against a live `init_tfsf_2d(141, 45, DX_M, DT_S, cpml_layers=20,
tfsf_margin=5, f0=TFSF_F0_HZ, bandwidth=0.05, theta_deg=45.0)`, 200 cells each:
`max|b_cfg − b_aux| = 2.977733e-08`, `max|c| = 7.393990e-09`, `max|kappa| = 0`. Cast to
float32 (the dtype `cfg` stores) it is bit-for-bit identical. Geometry matches:
`n2x 542`, `i0_x 225`, `src_x 203`. Independently re-derived in review.

Two new tests fail if the copy is reintroduced. `AUX_SIGMA_FACTOR` is gone entirely
because the shipped module derives `sigma_max` from `AUX_CPML_R_ASYMPTOTIC`.

### 3.2 `e_absorber` — the record law now answers for the auxiliary echo

`aux_lattice_field` gained `echo_free=True`: it keeps the auxiliary grid, its layout, its
soft source at the same `src_x` and the normalisation, and replaces **only** the
auxiliary CFS-CPML with the outgoing-wave termination `yee_lattice_full` already builds
for its own ends. Threaded through as `aux_echo_free`; `predict_settling`'s control is
now `ideal_absorber=True, aux_echo_free=True` — clean on **both** grids. The old
quantity is kept and reported as `e_absorber_main_only`, so the 3-D and auxiliary terms
stay separable.

Echo-free control validated three ways, and re-derived independently in review with a
different two-wave fit: node ratio equals `exp(−j kx dx)` to 3e-16; |B/A| falls from
1.95e-06 / 3.64e-05 / 6.27e-06 (0/45/60 deg, shipped) to 2.06e-14 / 2.20e-14 / 3.79e-15;
R and T reproduce `aux="plane"` to `max|dR| 1.6e-14`, `max|dT| 7.7e-12`. A deliberately
crippled auxiliary absorber (n=8, R=1e-2) reads 9.1e-02, so the fit has dynamic range.

The blindness demonstration, te_00 (where the 3-D term is small), record held fixed:

| aux setting | \|B/A\| | e_absorber OLD | e_absorber NEW |
|---|---|---|---|
| shipped 200 @ 1e-14 | 9.63e-07 | 9.553e-07 | 1.828e-06 |
| pre-#888-like 30/4/7 | 3.71e-02 | 1.451e-06 | 3.408e-02 |
| 8 cells @ R=1e-1 | 7.74e-02 | 4.194e-06 | 7.458e-02 |
| 6 cells @ R=0.5 | 4.01e-01 | 5.152e-06 | 4.680e-01 |
| **span** | **x 4.2e+05** | **x 5.4** | **x 2.6e+05** |

On the **shipped** rig the fix is a guard, not a re-baseline: te_45 at dx/2 moves
0.06673599 → 0.06673603 (4e-08 relative).

The note's claim that round 2 "would have failed honestly" is **confirmed** at the rung
round 2 actually ran (te_45 dx/2, pre-#888-like auxiliary grid): OLD `e_absorber`
0.0298424 reproduces the round-2 artifact's 2.984e-02 with `absorber_ok TRUE`; NEW is
0.0751804 (2.52x), `W_absorber_R_max` 0.111529 > `W_BIN` 0.074, `absorber_ok FALSE`.

### 3.3 `RECORD_DECLARED` — all 18 entries re-derived

Derived at `nfft = 2**19`, **not** at `predict_settling`'s `RECORD_NFFT = 2**17` default
(`oblique_fresnel.py:796`). At the default, te_45 and tm_45 at dx/2 **never settle** and
te_30 at dx reads 4584 instead of 3076 — a transform-length artifact (undecayed tail
wrapping onto the witness), not physics. Independently confirmed converged: an nfft
ladder on graze_vac at 2**17..2**21 gives `n_settle` 24625 / 24692 / 24784 / 24771 /
24771.

`n_settle`, old → new (2**19):

| key | old | new | Δ |
|---|---|---|---|
| te_00 dx | 1512 | 1511 | −0.1 % |
| tm_00 dx | 1512 | 1512 | — |
| te_30 dx | 3094 | 3076 | −0.6 % |
| te_30 dx/2 | 6387 | 6238 | −2.3 % |
| te_45 dx | 7362 | 6048 | −17.8 % |
| te_45 dx/2 | 14283 | 11082 | −22.4 % |
| te_60 dx | 12811 | 12153 | −5.1 % (jittery, see §6.2) |
| te_60 dx/2 | 26438 | 18614 | −29.6 % |
| tm_45 dx | 7362 | 6123 | −16.8 % |
| tm_45 dx/2 | 14283 | 11395 | −20.2 % |
| tm_60 dx | 12811 | 12153 | −5.1 % (jittery) |
| tm_60 dx/2 | 26438 | 18614 | −29.6 % |
| all six compact-box arms | 22008 | 24784 | **+12.6 % LONGER** |

`e_absorber`, the two that moved by more than a percent:

- graze_vac 2.7431e-14 → 1.3752e-02 (x5.0e+11). Main-grid-only part is 3.47e-14, so the
  **whole** term is auxiliary. That is **13.8x that arm's own `LEAK_BAR` = 1e-3**.
- te_30 dx/2 4.7908e-06 → 1.4156e-05 (x2.95), main-grid part unchanged.

Grazing depth ladder stays monotone: `e_absorber` 3.0561e-01 > 1.0366e-01 > 7.6371e-02 >
4.0968e-02 at 8 / 16 / 20 / 32 cells.

Cost of the re-derivation: ~3.1 h CPU across the nfft ladder.

---

## 4. Every gate's verdict

### 4.1 Committed pytest gates — **4 RED**

All four are RED *because* the comparator now models the rig that ships. **No window,
bar or tolerance was touched.**

| # | test | assertion as it fails | why |
|---|---|---|---|
| R1 | `test_records_are_the_declared_lattice_settling_steps` (`:470`) | `AssertionError: ('te_45', 1, 1.9888194672804997)` / `assert 1.9888… > 2.3` | 6 of 12 oblique rungs now fall below the >2.3x closed-form bar (te_45 1.989/1.841, te_60 2.432/1.886, tm_45 2.076/1.952, tm_60 2.538/1.970). Line `:474` would also fail: it pins `n_steps == 22008` for the graze arms against the declared 24784. The comparator comment was updated to "1.42x to 2.54x"; **the test bar (2.3) and its docstring ("2.4–2.8x") were left at the old claim** — code and test now assert different things about the same quantity. |
| R2 | `test_the_record_of_round_1s_settled_arms_is_reproduced` (`:485`) | `AssertionError: ('graze_pec', 1, 24784, 22001)` / `assert 24784 <= (22001 + 100)` | 22001 is a **measured FDTD settling** on hardware (run `cv26-oblique-r1-20260902T162340Z`), not a model number. The declared record now over-predicts it by 12.6 %, more than a whole extension quantum. te_30 dx/2 also falls 58 steps *below* its bracket. **This is the only check of the round-2 record law against a real run; everything else in §13 of the note is self-consistency of the same lattice model.** The record law is therefore empirically unvalidated. |
| R3 | `test_every_declared_falsifier_has_an_analytic_margin[graze_pec_sigma_half]` (`:548`) | `assert 1.7916934203089905 > 10` | Re-verified live: `graze_pec_sigma_half` ratio **1.7917** (was 16.5914 at HEAD, a 9.3x collapse), `bins_beyond_window` 37 (was 39). `graze_pec_depth_half` ratio **13.8167** (was 90.4127, a 6.5x collapse), bins 69. Both still fire (`predicted_fails_G6 True`), but the declared separation is 5.6x below its own bar. The 16.6 was riding on an accidental cancellation between the aux echo and the 3-D CPML's grazing error — i.e. on the broken auxiliary grid. |
| R4 | `test_predict_settling_reproduces_a_declared_record` (`:526`, **slow-marked**) | `assert 4.338e-07 <= (0.001 * 1.3939e-06)` | The test calls `predict_settling` at the `RECORD_NFFT = 2**17` default while the table is declared at 2**19, so `e_absorber` reads 1.828e-06 against the declared 1.394e-06. `n_settle` reproduces exactly (1511) at both lengths. **Even if it passed it would not be exercising the declared numbers** — the table's own comment warns that anything re-deriving it must pass `nfft` explicitly, and this test does not. |

### 4.2 Live case run — full baseline, all 10 arms, no falsifier

Artifact: `wf_baseline/rfx.json` (uncommitted scratchpad — see §7).
`wf_baseline/rfx.json::commit = 97e845ef2acf950f6b2701ee0d4d33e755439b3f`
`wf_baseline/rfx.json::verdict.exit_code = 1`
`wf_baseline/rfx.json::verdict.rfx_self_ok = false`
`wf_baseline/rfx.json::verdict.meep_present = false`

**4 PASS / 6 FAIL.**

| arm | `e2_ok` | gates_all | max\|dR\| | max\|dT\| |
|---|---|---|---|---|
| te_00 | **true** | all 8 True | 0.01864 | 0.01932 |
| te_30 | **true** | all 8 True | 0.005034 | 0.004996 |
| te_45 | false | G2_R, G2_T, G3_passivity **False** | 0.07017 | 0.07644 |
| te_60 | false | G2_R, G2_T **False** | 0.06469 | 0.07026 |
| tm_00 | **true** | all 8 True | 0.01864 | 0.01914 |
| tm_45 | false | G1_T, G2_R, G2_T, G3_closure, G3_passivity **False** | 0.04395 | 0.08661 |
| tm_60 | false | G1_T, G2_T, G3_closure, G3_passivity **False**; G_brewster True | 0.02321 | 0.08797 |
| graze_vac | **true** | G1_R, G1_T, G2_R, G2_T **False**; G_leak True | 0.9492 | 0.9492 |
| graze_pec | false | G1/G2 all False, G3_closure/G3_passivity False; **G6_absorber True** | 0.2246 | 0.1311 |
| graze_te | false | G1_R, G2_R, G2_T, G3_closure, G3_passivity False; **G7_R, G7_T True** | 0.1316 | 0.04247 |

`G3_tail` is True on all ten arms. F2 control (TM oracle on te_00): PASS.

Record / absorber, per arm (`wf_baseline/rfx.json::arms.<arm>.run.record.*`):

| arm | n_steps | extensions | e_absorber | W_abs R / T | absorber_ok |
|---|---|---|---|---|---|
| te_00 | 1511 | 0 | 1e-06 | 2e-06 / 3e-06 | true |
| te_30 | 6238 | 0 | 1.4e-05 | 1.8e-05 / 2.8e-05 | true |
| te_45 | **11282** | **1** | 0.0299 | 0.043002 / 0.060684 | true |
| te_60 | 18614 | 0 | 0.017836 | 0.027269 / 0.029966 | true |
| tm_00 | 1512 | 0 | 1e-06 | 2e-06 / 3e-06 | true |
| tm_45 | **11595** | **1** | 0.02239 | 0.021637 / 0.045281 | true |
| tm_60 | 18614 | 0 | 0.02653 | 0.012153 / 0.053763 | true |
| graze_vac | 24784 | 0 | 0.013752 | 0.027692 / 0.027692 | true |
| graze_pec | 24784 | 0 | 0.076371 | **0.158575 / 0.158575** | **false** |
| graze_te | 24784 | 0 | 0.066026 | **0.132899** / 0.05294 | **false** |

Two entries of `RECORD_DECLARED` are still marginally short at the rig: te_45 and tm_45
each needed one 200-step extension. Two grazing arms carry an absorber term **2.1x over
`W_BIN` = 0.074**, reported and *not gated* because they are declared diagnostic rungs.

**Exit code is 1, not 2, and both readings must be stated at once.** Meep is absent on
all six Meep arms, which alone would give exit 2; the script tests `any_fail` before
`any_meep_missing`, so a real rfx-side gate failure wins. The rfx-side FAIL stands on its
own evidence. The E4 Meep cross-check is separately **UNAVAILABLE** and must not be read
as agreement or disagreement.

### 4.3 Pre-declared falsifiers — 7/7 exit non-zero, but the criterion is hollow on 4 of them

| falsifier | exit | gate that fired | separation |
|---|---|---|---|
| te_60_angle_m5 (F1) | 1 | G1/G2/G3_passivity | mean\|dR\| 0.0425 / 0.0154 = **2.76x** |
| te_45_swap_tm (F2) | 1 | G1/G2/G3 | 22.28x |
| tm_60_swap_te (F2) | 1 | G1/G2/G3; **G_brewster True** | 30.00x |
| te_45_eps_x1p2 (F3) | 1 | G1/G2 | 15.37x |
| meep_te_45_k_2pi (F4) | 1 | te_45's own baseline redness | **the defect never ran** |
| graze_pec_depth_half (F5) | 1 | **G6_absorber False** | 25.49x per-bin; 69/69 bins |
| graze_pec_sigma_half (F5b) | 1 | **G6_absorber False** | 4.66x per-bin; 38/69 bins |

Clean baselines on the same rigs, no defect applied — **te_60, tm_60, graze_pec and
te_45 all exit 1 already**. So for F1/F2/F3/F4 the exit code attributes nothing; only
the separation does. **G6 is the only gate that discriminates cleanly**: graze_pec clean
`max|R − R_lat| = 2.88e-03` → F5 0.3123 (108x) and F5b 0.0878 (30x), `G6_absorber`
True → False.

Refutations of the pre-declaration, §8:

- **F4 is a silent falsifier wearing an exit 1.** Its defect lives entirely on the Meep
  leg; with no Meep JSON the run takes the E4 `[SKIP]` path and the rfx side runs the
  **undefected** te_45 arm, bit-for-bit identical to the clean te_45. On this rig F4 is
  **NOT-APPLICABLE** and must be read as such, never as a pass. Nothing in this run
  exercises `precheck.passed = false`.
- **F1's separation is 2.76x, below the pre-declared 3.0x and below the ≥2.9x the unit
  test asserts.** Its stated reading rule ("an F1 exit 1 counts only with G2_R = false")
  is void: G2_R is false on the clean te_60 too.
- **F2 `tm_60_swap_te` does not fail the Brewster gate**, contradicting the
  pre-declaration's named bin. `evaluate_brewster` gates `R_rfx[iB] <= window_R[iB]` and
  never compares the *oracle's* R; swapping the oracle to TE only widens `window_R`
  (0.0740 → 0.0798), so the swap makes the gate **easier**. As coded, this gate cannot
  fire on an oracle-swap falsifier at all.
- **F5's "excess up to 90x" does not reproduce**: measured excess \|R−1\| max 0.4446 vs
  a-priori absorber term 0.1346 is 3.3x.
- **F5 (`n_cpml = 10`) has no `RECORD_DECLARED` entry**, so its record falls back to the
  closed form (21501 grown to 24801 by 33 extensions) with `e_absorber = nan` and
  `G3_absorber` reading `'reported'`. It is judged against an arm whose record was
  derived by a different rule.

### 4.4 Ladders — clean; the gates on top of them are not

**Absorber-depth ladder (graze_pec)** — strictly monotone and matching each rung's own
a-priori prediction to 0.0–0.5 %, with a flat per-bin residual 2.75e-3–3.17e-3 at every
depth:

| n_cpml | measured max\|R−1\| | a-priori total (own geometry) | 3-D term | aux term |
|---|---|---|---|---|
| 8 | 6.9177e-01 | 6.9155e-01 | 6.5054e-01 | 7.8249e-02 |
| 16 | 1.9532e-01 | 1.9505e-01 | 1.6997e-01 | 5.9740e-02 |
| 20 | 1.3488e-01 | 1.3458e-01 | 1.1527e-01 | 5.7147e-02 |
| 32 | 5.7845e-02 | 5.7580e-02 | 5.0915e-02 | 5.3715e-02 |

Note the last column: **on the compact box the auxiliary term does not shrink with
`--n-cpml` at all** (it is the injection path), and at 32 cells it *exceeds* the 3-D
absorber term. The depth ladder is approaching an injection-path floor at its deepest
declared rung.

**dx ladder** — monotone on every arm and every metric; te_30 (the one arm whose record
ends before its echo) converges at **4.03x = clean second order**, while the four
echo-carrying arms converge at only **2.2–3.3x**, and their absorber term converges at
the same rate. That separation is the lane's cleanest single piece of physics: the
sub-second-order part of the error *is* the absorber.

Contradictions of the note, §4.6 and §10:

- §4.6 predicts "te_60 and tm_45 would FAIL G2 at dx by 0.5–5 %". Measured: **four** arms
  fail G2_R at dx, by **83–144 %**. Its per-arm lattice terms match only on te_30 — the
  one arm with no in-record echo. The §4.6 dx term looks to have been computed without
  the in-record absorber echo.
- §10's lattice-witness reading rule ("|rfx − lattice| ≫ 3e-4 says the solver differs
  from its own discrete model") misreads te_30, which measures 3.79e-02. It is not a
  defect: te_30 matches the **ideal** lattice (1.15e-04) because its record ends before
  the echo. The witness must be taken against the reference whose echo content matches
  the record; the note's rule does not say that.

### 4.5 The auxiliary-echo arrival invariant — 16 of 18 arms breach it

Measured with the slab family's own definition (`G.aux_echo_arrival`), reflector depth
taken as a bound (0.0 cells), at the Courant cell speed:

**PASS 2/18** (te_00 ratio 0.489, tm_00 0.490). **FAIL 16/18.**

| arm/rung | record | arrival | ratio |
|---|---|---|---|
| te_30 dx | 3076 | 3036 | 1.013 |
| te_30 dx/2 | 6238 | 5963 | 1.046 |
| te_45 dx | 6048 | 2884 | 2.097 |
| te_45 dx/2 | 11082 | 5659 | 1.958 |
| te_60 dx | 12153 | 2447 | **4.966** |
| te_60 dx/2 | 18614 | 4787 | 3.888 |
| tm_45 dx | 6123 | 2884 | 2.123 |
| tm_45 dx/2 | 11395 | 5659 | 2.014 |
| tm_60 dx | 12153 | 2447 | **4.966** |
| tm_60 dx/2 | 18614 | 4787 | 3.888 |
| all six compact-box arms | 24784 | −10802 | **inf** |

Robustness: at the *least* conservative defensible speed (slowest gated component),
**12 of 18 still breach**. te_30 is the only borderline case — it passes at the
fastest-gated convention (0.93 / 0.96) and breaches by 1.3 % / 4.6 % at the supremum.

**cv26 carries no arrival gate at all.** Round 2 removed the arrival cap and replaced it
with the amplitude statement `e_absorber` inside `W_BIN`. The working agreement requires
a replacement gate to be **shown** to kill the same defects. It does not:
`absorber_ok` reads **True on eleven of the sixteen arms that breach the arrival
invariant**, including te_60 and tm_60 at dx/2 (ratio 3.89) and graze_vac (ratio 98.74 at
the centre convention). **The amplitude gate and the timing gate are not substitutes for
each other on this case.**

---

## 5. Review findings

24 findings, **6 blockers**. Blockers first.

### 5.1 BLOCKERS

**B-1 — `rfx/sources/tfsf_2d.py`: #888 was derived at 70 deg; three cv26 arms run at
80–85 deg, where the fix buys a factor of 1.8, not 227.**
The redesign's "worst declared angle" comment cites 70 deg as "cv26's gate cap", but 70
is only the **primary**-rig cap (`THETA_GATE_MAX_DEG`). `graze_vac`, `graze_pec` and
`graze_te` are declared at `GRAZE_THETA_GATE_DEG = (80, 85)`. Measured with the project's
own instrument (`tests/_aux_absorber_reflection.measure_aux_reflection_2d`):

| angle | shipped 200-cell \|B/A\| mean / max | pre-#888 30-cell |
|---|---|---|
| 70 deg | 3.83e-05 / 2.33e-04 | — |
| **82 deg** | **1.01e-02 / 2.93e-02** | **5.15e-02 / 5.21e-02** |

Factor 227 at 70 deg; **factor 1.8 at 82**. Every fit residual is below
`FIT_RESID_LIMIT = 1e-2`, so all are valid by the lane's own rule. Corroborated
independently by `RECORD_DECLARED[('graze_vac',1,20,100)]['e_absorber'] = 1.375e-02`,
whose main-grid part is 3.5e-14. **The defect #888 exists to remove is essentially still
present on the arms that run nearest grazing, and no number anywhere in the lane states
it.** The derivation note has no angle above 70 in it.

**B-2 — `validation/crossval/26_oblique_slab_fresnel.py:458`: graze_vac prints "ALL
CHECKS PASSED … (exit 0)" with four of its own gates False.**
Line 458 *replaces* the E2 verdict with `arm_ok = lk["G_leak"] and
e2["gates"]["G3_tail"]`, silently dropping `G3_closure` and `G3_passivity` — which **are**
measurable on this rig — and line 459 overwrites `e2["e2_ok"] = arm_ok` so the artifact
records the arm green. The same pattern at `:468` and `:477` drops `G3_closure` from
graze_pec and graze_te. graze_vac's entire verdict is now one witness plus the tail, so a
loss of closure or passivity on the vacuum arm cannot make the case red.

**B-3 — `docs/…_predeclaration.md`: the note the comparator names as its authority still
describes the pre-#888 auxiliary grid, and no #888 section was ever appended.**
The comparator header says "Everything numeric is fixed by …predeclaration.md; change the
note (append-only) before changing a number here." §3.2 still reads "30 cells, 4th order,
κ_max 7, σ_max = 0.8·5·7/(η dx) = 74 S/m … source at node 33" against a shipped 200 / 3 /
1.0 / 0.8557 / node 203. §3.3's aux-echo and total columns and §12.4's "a-priori 0.2724
(3-D 0.115, aux 0.174)" recompute on the shipped grid to 0.1346 / 0.1153 / 0.0571. §3.3
still says the aux round trip is `2(nx + 55)/v_gx` where `i0_x` is now 225. Headings stop
at "## 16. What round 2 owes."

**B-4 — the falsifier margin collapse (R3 above), restated as a blocker.** The case's
E4-tier claim rests on a margin that is now 5.6x below its own bar, and no phase summary
before the review mentioned the margins moving at all.

**B-5 — §13.3's headline validation is false for 2 of its 4 rows (R2 above), restated as
a blocker.** "The law reproduces every arm round 1 actually settled, each inside one
extension quantum of its bracket." te_30 dx/2 lands 58 steps below its bracket;
graze_pec lands 2783 steps (12.6 %) above. This is the **only** empirical check of the
record law.

**B-6 — the brief's PYTHONPATH premise is false and its prescribed check cannot fail.**
The comparator inserts the repo root at `sys.path[0]` before any rfx import, so the case
script and pytest resolve the worktree `rfx` **with or without** `PYTHONPATH` — verified
by running graze_vac both ways: artifacts differ only in `date_utc` and `elapsed_s`.
The prescribed `python -c` check prints the worktree path and 200 without `PYTHONPATH`
too, because `-c` puts cwd on `sys.path`. **Net: the earlier phases' numbers are safe on
this axis, but not for the stated reason, and the stated check would not catch a real
contamination.** A real trap does exist — a script run from *outside* the repo picks up
the other checkout and dies with `AttributeError: module has no attribute 'AUX_N_CPML'`.

### 5.2 MAJOR (9)

- **M-1** `oblique_fresnel.py` — **no gate can see the grazing auxiliary echo.** G6/G7
  judge the run against `yee_lattice_full(..., aux="model")`, which carries the same
  auxiliary echo on **both sides** of the difference. Measured
  `max_aux_echo_term_R_gated`: graze_pec 5.3359e-02, graze_te 4.3454e-02 — **72 % and
  59 % of the whole bin window `W_BIN` = 0.074** — contributing nothing to
  `d = |R_meas − R_lat|`. G6/G7 passing is a self-consistency check against a model
  containing a 1–3 % non-physical injected wave, not a physics verdict at 80–85 deg.
- **M-2** `oblique_fresnel.py:1137` — `evaluate_e2` hard-codes `oracle_eps = EPS_R_SLAB`
  and never consults `spec["slab"]` or `spec["pec"]`, so all three compact arms are judged
  against a 4-eps slab that is not in the rig. On graze_vac the oracle gives `R_an`
  0.862–0.949 against rfx's 3.8e-11. Those G1/G2 Falses are a **wrong-oracle verdict**,
  not "not applicable, reported as such". `absorber_window` at `:881` *does* branch on
  `spec["slab"]` — the comparator knows the distinction and `evaluate_e2` does not.
- **M-3 / M-4 / M-5** — the four red tests (§4.1), each also carrying a stale docstring or
  bar that now contradicts the code comment beside it.
- **M-6** `oblique_fresnel.py` — the stated justification for the `e_absorber` fix ("the
  arm's own `aux_echo_term_R_gated_max` — 0.2431 at te_45 — was never answered for") is a
  number produced by the **stale 30-cell copy the same change deletes**. On the shipped
  grid the identical quantity is 1.0e-04, **2400x smaller**, and the total absorber term
  at that rung falls 0.30 → 0.068. The argument that the auxiliary echo is "the dominant
  term" is built on a number the fix itself falsifies.
- **M-7** `oblique_fresnel.py` — the headline "graze_vac 2.7e-14 → 1.375e-02, the
  auxiliary absorber's echo alone" over-states what that arm can see by 12 orders of
  magnitude: at the reflection probe the rig-vs-control difference is identical to 13
  digits in the total and the incident series (0.007242379444661775 vs
  0.007242379444662518), so it divides out, and the arm's own scat/inc witness reads
  4.35e-15 over the whole record.
- **M-8** `validation/crossval/manifest.json:695,702` — the **public group-A** claim
  scope states a falsifier margin the current code contradicts by 6.5x ("the exact lattice
  predicts a 90x larger excess" — now 13.82) and a CPU-runner exclusion rationale citing
  step counts now ~2x larger ("up to 9870 steps … ~22 000" against 18614 and 24784).
- **M-9** `docs/…_predeclaration.md` §13.4 — the entire 13-row table is stale and the
  argument drawn from it is falsified: "Every 45°/60° arm needs 1.8–2.4x its round-1 cap"
  is now **1.39–2.23x**.

### 5.3 MINOR (9, condensed)

- The F2 control at normal incidence is computed and printed but never enters `arm_ok` or
  `any_fail`; a live run in which it FAILS still exits 0.
- `derive_record`'s docstring still asserts a CPML-arrival rule "asserted by the case"
  that round 2 removed; the shipped table violates it on **4 of 7 primary arms**
  (te_45 dx/2 11082 > 7959, te_60 18614 > 10867, and the tm twins).
- G7 builds its oracle at the **run's** absorber depth while G6 deliberately builds at the
  declared depth — so a graze_te run with a defective absorber is compared against a model
  of that same defect and passes.
- `RECORD_AMP_FLOOR = 1e-9` (`:797`) is **inert** — it selects 100 % of positive-frequency
  bins at every length used (65535/65535 and 262143/262143), smallest kept relative
  amplitude 6.05e-07. The documented band selection does not happen; the un-truncated
  evanescent tail is the same content the note blames for transform-length sensitivity.
- The new `RECORD_DECLARED` comment states two arrival times (~1835 on te_30 at dx,
  ~7348 on the compact box) that **no code computes and that reproduce under no natural
  construction**; 7348 is below that arm's own source centre. Its direction is right; its
  magnitudes understate the breach by 1.7x and 29x respectively.
- The comparator now imports rfx (and jax) at module level. `scripts/crossval/meep_cv26_oblique_slab.py`
  still says "This script never imports rfx". **The Meep leg for cv22 and cv26 now needs
  rfx + jax importable in the Meep environment**; meep is not installed here, so neither
  leg could be run to confirm.
- Two unhandled `RuntimeWarning`s at `oblique_fresnel.py:179` (`s = ky*C0/(TWO_PI*f)`,
  divide-by-zero at f = 0); benign, the zero bin is outside every masked band.
- The artifact's per-arm `gates` key holds only the seven base E2 gates; the full dict
  lives under `gates_all`. A consumer reading `gates` silently misses G3_absorber,
  G_brewster, G_leak, G6_absorber, G7_R, G7_T. The per-arm boolean is `e2_ok`, not `passed`.
- `cv22_dispersive_gates.aux_echo_verdict` returns `ok: True` on a **negative** arrival
  (`{-10802, 24784} -> ok True`). All six cv26 compact-box arms produce a negative
  arrival, so the shipped helper would pass the single most contaminated family in the
  case. It needs a guard: `arrival_steps <= 0` must be a hard failure, never a ratio.

---

## 6. What is STILL RED or still UNKNOWN

### 6.1 Still red — nothing here was widened, xfailed or dropped

1. **Four pytest gates** (R1–R4, §4.1). Two encode round-2 numbers taken on the pre-#888
   rig (the 2.3x closed-form ratio, the graze_pec 22001 round-1 witness); one is a
   falsifier margin collapse; one is an nfft mismatch inside the test itself.
2. **Six of ten arms FAIL the live case** (§4.2), exit 1.
3. **`absorber_ok = false` on graze_pec and graze_te**, `W_abs` 2.1x over `W_BIN`,
   **reported and not gated** because they are declared diagnostic rungs.
4. **16 of 18 arms breach the auxiliary-echo arrival invariant** (§4.5), and the gate
   that replaced the arrival cap has **not been shown** to kill the same defect.
5. **Six blocker findings** (§5.1), of which B-1 is a physics gap in #888 itself, B-2 is a
   live green-washing path, and B-3/B-5 are the authority document contradicting the code.

### 6.2 Still unknown — these want a decision, not more measurement

- **te_60 / tm_60 at dx is not converged in nfft and I could not settle it**: 9622 at
  2**18, **12153** at 2**19, 9818 at 2**20 — a **26 % spread, non-monotone**. Cause is
  measured: the binding witness is the incident purity, which grazes its 1e-3 bar over a
  long stretch, so the first crossing moves with the length. 12153 was declared as the
  conservative reading. **It is a 26 % uncertainty on a declared record.**
- **`RECORD_NFFT` was left at 2**17** although the table is declared at 2**19 and the
  default is demonstrably too short (two keys never settle; te_30 at dx inflates its
  `e_absorber` by four orders of magnitude). Changing it is a code/gate decision.
  A caller invoking `predict_settling` at the default on te_45/tm_45 at dx/2 still gets
  `n_settle None` and a KeyError-shaped absence rather than a stated NOT-APPLICABLE.
- **Whether G6/G7 can stand at all on a rig whose injected incident carries a 1.4 % echo.**
  The compact box is 141 cells; `AUX_N_MARGIN_X = 25` is fixed in `tfsf_2d` and does not
  scale with the box, so **no record longer than ~250 steps is admissible on a 100-cell
  box at any dx or any 3-D `n_cpml`**. This is not a #888 regression — the pre-#888 layout
  had the same 25-cell margin — it is a defect #888 did not touch and this lane exposes.
- **The Meep leg has never run post-#888.** E4 is `[SKIP]` on all six Meep arms; F4 is
  NOT-APPLICABLE; and the comparator's new module-level rfx import may break the Meep
  environment at import if it is jax-free. **Unverified.**
- **The "pre-#888" auxiliary absorber used in the round-2 reconstruction is a
  reconstruction, not the shipped one** (the comparator can now only express `sigma_max`
  derived from a reflection target). It lands at |B/A| 4.37e-02 against the diagnosis'
  measured 4.81e-02. The round-2 verdict flip is far outside that gap, but the exact
  0.0752 should not be quoted as "what round 2 would have printed".
- `evaluate_e2`, `evaluate_grazing_pec` and `evaluate_grazing_slab` still build their
  ideal-lattice reference with `aux="plane"`, which replaces the source and normalisation
  rather than only the echo. Now that an echo-free auxiliary grid exists these could be
  re-expressed surgically; doing so would move numbers already written into committed
  artifacts, so it was deliberately not done.

---

## 7. Provenance caveat — read before citing anything above

**No cv26 run artifact is committed in this repo.** `validation/crossval/_26_oblique_results/`
does not exist. Every artifact cited above lives in an **uncommitted scratchpad**
directory under
`/private/tmp/claude-501/-Users-byungkwankim/50935fff-b722-4069-86b0-f259802f7b01/scratchpad/`
(`wf_baseline/rfx.json`, `wf_fals_<name>/rfx__falsifier_<name>.json`,
`wf_ladder_<tag>/rfx__<tag>.json`, and `aux-echo-arrival/gate_table.json`).

Consequence: **the #829 / #812 numeric-provenance gate cannot resolve the citations in
this note today**, because that gate requires a repo-relative committed `.json` path.
This note is deliberately **not** opted into `REQUIRED_SITES` for that reason. Landing
this lane requires committing the run artifacts under a repo path first, at which point
every `path.json::key = value` span above should be re-pointed and the note opted in.

All numbers in this note were produced at `commit 97e845ef` **plus the three uncommitted
working-tree files**, i.e. at a tree state that exists nowhere in git history.
