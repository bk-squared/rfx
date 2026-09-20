# #1043 review round 1, F1 — the PEC-short conformal gate moved

Frozen before any arm below ran. R2 (RF/EM intensifier): one pre-declared
attempt; a non-closing arm is recorded as non-closing. **No gate is touched
until the root cause is written.**

## 0. The observation

`tests/unit/geometry/test_subpixel_pec.py::test_pec_short_s11_with_conformal_face_pec`
(`:554`, gate `s11.min() >= 0.99`), rig `_pec_short_sim(conformal=True)`:
x CPML, y/z PEC with conformal faces, a one-cell PEC short at x = 0.084-0.087 m,
`normalize=False`, `num_periods=40`, 6 bins over 5-7 GHz.

| | `origin/main` | this branch |
|---|---|---|
| `\|S11\|` range | [0.9942, 1.0278] | [0.9877, 1.0145] |
| `\|S11\|` mean | 1.0032 | 1.0005 |
| gate `min >= 0.99` | passes | **FAILS** (0.9877) |
| extractor passivity self-check, max column power | 3.258 | **5.621** |
| max `\|S\|` | (to be measured on both) | 2.163 |

The mutation that ignores `inv_eps_r_update` reproduces main to 4 decimals, so
the move is the threaded coefficient and nothing else.

Two facts pull in opposite directions and that is the whole question. A lossless
PEC short must give `|S11| = 1` exactly, so **mean 1.0032 -> 1.0005 and max
1.0278 -> 1.0145 are both moves toward the physical answer**, and the
over-unity excursion — which is a passivity violation on its face — got
smaller. But the extractor's own column-power self-check got **worse**, and a
max `|S|` of 2.163 is not a small number.

## 1. Why this rig touches the changed code at all

`rfx/runners/uniform.py:279-303`: with `conformal=True` and PEC shapes,
`conformal_eps_correction(eps_base, w_ex, w_ey, w_ez)` sets `aniso_eps`
directly — `eps_eff = eps/w` with `w < 1` at wall cells. So this rig runs
`update_e_aniso`, and `eps_a = eps/w > eps_b = materials.eps_r` at every
conformal wall cell. The x CPML pad is `[:n_x, :, :]` / `[-n_x:, :, :]`, which
spans every y and z, so the y/z conformal wall cells at x inside the pads are
in the absorber. That is the amplifying direction of the #1043 inequality, and
before this branch those cells' psi coefficient used the staircase epsilon.

## 2. Arms, all pre-declared

**A — UPML control.** The same rig with `x="upml"`. `apply_upml_e` builds its
coefficients from `aniso_eps` itself (`upml.py:213-217`) and never had two
permittivities to disagree about, so it is the arm that already had the
consistent epsilon.
- *If UPML matches HEAD* (`|S11|` min below 0.99, mean near 1.0005): head is
  the consistent-absorber answer and main's numbers were held up by the
  inconsistent coefficient. That makes the 0.99 bar a pin on the OLD absorber.
- *If UPML matches MAIN*: the move is NOT "consistency", something else in the
  CPML path is doing it, and the fix needs re-examining at this rig. Recorded
  as such; do not land.
- *If UPML matches neither*: non-closing; report and stop.

**B — pad-restricted conformal.** The same rig, conformal correction applied
only at cells OUTSIDE the x CPML pads (wall cells inside the pads keep
`eps_base`). Isolates whether the move comes from the pad coefficient or from
the interior wall cells.
- *If B reproduces MAIN on head*: the whole move is the pad cells, i.e. exactly
  the cells #1043 is about.
- *If B still moves*: interior wall cells contribute too, and the attribution
  in section 1 is incomplete.

**C — per-bin dump (R5).** All 6 `|S11|` bins, both trees, plus the column-power
trace per bin, plus the preflight banner verbatim and the settling witness.
A range and a mean are not a result; the per-bin trace is.

**D — the self-check's own inputs.** What `max column power` and `max |S|` are
computed FROM on each tree — the raw S entries, which port/column carries the
5.621, and whether the same bin carries the min `|S11|`. A self-check number
that moved without its inputs identified is not evidence of anything.

## 3. Decision rule, frozen

- **Head is right and the bar was pinning the old absorber** — requires A to
  match head AND B to attribute the move to pad cells AND C to show the per-bin
  trace is tighter about 1 on head. Then, and only then, re-derive the bar from
  the measured envelope with the root cause written in the test docstring.
  The new bar is derived from the three-tree measurement, never invented:
  it is the head envelope widened by the main-vs-head spread, stated as a number
  with its provenance.
- **Head is worse for an identified reason** — any of A/B/C contradicting the
  above. Then that is a finding: say so, do NOT land, and do not touch the gate.
- **Non-closing** — A or B inconclusive. Report, stop, name what a second
  attempt would need.

residual `r_F1 = (0 if A matches head else 1) + (0 if B attributes to pads else 1)`;
gate `r_F1 = 0` for the first branch of the rule.

## 4. What is NOT in scope

Whether conformal PEC is ACCURATE (as opposed to stable) is not decided here.
The 2026-06-08 accuracy verdicts on the four conformal methods were taken with
the #1043 defect present and have to be re-measured separately; this note only
concerns the one committed gate that moved.

---

## 5. APPEND (2026-09-15 KST) — Arm A is not constructible; substitute declared

Written after the API refusal and BEFORE the substitute ran. Sections 0-4 stand.

**Arm A cannot be built.** `add_waveguide_port` refuses any non-CPML absorber:

> `ValueError: Waveguide port requires boundary='cpml'` (`rfx/api/__init__.py:2310`)

So "the same rig under UPML" does not exist — this rig's observable is only
reachable through the absorber the defect lives in. Recorded as a property of
the rig, not as a failed measurement, and Arm A is struck rather than quietly
reinterpreted.

That removes the arm that was going to say WHICH answer is right, so a
substitute is declared here, on an independent axis rather than on a second
absorber:

**A' — the analytic oracle.** A lossless PEC short in a lossless guide has
`|S11| = 1` exactly, at every bin. The physical score is therefore
`D = max_bin | |S11| - 1 |`, which is two-sided; the committed gate
`min >= 0.99` is one-sided and cannot see an over-unity excursion at all.
Measured on both trees.

**A'' — record length, the independent axis.** Repeat at
`num_periods = 40 / 80 / 160`. A PEC-short `|S11|` that deviates from 1 because
the record was cut settles toward 1 as the record grows; one that deviates from
a systematic error does not. The tree whose `D` falls with record length is the
one converging on the physics.

Frozen reading:
- **head is right** iff `D_head < D_main` at 40 periods AND `D_head` falls with
  record length. Then the 0.99 bar was pinning the old absorber's bias and is
  re-derived from the measured envelope, two-sided, with this note cited.
- **head is worse** iff `D_head > D_main`, or `D_head` does not fall while
  `D_main` does. Then say so and do not land.
- anything else: non-closing, report, stop.

residual `r_A' = max(0, D_head - D_main) + max(0, D_head(160) - D_head(40))`;
gate `r_A' = 0`.

Arm B (pad-restricted conformal) and arms C/D are unchanged and still run.
