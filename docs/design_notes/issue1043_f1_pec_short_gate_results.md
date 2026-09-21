# #1043 review round 1, F1 — result: the gate was pinning a diverging run

Run against the gates frozen in
`issue1043_f1_pec_short_gate_predeclaration.md` (§2-§3 and the §5 append).
Driver `scripts/diagnostics/cpml_subpixel_stability/f1_pec_short_gate.py`;
artifacts `f1_head.json` / `f1_main.json`. "main" ran against a `git archive`
of `origin/main` (499c8e1e) unpacked to a scratch directory.

## 1. Verdict

**Head is the physically right answer, and the committed `min >= 0.99` bar was
pinning the OLD inconsistent absorber — a run that is exponentially growing.**
The bar is re-derived below, two-sided, from the measurement.

## 2. The decisive measurement: record length

`|S11|` on the same rig, both trees, as the record is lengthened. This is the
independent axis, and it settles the question by itself.

Every number below is a value-checked citation into the committed artifacts,
so a transcription slip here fails the provenance contract rather than sitting
in prose. (One did, and did sit in prose: the 80-period head range read
[0.9969, 1.0014] until review — 0.9969 is bin 0, not the min. Opting this note
into the gated set is the fix for the class, not just the instance.)

| periods | `origin/main` min | `origin/main` max | this tree min | this tree max |
|---:|---|---|---|---|
| 40 | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml.s11_min = 0.9942` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml.s11_max = 1.0278` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml.s11_min = 0.9877` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml.s11_max = 1.0145` |
| 80 | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml_80periods.s11_min = 0.5724` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml_80periods.s11_max = 2.8001` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml_80periods.s11_min = 0.9552` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml_80periods.s11_max = 1.0014` |
| 160 | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml_160periods.s11_min = 9.3597` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml_160periods.s11_max = 12.0899` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml_160periods.s11_min = 0.9637` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml_160periods.s11_max = 1.0014` |

And the ring-down witness moves the two ways round:

| periods | main (port0 / port1) | head (port0 / port1) |
|---:|---|---|
| 40 | −39.20 / −5.09 dB | −40.05 / −15.01 dB |
| 80 | −6.60 / −7.62 dB | −66.26 / −22.54 dB |
| 160 | −10.33 / −10.58 dB | −76.57 / −26.62 dB |

**A settling witness that gets WORSE as the record grows is growth, not
ring-down.** main's does; head's improves monotonically. main's `|S11|` reaches
12.09 at 160 periods — the field is diverging, which is exactly the #1043
instability this rig is built to have (§3).

So main's 40-period numbers — mean 1.0032, max 1.0278, *above* unity on a
lossless passive short — are **early exponential growth stopped before it
became visible**, not a better answer. head's are the bounded ones, and its
interior bins improve monotonically with record length (|dev| 0.0023 → 0.0019
→ 0.0018). head's band min sits in bin 5 at every record length —
0.9877 → 0.9552 → 0.9637 — which is the near-cutoff edge bin discussed in §2b,
not a record-truncation effect.

## 2b. The pre-declared residual r_A' is NOT met — recorded as declared

§5 of the pre-declaration froze
`r_A' = max(0, D_head(40) - D_main(40)) + max(0, D_head(160) - D_head(40))`
with gate `r_A' = 0`, where `D = max_bin | |S11| - 1 |`.

| term | value |
|---|---|
| `D_head(40)` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml.D_max_dev_from_unity = 0.0145` |
| `D_main(40)` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml.D_max_dev_from_unity = 0.0278` |
| `D_head(160)` | `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml_160periods.D_max_dev_from_unity = 0.0363` |
| `max(0, 0.0145 - 0.0278)` | 0 |
| `max(0, 0.0363 - 0.0145)` | **0.0218** |
| **`r_A'`** | **0.0218** against gate 0 — **NOT MET** |

The first term is met: head is closer to unity than main at the record length
the committed gate uses. The second is not: `D_head` does not fall
monotonically with record length (0.0145 → 0.0448 → 0.0363), so the
pre-declared reading "head is right iff D falls with record length" does not
license the conclusion on its own terms, and is recorded here as not licensing
it.

**The interior-restricted reading DOES meet it, and is a RESTRICTION of the
declared observable, not the declared one.** Restricting `D` to interior bins
1-4 gives 0.0023 → 0.0019 → 0.0018, monotonically falling, so
`r_A'(interior) = 0`. That restriction was chosen AFTER seeing the data and is
therefore not the frozen gate; it is reported as a narrower observable that
behaves as the gate expected, alongside the full one that does not.

What actually carries the verdict is neither residual but the thing the
pre-declaration could not have written down in advance: **main diverges**
(|S11| → 12.09, settling degrading) while head stays bounded. That is a
categorical difference, not a residual comparison, and §2 is where it is
measured.

The whole `D_head` non-monotonicity lives in band-edge bin 5 (7.0 GHz, 0.93x
this guide's TE20/TE01 cutoff at 7.5 GHz) on a port that never reaches the
-40 dB settling bar: its |S11| runs 0.9877 → 0.9552 → 0.9637. That is a
near-cutoff extraction limit on an unsettled port, **not** record-length
truncation — lengthening the record does not remove it, which is why the
interior bins improve while `D` does not.

## 3. Why this rig has the defect at all

`rfx/runners/uniform.py:279-303`: with conformal faces and PEC shapes,
`conformal_eps_correction(eps_base, w_ex, w_ey, w_ez)` sets `aniso_eps` to
`eps_eff = eps/w`, `w < 1` at wall cells — so `eps_a > eps_b = materials.eps_r`
exactly there. The x CPML pad is `[:n_x, :, :]` / `[-n_x:, :, :]`, spanning
every y and z, so the y/z conformal wall cells at x inside the pads are in the
absorber. Before this branch their psi coefficient used the staircase epsilon.
That is the amplifying direction of the #1043 inequality.

## 4. Arm D — the self-check's inputs, which is where the 3.258 → 5.621 goes

The reviewer's "passivity self-check got worse" is real and is **not about
`S11`**. head, 40 periods, `|S|` per bin (rows = out, cols = in):

| bin | S11 | S12 | S21 | S22 |
|---:|---:|---:|---:|---:|
| 0 | 1.0145 | **1.6544** | 1.0343 | 0.9755 |
| 1 | 1.0017 | **1.9298** | 1.0009 | 0.9627 |
| 2 | 0.9988 | **2.0341** | 0.9990 | 0.9538 |
| 3 | 0.9981 | **1.7246** | 1.0107 | 0.9623 |
| 4 | 1.0023 | **1.5993** | 1.0146 | 0.9685 |
| 5 | 0.9877 | **2.1630** | 1.0004 | 0.9715 |

Column power for port 1 = |S12|² + |S22|²; at bin 5 that is
2.1630² + 0.9715² = 5.62, and the recorded maximum is
`scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml.max_col_power = 5.6226` with
`scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_head.json::arms.cpml.max_abs_S = 2.1630`. **That 5.6226 is the S12 entry and nothing
else.**
On a PEC short nothing transmits, so S12/S21 are a near-zero amplitude in a
denominator — and the right port never reaches the −40 dB settling bar on this
rig at any record length tested (−15.01 dB at 40 periods, −26.62 at 160).

main's own `scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml.max_col_power = 3.2583` with
`scripts/diagnostics/_artifacts/cpml_subpixel_stability/f1_main.json::arms.cpml.max_abs_S = 1.4875` is the same quantity computed from the same
meaningless entry, on a run that is growing.
Comparing 3.258 against 5.621 compares two numbers neither of which is about
the physics the gate names.

## 5. The one-sided bar cannot see the failure it exists to guard

`min(|S11|) >= 0.99` **PASSES** main's 160-period run, where `|S11|` is
9.36-12.09. A one-sided minimum is satisfied by a field that has grown by a
factor of twelve. That, not the 0.9877, is the defect in the gate.

## 6. Arm B — non-closing, and why

Restricting the conformal correction to cells outside the x CPML pads
(1240 cells restored to `eps_base`) gives **identical numbers on both trees**
(`|S11|` [0.9844, 1.0171], max column power 37.2753) and a port-1 settling of
**−4.69 dB**. That arm does not isolate anything: it leaves `conformal_weights`
applying at pad cells whose `aniso_eps` has been reverted, which is a *new*
inconsistency rather than the absence of the old one. **Recorded as
non-closing**, as the pre-declaration requires; the record-length arm answered
the question instead.

## 7. Arm A — struck, not measured

`add_waveguide_port` refuses any non-CPML absorber
(`ValueError: Waveguide port requires boundary='cpml'`,
`rfx/api/__init__.py:2310`), so "the same rig under UPML" does not exist. The
substitute (§5 of the pre-declaration) is the analytic oracle plus record
length, which is what §2 reports.

## 8. The re-derived gate

Old: `s11.min() >= 0.99` — one-sided, blind to over-unity, green over a
diverging run.

New, both in `tests/unit/geometry/test_subpixel_pec.py`:

1. `test_pec_short_s11_with_conformal_face_pec` (40 periods, unchanged rig)
   - all bins: `| |S11| − 1 | <= 0.03` — **two-sided**, which the old bar was
     not, and **a 40-period envelope only**. Provenance: the measured
     **40-period** edge worst, 0.0145 here and 0.0278 pre-fix, rounded up to
     the next percent. It is NOT record-independent: the same rig reads a max
     |dev| of 0.0448 at 80 periods and 0.0363 at 160, all of it band-edge
     bin 5, so this bar would not hold at a longer record and must not be
     quoted as one. The gate runs at 40 periods, which is what makes it
     applicable.
   - interior bins 1-4: `| |S11| − 1 | <= 0.005` — **tighter than the old
     0.01**. Provenance: the measured interior worst, 0.0033 pre-fix and
     0.0023 here, rounded up to the next half-percent.
   - The band-edge bins are excluded from the tight bar by measurement: bin 5
     (7.0 GHz) sits just under this guide's second-mode cutoff
     (40 mm × 20 mm → TE20/TE01 at 7.5 GHz), and the right port never settles.
2. `test_pec_short_conformal_stays_bounded_over_a_long_record` (slow, 160
   periods) — `max |S11| <= 1.05`. Provenance: head measures 1.0014, main
   measures 12.0899; the bar is the passivity bound plus the measured
   discretization headroom. **This is the gate that actually binds**, and it is
   red on main.

Net: tighter where the observable is trustworthy, two-sided everywhere, and a
new gate for the failure mode the old one structurally could not express.

## 9. What is NOT claimed

That conformal PEC is ACCURATE. The 2026-06-08 verdicts that falsified four
conformal methods were taken with this defect present; re-measuring them is a
separate piece of work and no accuracy claim is made here.
