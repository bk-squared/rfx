# cv14 (`validation/crossval/14_rect_cavity_pozar.py`) — gate re-declaration

Append-only. Corrections are added as new dated sections; nothing above is edited.

Issue: bk-squared/rfx#812 (crossval gate audit, 18 cases). cv14 finding: HIGH→critical.
Lane: `agent/regate-cv14`. Author session date: 2026-08-31.

**This is instrument work, not physics.** cv14's physics verdict is NOT challenged.
The audit measured this case's answer to be excellent — it reproduces the discrete-Yee
cavity eigenvalues on all seven declared modes to ≤ 0.15 MHz. What is challenged is the
*gate*: its ability to fail for its own stated reason.

---

## 1. What is wrong with the gate as committed (defect statement, with the measurement)

`main()` at `validation/crossval/14_rect_cavity_pozar.py:346-358` evaluates

```python
higher = {k: v for k, v in errs.items() if k != "TE101"}
best_name = min(higher, key=higher.get)     # <-- min over six
p = higher[best_name] < 2.0
```

Three independent instrument defects compose here:

1. **`min` over the candidate set.** Gate 2 is satisfied by the single *easiest* of six
   modes. Six modes are measured; one is gated. Five carry no gate weight at all.
2. **The candidate set contains a mode with a zero index on every axis.** The declared
   targets are TE101 (1,0,1), TM110 (1,1,0), TE011 (0,1,1), TM111 (1,1,1), TE201 (2,0,1),
   TM210 (2,1,0), TE102 (1,0,2). Every axis has at least one target that is blind to it:
   `n = 0` for TE101/TE201/TE102 (blind to `b`), `m = 0` for TE011 (blind to `a`),
   `l = 0` for TM110/TM210 (blind to `d`). Because `f_mnl` depends on an extent only
   through `m/a` etc., a mode with a zero index on the erroneous axis has an analytic
   frequency AND a physical frequency that are both unchanged by the error, so its
   `%err` is identically 0.000% for *any* magnitude of single-axis dimensional error.
   `min` then selects exactly that mode. Gate 2 is an **axis-blind minimum**: it cannot
   fail for a single-axis dimensional error, at any magnitude.
3. **NOT-FOUND is silently absorbed.** `print_table` (`:248-256`) skips a `None`
   frequency without adding it to `errs`, and Gate 2 reads `errs`. A mode that could not
   be extracted at all therefore *removes itself from the gate* instead of failing it.
   Only the degenerate case where **every** higher mode is missing reaches
   `if not higher: FAIL`.

Auditor's measurement (issue #812): shrinking the cavity's `a` extent 50 → 25 mm
(**−50 %**, the single largest single-axis error the domain can carry) drives Gate 1 to
FAIL at TE101 err **47.334 %** while **Gate 2 still reads PASS** at its best match.
47.334 % is exactly the analytic prediction: TE101 at (a,d) = (25,40) mm is
7.07559 GHz against the (50,40) mm oracle's 4.79902 GHz.

A fourth, separate gap: `run_leg` computes the effective wall separation
`eff = ((nx-1)*dx, (ny-1)*dx, (nz-1)*dz)` at `:208-209`, **prints** it at `:291-293`
next to the target — and never gates it. The one quantity in the script that is a
direct, zero-noise readout of the geometric defect class the audit exploited is
report-only.

---

## 2. Pre-declared thresholds

All four are declared HERE, in this commit, which precedes the commit that measures them.
No threshold below is derived from any cv14 run output. Derivation class is stated for
each; every class is geometry, first principles, or prior provenance, per SPEC-00 §0.2.2.

### T0 — wall-registration tolerance `WALL_REG_TOL_M = 1e-9` m (NEW gate, Gate 0)

Gate: `|eff_i − (a,b,d)_i| ≤ 1e-9 m` for i ∈ {x,y,z}, where `eff` is the quantity the
script already computes at `:208`.

*Derivation (geometry + IEEE-754, no measurement).* The mesh is chosen so that
dx = 1 mm divides a = 50, b = 30, d = 40 mm exactly, and rfx places PEC walls on the
first and last grid planes, so the effective separation is `(n−1)·dx` and must equal
(a, b, d) **exactly** — this is the premise the case's own docstring (`:46-58`) rests its
tight gate on. The only admissible deviation is the IEEE-754 double rounding of the
product `(n−1)·dx`: relative ≤ 2⁻⁵² ≈ 2.2e-16, i.e. ≤ **1.1e-17 m** absolute at 0.05 m.
The smallest geometric error the grid can *express* is one cell = **1e-3 m**. The window
1e-9 m therefore sits ~8 orders of magnitude above float noise and ~6 orders below the
smallest expressible real error: it cannot fire on correct code, and no real registration
error can slip under it. Nothing in this derivation refers to a run.

### T1 — Gate 1 unchanged: TE101 `%err < 1.0 %`

Prior provenance: the case's own published threshold (docstring `:68`, manifest
`claim_scope`, `docs/public/guide/benchmarks.mdx:61`, `validation/README.md:45`).
Unchanged and not widened.

### T2 — Gate 2 aggregator `min` → `max`, over **all seven** targets, at the **unchanged
2.0 %**; NOT-FOUND is a hard FAIL

Gate: every one of the seven declared targets must be extracted, and
`max_i %err_i < 2.0 %`.

*Derivation.* The 2.0 % number is prior provenance — the case's own published threshold,
carried over unmodified. Only the aggregator and the found/not-found handling change.
This is a strict tightening in the set-theoretic sense: the set of outcomes a
`max`-over-seven gate admits is a **subset** of what the committed `min`-over-six gate
admits, for every possible measurement. No widening occurs anywhere. The stated reason
for the gate — "rfx reproduces the analytic cavity spectrum" — is a claim about the
spectrum, i.e. about all declared modes; `min` gated one mode while the claim covers
seven, and `max` is the aggregator the claim already implied.

### T3 — NEW discrete-Yee residual gate (Gate 3): `|f_meas − f_yee| ≤ 0.1 / T`, every mode

`f_yee` is the **exact** eigenfrequency of the Yee scheme in this PEC box, closed form,
re-derived in-script and importing no producer-side helper:

```
sin(ω Δt / 2) = c Δt · sqrt( Σ_i sin²(k_i h_i / 2) / h_i² ),
k_x = mπ/a,  k_y = nπ/b,  k_z = lπ/d,  h_i = cell size on axis i
f_yee = ω / 2π
```

This is exact *only because* wall registration is exact — with walls on the first/last
grid planes and `(n−1)·dx = a`, the discrete field `sin(mπ x_i / a)` at `x_i = i·dx` is an
exact eigenvector of the discrete curl-curl operator with eigenvalue
`(2/dx)·sin(k_x dx/2)`, and leapfrog time stepping contributes the `arcsin`.
**Gate 3 is therefore conditioned on Gate 0** and is evaluated only when Gate 0 passes;
if the box is off the grid, `k_i ≠ mπ/a` and the prediction is void, not merely loose.

*Derivation of the 0.1 factor (estimator theory, no measurement).* `1/T` is the Rayleigh
limit: the frequency resolution of a **non-parametric** estimator on a record of span `T`.
harminv here is a parametric matrix-pencil estimator applied to a high-SNR sum of
*exactly undamped* complex exponentials — the model it assumes is the model the signal
literally is — whose Cramér–Rao bound scales as ~1/(SNR · T · √N), orders of magnitude
below `1/T`. Granting the estimator **one tenth of one Fourier bin** is therefore a loose
bound on the estimator and a tight bound on the physics. `T` is computed in-script from
the record actually handed to harminv (post excitation-skip, post any decimation), so the
budget follows the run length rather than being a frozen MHz number that silently
loosens if the run is shortened.

At the gated leg (dx = 1 mm, num_periods = 200, `Δt = 1.9066e-12 s`, 10491 steps, 251
skipped) this evaluates to `T = 1.9524e-8 s`, `1/T = 51.220 MHz`, budget
**≈ 5.122 MHz** — 0.107 % at TE101, 0.063 % at TE102.

**Prediction published before measurement.** From the closed form above at
Δt = 1.9066e-12 s, dx = 1 mm, the seven `f_yee` values and their offsets from the
continuum Pozar oracle are:

| mode | f_pozar (GHz) | f_yee (GHz) | f_yee − f_pozar (MHz) | rel |
|---|---:|---:|---:|---:|
| TE101 | 4.799021 | 4.798621 | −0.3994 | −0.0083 % |
| TM110 | 5.826918 | 5.825889 | −1.0288 | −0.0177 % |
| TE011 | 6.245676 | 6.244728 | −0.9480 | −0.0152 % |
| TM111 | 6.927916 | 6.927523 | −0.3929 | −0.0057 % |
| TE201 | 7.070591 | 7.068848 | −1.7432 | −0.0247 % |
| TM210 | 7.804846 | 7.803196 | −1.6507 | −0.0211 % |
| TE102 | 8.072159 | 8.067964 | −4.1949 | −0.0520 % |

The claim under test is that rfx's measured frequencies land on the `f_yee` column, not
merely near the `f_pozar` column. Gates 1 and 2 keep measuring against `f_pozar` (that is
the case's published claim and it stays); Gate 3 adds the sharper instrument.

*Detection power of T3, derived from geometry alone (no run).* The finest geometric error
this grid can express is **one cell on one axis**. With
`d ln f = −(k_i² / Σ k²) · (dL_i / L_i)`:

| perturbed axis | worst-shifted mode | shift | in MHz | vs the 5.122 MHz budget |
|---|---|---:|---:|---:|
| a: 50 → 51 mm | TE201 | 1.4382 % | 101.7 | **20×** |
| b: 30 → 31 mm | TM110 | 2.4510 % | 142.8 | **28×** |
| d: 40 → 41 mm | TE102 | 2.1552 % | 174.0 | **34×** |

Neither existing gate can do this: a one-cell error in `a` moves TE101 by only 0.7805 %
(inside Gate 1's 1 %) and moves the worst of all seven by only 1.4382 % (inside Gate 2's
2 %). T3 with `max` over all seven modes fires on a one-cell error on **every** axis.
This is the smallest defect in the case's own defect class, and it is the reason T3
exists rather than simply tightening the percentage in T2 — a percentage tight enough to
catch one cell would have no derivation, whereas `0.1/T` does.

---

## 3. Falsifiers this re-gate must satisfy (two-sided, both to be demonstrated)

* **(A)** The case still PASSES on today's correct code, with margin. Predicted from
  §2 T3: measured − f_yee ≤ ~0.15 MHz on all seven (prior provenance: the #812 audit's
  own measurement), i.e. ≥ 30× inside the 5.122 MHz budget. If any new gate comes near
  firing on correct code, the threshold is wrong, not the physics.
* **(B)** The audit's measured defect — `a` 50 → 25 mm, a −50 % single-axis shrink, with
  the oracle constants untouched — must now FAIL. It currently passes Gate 2.

Both, or a justified STOP.

---

## 4. Scope — what this lane does NOT do

* No physics verdict is changed. No claim in `manifest.json`, `validation/README.md`, or
  `docs/public/guide/benchmarks.mdx` is broadened; the claim text is amended only to
  describe the gate accurately (all seven modes, not "at least one higher mode").
* No existing gate is widened. T1 and the 2.0 % of T2 are carried over unchanged.
* No new solver, mesh, or geometry. The gated leg is byte-for-byte the same simulation.

---

## 5. CORRECTION (2026-08-31, appended after the first measurement) — the T3 budget's numeric evaluation was wrong

**What §2 T3 said:** "At the gated leg … `T = 1.9524e-8 s`, `1/T = 51.220 MHz`, budget
**≈ 5.122 MHz**", and the detection-power table quoted margins of 20× / 28× / 34×.

**Why that was wrong.** I evaluated `T` as the full post-excitation record,
`(10491 − 251) · Δt = 10240 · Δt`. It is not. `extract_modes` caps the record at
`_MAX_HARMINV_SAMPLES = 8000` and, when the decimation `step` rounds down to 1 (which it
does here, `10240 // 8000 = 1`), the `[:max_samples]` slice **truncates** rather than
decimates. The span actually handed to harminv is therefore `8000 · Δt`, not `10240 · Δt`.

**Correct values:** `T = 8000 × 1.9066e-12 = 1.5253e-8 s`, `1/T = 65.56 MHz`, budget
`0.1/T =` **6.556 MHz**. Corrected detection-power margins: x **15.5×**, y **21.8×**,
z **26.5×** — still decisive, and the conclusion of §2 T3 is unchanged.

**What did NOT change:** the declared rule. T3 was declared as `0.1/T` with `T` "computed
in-script from the record actually handed to harminv (post excitation-skip, post any
decimation)" — the script computes it that way (`harminv_record`, one source of truth
shared with `extract_modes`), so the gate that shipped implements the rule as declared.
Only my hand evaluation of that rule in the note's parenthetical was arithmetically wrong,
and it is corrected here rather than edited above. The `0.1` factor is untouched. The
extraction path itself is deliberately left as it was — changing what harminv sees would
change the measurement this lane exists to gate, not the gate.

---

## 6. RESULTS (2026-08-31) — both sides of the acceptance criterion, measured

Command: `PYTHONPATH=<worktree> .venv/bin/python validation/crossval/14_rect_cavity_pozar.py`
(grid (51,31,41), Δt = 1.9066e-12 s, 10491 steps, run wall ≈ 2 s).

### (A) The case still PASSES on today's correct code, with margin

```
Gate 0 [wall reg <= 1e-09 m]:  PASS   max |eff - target| = 0.000e+00 m
Gate 1 [TE101 < 1%]:           PASS   TE101 err = 0.0083%
Gate 2 [all 7 modes < 2%]:     PASS   worst TE102 err = 0.0529%   (37.8x inside)
Gate 3 [|meas-yee| <= 6.556 MHz]: PASS worst TE201 residual = 0.1504 MHz (43.6x inside)
VERDICT: PASS
```

Per-mode residual against the discrete-Yee prediction published in §2 T3 **before** this
run, in MHz: TE101 +0.0024, TM110 −0.0270, TE011 −0.0486, TM111 +0.0635, TE201 −0.1504,
TM210 +0.0855, TE102 −0.0755. Max 0.1504 MHz — independently confirming the #812 audit's
"≤ 0.15 MHz on all seven modes", and confirming the published `f_yee` table it was
predicted against. **The physics verdict is unchanged and was never in question.**

### (B) The audit's measured defect now FAILS

`a` 50 → 25 mm (−50 %, single axis), oracle constants untouched, everything else identical:

```
OLD Gate 2 [min over six higher < 2%]:  PASS   (best higher: TE011 err = 0.0154%)   <-- the defect
Gate 0 [wall reg <= 1e-09 m]:  FAIL   max |eff - target| = 2.500e-02 m  (2.5e7x the tolerance)
Gate 1 [TE101 < 1%]:           FAIL   TE101 not extracted
Gate 2 [all 7 modes < 2%]:     FAIL   not extracted: TE101
Gate 3:                        N/A    Gate 0 failed, so the Yee prediction is VOID
VERDICT: FAIL (exit 1)
```

TE011 is the `m = 0` mode: it scored **0.0154 %** on a cavity half the intended width,
exactly as the audit reported, and `min()` handed the gate to it. Under `max` over all
seven, TE101 is NOT FOUND (its true frequency, 7.0756 GHz, is outside the ±3 % search
window around the 4.799 GHz oracle) and NOT-FOUND is now a hard failure.

Note on Gate 1's failure mode: the audit reported "TE101 err 47.334 %". Through the
committed `extract_modes` (±3 % window) TE101 reports NOT FOUND rather than a percentage;
47.3 % is the analytic magnitude of the shift, and both are the same failure.

Worth recording, because it is why `max` matters more than any single sharp gate: at
`a`/2 the shrunk cavity's TE101 is **exactly degenerate** with the original box's TE201
(`k_x = 2π/0.05 = π/0.025`), so the "TE201" row reads 0.0246 % / 0.0045 MHz — it would
have passed Gate 3 on its own. Three other modes are 141–419 MHz off, i.e. 21–64× the
Gate-3 budget. A gate that reads one mode can be defeated by a degeneracy; a gate that
reads all seven cannot.

### (C) Detection power beyond the audited defect — a defect only Gate 3 sees

Cavity fill `eps_r = 1.002` instead of 1.0 (a 0.2 % permittivity error). Geometry
untouched, so this is invisible to Gate 0 by construction; the frequency shift is
`1/√1.002 − 1 = −0.0999 %`, far inside Gates 1 and 2. Predicted before the run from the
oracles alone: −4.79 to −8.06 MHz per mode. Measured:

```
Gate 0:  PASS   max |eff - target| = 0.000e+00 m
Gate 1:  PASS   TE101 err = 0.1081%
Gate 2:  PASS   worst TE102 err = 0.1542%
Gate 3:  FAIL   worst TE102 residual = 8.2486 MHz  (budget 6.556 MHz)
VERDICT: FAIL
```

Per-mode residuals −4.79, −5.85, −6.28, −6.92, −7.24, −7.74, −8.25 MHz against the
predicted −4.79, −5.82, −6.24, −6.92, −7.06, −7.79, −8.06. Every pre-existing gate passes;
Gate 3 alone fails. Gate 3 fires at ~0.163 % permittivity error against Gate 2's ~4 %:
**~25× more sensitive** on this defect class. This is the answer to "why not just tighten
the 2 %" — a percentage tight enough to do this would have no derivation, whereas `0.1/T`
does.

### Falsifiers committed

`tests/crossval/test_crossval_cv14_cavity_gates.py`, 16 tests: 12 synthetic (fast lane, no FDTD)
pinning the gate math, the axis-blind `min()` defect, NOT-FOUND handling, the unchanged
2 % threshold, Gate 0 on a one-cell error on each axis, Gate 3's void-not-passing
behaviour when Gate 0 fails, and the `0.1/T` scaling; plus 3 end-to-end FDTD legs marked
`slow` reproducing (A), (B) and (C) above. Each of (B) and (C) first asserts that the
PRE-#812 gate still passes on that leg, so the regression is anchored to the defect rather
than to the new code.
