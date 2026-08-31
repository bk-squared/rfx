# cv10 re-gate — PMC realization + image-doubling control arm

Lane: `agent/regate-cv10`. Case: `validation/crossval/10_pmc_cpml_half_symmetric.py`.
Tracker: issue #812 (crossval gate audit), critical tier, first entry.
Status of this section: **PRE-DECLARATION**. Written and committed BEFORE the
measurement that judges it. Append-only — later sections correct, never rewrite.

This is instrument work. **No physics verdict of cv10 is challenged.** The
v1.7.5 PMC/CPML composition fix that cv10 locks is not in question; what is in
question is whether cv10's gate can fail when that fix is absent.

## 1. What the audit measured (prior measurement, not this lane's)

Issue #812 reports that monkeypatching `rfx.boundaries.pmc.apply_pmc_faces` to
a no-op — i.e. deleting the boundary condition the case is *named for* — makes
cv10 score **better**, not worse:

| leg | peak spread `(max−min)/max` | gate |
|---|---|---|
| uniform, correct code | 0.2487 % | < 2 % PASS |
| uniform, `apply_pmc_faces` → no-op | 0.0667 % | < 2 % PASS (better) |
| non-uniform, correct code | 0.00784 % | < 2 % PASS |
| non-uniform, `apply_pmc_faces` → no-op | 0.0201 % | < 2 % PASS |

with the probe peaks scaled by 0.5077 and the field bit-identical to a PEC wall.

## 2. The mechanism, stated so the repair is forced to address it

Every current cv10 gate is a **within-path relative spread**:

```
peak_range = (peak_max - peak_min) / peak_max      # over cpml_layers ∈ {2,4,6,8}
```

`peak_range` is invariant under `peak_i → c · peak_i` for any constant `c > 0`.
Deleting a boundary condition on a face that the direct source→probe path never
touches is *exactly* such a constant factor: the array truncation silently
supplies a PEC (`E_tan = 0`) wall in place of the absent PMC (`H_tan = 0`) wall,
which flips the image-source sign and rescales every leg of the sweep by the
same factor. The gate divides that factor out by construction.

So the repair cannot be a tighter relative spread. It must add (i) a **direct
realization check** — is the wall the thing it claims to be — and (ii) an
**absolute amplitude reference** that a constant factor cannot cancel.

## 3. Gate 3 (new) — PMC realization, bit-exact

**Statement.** At the end of every swept run, on the declared `y_lo` face:

```
max |Hx[:, 0, :]| == 0.0   and   max |Hz[:, 0, :]| == 0.0    (bit-exact)
```

with the non-degeneracy guard `max |Hx| over the whole array > 0` (so an
all-zero field cannot satisfy the gate vacuously).

**Threshold derivation — definitional, nothing fitted.** `0.0` is not a
tolerance. `rfx/boundaries/pmc.py::apply_pmc_faces` writes the literal `0.0`
into `hx[:, 0, :]` and `hz[:, 0, :]` on a `y_lo` face, and the scan body
(`rfx/simulation.py:1338-1340`, `rfx/nonuniform.py:1326-1327`) applies it after
the CPML-H stage, so nothing re-populates those cells before the state is
returned. Any code path in which the wall is not realized leaves the ordinary
H update in place and the face is generically non-zero. There is no window to
widen and no number to fit. The guard is likewise threshold-free (`> 0`).

**Both paths, all four `cpml_layers` values.** Free — no extra run; it reads
`result.state` of runs the case already performs. This is the leg that covers
`84b11aa` (the NU scan body previously never called `apply_pmc_faces` at all),
which is the historical defect cv10 was written for and which its relative
spread cannot see.

## 4. Gate 4 (new) — image-doubling control arm, absolute amplitude

**Geometry, derived from the PMC-plane convention this file already states.**

Half domain: `y' ∈ [0, 20] mm`, `dx = 1 mm`, 21 y-nodes `j' = 0…20` at
`y' = j'·dx`. `apply_pmc_faces` zeros `Hx`/`Hz` at array index 0, and those
components sit at `y' = 0.5 mm`. `H_tan = 0` there ⇒ `Ez` is **even** about
`y' = 0.5 mm` ⇒ node `j'` mirrors onto node `1 − j'`.

Images of the interior nodes `j' = 1…20` therefore land on
`y' = 0, −1, …, −19 mm`. The full symmetric domain is `y' ∈ [−19, 20] mm`
= **39 mm**, 40 nodes, 19.5 mm on each side of the plane. Map
`y_full = y' + 19 mm`; the mirror plane is at `y_full = 19.5 mm`.

- half-domain source `Ez` at `y' = 1 mm` ⇒ full-domain `Ez` **pair** at
  `y_full = 20 mm` and `y_full = 19 mm`, equal amplitude, **same sign** (a PMC
  images a tangential electric current in phase; a PEC would reverse it — this
  is the sign the defect flips);
- half-domain probe `y' = 5 mm` ⇒ full-domain probe `y_full = 24 mm`;
- full domain: CPML on **both** y faces, x/z extents and `cpml_layers`
  identical to the half domain.

The full-domain problem is then mirror-symmetric about `y_full = 19.5 mm` in
geometry, sources, and both y absorbers. Its `H_tan` is therefore identically
zero on that plane for all time — exactly the condition the half domain
imposes. The two runs are the same discrete problem, so

```
R = peak|Ez(probe)|_half-PMC / peak|Ez(probe)|_full-image   ==  1
```

identically, in exact arithmetic. The reference leg contains **no PMC face**,
so it is untouched by any defect in `apply_pmc_faces` — the control arm is an
independent measurement of the amplitude the relative spread threw away.

**Gate.** `|R − 1| < 0.02`, evaluated on each path at
`cpml_layers = max(CPML_VALUES) = 8`.

**Threshold derivation — three classes, none of them the data being judged.**

1. *Prior provenance.* `0.02` is the tolerance gate 1 has carried since the
   v1.7.5 lock (2026-04-20). It is re-used **unchanged and not widened**, now
   applied to an ABSOLUTE amplitude comparison where a constant multiplicative
   factor cannot cancel. Moving a number sideways onto a stricter comparator is
   not a new window.
2. *First-principles residual budget.* The discrete image identity above is
   exact, so the only residuals are (a) float32 accumulation over 800 steps,
   `≈ √800 · 2⁻²⁴ ≈ 2e−6` relative, and (b) the CPML reflection floor, budgeted
   at the conventional **−40 dB = 1 %** for an 8-layer polynomial-graded
   absorber. `0.02` is ≥ 2× that floor and ~10⁴× above (a).
3. *Discrimination margin.* The competing hypothesis — the wall is not
   realized, so array truncation supplies a PEC wall — **reverses the image
   sign**, turning a constructive pair into a destructive one. That is an O(1)
   change in absolute amplitude for any sub-wavelength plane offset (here
   `k·d = 2π·0.5 mm / 60 mm = 0.052 rad`), i.e. tens of percent at minimum. A
   2 % window sits more than an order of magnitude inside it.

`cpml_layers = 8` is chosen a priori as the thickest absorber in the existing
sweep — the leg for which the −40 dB budget in (2b) is best justified — not
because of anything observed.

**Cost.** Two extra runs (one per path), same scale as the eight the case
already performs.

## 5. Correction to the module docstring of `10_pmc_cpml_half_symmetric.py`

The docstring at `649b2cf` states, of the half-cell PMC-plane offset:

> "UNLIKE cv09, this offset biases NO comparator here: this crossval is a
> self-consistency / regression lock … so the half-cell shift changes no
> pass/fail gate … there is no gate-3 class comparison against a mirrored full
> geometry to protect."

**That was true only while every gate was a within-path relative spread, and
gate 4 makes it false.** Gate 4 is precisely a comparison against a mirrored
full geometry. The `y' = 0.5 mm` plane location is now load-bearing: it fixes
the full-domain length at **39 mm, not 40 mm**, and the image pair at
`y_full = 19/20 mm`. Getting the plane wrong by a half cell would move the
control-arm geometry and break gate 4. The docstring is updated accordingly;
the superseded sentence is recorded here rather than merely deleted.

## 6. Acceptance criteria for this lane (two-sided, both required)

- **(A)** the re-gated case still exits 0 on today's `main`;
- **(B)** with `rfx.boundaries.pmc.apply_pmc_faces` monkeypatched to
  `lambda state, faces: state`, the re-gated case exits 1, and it does so on
  gates 3 and 4 rather than on gate 1 (gate 1 is expected to keep passing —
  that is the audit's whole point, and it is not a defect of gate 1 but of
  relying on gate 1 alone).

(A) alone is cosmetic; (B) alone means the case is broken. A permanent pytest
falsifier pins (B) so it cannot silently regress.

---

## 7. MEASUREMENT (appended 2026-08-31, after the freeze at commit `a00a53d`)

Implementation frozen at `bfbd196`. Host: local CPU JAX venv
(`~/Documents/rfx/.venv`), float32. Command:

```
PYTHONPATH=<worktree> .venv/bin/python validation/crossval/10_pmc_cpml_half_symmetric.py
```

### 7.1 Criterion (A) — the case still passes on today's correct code. **exit 0**

| path | cpml | peak \|Ez(probe)\| | max\|H_tan\| on y_lo | max\|H\| in array |
|---|---:|---:|---:|---:|
| uniform | 2 | 1.569e−02 | **0.000000e+00** | 1.204e−06 |
| uniform | 4 | 1.571e−02 | **0.000000e+00** | 2.654e−10 |
| uniform | 6 | 1.572e−02 | **0.000000e+00** | 4.944e−10 |
| uniform | 8 | 1.572e−02 | **0.000000e+00** | 3.650e−10 |
| nonuniform | 2 | 1.573e+07 | **0.000000e+00** | 2.583e+03 |
| nonuniform | 4 | 1.573e+07 | **0.000000e+00** | 6.531e−01 |
| nonuniform | 6 | 1.573e+07 | **0.000000e+00** | 1.754e−01 |
| nonuniform | 8 | 1.573e+07 | **0.000000e+00** | 5.039e−01 |

| gate | uniform | nonuniform |
|---|---|---|
| G1 peak spread `< 2 %` | 0.249 % PASS | 0.008 % PASS |
| G2 no NaN/Inf | PASS | PASS |
| G3 `max\|H_tan\|` on y_lo `== 0.0` | `0.0` PASS | `0.0` PASS |
| G4 `\|R − 1\| < 2 %` @ cpml = 8 | R = 0.999968, 0.0032 % PASS | R = 0.999967, 0.0033 % PASS |

G4 legs: uniform half 1.572448e−02 vs full-image 1.572498e−02; nonuniform half
1.572944e+07 vs full-image 1.572996e+07. (The two paths' absolute scales differ
by the documented legacy `amplitude_kind=None` per-path convention — a
`Cb`-normalized field add on uniform, a current in amperes on NU — which is why
G4 is evaluated per path and never across paths.)

The measured `|R − 1|` of 3.2e−5 / 3.3e−5 sits ~600× inside the 2 % window and
~1.6× above the √800·2⁻²⁴ ≈ 2e−6 float32 budget of §4(2a) — i.e. the residual is
at the arithmetic floor, as the exactness argument predicted, and nowhere near
the −40 dB CPML allowance the window was sized against. The window is **not**
tightened on the strength of that: it was frozen at `a00a53d` and stays there.

Wall time: 10 runs, 3.0–4.3 s each, ~37 s total. The two control-arm runs cost
4.1 s (uniform) and 3.9 s (nonuniform) — the audit's estimate of two extra ~4 s
runs was right.

### 7.2 Criterion (B) — the new gates fail on the defect the audit measured

Defect reproduced verbatim: `rfx.boundaries.pmc.apply_pmc_faces` replaced by
`lambda state, faces: state`, then the re-gated script run through its own
`main()`. **exit 1.**

| path | cpml | peak \|Ez(probe)\| | max\|H_tan\| on y_lo |
|---|---:|---:|---:|
| uniform | 2 | 7.978e−03 | 8.662774e−08 |
| uniform | 4 | 7.982e−03 | 4.013157e−10 |
| uniform | 6 | 7.983e−03 | 4.396453e−10 |
| uniform | 8 | 7.983e−03 | 7.516874e−10 |
| nonuniform | 2 | 7.982e+06 | 1.007089e+02 |
| nonuniform | 4 | 7.984e+06 | 3.418385e−01 |
| nonuniform | 6 | 7.984e+06 | 9.125803e−01 |
| nonuniform | 8 | 7.984e+06 | 2.579114e−01 |

| gate | uniform | nonuniform |
|---|---|---|
| G1 peak spread `< 2 %` | 0.067 % **PASS (blind — this is the finding)** | 0.020 % **PASS (blind)** |
| G2 no NaN/Inf | PASS (blind) | PASS (blind) |
| G3 `max\|H_tan\|` on y_lo `== 0.0` | 8.66e−08 **FAIL** | 1.007e+02 **FAIL** |
| G4 `\|R − 1\| < 2 %` | R = 0.507689, 49.23 % **FAIL** | R = 0.507564, 49.24 % **FAIL** |

The audit's numbers reproduce to the digits it published: no-op uniform spread
**0.067 %** vs 0.0667 %, no-op NU spread **0.020 %** vs 0.0201 %, correct
uniform **0.249 %** vs 0.2487 %, correct NU **0.008 %** vs 0.00784 %, peak
scale factor **0.5077**. Separately confirmed in the same session: the no-op
half-domain time series is `np.array_equal`-identical to the same run with
`y_lo` declared `"pec"` — the array truncation really does supply a PEC wall.

Both new gates fail on both paths, and they fail for the reason they were
written: G3 because the face is no longer a magnetic wall, G4 because the image
sign flipped and the absolute amplitude halved. Gate 1 is confirmed still
passing under the defect — that contrast is the point, and it is asserted
explicitly (not merely observed) in
`tests/test_crossval10_pmc_regate.py::test_noop_pmc_defeats_the_old_relative_gate_and_is_caught_by_the_new_ones`,
so a future refactor cannot silently return cv10 to a state where deleting the
wall is undetectable.

### 7.3 Test suites

`pytest -o addopts="" -m "not gpu"`:
`tests/test_crossval10_pmc_regate.py`, `tests/test_boundary_pmc_composition.py`,
`tests/test_pmc_plane_convention.py`, `tests/test_crossval_manifest_contract.py`,
`tests/test_boundary_pmc_hi_faces.py` → **28 passed**;
`tests/test_example_fidelity_contract.py` → **143 passed**;
`-k "crossval or benchmark or docs"` → **89 passed, 2 skipped**;
public-carrier and crossval gate-logic suites → **41 passed**.

### 7.4 Ledger entries that depend on this case

The authoritative ledger is READ-ONLY from this lane (SPEC-00 §0.3); nothing
below was written. Three entries reference cv10:

1. `docs/agent-memory/index.md:266` — the **Rule** that a crossval may stand on
   "a self-invariant that detects architectural regressions (e.g. crossval 10's
   peak-stability invariant locks the v1.7.5 PMC+CPML composition fix)". **This
   is the entry the change affects, and it was false as written**: §7.2 shows
   the peak-stability invariant does *not* detect the `84b11aa` architectural
   regression — deleting `apply_pmc_faces` leaves it passing, better. After
   this lane the sentence's *conclusion* is true of cv10 as a whole, but its
   *named mechanism* is wrong; the exemplar should read "crossval 10's
   bit-exact PMC-realization and image-doubling gates".
2. `docs/agent-memory/index.md:247` — row 10, Reference column "self
   (peak-stability invariant)", Status **A**. The status letter stands (and is
   now earned); the Reference column is incomplete — it should also name the
   realization and image-doubling invariants.
3. `docs/agent-memory/rfx-known-issues.md:~3976` — the resolved "PMC + CPML
   composition on the same axis" entry lists as one of its two regression locks
   the "free-space PMC+CPML reproducer (peak stable across cpml ∈ {2,4,6,8})".
   For the `84b11aa` half of that fix ("wire `apply_pmc_faces` into the NU scan
   body — pre-fix the NU 'PMC' was effectively a free boundary") that lock was
   measurably ineffective; gate 3 is what locks it. The *physics verdict* of
   the entry — the composition bug is fixed — is unchanged and unchallenged.

`docs/agent-memory/accuracy_class_sweep_20260712.md:22` also lists cv10, under
E4 truncation, as "applies(form only; ~0 on within-path relative gate)". That
remains true: gate 4 compares two legs recorded over the same 800 steps, so
record truncation is common-mode and cancels between them.
