# cv09 re-gate — pre-declaration (issue #812, Phase 1)

Lane: `validation/crossval/09_half_symmetric_waveguide.py` (claims-bearing).
Author session: 2026-08-31. **This file is APPEND-ONLY.** Sections below the
`## Pre-declaration` heading were committed BEFORE any measurement that judges
them (burned-data rule, SPEC-00 §0.2.2); measurements are appended afterwards
in their own section and never edited back into the pre-declaration.

## The defect this re-gate exists to close

Issue #812's audit measured, of this case:

> The #722 half-cell mirror-plane offset this case was rewritten for (PR #762)
> is a **no-op at this mesh** — both declarations build grid (24,21,61),
> `f_half = 8.19589 GHz` identical, gate 3 = 0.001%; and gate 3's 5% window
> tolerates ±1 cell of mirror-plane error.

Reproduced here at grid-build level (no solve required — `Grid.__init__` plus
`rfx.fidelity.fidelity_report`, 2026-08-31, this worktree):

| declared `HALF_X` | dx (mm) | grid shape | x cells | mesh line | realized H_tan wall |
|---|---|---|---|---|---|
| `a/2 + dx/2` (this file, post-#762) | 0.508 | (24, 21, 61) | 23 | 11684.0 um | 11430.0 um |
| `a/2` (pre-#762 convention) | 0.508 | (24, 21, 61) | 23 | 11684.0 um | 11430.0 um |
| `a/2` (pre-#762, at its own mesh) | 0.635 | (19, 17, 49) | 18 | 11430.0 um | 11112.5 um |

The two 0.508 mm rows are byte-identical because `Grid` takes
`n_cells = ceil(L/dx)` and `a/2 = 22.5 dx` is not an integer number of cells:
`ceil(22.5) = ceil(23.0) = 23`. The PR #762 declaration change therefore
carries no information at this mesh — **the mesh change 0.635 -> 0.508 mm is
what removed the half-cell bias, not the `+ dx/2`** — and no gate in the file
reads the realized mirror plane at all. Gate 3 (`5%`) is 1.7x wider than the
1.72–1.84% signature of a half-cell error and 1.7x wider than the 2.70–3.00%
signature of a one-cell error, so it is blind to both.

## The quantity that was never gated

`rfx/boundaries/pmc.py` zeros `H_tan` at array index `-2` on a `_hi` face,
i.e. **0.5 dx inside the declared mesh line** (solver physics, pinned by
`tests/unit/boundaries/test_boundary_pmc_hi_faces.py`; not touched here). With
`n = ceil(HALF_X/dx)` cells the mesh line is at `n*dx` and the realized mirror
plane is therefore

    x_m = (n - 0.5) * dx

and the half domain is the image half of a full guide whose broad wall is

    a_eff = 2 * x_m = (2n - 1) * dx.

`a_eff` is the physical quantity PR #762 was about. It is realized, not
declared, and it is read here off the production reporter
(`rfx.fidelity.fidelity_report`, domain row, `realized_um[1]`), which already
applies the same half-cell rule — so the gate and the solve cannot disagree
about the convention by construction.

**Cross-check against the #722 / PR #762 PMC-plane convention: they agree.**
The convention is realize-declared with an odd cell count: `a = 45 dx` (odd)
makes `a/2 = 22.5 dx` an H-node plane, and the declared hi face at `23 dx` puts
the index `-2` zero exactly there. `a_eff = (2*23 - 1) dx = 45 dx = a`. The
`a_eff` definition above is that same statement with the declaration removed
from it, which is precisely why it can see the case where declaration and
realization part company.

## Pre-declaration

### Gate 0 (NEW) — realize-declared geometry, both runs

    |realized_extent(axis) - declared(axis)| < DX/4    for x, y, z of the full
                                                       cavity and y, z of the half
    |a_eff - a| < DX/4                                 for the half cavity's mirror

Derivation of `DX/4`, from the lattice alone — no measured frequency enters it:
`a_eff = (2n - 1)*dx` is an ODD multiple of `dx`, so the values `a_eff` can
take form a lattice of spacing `2*dx`. The smallest **nonzero** misregistration
this mesh family can express is therefore `dx` (reached when `a/dx` is an even
integer — exactly the dx = 0.635 mm case, `a = 36 dx`, best `a_eff = 35 dx`),
and `2*dx` when `a/dx` is odd (this file, `a = 45 dx`). `DX/4` is a quarter of
that minimum: it admits only exact registration (residual 0 up to float
round-off) and rejects every expressible error by a factor of >= 4, with a
factor-2 guard band below `dx/2` so a future mesh cannot sit on the gate edge.
The same quarter-cell budget states REALIZE-DECLARED-BY-MESH (#722/#724) for
the other axes: `dx` is required to divide `a`, `b`, `d`, so their residual
must be 0 too.

### Gate 3 (TIGHTENED) — 5% -> the frequency image of Gate 0

    G3_TOL = |d ln f / d ln a| * (DX/4) / a
           = (d^2 / (a^2 + d^2)) * (DX/4) / a

from Pozar `f_101 = (c/2) sqrt(1/a^2 + 1/d^2)`, whose logarithmic derivative is
`d ln f / d ln a = -(1/a^2)/(1/a^2 + 1/d^2) = -d^2/(a^2 + d^2)`. With
`a/d = 22.86/30.48 = 3/4` this coefficient is exactly `16/25 = 0.6400`, so at
`DX = 0.508 mm`

    G3_TOL = 0.6400 * 127.0 um / 22.86 mm = 3.5556e-3 = **0.3556%**

(the audit's proposed 0.36% to two significant figures; re-derived here, not
adopted). Gate 3 is thus the same statement as Gate 0 expressed in frequency:
it refuses any full-vs-half discrepancy larger than what a quarter-cell
mirror-plane misregistration would produce. The form is kept as the closed
expression rather than a frozen literal so that changing `DX` moves the gate
with the physics instead of leaving a stale number (SPEC-00 §0.2.4).

Gates 1 and 2 (10% vs the Pozar closed form) are NOT changed and NOT widened.

### FFT fallback (REMOVED from the judged path)

`_extract_mode_near` currently falls back to a windowed-FFT argmax when Harminv
returns nothing, and the gates then judge that number silently. The fallback's
frequency quantum is `1/(N_ringdown * dt) = 1/(3072 * 0.969 ps) = 335.9 MHz =
4.099% of f_101` — 11.5x the new `G3_TOL`. A 0.3556% gate cannot be honestly
judged by a 4.099%-quantised estimator, so Harminv returning no candidate
becomes a hard FAIL and the FFT spectrum survives only as printed diagnostic.

### Falsifier predictions (first-principles, stated before measurement)

Evaluated from Pozar at the `a_eff` each configuration realizes — these are
predictions the measurement must confirm, not thresholds fitted to data:

| configuration | realized `a_eff` | `|a_eff - a|` vs `DX/4` | predicted gate 3 | expected |
|---|---|---|---|---|
| this file, `a/2 + dx/2`, dx = 0.508 | 22.8600 mm | 0.0 um < 127.0 um | ~0 (exact discrete image) | **PASS** |
| one-cell hi error (`n = 24`) | 23.8760 mm | 1016.0 um | 2.702% | **FAIL** |
| one-cell lo error (`n = 22`) | 21.8440 mm | 1016.0 um | 3.001% | **FAIL** |
| pre-#762 `a/2` at dx = 0.635 | 22.2250 mm | 635.0 um vs 158.75 um | 1.838% | **FAIL** |
| pre-#762 `a/2` at dx = 0.508 | 22.8600 mm | 0.0 um | ~0 | **PASS** (correct: identical grid) |

The last row is not a gap: at this mesh the two declarations realize the same
mirror plane, so a gate on the realized plane must pass on both. The
declaration-level defect is that `+ dx/2` buys nothing here; the physics-level
defect it was meant to fix is the dx = 0.635 row, and that is the row the new
gate must fail. The historical 1.835% / 1.722% gate-3 readings recorded in the
script docstring are used only as a cross-check that this `a_eff` model
reproduces past measurements; the thresholds above do not descend from them.

---

## Measurements (appended 2026-08-31, AFTER the pre-declaration above)

Host: this pod, CPU JAX, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.
Code: `agent/regate-cv09`, gates implemented in the commit that follows the
pre-declaration commit `f1b3e68`.

### Criterion (A) — the case still passes on today's correct code

`python validation/crossval/09_half_symmetric_waveguide.py` -> exit 0,
`ALL CHECKS PASSED`, 3.5 s + 1.3 s of solve:

```
PASS: full cavity a     = 22.8600 mm vs declared 22.8600 mm, |resid| =   0.0 um < 127.0 um
PASS: full cavity b     = 10.1600 mm vs declared 10.1600 mm, |resid| =   0.0 um < 127.0 um
PASS: full cavity d     = 30.4800 mm vs declared 30.4800 mm, |resid| =   0.0 um < 127.0 um
PASS: half cavity a_eff = 22.8600 mm vs declared 22.8600 mm, |resid| =   0.0 um < 127.0 um
PASS: half cavity b     = 10.1600 mm vs declared 10.1600 mm, |resid| =   0.0 um < 127.0 um
PASS: half cavity d     = 30.4800 mm vs declared 30.4800 mm, |resid| =   0.0 um < 127.0 um
PASS: full-cavity f = 8.1958 GHz, |err| = 0.007% < 10%
PASS: half-cavity f = 8.1959 GHz, |err| = 0.007% < 10%
PASS: |f_full - f_half| / f_full = 0.0006% < 0.3556%
```

Gate 0 residuals are exactly 0 um; gate 3 clears its new tolerance by 590x.
The physics verdict is unchanged from the pre-#812 run (full 8.1958 GHz,
half 8.1959 GHz) — this re-gate changes what is *judged*, not what is solved.

### Criterion (B) — the new gates fail on the defects the old ones passed

Full sweep, every leg an independent FDTD solve pushed through the script's
own `geometry_gate` / `G3_TOL`:

| mesh | half-domain declaration | n | realized `a_eff` | `\|a_eff-a\|` | gate 0 | `f_half` | gate 3 | NEW | OLD 5% |
|---|---|---|---|---|---|---|---|---|---|
| 0.508 | `a/2 + dx/2` (post-#762, shipped) | 23 | 22.8600 mm | 0.0 um | PASS | 8.19589 | 0.0006% | PASS | PASS |
| 0.508 | `a/2` (pre-#762 convention) | 23 | 22.8600 mm | 0.0 um | PASS | 8.19589 | 0.0006% | PASS | PASS |
| 0.508 | `a/2 + 3dx/2` (one-cell HI) | 24 | 23.8760 mm | 1016.0 um | **FAIL** | 7.97440 | 2.7018% | **FAIL** | PASS |
| 0.508 | `a/2 - dx/2` (one-cell LO) | 22 | 21.8440 mm | 1016.0 um | **FAIL** | 8.44174 | 3.0003% | **FAIL** | PASS |
| 0.635 | `a/2 + dx/2` (naive control) | 19 | 23.4950 mm | 635.0 um | **FAIL** | 8.05451 | 1.7224% | **FAIL** | PASS |
| 0.635 | `a/2` (pre-#762, its own mesh) | 18 | 22.2250 mm | 635.0 um | **FAIL** | 8.34604 | 1.8347% | **FAIL** | PASS |

(`f_full` = 8.19584 GHz at dx = 0.508 mm, 8.19567 GHz at dx = 0.635 mm; gate-0
tolerance 127.00 / 158.75 um, gate-3 tolerance 0.3556% / 0.4444%.)

Every pre-declared prediction is confirmed to within the measurement:
2.702% predicted / 2.7018% measured; 3.001% / 3.0003%; 1.838% / 1.8347%;
1.721% / 1.7224%. The last two also reproduce the script docstring's own
historical readings (1.835% and 1.722%), which is the independent check that
the `a_eff = (2n-1)dx` model is the right model and not a coincidence.

**The old 5% gate passes all four defects.** That is the blindness the audit
measured, now stated as an executable assertion
(`test_old_five_percent_gate_was_blind_to_every_defect_above`).

End-to-end falsifier — the shipped script with `HALF_X = a/2 + 3dx/2`
substituted (one-cell mirror error), run unmodified otherwise:

```
FAIL: half cavity a_eff = 23.8760 mm vs declared 22.8600 mm, |residual| = 1016.0 um >= DX/4 = 127.0 um
PASS: full-cavity f = 8.1958 GHz, |err| = 0.007% < 10%
PASS: half-cavity f = 7.9744 GHz, |err| = 2.709% < 10%
FAIL: |f_full - f_half| / f_full = 2.7018% >= 0.3556%
SOME CHECKS FAILED                                          [exit 1]
```

Note that gates 1 and 2 (10%, unchanged, not widened) both PASS on this
defect — they never were the instrument for it, which is why gate 0 exists.

### The #762 declaration change is a no-op — as a proposition, not a datum

`n = ceil(HALF_X/dx)` and `a_eff = (2n - 1)*dx`, so `a_eff = a` requires
`n = (a/dx + 1)/2`, which is an integer only when `a/dx` is ODD. When it is,
that `n` is produced by ANY declaration in `((a/dx - 1)/2, (a/dx + 1)/2] * dx`
— an interval containing both `a/2` and `a/2 + dx/2`. When `a/dx` is even,
`a_eff` is an odd multiple of `dx` and `a` is an even one, so no declaration
registers. Hence **on a ceil-based grid the `+ dx/2` term never converts a
wrong mirror plane into a right one**, at any mesh. Rows 1-2 and 5-6 of the
table are the measurement of both halves of that statement.

This is why the new gate reads the realized plane and not the declaration: it
reports rows 1 and 2 as identical (correctly — they are the same solve) while
failing every configuration whose mirror plane is actually wrong.

### Cross-check against the #722 / PR #762 PMC-plane convention

They agree. The convention is realize-declared with an odd cell count:
`a = 45 dx` makes `a/2 = 22.5 dx` an H-node plane, and the index `-2` H_tan
zero on the declared hi face at `23 dx` lands exactly there —
`a_eff = (2*23 - 1) dx = 45 dx = a`, which is what gate 0 measures (0.0 um).
The one place this note goes further than the convention text is the
attribution: the convention text credits the `+ dx/2` declaration, and the
measurement credits the odd-cell mesh. The convention's own load-bearing
clause ("the ODD-cell condition is load-bearing, not incidental") is the part
that survives; the declaration term is decoration on a ceil-based grid.

### Not changed

- Gates 1 and 2 stay at 10% of the Pozar closed form. They are loose
  (measured 0.007%), but no defect in #812's cv09 finding is named against
  them, and tightening them needs its own derived budget (the discrete-Yee
  eigenvalue, not the continuum closed form) plus its own two-sided
  demonstration. Left for a separate lane rather than fitted here.
- `rfx/boundaries/pmc.py`'s half-cell placement is solver physics pinned by
  `tests/unit/boundaries/test_boundary_pmc_hi_faces.py` and is untouched.
- `HALF_X` keeps the `+ dx/2` form; see above.
