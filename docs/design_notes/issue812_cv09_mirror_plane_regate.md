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
`tests/test_boundary_pmc_hi_faces.py`; not touched here). With
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
