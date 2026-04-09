---
title: "Meep crossval deep diagnosis — waveguide bend field comparison"
date: 2026-04-09
type: handoff
project: rfx
tags: [crossval, meep, upml, subpixel, field-comparison, waveguide-bend]
status: validated
---

## What Was Done

Systematic field-distribution-based diagnosis of rfx vs Meep discrepancy for
the waveguide bend tutorial (2D TMz, ε_r=12, resolution=10, PML boundaries).

### Root Causes Identified

| Suspect | Result | Impact |
|---------|--------|--------|
| PML physics | **IDENTICAL** — integrated absorption ratio = 1.000000 | None |
| PML polynomial order | Both use order=2, R=1e-15 | None |
| Courant number (S=0.70 vs 0.50) | Tested S=0.50 in rfx — **no change** in fields | None |
| Grid offset (0.5×dx) | Confirmed, interpolation test — **minimal impact** | None |
| Subpixel overlap bug | **FIXED** — union SDF for same-material shapes | eps 9.25→6.50 |
| Source mechanism | Soft E-field (rfx) vs J→D→E (Meep) — **phenomenon** | ~113 fs timing |

### Subpixel Fix (smoothing.py)

The L-shaped waveguide uses two overlapping boxes. The old code processed shapes
sequentially: shape 2 used shape 1's smoothed boundary value as "outside eps",
giving double-smoothed eps=9.25 at the waveguide corner edge (correct: 6.5).

Fix: group shapes by eps_r, compute union SDF via `min(sdf1, sdf2, ...)` for
same-material shapes, then apply smoothing once per group.

### Field Comparison Results (after fix)

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Envelope xcorr | 0.947 | 0.952 |
| Mean field env corr | 0.854 | 0.882 |

### Qualitative Field Assessment

6-frame time-domain field progression review at t = 0.10, 0.20, 0.35, 0.50, 0.80, 1.20 ps:

1. **Guided mode profile** — rfx transverse confinement pattern matches Meep
2. **Waveguide confinement** — energy stays within dielectric in both
3. **Bend scattering** — 90° corner radiation and mode coupling match
4. **Standing waves** — same fringe count, spacing, and envelope in vertical arm
5. **PML absorption** — no boundary reflections visible in either code
6. **Difference maps** — all show spatial-shift-type residuals (wavefront dipole), not structural discrepancies

**Verdict: Valid field-level cross-code validation.**

### Key Diagnostic Findings

- PML σ_max formula, polynomial order, and R_asymptotic all match Meep exactly
- Meep default PML profile is u² (verified via `pml.pml_profile(u)`)
- rfx Courant S=0.700 ≈ magic time step (1/√2 = 0.707) gives minimal numerical
  dispersion for axis-aligned waves — this is an advantage, not a problem
- Grid offset: Meep cells at `(i+0.5)×dx`, rfx at `i×dx` — inherent convention difference
- Source timing offset (~113 fs) is from Meep's J→E build-up delay, well-documented

### Files Changed

```
rfx/geometry/smoothing.py  — union SDF for overlapping same-material shapes
rfx/boundaries/upml.py     — half-cell σ offset, no n/2 scaling (from prev session)
rfx/sources/sources.py     — ModulatedGaussian cutoff parameter
rfx/api.py                 — add_source uses J source (Cb-normalized)
examples/crossval/01_field_progression_review.py — 3-step field comparison
```

### Diagnostic Scripts (not committed)

```
examples/crossval/02_deep_field_diagnostic.py  — epsilon/sigma/PML zoom
examples/crossval/03_grid_aligned_comparison.py — grid alignment test
examples/crossval/04_courant_test.py           — Courant S=0.5 vs 0.7 test
```

### Next Steps

1. Validate next Meep tutorial example (ring resonator or straight waveguide)
2. Run full test suite to verify subpixel fix doesn't regress
3. Consider higher resolution comparison for tighter field agreement
