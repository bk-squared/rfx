# issue #280 — two-run incident-reference subtraction for RCS bistatic

Validates `compute_rcs(subtract_incident_reference=True)` against the **exact Mie
bistatic series** on a PEC sphere (an analytic reference at every angle — no
external solver needed).

## The bug (issue #280)

rfx's RCS **bistatic** pattern read too high at forward-oblique observation angles
(spurious lobe ~10 dB above Mie near 25–55°), while the **monostatic/backscatter**
bin was correct. Prior guesses blamed the staircased PEC surface.

## Diagnosis (mechanism isolated)

An **empty-domain** run (no scatterer) reproduces the *same* forward-oblique lobe.
A perfect TFSF+NTFF gives zero far-field with no target, so this is pure
**residual incident-field leakage** from the discrete TFSF boundary that the NTFF
box integrates into a spurious far-field. It **nulls at backscatter** (≥ 80 dB
below its forward-oblique peak here) — which is exactly why the monostatic bin
stayed clean. The leakage is target-independent (the TFSF injects the same
incident field with or without a scatterer).

## Fix

Two-run reference subtraction at the **complex** far-field level:
`E_scat = E_far[target] − E_far[vacuum]`, then `σ = 4π|E_scat|²/|E_inc|²`. This is
the standard total-field/scattered-field normalization; it cancels the leakage
exactly. `compute_rcs(subtract_incident_reference=True)` runs the vacuum reference
and subtracts (doubles the solve cost). Default `False` keeps the validated
monostatic path **byte-identical**.

## Result (PEC sphere, ka≈1, exact Mie reference)

Regenerated 2026-09-13 on the converged 24-cell CPML and the derived TF/SF
auxiliary absorber (#888). The previous row of this table — 10.49 / 1.18 /
−0.06 dB / 0.42 dB / 0.965 — was taken on an 8-cell CPML with an auxiliary grid
that reflected 4–6 % back into the injected field.

| metric | uncorrected | corrected |
|---|---|---|
| forward-oblique (15–90°) max vs exact Mie | **10.66 dB** | **1.46 dB** |
| backscatter vs exact Mie | — | +0.19 dB |
| full-curve mean \|distance\| | 3.10 dB | 0.70 dB |
| H-plane shape correlation (dB) | −0.09 (uncorrelated) | **0.977** |

The corrected pattern's mean distance got **worse**, 0.42 → 0.70 dB, and that is
a property of the #280 correction rather than of the depth: sweeping the two
absorbers independently, both converge in CPML depth and they converge to
different values (auxiliary 20 cells: 0.481 / 0.404 / 0.408 / 0.412 at cpml
8 / 16 / 24 / 32; auxiliary 200 cells: 0.886 / 0.719 / 0.705 / 0.714). The
subtraction was cancelling more of the pattern error with a contaminated
reference field present. Why is open, and belongs to #280.

The spurious lobe is removed and the corrected pattern tracks the exact analytic
Mie bistatic. The remaining ~1 dB residual is **not** leftover leakage; its
components were isolated during the #280 diagnosis (see the `compute_rcs`
docstring for the full list and the converged-bistatic recipe): curved-surface
**staircase** (shrinks with resolution; an independent Meep run at matched
resolution shows the same order — FDTD-generic), a **deep-pattern-null bias**
from the default NTFF box sitting in the radiating near field (cured by
enlarging the domain), and a ±1–2 dB bright-bin **placement sensitivity**
(thicker CPML reduces its normal-reflection component). The fix was also
validated case-independently (PEC ka 0.8–2.0 and dielectric spheres vs the
exact Mie series) during the same diagnosis.

## Files

- `fixture.json` — committed rfx bistatic (corrected + uncorrected) + exact Mie +
  the empty-domain leakage + metrics. Clean-checkout durable.
- `generate.py` — offline producer (pure rfx + scipy Mie oracle; no external
  solver, no CI deps). Emits `fixture.json` mechanically.
- Gated by `tests/unit/farfield/test_rcs280_reference_subtraction.py`, which re-derives Mie from
  the committed analytic oracle and uses shape-robust metrics (lobe removal,
  correlation, mean, backscatter) — not max |distance| (a dB-amplified floor at
  deep pattern nulls). Additive; no existing gate touched.
