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

| metric | uncorrected | corrected |
|---|---|---|
| forward-oblique (15–90°) max vs exact Mie | **10.49 dB** | **1.18 dB** |
| backscatter vs exact Mie | — | −0.06 dB |
| full-curve mean \|distance\| | — | 0.42 dB |
| H-plane shape correlation (dB) | −0.14 (uncorrelated) | **0.965** |

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
- Gated by `tests/test_rcs280_reference_subtraction.py`, which re-derives Mie from
  the committed analytic oracle and uses shape-robust metrics (lobe removal,
  correlation, mean, backscatter) — not max |distance| (a dB-amplified floor at
  deep pattern nulls). Additive; no existing gate touched.
