# Cross-validation verdict — the "beats classical" headline is RETRACTED (2026-08-26)

Tier-1 (imperative hard-PEC geometry + independent extractor, 80-period window,
all runs settled at ≈ −119 dB with 82/82 reliable bins) and tier-1b (mesh-
calibrated classical baseline). Jobs 369367256312-321. Raw JSON in
`out_vessl/xval1/`.

## The number that changes the conclusion

Sweeping the classical stub length on THIS mesh (the calibration any engineer
would do, and the same calibration the paper applies to its own notch):

| stub (mm) | notch @ 6 GHz | contrast | own minimum |
|-----------|---------------|----------|-------------|
| 5.80 | −6.3 dB | −5.9 dB | −42.9 @ 7.20 GHz |
| 6.10 | −8.0 dB | −0.5 dB | −48.0 @ 6.90 GHz |
| 6.40 | −12.3 dB | +6.8 dB | −51.3 @ 6.50 GHz |
| 6.70 | −17.6 dB | +13.1 dB | −32.6 @ 6.30 GHz |
| **7.00** | **−35.5 dB** | **+31.6 dB** | **−35.5 @ 6.00 GHz** |
| 7.37 (analytic) | −16.3 dB | +12.8 dB | −36.8 @ 5.70 GHz |

Against that calibrated baseline:

| design | contrast | verdict |
|--------|----------|---------|
| **classical stub, mesh-calibrated (7.00 mm)** | **+31.6 dB** | best |
| C free-form (no seed) | +30.5 dB | parity (−1.0 dB) |
| B stub-seeded (damped gray) | +18.8 dB | behind |
| classical stub, analytic length (7.37 mm) | +12.8 dB | mis-targeted by the mesh |

**The free-form design reaches classical parity; it does not beat classical.**
The 2.2x margin reported on 2026-08-25 was the product of two optimistic
biases, both of which this cross-check was built to find:

1. **Un-calibrated baseline.** The analytic λ/4 length puts its notch at
   5.70 GHz on this mesh (the ε_eff staircase the paper documents), so a large
   part of the margin measured the baseline being mis-targeted, not the
   gradient winning. Give the classical design the same mesh that the
   optimizer was given, and it lands −35.5 dB exactly on target.
2. **Same-operator evaluation.** The phase-1c ranking (B 21.5 > C 18.9) came
   from the differentiable operator at its PEC limit. On independent hard-Box
   geometry the ranking FLIPS (C 30.5 > B 18.8) because B's narrow high-Q stub
   shifts to 6.10 GHz when rasterized as real geometry, while C's distributed
   50-box structure stays on 6.00 GHz. B's operator-path score was optimistic.

## What still stands

- **Free-form binary metal topology optimization works in rfx.** From a random
  low-fill start with NO seed and no transmission-line theory, it found a
  design whose independently-verified notch sits exactly on target at
  −34.9 dB, matching a hand-derived classical stub. The accepted paper lists
  free-form binary metal as the principal open limitation; parity from an
  unseeded start is a real step across that line, and it is the honest claim.
- The probe-01 landscape measurements (spurious resonance on the Kottke path,
  90 %-flat plateau on the legacy path) are measurements, not comparisons —
  unaffected.
- Damped conductive gray still fixes the SOFT traversal (smooth 100x descent
  vs arm A's oscillation). It did not produce the best hard design here.
- Passband loss is better than first reported: −4.2 dB (independent absolute
  path) rather than −8 dB (diverged plane extractor).
- The empty line reads 0.00 dB on the independent path — the absolute
  calibration of this lane is sound.

## What this says about the demonstration problem

A single-frequency notch on a uniform line has ONE effective degree of freedom
and an exact closed-form answer. A 2 256-variable search cannot beat a λ/4
stub at the task the λ/4 stub was derived for; the most it can do is find it.
That is what happened, and it is the same sentence the accepted paper already
uses for the dielectric taper: the gradient recovers classical-synthesis
performance, it does not exceed it.

**Phase-2 redirection**: to show a gradient advantage rather than parity, the
demonstration must be a problem with no closed form — multi-band or asymmetric
notch, an area- or feature-size-constrained layout, a co-designed
notch-plus-match, or a geometry where the stub cannot be placed. Parity on the
closed-form problem then becomes the credibility check, not the headline.

## Method note (keep)

Both biases were invisible from inside the optimization loop and both were
caught by changing ONE thing at a time: first the physics path (occupancy
operator -> real PEC boxes), then the baseline's calibration. Every future
verdict in this line should carry: (a) an independent-geometry re-solve, (b) a
baseline calibrated on the same mesh, (c) the solver's own ring-down/passivity
verdict (all runs here settled ≈ −119 dB, 82/82 reliable).
