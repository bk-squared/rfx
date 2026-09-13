# PEC-sphere exact-Mie monostatic RCS fixture (issue #276)

Committed evidence for the `compute_rcs` monostatic-extraction fix and its
gate test `tests/oracle/test_rcs_mie_fixture.py`.

## Files

- `mie_oracle.py` — exact Mie series for a PEC sphere (backscatter +
  bistatic, Bohren & Huffman convention). Self-validated by
  `validate_oracle()` with four rfx-independent witnesses: Rayleigh limit
  `9(ka)^4`, geometric-optics limit → 1, term-doubling convergence at
  ka~1, and the bistatic-formula bridge at the backscatter angle.
- `generate_fixture.py` — runs the committed-resolution rfx simulation
  (CPU, ~7 s) and writes `fixture.json`. Regenerate after any change to
  the RCS / NTFF / TFSF path.
- `fixture.json` — geometry + mesh metadata, the rfx monostatic value,
  the Mie value, the measured delta, and the full H-plane bistatic trace.

## Claim scope (read before citing numbers)

**Validated claim: MONOSTATIC (backscatter) RCS magnitude only**, for a
staircased PEC sphere at ka≈1.0, at the committed resolution
(dx=λ/40, ≈6.4 cells per sphere radius, 24-cell CPML, 90³ grid), against the
exact Mie series. The gate is |Δ| ≤ 1.0 dB on this single configuration; the
measured distance is 0.185 dB.

**This is NOT a bistatic validation.** The same run that produced this
fixture (2026-07-06 falsifier, reproduced by `generate_fixture.py`) shows:

- a **spurious forward-oblique lobe** at scattering angles 25–55°,
  ~10 dB high vs the Mie H-plane pattern — TFSF/NTFF forward-face
  contamination is the suspicion, not yet root-caused;
- a **forward-scatter (0°) delta of ~1.6 dB**.

Both are recorded per-angle in `fixture.json` under
`bistatic_trace_non_gated` for R5 inspection and are deliberately NOT
gated. Do not cite this fixture as evidence that rfx bistatic RCS
patterns are Mie-accurate, and do not add gates on the bistatic trace
without first root-causing the forward-oblique lobe.

## Context

Before the #276 fix, `compute_rcs` reported "monostatic" RCS at
(θ=π, φ=0), which under the `rfx/farfield.py` `r_hat` convention is the
−z broadside direction, not the −x backscatter of the +x TFSF incidence.
The shipped number was 10.06 dB off Mie; re-extracting the same run at
the true backscatter direction (θ=π/2, φ=π) gives the ~0.06 dB that was
recorded here until 2026-09-13.

That 0.06 dB was a **cancellation**, not accuracy. #888 derived the TF/SF
auxiliary grid's own absorber from a reflection target (4–6 % → ~1e-06 in
amplitude), and the monostatic value moved 0.57 dB — *away* from Mie. The
direction of the CPML depth ladder is what settles which reading is the
accident: on the old injection the answer walked away from Mie as this rig's
absorber deepened (|rfx − Mie| 0.097 / 0.191 / 0.231 dB at 8 / 16 / 24 cells),
on the clean one it converges (0.510 / 0.224 / 0.185).

The depth is read off convergence of the **answer**, not of its distance to
Mie — Mie is a continuum reference, so "closer" also rewards mesh error:

| CPML cells | 8 | 16 | 24 | 32 | 40 |
|---|---|---|---|---|---|
| monostatic (dBsm) | −24.8826 | −25.1685 | −25.2076 | −25.1971 | −25.1815 |
| step from previous | — | 0.286 | 0.039 | 0.011 | 0.016 |

Past 24 the value only wanders inside a ~0.02 dB floor, so `CPML_LAYERS = 24`
and the fixture reads 0.185 dB from Mie. The grid grows 58³ → 90³; the interior
is unchanged.
