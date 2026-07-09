# PEC-sphere three-way RCS — exact-Mie / rfx-FDTD / Bempp-BEM (campaign Lane 1)

The **first independent integral-equation cross-check** in rfx's RCS pipeline.

## What / why

rfx's RCS is cross-checked against the exact **Mie** series (analytic) and, at
coarse test-scale, that is the only reference. Every other candidate lane
(Meep / openEMS) is **also FDTD** — it shares rfx's staircase-and-time-domain
error class, so FDTD-vs-FDTD agreement cannot rule out a shared-method artefact.

**Bempp-cl** solves the Maxwell EFIE on a **triangulated true curved surface**
(surface integral equation, RWG basis) — a fundamentally different discretization
with **no staircase**. BEM-vs-Mie agreement therefore confirms the Mie reference
with an independent method, and isolates rfx's coarse-ladder error as rfx-side
resolution (staircasing + the near-field NTFF box; the sibling fixture also
records a large domain-size swing), not a comparator/extraction artefact.

## Result (measured, this fixture)

Backscatter `sigma/(pi a^2)`, PEC sphere, `a = 0.015 m`, incidence `+x`, `E‖z`:

| ka  | Bempp (BEM) | exact Mie | Bempp − Mie |
|-----|-------------|-----------|-------------|
| 0.8 | 2.563 | 2.596 | −0.056 dB |
| 1.0 | 3.614 | 3.638 | −0.028 dB |
| 1.5 | 1.098 | 1.076 | +0.091 dB |
| 2.0 | 0.974 | 1.008 | −0.151 dB |

Bempp reproduces exact Mie to **≤ 0.15 dB** across the ladder. `h`-refinement at
ka=1 converges monotonically (`a/3 → a/5 → a/8`: −0.109 → −0.046 → −0.018 dB).

**Three-way at ka≈1** — three independent methods within 0.07 dB:

| method | sigma/(pi a^2) | vs Mie |
|--------|----------------|--------|
| exact Mie (scipy) | 3.638 | — |
| rfx FDTD, fine (λ/40, 6.4 cells/radius) | 3.585 | −0.063 dB |
| Bempp BEM (h=a/6, N=1884) | 3.614 | −0.028 dB |

The rfx **coarse** E4 ladder (λ/10–15, 1–5 cells/radius; `rcs_mie_e4`, 13.9 dB
envelope) carries 4.7–9.3 dB error; Bempp confirming Mie to ≤0.15 dB at the same
ka is a non-FDTD witness that this is rfx **resolution** (staircasing + near-field
NTFF box), consistent with the two-regime finding (fine = 0.06 dB). Stated as an
rfx-centric distance; no solver is framed as wrong.

## Files

- `fixture.json` — committed Bempp column + re-derived Mie + rfx (sourced from
  `rcs_sphere_mie/`) + convergence witness + provenance. Clean-checkout durable
  (never `.omx`).
- `generate_bempp.py` — offline producer for the Bempp column. Bempp is **not** a
  CI/runtime dependency of rfx; run manually to regenerate. **Env:** `pip install
  bempp-cl gmsh` (import name `bempp_cl` in 0.4.x, not `bempp`); set
  `OPENBLAS_NUM_THREADS<=64` on many-core boxes.
- Gated by `tests/test_rcs_sphere_three_way_gates.py`, which **re-derives Mie
  from `scipy.special`** (does not import the producer) so the reference cannot
  self-certify. Additive — it relaxes no existing RCS gate.
