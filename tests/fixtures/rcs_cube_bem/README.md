# PEC-cube RCS — rfx FDTD vs independent Bempp BEM (campaign Lane 3)

Extends the independent integral-equation cross-check (Lane 1, sphere) to a shape
with **no closed form**.

## What / why

rfx's RCS has an analytic reference only for the sphere (Mie) and the flat plate
(physical optics). For everything else there is **no reference**. This lane uses
the **Lane-1-validated Bempp EFIE harness** — which reproduces exact Mie to
≤0.15 dB — as an independent arbiter for an axis-aligned PEC cube.

Three things make the referee rigorous:
1. Bempp meshes the true faceted surface (surface IE, no FDTD staircase) — a
   different error class than any FDTD sibling (Meep/openEMS).
2. Bempp is shown **converged at every H-plane angle** (main-vs-fine mesh ≤ 0.016
   dB), so it can referee the oblique bins, not just backscatter.
3. The *same* harness reproduces exact Mie to ≤0.15 dB **in-tree**
   (`tests/fixtures/rcs_sphere_three_way/`, Lane 1 / PR #297) — so "converged" is
   backed by "correct." Two further in-tree defences against "converged ≠ correct":
   `kL=3.77` sits **below** the first PEC-cube interior resonance `kL=π√2≈4.44`
   (no EFIE spurious-resonance band), and rfx agrees with Bempp at **both**
   backscatter (−0.42 dB) and forward-scatter (−0.97 dB), so a Bempp error would
   need an implausible angular structure — see `fixture.json`
   `arbiter_correctness_basis`.

An axis-aligned cube is also **grid-perfect in FDTD** (no staircase) — a different
regime than the curved sphere.

## Result (measured, this fixture)

PEC cube `L=0.03 m`, `f0=6 GHz` (`kL=3.77`, resonant region), incidence `+x`,
`E‖z`, H-plane bistatic `θ=π/2`, `φ=π` backscatter:

| region | rfx vs Bempp | reading |
|--------|--------------|---------|
| **backscatter (φ=180°)** | **−0.42 dB** | rfx monostatic **validated** on a non-closed-form shape |
| near-backscatter (φ≥135°) | ≤ 1.06 dB | validated region |
| forward-oblique (φ 15–120°) | **up to +13 dB** | rfx reads high — see below |

The forward-oblique gap is **rfx-side**: the Bempp arbiter is converged there
(≤0.016 dB), so the discrepancy is rfx's documented **bistatic forward-oblique
contamination** (TFSF/NTFF forward-face, issue #280) — now confirmed on a **second
shape** (cube) by a non-FDTD method, i.e. it is not a sphere-specific artefact.
This is **recorded, not gated**, matching the sphere's non-gated bistatic posture.

Physical-optics sanity: rfx and Bempp both give `backscatter/σ_PO ≈ 2.2–2.4`.
`σ_PO = 4π L⁴/λ²` is the `kL→∞` flat-plate asymptote, so at `kL=3.77` (resonant)
it is only an order-of-magnitude anchor — the meaningful independent content is
that **both** land at order 2× PO. (Their *ratio* agreement is arithmetically the
backscatter match above, since PO cancels — not a second independent witness.)

## Files

- `fixture.json` — committed rfx FDTD + Bempp BEM (two mesh densities) H-plane
  bistatic patterns + PO witness + provenance. Clean-checkout durable.
- `generate.py` — offline producer; runs rfx (JAX) then Bempp (Numba) in one
  process and emits the full `fixture.json` mechanically. Bempp is **not** an rfx
  CI/runtime dep. **Env:** `pip install bempp-cl gmsh` (import name `bempp_cl` in
  0.4.x); `OPENBLAS_NUM_THREADS≤64` on many-core boxes.
- Gated by `tests/test_rcs_cube_bem_gates.py`, which recomputes every dB distance
  from the committed raw sigma arrays (the producer's derived dB is not trusted).
  Additive — relaxes no existing RCS gate. humble-crossval: rfx-centric distances;
  Bempp is the converged arbiter, not a verdict that rfx is "wrong" in a
  documented non-validated region.
