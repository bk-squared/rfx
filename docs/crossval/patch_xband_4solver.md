# X-Band Inset-Fed Patch — Four-Solver Cross-Validation (rfx / openEMS / Palace / CST)

This document is the complete, self-contained record of the four-solver patch
cross-validation summarized in Sec. III-A of the T-MTT paper *"rfx: An
End-to-End Differentiable 3-D FDTD Simulator for RF and Microwave
Engineering"* (Fig. 2 and the X-band rows of Table II there). The paper keeps
only the conclusions; everything needed to verify or reproduce them is here.

Campaigns: 2026-07-04 → 2026-07-07 (rfx / openEMS / Palace), 2026-07-12 (CST,
run independently within REMI Lab from the dimensional specification below,
without access to the other solvers' setups).

## 1. Geometry (all dimensions mm)

Substrate: Rogers **RO4003C**, thickness **h = 0.787** (31 mil), nominal
εr = 3.38, tan δ ≈ 0.0027 @ 10 GHz. 1-oz (35 µm) copper both sides; the
entire bottom face is ground (no etching).

Top metal (single layer):

| Parameter | Value | Note |
|---|---|---|
| Patch length L (resonant, x) | **8.595** | feed direction |
| Patch width W (y) | **10.129** | |
| 50-Ω feed-line width w | **1.80** | |
| Inset depth d | **2.40** | feed line penetrates into the patch |
| Notch gap g | **0.90** (each side) | etched slots flanking the feed line |
| Feed-line length | ≥ 12 (board edge → patch) | straight run |
| Board size | ≥ 31 × 31 (35 × 35 recommended) | ≥ 10 clearance around the patch |

## 2. Two-tier protocol

The study deliberately separates **geometry fidelity** from **feed modeling**:

- **Tier 1 — shielded eigenmode.** The identical structure enclosed in an
  all-PEC box, x 30.595 × y 30.129 × z 10.787 mm (ground = bottom face; the
  feed line shorted to the x = 0 wall). No ports, no absorbing boundaries —
  a pure geometry/material equivalence check across solvers.
- **Tier 2 — radiating |S11|.** Open boundaries, each solver's native feed
  model. Because feed models differ, Tier-2 comparisons are made **within a
  common feed type** (see §4).

## 3. Tier-1 results — shielded fundamental resonance

| Solver | Method | f0 (GHz) | 2nd mode (GHz) |
|---|---|---|---|
| rfx | FDTD + harmonic inversion (Harminv) | 9.131 | 10.764 |
| openEMS | FDTD ring-down | 9.194 | 10.799 |
| Palace | FEM eigenmode | 9.199 | 10.797 / 10.806 |
| CST | eigenmode solver | 9.221 | 10.777 |

**Four-solver spread: 1.0 %.** Known systematic residuals: CST's adaptive
mesh was still rising ≈ +0.6 %/pass at pass 5, and CST alone modeled the
metal at its physical t = 35 µm (the others use sheet/one-cell metal) — both
push CST slightly high and account for its position at the top of the band.
Bonus corroboration: CST finds modes at 6.45/6.73/6.81 GHz where rfx's
below-window Harminv lines sat (6.627/6.791 GHz), confirming those were real
modes rather than numerical artifacts.

## 4. Tier-2 results — radiating |S11|

| Solver / feed model | Feed type | Dip (GHz) | Depth (dB) |
|---|---|---|---|
| rfx, MSL port (dx = 197 µm) | guided | 9.250 | −7.0 |
| openEMS, MSL port | guided | 9.2625 | −9.5 |
| CST, waveguide port (t = 35 µm, tan δ = 0.0027) | guided | 9.33 | −12.1 |
| CST, discrete port (same model, port swapped) | lumped | 9.12 | −20.2 |
| Palace, lumped port (driven) | lumped | 9.05* | −13.0* |

**Headline finding — feed-model dominance.** The responses cluster by *feed
model*, not by solver: the guided-feed models agree at 9.25–9.33 GHz with a
mean absolute |S11| deviation of 0.5–0.9 dB between curves over 8–11.5 GHz,
while a **controlled port swap within CST alone** — geometry, mesh, and
boundaries identical — moves the dip 210 MHz lower and 8 dB deeper. The feed
model, not the field solver, is the dominant residual between packages, and
dip depth is comparable only within a common feed type (depths are further
non-comparable across solvers wherever loss models differ).

\* **Palace-driven caveat.** The Palace driven-port leg used a tight (~10 mm)
radiation box with a first-order absorbing boundary, which demonstrably pulls
the driven dip low (the same mesh's shielded eigenmode agrees with the FDTD
legs at 9.199 GHz); its record was also produced with a pre-correction
LossTan = 1e-3 setting. It is therefore **excluded** from radiating-tier
comparisons (and from the paper's Fig. 2); Palace corroborates the study at
the shielded tier.

**Excluded rfx leg.** An rfx dx = 98 µm refinement leg exists but carries an
unresolved passivity violation and must not be used in comparisons.

## 5. Data and reproduction

- **Scripts, falsification ledger, evidence** (including both CST Touchstone
  files, `cst_s11_{waveguide,discrete}_port_t35um_hw260712.s1p`): this
  repository, branch `research/calibration-inverse`, under
  `scripts/research/calibration/crossval/` (start with `verdicts.json` and
  `REPRODUCE.md`).
- **Figure-source copies** used by the paper's Fig. 2 ship with the paper's
  figure script (`figures/generate_patch4solver_figure.py` +
  `figures/data/patch4solver/` in the paper repository).
- CST runs: CST Studio Suite (FIT time-domain for S11; eigenmode solver for
  Tier 1), performed in REMI Lab, 2026-07-12.
- Follow-on: fabrication and VNA measurement of this exact board are in
  progress; the measured `.s1p` will be compared against the guided-feed
  cluster (SMA end-launch is a physically guided feed).

## 6. Documented limits (do not un-learn)

1. The patch |S11| **dip frequency is mesh-limited and an unstable argmin** —
   never use it as a pass/fail gate; gate on band-integrated curves or
   recovered material parameters.
2. **Dip depth is not comparable across feed models or loss models** (see §4).
3. The shielded tier is the only solver-pure comparison; every radiating
   number carries its feed model with it.
