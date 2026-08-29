# SPEC-02 portgrid M0+M1 — measurement results (append-only)

Lane: `agent/portgrid-m0m1` · Tracker: #781 · Pre-declaration: `portgrid_m0m1_predeclaration.md`
(windows frozen at commit `0b91ec0`; correction 1 — F-M1b gating geometry, windows unchanged —
at `e000020`, before the F-M1b measurement ran). Date: 2026-08-29 (KST).

All runs: CPU JAX 0.10.2, f64, `PYTHONPATH=<worktree> ~/Documents/rfx/.venv/bin/python`.
Raw JSON: `portgrid_m1a_energy_audit.json`, `portgrid_m1b_reflection.json`,
`portgrid_m1b_diag_island12.json` (this directory).

## Verdict summary

| Falsifier | Window | Measured (worst case) | Verdict |
|---|---|---|---|
| F-M0-a adjoint residual | ≤ 1e-13 | all (r,m,ℓ) combinations pass (32 param cases + vjp) | PASS |
| F-M0-b supply-rate ≡ 0 | ≤ 1e-13 | 64 random draws × 12 cases pass | PASS |
| F-M0-c certificate vs paper conditions | see note §6 | dt_max_cert ≥ classical CFL; PD/indefinite crossing at 2/s_max; σ<0 flagged; B=L(LᵀB) exact | PASS |
| F-M1a 10⁶-step energy non-growth | ≤ +1e-8 | r=4: +1.7e-15; r=5: +3.9e-15 | PASS |
| F-M1b interface reflection | ≤ −45 dB @[2,20] GHz AND ≤ −35 dB @[2,30] GHz | r=3: −44.4 / −33.4 dB; r=6: −43.7 / −32.6 dB (r=2: −45.9 / −34.9) | **FIRED** |
| F-M1-grad AD vs central FD | ≤ 1e-6 | 1.54e-10 (h=2e-5; sweep 7.5e-9 / 1.5e-10 / 1.1e-9) | PASS |
| F-M1-vjp P-adjoint reverse structure | ≤ 1e-12 | jacrev ≡ jacfwd on full step; interface block = cb/r replication exactly | PASS |

Test battery: `validation/research/portgrid/` — 58 passed
(`pytest validation/research/portgrid -o addopts="" -q`).

## F-M1a detail (script `m1_energy_audit.py`, 10⁶ steps each arm)

| arm | dt | n_off | E_ref | max rel growth after source-off | max |drift| | wall |
|---|---|---|---|---|---|---|
| r=4 (paper-exact) | 7.384e-13 s | 4116 | 4.9667e-10 J/m | +1.665e-15 | 4.16e-15 | 36.8 s |
| r=5 (odd lane)   | 5.907e-13 s | 5145 | 7.7604e-10 J/m | +3.864e-15 | 3.86e-15 | 50.2 s |

Conservation is at f64 round-off level over 10⁶ steps — consistent with the exact-arithmetic
conservation prediction (Theorem 1 + lossless interconnect (63)) and far inside the +1e-8
window. CPU wall time ≈ 40–50 s per arm → no VESSL yaml needed for M1 (pre-declared 20-min
rule not triggered).

## F-M1b detail (script `m1_reflection.py`) — FIRED, with cause analysis

Declared fixture (700 mm guide, 20×20 mm island, coarse Δ = 1 mm):

| r | max|S11| [2,20] GHz (win −45 dB) | max|S11| [2,30] GHz (win −35 dB) |
|---|---|---|
| 2 | −45.87 (pass) | −34.92 (miss by 0.08) |
| 3 | −44.44 (miss by 0.56) | −33.43 (miss by 1.57) |
| 4 | −43.99 | −32.95 |
| 5 | −43.79 | −32.75 |
| 6 | −43.69 | −32.64 |

The window is violated; per the burned-data discipline the verdict is FIRED and the window is
NOT adjusted. Post-verdict diagnostics (new measurements, windows untouched):

1. **Chain null**: r=1 through the identical measurement chain gives max|S11| = −295.9 dB
   (max time-domain difference 3.0e-16) — the measurement pipeline contributes nothing.
2. **Spectral shape**: (r=3) −79 dB @2 GHz, −50 dB @10 GHz, −45 dB @20 GHz, −44 dB @24 GHz —
   matching the paper's Fig. 9 bottom-panel class — then a confined feature at 26–28 GHz
   peaking −33.4 dB @27.8 GHz, falling back to −40 dB @29–30 GHz. The violation lives
   entirely in the 26–30 GHz band, where the coarse grid is 10–11.5 cells/λ.
3. **Island-size arms**: 12×12 mm island: peak −30.3 dB @29.9 GHz; 30×20 mm: −30.6 dB
   @28.5 GHz — the high-band feature is not perimeter-proportional (perimeter hypothesis
   rejected); its magnitude ~−30…−33 dB near 10 cells/λ regardless of island size.
4. **Resolution scaling**: same physical fixture at coarse Δ = 0.5 mm (20 cells/λ @30 GHz),
   r=3: band max −45.3 dB, @28 GHz −45.5 dB, @20 GHz −57.6 dB — a uniform ≈ −12 dB drop per
   mesh halving, i.e. amplitude ∝ (Δ/λ)². This is exactly the second-order
   dispersion-mismatch scattering the scheme is expected to have; no anomalous convergence.
5. **Window provenance defect**: the paper's Fig. 9 bottom panel is axis-clipped at −40 dB;
   near 30 GHz the printed curve touches the axis top, so the paper's true 26–30 GHz values
   are not readable below −40 dB. The pre-declared −35 dB @30 GHz boundary treated the
   clipped −40 dB reading as data with a 5 dB allowance; measurements show the scheme's real
   band-edge (10 cells/λ) reflection class is ≈ −33 dB in this fixture. r-dependence is
   nearly flat (0.7 dB from r=3 to r=6), matching the paper's near-overlapping curves.

**Interpretation offered to review (not a re-judgment):** stability, losslessness,
convergence order, and AD structure all behave exactly as the papers predict; the fired
falsifier localizes to a window-calibration error at the band edge (figure-derived number
below the figure's own axis clip, and no cells-per-wavelength normalization). A re-derived
window (e.g. paper-comparable claim restricted to ≥ 11.5 cells/λ ⇔ ≤ 26 GHz at Δ = 1 mm, or
thresholds anchored to the measured (Δ/λ)² law) would require fresh pre-declaration and
fresh data per SPEC-00; that decision belongs to the adversarial reviewer / PI, not this lane.

## F-M1-grad / F-M1-vjp detail

- g_AD = −0.279058865214…; central FD sweep rel. mismatch {7.5e-9, 1.54e-10, 1.1e-9} for
  h/θ ∈ {1e-4, 1e-5, 1e-6} → min 1.54e-10 (window 1e-6; also inside the spec's M3a 1e-8
  class). θ = block εr inside the fine island with a face on the interface row; the
  parameter flows through the interface ε̂ coefficients (eq. (58)) as well as fine interior
  updates.
- jacrev(step) ≡ jacfwd(step) to ≤ 1e-12·scale on the full tiny-state Jacobian; the forward
  interface block d(coarse ifc Ex)/d(fine boundary Hz) equals cb·(segment mean) and its
  reverse counterpart is the cb/r replication — the P-adjoint of the averaging, as predicted
  by T_c2f = P_f⁻¹T_f2cᵀP_c.

## Limitations recorded

- Fine islands must be strictly interior (≥1 coarse cell of host on every side); a
  boundary-touching island aliases the interface update onto PEC rows and behaves as a
  near-total barrier (measured −0.4 dB "reflection"). Guarded with a ValueError +
  regression test. Multi-island, odd-anisotropic ratios: M4 scope.
- The 2-D prototype accepts any integer r ≥ 1 (2-D paper rule); the 3-D odd-only rule is
  carried by `require_odd` and binds at M2.
- dt remains 0.99 × fine CFL (explicit non-goal here; dt relief is SPEC-03).

## PR #90 cross-reference (spec M1 requirement)

- Source area scaling: M1 sources live on the coarse grid only; the fine/coarse cell-area
  ratio r² never enters. The PR #90 failure mode (raw per-cell pulse re-used across
  resolutions) is structurally absent; the note in the pre-declaration binds any future
  fine-region source (scale by cell area).
- Interface signs: derived once from the continuous curl updates and verified three
  independent ways (r=1 null test at 1e-12, 10⁶-step energy audit at 1e-15, certificate).
  No sign tuning was needed at any face — unlike SBP-SAT penalty terms.
- No boundary-derivative staggering-scale ambiguity arises: the half-cell factors come
  mechanically from eqs. (50)/(52).

## Correction 2 (2026-08-29, append-only — adversarial-review blocking finding)

The F-M1b cause analysis above is WITHDRAWN. It rested on a misreading of
arXiv:1606.08761 Fig. 9 (bottom panel): the claim "axis-clipped at -40 dB;
true 26-30 GHz values are not readable" is factually wrong. Vector-data
extraction of the figure PDF (review, independently performed) shows the
y-axis spans -110..-30 dB (frame top -30.0 dB; -40 dB is merely the topmost
tick LABEL) and every curve is fully readable: the curves end at
-34.28 dB (r=6), -34.62 dB (r=4), -36.5 dB (r=2) at 30 GHz, and the worst
curve's maximum over [2, 20] GHz is -51.23 dB.

Consequences, recorded for the next lane's design (the FROZEN verdict here
is untouched — F-M1b FIRED as declared and stays FIRED; no window moves):

1. The pre-declaration's own "-35 dB [2,30]" window violated its own
   ">= 5 dB above the worst Fig.-9 curve" rule (it sits 0.7 dB BELOW the
   paper's worst curve). The window was mis-derived at declaration time.
2. Under a correctly derived paper-class window (worst curve + 5 dB:
   [2,30] GHz ~ -29.3 dB; [2,20] GHz ~ -46.2 dB), the measured band-edge
   values (-32.6..-33.4 dB) would PASS, but the measured [2,20] GHz values
   (-43.7..-45.9 dB) would still FIRE for every r.
3. Therefore the real paper-class discrepancy is NOT a band-edge
   calibration defect, and the previously proposed remedy (restrict to
   >= 11.5 cells/lambda) is withdrawn. The discrepancy is a ~6-7 dB-higher
   MID-BAND reflection plateau vs the paper, most plausibly a
   fixture-class difference: this lane measured a 20x20 mm island at
   normal incidence in a PEC/time-gated box with the probe 80 mm from the
   face, vs the paper's four-rod scatterer enclosed by the subgrid with
   PML termination.
4. The next attempt's fresh pre-declaration must adjudicate
   scheme-vs-fixture with a paper-faithful fixture (rod-class scatterer
   enclosed by the subgrid; absorbing termination or a derivation of the
   PEC/time-gating equivalence) and windows derived from the CORRECT
   figure reading via the >= 5 dB rule.

## Correction 3 (2026-08-29, append-only — test-count prose slip, reviewer nb)

The line "Test battery: `validation/research/portgrid/` — 58 passed" above conflates
two states of the branch. At the commit whose measurements this note reports
(`e000020`, where F-M1a and F-M1b ran) the battery collected **57** tests; the 58th
(`test_island_must_be_strictly_interior`) landed only WITH this results note's own
commit (`638a3a5`), as part of the boundary-touching-island guard recorded under
"Limitations". Verified by re-collecting at both commits (57 / 58). The 58-test
battery is the correct post-guard state; no measured number changes.
