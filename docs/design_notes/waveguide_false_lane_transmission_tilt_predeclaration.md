# Pre-declaration — what carries the empty-guide transmission tilt on `normalize=False`

Lane: diagnostic evidence for the waveguide `normalize=False` extractor (issue #873,
second and last permitted attempt). Written **before** any number in this attempt was
computed. Every measured number will come from the frozen chain-battery artifact
`tests/fixtures/waveguide_chain_battery/fixture_v18_close.json` (run 3, VESSL run
`369367258638`, commit `f914a7ca`), which is the first battery run made on the
**corrected** port (the N+1-cell transverse eigenproblem, #868, was fixed by PR #889).

## 1. The observable

The empty guide (`dut = "thru"`), `lane = "false"`, three mesh rungs
(dx = 2.54 / 1.27 / 0.635 mm; the guide's wide wall a = 22.86 mm is 9 / 18 / 36 cells),
17 frequency bins from 8.4 to 11.6 GHz.

**Observable: `|S21|² − 1` per bin, per rung**, taken from `cells[].s_params.S21` on
those cells. Its relation to the reported excess is
`column_power_per_bin = |S11|² + |S21|²`, so `column power − 1 = |S11|² + (|S21|² − 1)`.

On run 3 the reflection part has collapsed (worst-bin `|S11|²` = 6.8e-4 / 7.5e-5 /
8.2e-6) and the excess is carried by the transmission part: worst-bin `|S21|² − 1` =
+5.45e-3 / +1.09e-3 / +2.46e-4, and the per-bin curve runs **negative at the bottom of
the band, crosses zero near bin 10-12, and is positive at the top**. Rung ratios of the
column-power excess are 5.28 and 4.56 — second order in dx, slightly faster. The `flux`
extractor on the same fields reads 4.8e-5 / 6.6e-6 / 5.2e-6.

This is a different observable from the one PR #880 (attempt 1) chased. Attempt 1's three
suspects S1/S2/S3 were each a named factor of `Z_seen / Z_used`, i.e. a spurious
**reflection**; it ran on run 1, whose ladder had the #868 port defect folded in. That
defect is the identified implementation defect in attempt 1 that workspace rule R2's RF
intensifier requires before a second attempt on this lane.

## 2. What the code says before any number is computed

Read in `rfx/sources/waveguide_port.py` at the commit this branch is based on:

* `modal_voltage` (L1067) integrates E over the aperture at the **node** plane `ref_x`;
  `modal_current` (L1076) integrates H after `_plane_h_field` (L1106) averages the two
  half-cell planes `ref_x-1` and `ref_x`, i.e. a centred average onto the same node.
* `_co_located_current_spectrum` (L1560) multiplies the I spectrum by `exp(+jω·dt/2)`
  to undo the leapfrog half-step.
* `_extract_global_waves` (L1586): `forward = (V + Z·I)/2`, `backward = (V − Z·I)/2`,
  with `Z = _compute_mode_impedance` (L1517) built on `_compute_beta` (L1465) from
  `cfg.f_cutoff`, `dt`, `dx`.
* `_extract_port_waves` (L1656) relabels, for a `-x` port, (incident, outgoing) =
  (backward, forward). So at BOTH ports of an x-directed thru the quantity that enters
  S21 — `a1` at the driven port and `b2` at the receiving port — is the **same**
  combination `(V + Z·I)/2`, the global +x wave.
* `extract_waveguide_port_waves` (L1818) forms both from `cfg.v_ref_t` / `cfg.i_ref_t`,
  the recorded modal V/I time series at the reference plane, through the full-record
  rectangular DFT `_rect_dft` (L1605).
* `extract_waveguide_s_matrix` (L2111) sets `a_drive, b_drive =
  extract_waveguide_port_waves(final_cfgs[drive_idx], ...)` and then
  `col_slices.append(b_recv / safe_a)`.
* `rfx/sparams/waveguide.py` L1035 is the `normalize=False` caller.

Two consequences follow **from the code alone**, and they shape the candidate list:

**(a) `a1` is MEASURED, not nominal.** The denominator of every S column is the wave
decomposed from port 1's own recorded V/I at its reference plane. `cfg.v_inc_t` (the
analytic drive waveform) and `cfg.src_amp` are recorded but are not read by this lane.
Candidate D of the brief — "a1 taken from the drive's nominal injected amplitude /
mode-overlap model" — is therefore **excluded by code reading**, before any arithmetic.

**(b) Any port-local V/I measurement error that is COMMON to the two ports cancels
exactly in `|S21|`.** Write, at either reference plane, `V_m = α_V·(F + B)` and
`I_m = α_I·(F − B)/Z_g`, with `F`, `B` the true forward/backward modal amplitudes, `Z_g`
the grid's own V/I ratio, and `α_V`, `α_I` arbitrary complex frequency-dependent
transfer factors of the measurement. Put `q = Z_u·α_I/(Z_g·α_V)` and
`Γ = (1−q)/(1+q)`. Then

    a_m = g·(F + Γ·B),   b_m = g·(Γ·F + B),   g = α_V(1+q)/2

at each plane. On the thru, `F₂ = F₁·e^{-jθ}` and `B₁ = B₂·e^{-jθ}` with `θ = β·L`, `β`
REAL above cutoff, `L = 0.08128 m` (the declared plane separation; the reported cells
carry `reference_planes_m = [0.02032, 0.10160]`). Both ports are geometrically identical
and carry the same `f_cutoff`, `dx`, `dt`, so `g` and `Γ` are the same at both. With
`w = B₂/F₁`:

    S11 = (Γ + w·e^{-jθ}) / (1 + Γ·w·e^{-jθ})
    S21 = (e^{-jθ} + Γ·w) / (1 + Γ·w·e^{-jθ})                                    (†)

`g` cancels. When `w = 0`, `|S21| = 1` for ANY complex `Γ`; when `Γ` is real and `w = 0`
the column power is 1 exactly (`|S11|² + |S21|²` collapses algebraically). So a
transmission tilt of order 5e-3 cannot come from a common port-local V/I scaling alone.
It needs either a nonzero `w` beating against `Γ`, or a mechanism outside this class.

## 3. Candidates, each made a computable prediction of `|S21|² − 1 (f, dx)`

All of A-C are port-local V/I scalings, so (†) governs them: each fixes `Γ(f, dx)` and
each predicts a tilt only through the `Γ·w` product.

* **A — spatial half-cell offset along the port normal.** `modal_current` averages H over
  `ref_x-1` and `ref_x`. For a wave `e^{∓jβx}` a centred average scales I by
  `cos(βdx/2)`, the SAME real factor for forward and backward, so it enters `a1` and `b2`
  with the same sign. Prediction: `Γ_A = (1 − cos(βdx/2))/(1 + cos(βdx/2)) =
  tan²(βdx/4)`, real and positive, second order in dx. With `w = 0` it predicts
  `|S21|² − 1 = 0` EXACTLY; the brief's "same or opposite sign" alternative (I effectively
  sampled half a cell off the E node, `e^{∓jβdx/2}` with OPPOSITE sign on the two
  directions) is evaluated as well: it makes `q` direction-dependent, which splits `g`
  into `g_a ≠ g_b` and predicts `|S21|² − 1 ≈ −ψ·β·δ` with `ψ = arg(q)` and `δ` the
  residual offset. Both variants are computed per bin.
* **B — temporal half-step.** If the `exp(+jω·dt/2)` correction were absent, or applied
  with the continuous `ω·dt/2` where the discrete relation wants a different phase, `q`
  acquires a unit-modulus phase `φ`. `Γ_B = (1 − e^{jφ}cos(βdx/2))/(1 + e^{jφ}cos(βdx/2))`.
  With `w = 0` this still predicts `|S21|² − 1 = 0`. With `w ≠ 0` the leading term is
  `−4·Im(Γ·w)·sinθ`, which OSCILLATES across the band (θ sweeps ≈ 7.3 rad, ≈ 1.2 cycles).
  Variants evaluated: correction absent (φ = −ω·dt), correction doubled (φ = +ω·dt), and
  correction continuous-vs-discrete.
* **C — `Z_TE` from a β that is not the grid's discrete β.** `Γ_C = (Z_g − Z_u)/(Z_g + Z_u)`
  with `Z_u` from `_compute_mode_impedance(f, cfg.f_cutoff, dt, dx)` and `Z_g` the same
  formula evaluated on the guide's own discrete cutoff / on the continuum
  `β = √(k² − kc²)`. This predicts REFLECTION, real `Γ`, hence `|S21|² − 1 = 0` with
  `w = 0`. It is carried expressly to show it does **not** fit the tilt; on run 3 it is
  additionally near zero because `port_f_cutoff_hz == fc_discrete_guide_hz` (#889).
* **D — `a1` nominal rather than measured.** EXCLUDED by code reading, §2(a). Recorded
  here so the exclusion is on the record, and checked once numerically by confirming that
  `a1` reconstructed from `cfg.src_amp` is not what the lane divides by (structural, not
  a fit).
* **E — the `Γ·w` product (absorber reflection beating against the port mismatch).** The
  only mechanism inside (†) that can lift `|S21|` off 1. Prediction
  `|S21|² − 1 = |e^{-jθ}+Γw|²/|1+Γw e^{-jθ}|² − 1`, evaluated with `Γ = Γ_A` and `w` from
  an absorber-reflection model `w = |w|·e^{-2jβd}`, `d` the plane-to-CPML-face distance.
  Independently, **(†) is exactly invertible**: given the measured complex `S11`, `S21`
  and the known `θ`, solve `P = Γw = (u − S21)/(S21·u − 1)` with `u = e^{-jθ}`, then
  `Γ² − K·Γ + P·u = 0` with `K = S11·(1 + P·u)`; the two roots are `Γ` and `w·u`. So the
  model class is falsifiable per bin: if the inverted `Γ` is not close to a computable
  candidate, or if `|w|` comes out implausibly large (an empty guide terminated by a
  17-layer CPML cannot present a 10 %+ reflection while the measured band-mean `|S11|`
  is a few times 1e-2), the whole class (A-C, E) is falsified together.
* **F — full-record DFT truncation.** Both ports are DFT'd over the same `n_steps`, but
  the port-2 record is the port-1 record delayed by `L/v_g`, and `v_g` is smallest at the
  bottom of the band. Prediction: the tilt tracks the per-drive ring-down witness
  (`settling_db` −84.9 / −94.6 / −98.0 dB → tail amplitudes 5.7e-5 / 1.9e-5 / 1.3e-5,
  ratios 3.06 and 1.48) and the band-edge group delay, not `dx²`. Computed as the
  amplitude ladder implied by the witness and compared with the measured ratios 5.28,
  4.56.
* **G — per-port gain asymmetry `g₂/g₁`.** If the two ports' measurement operators are not
  identical, `|S21|` carries `|g₂/g₁|` directly and at first order. The code reading in
  §2 finds no asymmetry (`_plane_h_field` averages `(p−1, p)` in GLOBAL indices at both
  ports; `hy_profile`/`hz_profile`, `aperture_dA`, `f_cutoff`, `dt`, `dx` are the same
  objects; `_shift_modal_waves` applies a pure unit-modulus phase, and the `step_sign`
  conversion makes the two local shifts equal). This candidate is therefore carried as
  the **residual bucket**: whatever (†) cannot absorb is reported as an implied
  `|g₂/g₁| − 1` ladder, with its dx order, and NOT attributed to a mechanism.

Nothing else is reached for. If the measured curve is outside A-G, that is branch (iii)
below.

## 4. Decision rule, fixed here

A candidate **closes** when, simultaneously:

1. its predicted `|S21|² − 1` matches the measured value at the worst bin within a factor
   **1.25** in magnitude, at **all three** rungs;
2. it reproduces the **sign crossing**: the bin index at which the predicted curve
   changes sign is within **±2** of the measured one, at all three rungs;
3. its successive-rung ratios are within **±25 %** of the measured 5.28 and 4.56.

Branches:

* **(i) exactly one candidate closes** → identified; go to §5.
* **(ii) more than one closes** → not distinguished; report the tie and name the
  measurement that would break it. No fix.
* **(iii) none closes** → DOES NOT CLOSE. Say so plainly, report the inverted `Γ` and `w`
  of §3-E and the residual ladder of §3-G as the written envelope, and stop. No third
  attempt, no fourth candidate reached for in this run, no gate / tolerance / golden
  moved.

Per-bin traces for every rung, every candidate, are dumped to
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json` (a NEW file;
attempt 1's `suspects.json` is not touched) with provenance: commit, fixture path and
sha256, freqs, and the preflight / warning text of each source cell quoted verbatim.
Headline numbers without the per-bin trace are not a result (workspace rule R5).

## 5. Falsifier, fixed here

Only reached on branch (i). Re-run the three thru cells (`false` lane, coarse / mid /
fine) on CPU with the closing candidate's ingredient corrected — or removed, if the
candidate is an ingredient that should not be there — using the cell builder in
`scripts/diagnostics/waveguide_chain_battery_measure.py` /
`tests/_waveguide_chain_battery_fixture.py`. Predicted **before** the run:

* the worst-bin `|S21|² − 1` collapses to the `flux` lane's level on the same fields,
  i.e. **below 1e-4 at every rung** (flux reads 4.8e-5 / 6.6e-6 / 5.2e-6), and the
  monotonic sign crossing disappears;
* `|S11|` is **unchanged** to within 5 % band-mean at every rung — the candidate is a
  transmission-normalization term and must not move the reflection;
* the per-drive ring-down witness stays at or below the recorded −84.9 / −94.6 / −98.0 dB
  (a correction that changes the settling is changing the solve, not the extraction).

Every preflight line and warning of the re-run is quoted verbatim beside the numbers.
If the re-run does not reproduce this prediction, that is the result: it is written down
and the attempt stops. **No second correction is tried.**

## 6. Constraints on any fix

A fix is admissible only if it moves no committed gate, tolerance or golden anywhere in
the repo. If it would move one, the blast radius is reported instead and nothing is
changed. Any unit test added must red on `main` and green on the branch, and must carry
its derivation in the test docstring. The `flux` lane is not touched.

## 7. Revisions

| date | change |
|---|---|
| 2026-09-16 | first version, committed before any number was computed |
