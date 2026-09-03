# Pre-declaration — which factor carries the `normalize=False` column-power excess

Lane: diagnostic evidence for the empty-guide column-power observation (issue #873).
Written **before** the comparison numbers in
`tests/fixtures/waveguide_false_lane_column_power/suspects.json` were produced.
No new FDTD solve is authorized here: every measured number comes from the frozen
chain-battery artifact `tests/fixtures/waveguide_chain_battery/fixture.json`
(VESSL run `369367257823`, commit `ca168584`).

## The observable

On the thru (empty guide) the `normalize=False` extractor reports column power above
unity: `cells[]` with `dut="thru", lane="false"`, key `column_power_max`

| rung | dx (mm) | column power − 1 |
|---|---|---|
| coarse | 2.54 | 1.8253e-02 |
| mid | 1.27 | 4.0817e-03 |
| fine | 0.635 | 9.8341e-04 |

Successive ratios 4.47 and 4.15 — second order in dx. An empty guide neither absorbs
nor reflects, so the number is an extraction artefact.

## What is being tested

The excess is a **spurious reflection**: `S_ij = b_i / a_j` with
`a = (V + Z·I)/2`, `b = (V − Z·I)/2` (`rfx/sources/waveguide_port.py::_extract_global_waves`).
On a purely forward wave `b = 0` **iff** the `Z` the extractor uses equals the `V/I` the
grid actually presents. So

    Γ_spurious = (Z_seen − Z_used) / (Z_seen + Z_used)

and the column-power excess is `|Γ|²` plus whatever the transmission term contributes.
Each suspect is a named, computable contribution to `Z_seen / Z_used`.

## Suspects (issue #873's list, made computable)

* **S1 — modal cutoff → wave impedance.** `Z_used = _compute_mode_impedance(f, cfg.f_cutoff, dt, dx)`.
  `cfg.f_cutoff` is the discrete eigenvalue of the port **aperture**, which is one cell
  wider than the guide. `Z_seen` uses the guide's own discrete cutoff.
  Predicted `Γ1 = (Z(fc_guide) − Z(fc_port)) / (Z(fc_guide) + Z(fc_port))`.
* **S2 — Yee half-cell offset along the port normal.** `_plane_h_field` averages H over
  `x_idx−1` and `x_idx`; for a forward wave that scales I by `cos(β·dx/2)`.
  Predicted `Γ2 = tan²(β·dx/4)`.
* **S3 — aperture weighting / transverse pairing.** The extractor smooths the simulated H
  with a `[1,2,1]/4` stencil per transverse axis (`_plane_h_field_at_dual`, `h_offset=(0.5,0.5)`)
  and pairs it with an **already** smoothed profile (`hy = −_shift_profile_to_dual(ez, h_offset)`),
  while the `+face` aperture cell is dropped by weight. Predicted from the built port
  profiles and the guide's own mode, no solve.

The guide's own cutoff is not assumed: the battery already fitted it from the thru's S21
phase (`port_cutoff.per_rung[*].fc_fit_hz`, rms residual in `rms_deg_at_fit`).

## Decision rule, fixed here

A suspect **reproduces the ladder** when, at all three rungs simultaneously,

1. its predicted band-mean `|Γ|` is within a factor **1.25** of the measured band-mean
   `|S11|` on the thru (equivalently within 1.25² = 1.56 in power), and
2. its predicted successive-rung ratio is within **0.20** of the measured ratio.

Both conditions, all three rungs. The factor 1.25 is what a leading-order discretization
argument can be expected to deliver, not a number chosen from the residual.

* **(i) exactly one suspect (or the stated product of all three) reproduces the ladder** →
  identified. A fix is admissible only if it moves no committed golden, tolerance or gate
  anywhere in the repo, checked by running the full fast suite. If it would move one, stop
  and report the blast radius instead.
* **(ii) more than one reproduces it** → not distinguished; report the tie and name the
  measurement that breaks it.
* **(iii) none reproduces it** → NON-CLOSING. Say so plainly; do not reach for a fourth
  suspect in this run.

## Disclosure

Before this note was written, S1 was hand-evaluated at the coarse rung's lowest bin
(`Γ1 ≈ 0.064` against a measured `|S11| = 0.115`), so it was already known that S1 alone
does not close on magnitude. The 1.25 bar above is therefore set knowing S1 alone fails
it; what remains open at declaration time is whether the **product** S1·S2·S3 closes, and
what the residual's dx order is.

## Modelling assumptions, stated up front

* The guide's transverse mode is `sin(π·j/N)` at array index `j` (walls on nodes,
  `N = a/dx` = 9 / 18 / 36). Backed by the S21 phase fit selecting
  `fc = (2/dx)·sin(π·dx/2a)` over `c/2a` at every rung.
* `Ez` and `Hy` are co-located in the transverse plane on rfx's Yee grid
  (`rfx/core/yee.py::update_h`: `curl_y = dEx/dz − dEz/dx`, forward-staggered in x and z
  only), so the true H sample shares the E sample's transverse index. The alternative
  (H offset by half a cell in u) is reported as a sensitivity, never as the headline.
