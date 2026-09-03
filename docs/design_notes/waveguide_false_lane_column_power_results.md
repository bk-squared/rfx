# Results — which factor carries the `normalize=False` column-power excess

Verdict: **branch (iii), NON-CLOSING.** None of the three suspects issue #873 lists
reproduces the ladder under the rule fixed in
`waveguide_false_lane_column_power_predeclaration.md`, and neither does their product.

Everything below is recomputed from the frozen chain-battery artifact
(`tests/fixtures/waveguide_chain_battery/fixture.json`, VESSL run `369367257823`,
commit `ca168584`) by `scripts/diagnostics/waveguide_false_lane_column_power_suspects.py`
into `tests/fixtures/waveguide_false_lane_column_power/suspects.json`, replayed by
`tests/oracle/test_waveguide_false_lane_column_power_suspects.py`. **No new FDTD solve
ran.** Preflight on all three thru cells is empty; the only warning is the documented
`normalize=False` dispersion notice, quoted verbatim in the JSON
(`per_rung.*.warnings`). Ring-down per drive: −84.9 / −85.3 dB (coarse), −94.6 / −95.6
(mid), −98.0 / −98.1 (fine).

## 1. The excess is a spurious reflection

At the worst bin of column 0 (`excess_decomposition_worst_bin_col0`):

| rung | column power − 1 | `\|S11\|²` | `\|S21\|² − 1` | reflection share |
|---|---|---|---|---|
| coarse | 1.8253e-02 | 1.3256e-02 | +4.997e-03 | 0.726 |
| mid | 4.0817e-03 | 4.2612e-03 | −1.795e-04 | 1.044 |
| fine | 9.8341e-04 | 1.0254e-03 | −4.202e-05 | 1.043 |

An empty guide cannot reflect, so `|S11|` is the thing to explain. `b = (V − Z·I)/2`
vanishes on a forward wave only when the extractor's `Z` equals the `V/I` the grid
presents, so every suspect is a named factor of `Z_seen / Z_used` and

    Γ = (Z_seen − Z_used) / (Z_seen + Z_used).

## 2. The guide propagates a cutoff the port config does not carry

Fitted from the thru's S21 phase between the two declared planes (L = 0.08128 m),
already in the battery artifact at `port_cutoff.per_rung`:

| rung | port config `f_cutoff` | guide, discrete | guide, S21-phase fit | rms at port fc | rms at guide fc |
|---|---|---|---|---|---|
| coarse | 5.8772 GHz | 6.5239 GHz | 6.5230 GHz | 8.613° | 0.080° |
| mid | 6.2050 GHz | 6.5488 GHz | 6.5490 GHz | 5.084° | 0.017° |
| fine | 6.3780 GHz | 6.5551 GHz | 6.5550 GHz | 2.753° | 0.004° |

The port's effective aperture is 10.04 / 19.02 / 37.01 cells for a 9 / 18 / 36-cell
guide. That is issue #868's defect (the discrete eigenproblem solved on N+1 cells), seen
from the impedance side rather than the reference-plane side: the same `f_cutoff` feeds
`_compute_beta` and `_compute_mode_impedance`.

## 3. Suspect ladders, band-mean `|Γ|`

| band-mean `\|Γ\|` | coarse | mid | fine | successive ratios | pred/measured |
|---|---|---|---|---|---|
| measured `\|S11\|` | 0.09479 | 0.04327 | 0.02065 | 2.19, 2.09 | — |
| S1 cutoff → Z | 0.03636 | 0.02056 | 0.01096 | 1.77, 1.87 | 0.384 / 0.475 / 0.531 |
| S2 normal-axis H average | 0.01044 | 0.00258 | 0.00064 | 4.05, 4.01 | 0.110 / 0.060 / 0.031 |
| S3 aperture transverse pairing | 0.01436 | 0.00370 | 0.00094 | 3.88, 3.94 | 0.151 / 0.085 / 0.045 |
| product S1·S2·S3 | 0.06109 | 0.02683 | 0.01255 | 2.28, 2.14 | 0.645 / 0.620 / 0.607 |

As column power (`|Γ|²`, worst bin): measured 1.8253e-02 / 4.0817e-03 / 9.8341e-04;
product 6.864e-03 / 1.747e-03 / 4.490e-04; S1 alone 4.052e-03 / 1.360e-03 / 3.979e-04.

## 4. What that settles and what it does not

**Settled.** S1 is the only suspect with first-order-in-dx content. S2 and S3 fall ~4×
per halving, so their column-power contribution is fourth order and neither can carry an
excess that falls ~4× per halving. The dx-SCALING of the excess therefore belongs to the
cutoff/impedance channel — which is #868, not a separate defect.

**Not settled — and this is the load-bearing negative result.** The product reproduces the
scaling (2.28, 2.14 against 2.19, 2.09) but only 61–65 % of the magnitude at every rung,
and the remainder, measured minus product, is 0.0337 / 0.0164 / 0.0081 with successive
ratios 2.05 and 2.03 — itself first order in dx. A second-order remainder would be a
rounding story about the three suspects. A first-order one means a fourth first-order
channel exists in the extraction that none of the three names. The per-bin shape agrees:
the product falls monotonically across the band (0.0828 → 0.0521 at the coarse rung)
while the measured `|S11|` stays between 0.076 and 0.135 with a ripple whose period
matches a round trip to the near absorber, so the model does not carry the frequency
dependence either.

**Not established here**, stated so it is not mistaken for a finding: expressed as the
impedance factor the three modelled channels leave over, the remainder is 1.0702 / 1.0335 /
1.0163, i.e. 0.631 / 0.602 / 0.589 times `dx/a`
(`per_rung.*.residual.implied_extra_z_factor{,_minus_one_over_dx_over_a}`). That is the
order of a one-cell edge effect in a transverse aperture, but nothing in this run
instruments one. Naming it needs its own pre-declared comparison; per the rule fixed
before this ran, no fourth suspect is reached for here.

**Framing correction to #873.** The 550× thru gap against the flux lane overstates the
case. `extract_waveguide_s_matrix_flux` builds the diagonal from `|F_ref − F_dev|`, and on
an empty guide the reference run IS the device run, so its `|S11|` is identically 0 and
`|S21|` identically 1 — the thru number is a construction, matching the closure contract's
note that the empty-guide identity is vacuous (#395). The load-bearing flux comparison is
the slab: 1.759e-02 (`false`) against 1.09e-04 (`flux`).

## 5. No fix

Branch (iii) authorizes none, and the pre-declaration made a fix conditional on
identification. No gate, tolerance or golden moves; `rfx/` is untouched by this lane.
