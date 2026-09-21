# Run summaries for `msl_phase_referee_dx_ladder.py`

The three `summary_*.json` files are the script's own summary output, copied
unchanged from VESSL CPU runs of 2026-09-20 made at the commit that introduced
the script (before it was rebased for merging; the script is unchanged by the
rebase). Run ids recorded by the submitter: dx50 369367262240, dx25
369367262242; the box-doubled run's id was not recorded. A fourth rung
(12.5 um) did not finish and has no summary.

`gated` covers 3.0-4.5 GHz. `production` is the extractor's own `beta`
as it was on that day; the scan-node bias of that estimate (up to 0.17 %) was
removed afterwards, so read `refit_float64_lsq`, an independent float64
least-squares refit of the same probe phasors. `signed_beta_dev_frac` is
`beta / beta_HJ - 1` against `hammerstad_jensen_z0_eps_eff` for the board the
grid realizes (W = 600 um, 250 um of RO4350B under the strip).
