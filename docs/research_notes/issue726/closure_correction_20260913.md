# Correction to the September 13 closure of #726

The closure reused a retired shorted-line experiment as if it were corrected
evidence. That attribution is withdrawn. The original text is preserved below;
the live issue is reopened for its unfinished MSL power work.

## What the evidence actually establishes

- The historical shorted-line arms changed the source/load; the near arm's last
  probe lies on the realized PEC short, and its finite-ground/CPML structure is
  not an exact single-mode |S11|=1 oracle. The old 150/300/600-period observations
  cannot establish a general failure to settle or a clearance-induced bias.
  The executable is retired and its build geometry is retained for audit.
- The later fixed-source cv06b comparison is a different, valid comparison of
  observation offsets. Both arms settled below -118 dB. At 3.77125 GHz its raw
  S11 readings were +0.026895 and +0.018065 dB. Near-arm beta estimates hit the
  scan rail in 51/51 bins; the existing low-signal check made the producer
  verdict `not_read`. Neither result is an accuracy PASS.
- The H/E collocation and positive-real-reference power-wave changes remain
  supported by their separate RF/AD records. They do not eliminate the
  equal-reference raw coherent power excess.
- The subsequent source-work and six-face flux analysis in
  [code_contract_audit.md](power_budget/code_contract_audit.md) distinguishes
  source-model work from the desired first-plane MSL V/I. A passive result for
  the former cannot substitute for the latter. Lateral incoming flux in those
  fixtures is not a general qualification of the production MSL port.

The remaining work is the user-authorized improvement of excitation and
observation under a common modal/power definition while preserving the existing
S reference plane and port AD. Any production replacement must pass both RF
qualification and AD/finite-difference checks. It is not marked completed by
turning it into a version-plan limitation.

## Evidence links

- Original invalid-geometry audit: https://github.com/bk-squared/rfx/issues/726#issuecomment-5640694651
- Fixed-source comparison: https://github.com/bk-squared/rfx/issues/726#issuecomment-5641347165
- Remaining equal-reference power: https://github.com/bk-squared/rfx/issues/726#issuecomment-5646369067
- Retired diagnostic: `scripts/diagnostics/msl_probe_clearance_shorted_line.py`
- Always-on retirement check: `tests/crossval/test_msl_probe_clearance_experiment.py`

## Original closure text, withdrawn as specified above

Source comment: https://github.com/bk-squared/rfx/issues/726#issuecomment-5652568519

> The contradiction in this issue is resolved and the issue can be closed.
> 
> The corrected shorted-line comparison used the same source, DUT and probe
> comb. A clean ladder settled below the existing -40 dB threshold and measured
> |S11| within 0.23 dB of the exact PEC-short value. The near-reflector ladder
> could not satisfy the settling witness at 150, 300 or 600 periods, even
> though its un-gated |S11| stayed within 1.86 dB of that exact value. Thus the
> old preflight sentence predicting a 5–10 dB |S11| bias was not reproduced.
> The safe common statement is that insufficient probe clearance can make a
> record unquotable through the settling/fit conditions; a longer run is not a
> general remedy.
> 
> The implementation now keeps the meanings separate: `reliable` retains its
> relative low-signal test, while `probe_clearance` reports `satisfied`,
> `insufficient` or `unavailable` from the realized conductor geometry. The
> H-to-E spatial interpolation and positive-real power-wave reference
> normalization were merged in PRs #986 and #987, with raw side currents and
> reference metadata retained for replay. Their AD paths remain intact.
> 
> The remaining equal-reference raw power excess is not claimed solved by these
> changes. Same-run flux budgets show that the two-port V/I observable has
> unrepresented lateral incoming power, and the mixed coax↔MSL version is now
> tracked as the explicitly experimental limitation in #838. No fixture-specific
> rescale, passivity clipping or weakened gate is appropriate. A calibrated
> MSL chain remains part of the v2.0 plan (#825). This closes the misleading
> warning/diagnostic contradiction while preserving the unresolved physics as
> an explicit support limitation.
