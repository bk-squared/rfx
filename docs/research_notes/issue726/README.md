# Issue #726: invalid comparison inputs and contradictory accuracy claims

Status: partial input/measurement repair based on main `0662c3b9`. No new RF
measurement has been completed. The core warning wording is corrected; result-metadata policy
is still pending; this checkpoint does not close #726.

The 2026-08-27 notch diagnostic moved the physical source and termination
along with the observation ladder. Its executable verdict ignored settling.
The 2026-08-28 short diagnostic used a finite open microstrip with all-face
CPML as an exact unit-reflection reference, without ruling out radiation or
other outgoing channels. Source motion, model error and observation error
were not separated. The short's non-settling records do not prove failure at
all possible durations, nor do they instrument an evanescent-field cause.

The baseline short script also compares an unsnapped last-probe coordinate
against the declared BACK of its short volume. On the current H_SUB/3 grid,
the actual near probe is exactly on the realized FRONT at 19.896666667 mm.
All three substrate-normal Ez edges there are PEC. Its old check instead
reports 76 um clearance and passes. Both arms' first probes violate the
existing source standoff (10 versus 15 cells). `original-short-build.json`
and its full log reproduce these facts without advancing any fields;
`audit_original_short.py` loads and hashes the exact baseline script through
git, with run/forward/S-matrix execution forbidden. The script is retained as
a historical geometry builder but refuses to run an accuracy experiment.

The replacement `scripts/diagnostics/msl_probe_clearance_bias.py` uses current
cv06b with its asserted realized sheet stack. It clones one fixed input and
changes only p1's observation offset. Both physical ports, loads, waveforms,
materials and boundary conditions stay fixed; p2's ladder is identical.
All input validation finishes before either numerical run. The control's
last probe is 8.89 mm before the actual stub; the near one is 63.5 um before
it. All voltage paths are live; both first planes clear the source standoff.
`build-v1/inputs.json` records the first build-only certificate and the exact
case/driver hashes that produced it (not a later driver's output).

The driver saves raw phasors and S with passivity projection disabled. A
missing/nonfinite or > -40 dB settling witness prevents a comparison; a
failed control skips the near run. It reports both arms at the SAME bin,
selected by control S21's minimum inside the predeclared 3--5 GHz band, and
checks the existing low-signal mask at that bin. It does not select separate
arm peaks or assign a winner using the old arbitrary 2 dB threshold. A
reported difference describes observation-offset sensitivity. Production S
is assembled at each first probe plane without transport to the feed. Its
fixed analytic reference impedance may differ from the realized line
impedance, allowing magnitude changes even for purely propagating fields. Neither
zero difference nor a large difference certifies full-wave accuracy or
identifies evanescence. The ideal quarter-wave circuit remains a model.

Validation: 14 build/reporting checks passed; scoped Ruff and whitespace checks
passed. These are geometry and reporting contracts, not RF validation.

Remaining work: complete the paired RF record, inspect raw waves and all
quality diagnostics, resolve the user decision on separate clearance status
versus extending `reliable`, and align the core messages without turning a
layout heuristic into an accuracy theorem. The old raw RF evidence and the
#953 frequency-gate policy are unchanged.

## Pinned GPU staging check

The first launch, VESSL 369367260507, failed before a field step because its
source archive omitted cv06b's relative `comparators/spectral_features.py`
import. The failure's full provider log, execution log, environment, source
hash and launch configuration are preserved in `gpu-369367260507/`; the
terminal run was deleted only after lossless backups and hashes were checked.

The replacement archive includes the complete crossval subtree. It was
unpacked in a separate local directory, imported from that directory with
`JAX_ENABLE_X64=0`, and built with run/forward/S-matrix execution forbidden.
Its realized geometry equals the reviewed build. The resulting certificate
is `complete-archive-build/inputs.json`. The source commit is
`130b9071e5e6aaa0efda0dd1d6a36fb7e8d519bd`; source archive SHA-256 is
`f78ad7c25ed326e56d7922f8fc6033938dfe50bf5d5a3c8586880e97f53b8323`.
This is a staging validation, not an RF result.

## Warning and reference-plane correction

The two-wave fit includes both travelling directions. The unchanged
lambda_g/4 layout proxy cannot predict a universal S error in dB or certify
modal purity when its warning is absent. Core preflight now describes that
limited scope and the full source/ladder/reflector space needed when its
offset interval is empty. The Z0/beta guards state only that those fitted
numbers do not enter S; they no longer certify the shared measured V/I.
Result and extraction docs identify the first-probe reference planes and
distinguish fitted Z0 from the analytic normalization impedance.

`math-and-dataflow.md` gives the exact algebra, synthetic intervention and
limitations. The focused suite passed 98 checks; after the final reference
metadata wording and three-frequency intervention, both affected tests
passed again. Logs are retained. Source review confirmed that thresholds,
S/fit/mask algorithms and public result fields are unchanged.

The existing low-signal mask's blanket whole-matrix-error wording remains a
known unresolved item, documented by the exact transmission-zero
counterexample. Its formula and the experiment's acceptance screen have not
been relaxed. The separate clearance-result policy still awaits the user.
The GPU experiment remains pinned to 130b9071; these subsequent production
changes affect prose/metadata only, not the fields or extracted numbers.
