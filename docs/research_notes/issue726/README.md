# Issue #726: invalid comparison inputs and contradictory accuracy claims

Status: partial input/measurement repair based on main `0662c3b9`. No new RF
measurement has been completed. The core warning and result-metadata policy
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
reported difference describes observation-offset sensitivity, with analytic
reference-plane transport error still among the possible causes. Neither
zero difference nor a large difference certifies full-wave accuracy or
identifies evanescence. The ideal quarter-wave circuit remains a model.

Validation: 14 build/reporting checks passed; scoped Ruff and whitespace checks
passed. These are geometry and reporting contracts, not RF validation.

Remaining work: complete the paired RF record, inspect raw waves and all
quality diagnostics, resolve the user decision on separate clearance status
versus extending `reliable`, and align the core messages without turning a
layout heuristic into an accuracy theorem. The old raw RF evidence and the
#953 frequency-gate policy are unchanged.
