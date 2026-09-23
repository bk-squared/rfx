# B0 measurement report

The 24 mm PEC cube, excited by an Ez pulse, has analytic TM110 = 8.832720 GHz; `run()` measured 8.830532 GHz at dx = 1 mm and 8.832173 GHz at dx = 0.5 mm. The PMC-sided parallel-plate line has closed-form |S11| = 1/3, 0, 1/3 for R/Zc = 1/2, 1, 2; at x = 1 mm the recorded wire-port magnitudes at 1 GHz are 1.00000763 for all three loads through both entry points. In the eight-layer CPML box, final probe Ez was −8.74824536e−7 V/m through `run()` and −8.70895860e−7 V/m through `forward()`; the backing-PEC-removed arm equals the forward probe record sample for sample.

**Coverage:** inventory and all 96 requested matrix combinations recorded; W1, W3, W4 measured; **W2 NOT MEASURED** because the referenced issue does not specify its cavity geometry and face assignment.

Source: `798ec64e5cda057318664bc431e528819f9e371a`. All scripts, output and the source archive are under `B0/`. No repository file edits, pushes, PRs or GitHub comments. The detached worktree has been removed.

## A1–A5

| Item | Status | Recorded evidence |
|---|---|---|
| A1 | CONFIRMED | All-CPML `run`: E zeroing plus seven nonzero-sigma layers per face; `forward`: seven nonzero-sigma layers, no E-zeroing call. `matrix/cpml__run_cpu.json`, `matrix/cpml__forward_cpu.json`; full matrix in `MATRIX.md`. |
| A2 | CONFIRMED for zero-sample positions | Tangential H zeroed at 0 / N−2; physical samples +dx/2 / −dx/2 from declared low/high node planes. `operator_measurement.json`. W2 resonance comparison remains NOT MEASURED. |
| A3 | CONFIRMED in accepted waveguide cases | y/z tangential-E zeroing recorded for CPML, PMC and PEC declarations in run, forward, sweep, NU and distributed cases; PMC cases also record H zeroing in several lanes. `MATRIX.md`, waveguide tables. |
| A4 | CONTRADICTED across entry points | Uniform run/forward/sweep: transverse wrap. Constant-dz NU: CPML updates on xyz, seven nonzero-sigma layers on y/z, no wrap recorded. Both preflight logs say all checks passed. `matrix/tfsf__run_cpu.json`, `matrix/tfsf__nonuniform_nu_measured.json` and adjacent `.log` files. |
| A5 | CONTRADICTED as a universal statement | W4 all-CPML box: run has backing PEC and forward does not. Waveguide-port run: x absorbers active, `pec_axes="yz"`, no x backing PEC. `matrix/waveguide-pec__run_cpu.json`; W4 comparisons in `WITNESS.json`. |

Conclusion: leader fills.

## Deliverables

- `INVENTORY.md`, `inventory.csv`: 1518 sites in 79 files; DECLARE 196, DERIVE 252, APPLY 92, DECIDE 452, PASS 526. Category × lane counts, source lines, scopes and applied operators included.
- `MATRIX.md`, `MATRIX.json`: 12 boundary cases × 8 entry points; 66 MEASURED, 30 REFUSED with full exception text; 45 additional GPU observations. Face indices, nonzero-sigma counts, wrap/Bloch arguments, all measured disagreements and all declaration mismatches included.
- Matrix lists 844 pairwise face disagreements and 356 declaration-versus-observation cells, including supplemental GPU records.
- `WITNESS.md`, `WITNESS.json`, `witness/*.json`, `witness/*.npz`: W1 eight arms, W3 twenty-four arms, W4 three arms; realized nodes/pads, verbatim preflight logs, per-step energy arrays and run ids. W1 uses repository Harminv on the ring-down record.
- W2: `witness/W2_NOT_MEASURED.json`; issue body/comments saved as `issue1164.json`. The missing setup was requested asynchronously; no setup was supplied. No cavity was substituted.
- W3 reference fixture is absent on measured main. It was retrieved from PR #1162 at `89326838939bf9573b11352ad832688f5548d4f6` into `reference_known_load_line.py`; only its geometry/port/load setup was used with pinned main solver code.
- W4 end/post-source peak energy is about −16.53 dB, above the workspace −40 dB criterion; recorded workspace label: truncation-suspect. Final run−forward ΔEz = −3.92867605e−9 V/m; maximum absolute probe difference = 3.02679837e−8 V/m.

## FACT counter-record

- The brief says the face-level PMC routine runs as well. For `run()` on GPU with x PMC / y,z PEC, the trace contains `precompute_coeffs(pec_axes="xyz")` and no `apply_pmc_faces` call. CPU records the PMC call. Evidence: `matrix/pmc-pec__run.json`, `matrix/pmc-pec__run_cpu.json`; both appear in `MATRIX.md`. No frequency attribution is assigned.
- The other cited FACT sites are in the inventory.

Conclusion: leader fills.

## Execution and verification

- `instrumentation_verification.json`: original versus observed probe arrays have maximum difference 0 through run, forward and sweep; energy instrumentation also gives maximum difference 0. Source export: 161 Python files, zero hash mismatches.
- `verification.json`: all 96 matrix keys, 35 physical arms, 1,518 source-line references, twelve run-id/log pairs checked; zero recorder errors.
- `COMMANDS.md` contains invocations and last lines, driver corrections and all provider-log paths. Original observer metadata errors are retained but excluded from physical refusal counts.

| Command | Last line / result |
|---|---|
| `git ... fetch origin` | Exit 0; no output |
| `git ... worktree add --detach ... origin/main` | `HEAD is now at 798ec64e ... (#1200)` |
| `inventory.py <worktree>` | `{"sites": 1518, "files": 79}` |
| `reduce_matrix.py` | `{"status": {"MEASURED": 66, "REFUSED": 30}, "entry_disagreements": 844, "declared_disagreements": 356}` |
| `operator_measurement.py` | `Measured 12 face/operator combinations` |
| `profile_measurement.py` | `UPML/ADI six-face sigma profiles measured` |
| `verify_artifacts.py` | Exit 0; last JSON line retained in `verification.log` |
| `git ... status`, `diff --exit-code`, ignored-file listing | Exit 0; all empty |
| `git ... worktree remove .../rfx-wt-bnd-b0` | Exit 0; no output |

## VESSL run ids

| Lane | Run id | State |
|---|---|---|
| W1 | 369367263562 | completed; provider log saved |
| W3 | 369367263563 | completed; provider log saved |
| W4 | 369367263564 | completed; provider log saved |
| matrix-adi | 369367263561 | completed; provider log saved |
| matrix-distributed | 369367263560 | completed; provider log saved |
| matrix-forward | 369367263557 | completed; provider log saved |
| matrix-nonuniform | 369367263558 | completed; provider log saved |
| matrix-run | 369367263554 | completed; provider log saved |
| matrix-subgridded | 369367263559 | completed; provider log saved |
| matrix-sweep | 369367263556 | completed; provider log saved |
| matrix-wire-fast | 369367263555 | completed; provider log saved |
| nu-observation | 369367263567 | completed; provider log saved |

Each job was submitted once. No runs were deleted; the cited reference records are retained.

Conclusion: leader fills.
