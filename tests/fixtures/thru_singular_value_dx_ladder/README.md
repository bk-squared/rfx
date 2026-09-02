# thru_singular_value_dx_ladder — the three rung records and their adjudication

The wire-THRU singular-value dx ladder: the 2-port wire THRU fixture of
`tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py::_build_thru` run at dx, dx/2 and
dx/4 (0.5 / 0.25 / 0.125 mm) with the CPML physical thickness (4 mm) and the physical
run time (4000 steps at dx) held. Pre-declared before any rung ran in
`docs/design_notes/thru_singular_value_dx_ladder_predeclaration.md` (sections 1–8);
adjudicated in section 9 of the same note. Producer:
`scripts/diagnostics/thru_singular_value_dx_ladder.py` (one rung per call); the job that
produced these files is `scripts/vessl_thru_singular_value_dx_ladder.yaml`.

## The run

| item | value |
|---|---|
| VESSL run | 369367257803 (`rfx-thru-sv-dx-ladder`, remilab-c0 gpu-rtx4090, image nvcr.io/nvidia/jax:24.10-py3), all rungs rc=0 |
| rfx commit | 088281899727fcc644814f0ae9451b6b89a26af8 (SHA-pinned fetch; PR #858's head, on main as 30fad4bb) |
| run dir (originals) | `claude-workspace/rfx/runs/thru_sv_dx_ladder/20260902T101202Z-08828189/` on the personal-workspaces NFS mount — `rung_dx_over_{1,2,4}.{json,log,rc}` |
| job log backup | `docs/vessl-logs/rfx-thru-sv-dx-ladder_369367257803_completed.log` in the primary checkout (gitignored) |
| stack | jax 0.4.33.dev20241023+e3c6d6430, gpu (cuda:0), `JAX_ENABLE_X64=0`, default field dtype, python 3.10.12 |
| wall time (total) | 6.6 s / 12.2 s / 160.2 s |

## Files

| file | what it is | sha256 |
|---|---|---|
| `rung_dx_over_1.json` | dx = 0.5 mm rung, byte-identical copy of the run's file | `efcb60687b591cd0fa06e3e4a10bb84d281d77fb0615c74e3b7091d85d6e8854` |
| `rung_dx_over_2.json` | dx/2 = 0.25 mm rung, byte-identical copy | `b21288d45d084599d0631ce7701fdfc658b046c1472c7e13de94ac607b2fa1be` |
| `rung_dx_over_4.json` | dx/4 = 0.125 mm rung, byte-identical copy | `1c3cdb12f0a47a06495de60c455d40a9c6ed7a1c415f19908bdced5d7d680d2d` |
| `verdict.json` | the adjudication against the pre-declared outcome table, computed from the three rung files (harvest round, 2026-09-02) | — |

The rung JSON schema is the one the producer writes (see its module docstring): `rung`
(dx, CPML layers, step count, dt, physical time), `fixture`, `preflight` (codes +
`messages_verbatim`), `warnings_verbatim`, `rasterization` (finite-PEC cells, wire-port
cells / live cells), `freqs_hz`, `s_matrix` (`re`/`im`, `S[i][j][k]`, i = receive,
j = drive), `abs_s`, `singular_values` (`max_per_bin`, `min_per_bin`, `sv_max`,
`excess_3ghz`, `monotone_decreasing_in_f`, `delta_vs_battery_sv_max`), `column_power`,
`reciprocity_abs`, `settling` (per drive, MSL-lane definition), `wall_time_s`, `provenance`.

## Headline (verdict C — non-closing; details in the note's section 9)

| rung | sv_max at 3 GHz | e = sv_max − 1 | settling_db per drive |
|---|---|---|---|
| dx | 1.0032274714899068 | +3.2274714899e-3 | −138.3 / −141.8 |
| dx/2 | 1.0003216974938964 | +3.2169749390e-4 | −134.7 / −134.8 |
| dx/4 | 0.9991541764781098 | −8.4582352189e-4 | −129.2 / −126.2 |

First halving: e falls 10.03× with the sign held. Second halving: the sign changes
(|e4| = 8.46e-4, above the pre-declared 1e-5 floor). Outcome A (both pairs ≥ 2× with one
sign) and outcome B (< 20 % change) are both false; outcome C ("a sign flip, a
non-monotone e") holds. The gate `_THRU_MAX_SINGULAR_VALUE = 1.01` and every measured
number are untouched.

## Replay gate

`tests/unit/sparams/test_thru_singular_value_dx_ladder_replay.py` (fast lane, no FDTD) re-derives the
per-bin singular values from the stored S matrices, re-applies the outcome table to the
stored excesses, checks the five validity gates from the stored witnesses, and compares
everything with `verdict.json` and with the headline numbers above. A silent edit to any
of these files, or to the note's verdict, goes red there.

## Re-capture

Not a re-capture target: the pre-declaration allows one run (R2, RF/EM threshold), and
that run is this one. A new ladder is a new pre-declaration with its own fixture
directory. The producer command, for the record:
`python scripts/diagnostics/thru_singular_value_dx_ladder.py --dx-divisor {1|2|4} --output <file>`
under `JAX_ENABLE_X64=0`; dx/4 on the GPU lane only.
