# T3 — recompute (#931, branch `feat/931-t3-nu-runners-grid-subgrid`)

Machine rule (PI, 2026-09-07): the shared pod runs no solve longer than about
a minute and one small pytest at a time. Everything below goes to VESSL, one
run per case, preset `gpu-rtx4090`, `JAX_PLATFORMS=cpu` inside the job, source
read from this worktree — so the branch is committed before each submission.

No fixture in this group pins a solved NUMBER: the gates are lane parity,
self-comparison digests, loose ratios and one-sided thresholds. What the two
runs below produce is (a) a green/red verdict for the four directories
including the slow marks, and (b) replacements for three MEASURED values that
live in docstrings and are pre-#931. Nothing is hand-edited from either.

## Run 1 — `rfx-931-post-t3-pytest`

* yaml: `docs/design_notes/931_migration/t3_vessl_pytest.yaml`
* command inside the job:
  `python -m pytest -q -n 8 -m "" tests/unit/nonuniform tests/unit/runners tests/unit/grid tests/unit/subgrid`
  with `XLA_FLAGS=--xla_force_host_platform_device_count=2`
* outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-t3-pytest-<ts>/`
  (`pytest.log`, `junit.xml`, `reds.txt`, `pytest.rc`)
* expected runtime: ~35-60 min. The four directories are ~860 tests; the two
  `@pytest.mark.slow` WR-90 iris tests are two-run waveguide S-matrices at
  `num_periods=20` and dominate.
* what a red means: every migrated fixture asserts realized == drawn at BUILD
  time, so a red in one of those build-time gates is a realization defect, not
  a tolerance. Distinguish it from a red in a physics gate before touching
  anything.

## Run 2 — `rfx-931-post-t3-measure`

* yaml: `docs/design_notes/931_migration/t3_vessl_measure.yaml`; producer
  `docs/design_notes/931_migration/t3_remeasure.py`
* command inside the job:
  `python -u docs/design_notes/931_migration/t3_remeasure.py "$OUT/t3_remeasure.json"`
* outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-t3-measure-<ts>/t3_remeasure.json`
* expected runtime: ~40-90 min (dominated by the two iris S-matrices).
* which values it replaces, and why each moves:

  | docstring | pre-#931 value | why it moves |
  |---|---|---|
  | `test_nonuniform_pec_scatterer_limit.py` module docstring: uniform iris | `\|S11\|` ~ 0.6-2.1 | the waveguide S-matrix lane stopped folding `pec_mask` cells into `sigma = 1e10` and applies the realized PEC edges (commit 0184d64c); the fins are also drawn on the node line now |
  | same, NU (graded-dy) iris | `\|S11\|` ~ 1.4-1.6 | same |
  | `test_nu_wire_port_lane_parity.py` module docstring: three-load table, `S11(0.2 GHz)` | `-0.71429+0.00027j` (pec_plates), `-0.60000+0.00034j` (vacuum), "-0.600000 for all three" | the PEC plates realize BOTH drawn faces and short their own interior; before the contract a one-cell body realized one wall and left its normal E live, so the "plates" were films |
  | `test_run_progress_reporting.py::_msl_thru` prose: `Z0`, `beta` | not quoted numerically in the file | the trace moved from a one-cell PEC Box to a sheet on the substrate-top node plane, so it no longer owns cell k=4's material (the #702 backfill is deleted) |

  The gates in all three files are load-independent / ratio / digest gates and
  do not bind on these numbers, so a difference is evidence, not a failure.
  The wire-port row is the one to read carefully: the module's whole argument
  is that the passive reading does NOT move with the load, and the PEC arm is
  now a genuinely conducting body. If the reading starts moving with the load,
  the `#313/#318` normalization story in that file needs re-reading — say so,
  do not adjust the gate.

## Run ids

Submitted 2026-09-07 11:10 UTC from commit 8e223a34, preset `gpu-rtx4090`,
cluster `remilab-c0`, `JAX_PLATFORMS=cpu` inside the job.

| run | id | link |
|---|---|---|
| `rfx-931-post-t3-pytest` | 369367259208 | https://app.vessl.ai/remilab/runs/byungkwan/369367259208 |
| `rfx-931-post-t3-measure` | 369367259209 | https://app.vessl.ai/remilab/runs/byungkwan/369367259209 |
| `rfx-931-post-t3-pytest-r2` | 369367259214 | https://app.vessl.ai/remilab/runs/byungkwan/369367259214 |
| `rfx-931-post-t3-pytest-r3` | 369367259223 | https://app.vessl.ai/remilab/runs/byungkwan/369367259223 |

Run 369367259208 read commit 6b8f9fae and returned **865 passed, 4 failed,
4 xfailed in 28 min**. All four reds are accounted for:

* `test_distributed_nu_kernel.py` seam pair — the measured lane divergence,
  now `xfail(strict=True)` (see the findings section of the group note);
* `test_auto_config.py::test_auto_mesh_trigger_fires_thin_only_end_to_end` —
  a missing import of the shared helper, fixed in `eadb30b3`, the file's 27
  tests green;
* `test_runner_import_binding.py::test_coax_then_refplane_order_does_not_leak_fake_run`
  — pre-existing slow-lane brittleness, unrelated to #931 (its nested pytest
  passes; the assertion greps the whole stdout for "failed" and a warning
  says "one drive that failed to excite").

`rfx-931-post-t3-pytest-r2` (369367259214) re-ran the same lane on
`7b8d9921` and returned **866 passed, 1 failed, 6 xfailed in 32.5 min**
(`.../issue931-post-t3-pytest-20260907T120828Z/`). The single red is the
`test_runner_import_binding.py` slow-lane brittleness above, which does not
touch #931. The predicted verdict held; the four directories are green under
the contract.

`rfx-931-post-t3-pytest-r3` (369367259223, submitted 2026-09-07 14:19 UTC
from commit `4856b5c2`, same yaml with the name changed) is the verdict on
the FINAL tree — the docstring re-measurements, the `_nu_lane_shim` ->
`_wall_planes_m` rename and the merge of `feat/931-lattice-ownership`.
Expected: the same one unrelated red plus the two strict-xfail seam tests,
~35 min. Locally verified before submitting: the four gates that import the
renamed helper (WR-90 iris fins, NU progress-chunking block, in-plane
grading guards, wire-port PEC plates on both lanes) 8 passed in 58 s, and
`test_msl_thru_realizes_the_trace_where_it_is_drawn` 1 passed in 26 s.
Read `commit.txt` in the output directory before reading its verdict.

## Results — measure run 369367259209

Read commit `7b8d9921`, rc 0, four cases in 21 s of solve
(`.../issue931-post-t3-measure-20260907T123234Z/t3_remeasure.json`). The
values are folded into the docstrings in commits `8bcc2303` (iris, wire
port) and the one this paragraph ships in (MSL thru); none of
them is an assertion.

| case | measured | pre-#931 prose |
|---|---|---|
| WR-90 iris, uniform | `\|S11\|max` 2.170 | ~0.78-2.1 |
| WR-90 iris, graded-dy | `\|S11\|max` 1.807 | ~1.4-1.6 |
| wire port, 5 mm gap, vacuum / PEC plates (n_live 6) | -0.7142854 / -0.7142860 (-2.3e-05j) | -0.71429+0.00027j |
| wire port, 3 mm gap, vacuum / eps_r=10 (n_live 4) | -0.6000000 / -0.6000003 | -0.60000+0.00034j |
| MSL thru, `\|S21\|` 2-18 GHz | 0.99999 flat | not quoted |
| MSL thru, `beta/k0` | 0.872 on 8 of 12 points | not quoted |

The wire-port rows are the interesting ones: the real parts sit on the
module's own closed form `(1-n_live)/(1+n_live)` to seven digits and do not
move with the load, which is what the module claims, and they do not move
now that the PEC plates actually short their interior. Only the small
imaginary part changed.

The MSL row is an OPEN QUESTION, not a result of this migration — see the
group note, finding 3.

Both read this worktree
(`/root/workspace/byungkwan-workspace/research/rfx-931-T3-nu-runners-grid-subgrid`)
as it stands on disk, so a later commit on this branch is NOT in them — check
`commit.txt` in the run's output directory before reading a verdict.

---

## Phase 2b — ingest on the merged base (2026-09-07)

The base branch `feat/931-lattice-ownership` was merged in (fast-forward:
the merge agent had already taken this branch's tip at `53a5602f`, so the
worktree simply moved to the base head `770c4e6c`). Two things on the base
reach this group's directories.

### The seam xfails are gone — the fix landed, not the marker

`feat/931-core-distributed-seam` is merged into the base (`b096d464`, fix
`ac782d4f`). It deletes the two `xfail(strict=True)` markers in
`tests/unit/runners/test_distributed_nu_kernel.py` in the same commit that
makes them pass, and the one-cell fixtures stay one cell. Nothing to do at
ingest but confirm it: the xfail count in the four directories went from 6
to 4 between run 369367259223 and run 369367259276.

### R8 (half-open extent) put two closed-form port oracles one cell out

`6d66ac65` made a wire port's extent HALF-OPEN in edges. Nine tests in
`tests/unit/nonuniform` went red against it — measured, VESSL run
369367259276 (`rfx-931-post-t3-pytest-r4`, commit `770c4e6c`):
**10 failed, 861 passed, 4 xfailed in 35.7 min**, the tenth red being the
pre-existing `test_runner_import_binding.py` slow-lane brittleness.

Both failures are the test's side, and both are fixed in `9ca7d595`:

1. `test_nu_wire_port_lane_parity.py::_n_live` carried a SECOND COPY of the
   rasterization rule and kept the retired endpoint-inclusive `+ 1`. It now
   reads `rfx.sources.sources.wire_port_edge_span`, and a new build-time
   gate pins it against `sim._wire_port_cell_centers`. No gate constant
   moved — that file computes its gates from `_n_live`, not from literals.
2. `test_nu_port_sigma_dual_spacing.py`'s oracle-2 extents each lost a
   realized cell and tripped the fixture's own `assert n_live >= 2`. The
   extents were re-declared to realize the counts the oracle was built on
   (per component, since the port sits in the coarse run): ez 2D->5D,
   ez 6D->9D, ex 2D->3D, ey 2D->4D, all measured from the stamped array.
   The guard stays at `>= 2`; at `n_live = 1` the closed form is `S11 = 0`,
   which any wrong cell resistance satisfies, so lowering it would have
   left a green test that no longer discriminates.

### Run 3 — `rfx-931-post-t3-measure-r2` (369367259274)

Read commit `770c4e6c`, rc 0, output
`.../issue931-post-t3-measure-20260907T191140Z/t3_remeasure.json`. This is
the run that caught R8's effect on the docstring table: the solved `S11`
had moved to the closed form for the SMALLER count while the JSON's own
`n_live` column — which calls the stale helper — still said 6 / 4.

| case | 369367259209 (7b8d9921) | 369367259274 (770c4e6c) |
|---|---|---|
| WR-90 iris, uniform | `\|S11\|max` 2.1695750 | 2.1695750 (unchanged) |
| WR-90 iris, graded-dy | `\|S11\|max` 1.8067741 | 1.8067743 (unchanged) |
| wire port, 5 mm gap, vacuum / PEC plates | n_live 6, -0.7142854 / -0.7142860 | n_live 5, -0.6666666 / -0.6666666 |
| wire port, 3 mm gap, vacuum / eps_r=10 | n_live 4, -0.6000000 / -0.6000003 | n_live 3, -0.4999999 / -0.5000007 |
| MSL thru, `\|S21\|` 2-18 GHz | 0.99999 flat | 0.99999 flat (unchanged) |

The iris and MSL rows do not move: neither fixture has a wire port. Every
wire-port row still sits on `(1-n)/(1+n)` for its own count to seven digits
and still does not move with the load, which is the module's actual claim.

### Run 4 — the re-measure on the fixed tree (369367259310)

`rfx-931-post-t3-measure-r3` (369367259294, submitted 19:32 UTC) never
scheduled — no output directory after an hour on a congested cluster — and
was resubmitted unchanged as `rfx-931-post-t3-measure-r4`, **369367259310**,
which read commit `be40dddf`, rc 0, output
`.../issue931-post-t3-measure-20260907T204252Z/t3_remeasure.json`. That file
is committed as
`docs/design_notes/931_migration/t3_remeasure_369367259310.json`.

The three original cases are unchanged from 369367259274 to the digit; what
the fixed helper changed is the JSON's own `n_live` column, which now reads
5 / 3 — the count the solved `S11` had been sitting on all along.

The fourth case is the new one, `test_nu_port_sigma_dual_spacing`'s oracle-2
table on the re-declared extents:

| row | extent was -> now | n_live | expected | worst rel was -> now |
|---|---|---|---|---|
| ez | 2D -> 5D | 2 | -1/3 | 2.5e-05 -> 2.3e-05 |
| ez | 6D -> 9D | 3 | -1/2 | 6.9e-06 -> 6.1e-06 |
| ex | 2D -> 3D | 2 | -1/3 | 7.8e-05 -> 8.4e-05 |
| ey | 2D -> 4D | 2 | -1/3 | 1.9e-05 -> 1.9e-05 |

`n_live`, the expected column and the deviations all land where they were.
That is the check on the re-declaration: if the new extents had changed what
the fixture measures rather than only how much extent it takes to realize
the same port, these numbers would have moved.

### Run 5 — the verdict (369367259300)

`rfx-931-post-t3-pytest-r5`, read commit `be40dddf`:
**1 failed, 876 passed, 4 xfailed in 25.9 min**
(`.../issue931-post-t3-pytest-20260907T201554Z/`). The single red is
`test_runner_import_binding.py::test_coax_then_refplane_order_does_not_leak_fake_run`,
the pre-existing slow-lane brittleness of finding 2 — not #931, not this
group's to fix. Every one of the nine R8 reds is gone and the xfail count
is 4, i.e. the seam pair is not merely passing but deleted.

The pre-declared expectation for this run was written before it ran (the
paragraph this one replaces) and it held exactly.

### Local fast lane, `-n 4`, on commit `be40dddf`

Default marker expression (`not gpu and not slow and not slow_physics`),
`JAX_PLATFORMS=cpu`, one directory at a time on the shared pod:

| directory | result |
|---|---|
| `tests/unit/nonuniform` | **244 passed** (18.2 min) |
| `tests/unit/runners` | **256 passed** (5.1 min) |
| `tests/unit/grid` | **137 passed** (0.7 min) |
| `tests/unit/subgrid` | **166 passed, 2 xfailed** (8.7 min) |

803 passed, 2 xfailed, 0 failed. `runners`, `grid` and `subgrid` were run at
`770c4e6c`; the ingest commits touch only two files in
`tests/unit/nonuniform` plus docs, so those three counts stand. `nonuniform`
went 229 passed / 9 failed at `770c4e6c` to 244 passed / 0 failed after the
fix — the +6 is the new build-time gate (two lanes x three extents).

The one red the VESSL lane reports does not appear here because it is
`@pytest.mark.slow` and the fast lane deselects it.
