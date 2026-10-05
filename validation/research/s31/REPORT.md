S3-1 grouped source injection is implemented on `perf/s3-1-grouped-source-injection`.
Timing remains incomplete: the single authorized VESSL job failed during environment bootstrap, before either board ran. A replacement submission is awaiting user approval.

`Drives` in `rfx/core/drives.py` holds per-component nodes, coefficients, waveform indices, waveform table, drive owners, and optional per-waveform `source_end_step`. Existing sources retain coefficient 1 and their precomputed waveform columns. `inject_drives` uses one scatter-add per populated component. It makes no unique-index promise, so coincident sources accumulate. Electric and magnetic injection retain their original ordering relative to the Yee updates and measurements. Decay runs construct drives after their existing waveform padding/truncation.

| Path | Disposition | Evidence / scope |
|---|---|---|
| Uniform `simulation.py`, fixed-length and decay | fixed | Compiled-step scatter contract; point/wire/MSL fields and probes; plane DFT and wire/MSL S; waveform gradients; unequal-length decay waveforms |
| Uniform magnetic `mag_src_meta` | fixed | Same helper after H update; production equivalence includes overlapping H sources |
| Graded `nonuniform.py` | fixed | Compiled-step contract; point/wire/MSL equivalence; plane/port DFT, S and waveform gradients |
| Subgrid fine and coarse-shadow `jit_runner.py` | fixed | Four-step fine/coarse reference comparison, including overlapping nodes and multiple components |
| `vmap_sweep.py` field and current sources | fixed | Both source lists combined into one table; existing PEC current-source sweep regression passes |
| Distributed uniform `distributed_v2.py` | fixed | Shared slab-local helper; two host-device production contract and point-source field/probe equivalence |
| Distributed graded `distributed_nu.py` | fixed | Same shared helper; two host-device production contract and point-source field/probe equivalence |
| ADI `adi.py` | out of scope | Loops unchanged, as directed |

These dispositions concern existing source-list injection. Existing feature admission/refusal rules are unchanged. This does not claim a new carrier for magnetic, modal, or port features on lanes that previously refused them. S0's full future matrix is not claimed: the targeted numerical witnesses are listed above.

The contract lowers and compiles the actual production scan body, then counts full-field scatter operations in that step's StableHLO. StableHLO is used because CPU optimization expands scatter operations into loops. Counts include boundary scatters as well as source scatters.

| Compiled production step | 1 source | 64 sources |
|---|---:|---:|
| Uniform | 14 | 14 |
| Graded | 14 | 14 |
| Distributed uniform, 2 CPU devices | 20 | 20 |
| Distributed graded, 2 CPU devices | 18 | 18 |

Validation, JAX 0.10.2, serial on Mac:

- `tests/contracts/test_grouped_drives.py`: **21 passed**, 48 existing short-record/preflight/dtype warnings, 28.82 s. Per-cell test oracle; fields/probes within 9 float32 ULP of peak; DFT/S and waveform gradients within 1e-4 of peak. The short S records are equivalence witnesses, not settled physical S-parameter certificates.
- `tests/unit/runners/test_vmap_sweep.py::test_explicit_current_sweep_is_bit_identical_to_run[pec]`: **1 passed**, 3.52 s. Explicit `-o addopts=''` selects this small existing test despite its module marker. The initial command without that override selected no tests and is not counted as a pass.
- Mutation (a), `S31_CHECKS_OFF=1`: **red**, negative control fails because the scatter-growth checker no longer raises.
- Mutation (b), restoring the uniform per-cell loop with `inject_drives` retained: **red**, scatter counts become **15 versus 78**. The mutation was restored in `finally`; no mutation remains.
- New files pass Ruff. Modified existing files have the same 17 pre-existing unused-import diagnostics as the base; no new lint diagnostics. `git diff --check` passes.
- The YAML parses; its extracted `run` command and `job.sh` both pass `bash -n`.

Timing method prepared: warmed 200- and 1200-step calls, three sequential pairs, median of `(t1200 - t200) / 1000` in ms/step, synchronizing every result array. Source processes are separate for each board/revision. A3-d G-wire uses the frozen V6 constants and exact 0.25-mm cell-count gate. MSL uses the original graded arm-A model and parameters. The requested acceleration checkout lacked `bench/reference/throughput/`; the files were read from `~/rfx-research/rfx-benchmark/bench/reference/throughput/` and copied with hashes into the allowed NFS run folder. Both boards keep `compute_s_params=False` for stepping timings and accept declared geometry (`snap='declared'`).

| Board | Before marginal ms/step | After marginal ms/step | Status |
|---|---:|---:|---|
| MSL, graded arm A | unavailable | unavailable | Bootstrap failed before measurement |
| A3-d G-wire, 0.25 mm | unavailable | unavailable | Bootstrap failed before measurement |

VESSL [369367267680](https://app.vessl.ai/remilab/runs/byungkwan/369367267680), `gpu-a6000-1-hi-mem`, failed because `nvcr.io/nvidia/jax:24.10-py3` lacks `ensurepip`. No timings were produced. The complete 50-line provider log was harvested using `harvest.sh 369367267680 s31 keep`; only the copied harvest script's output directory was changed to `/private/tmp/s31/harvest` to obey the write restriction. The run remains kept. The bootstrap fix installs `uv` with `pip --target` under `/private/tmp/s31`, without `ensurepip`; no second job has been submitted.

The attempted job's archived before ref is `9ed83eb804874feec1f05f0cee75992c05da2182` (`origin/main` at start); its archived after ref is `e5114ec2`. Neither archive reached benchmark execution. A replacement would archive the final runtime commit `6d57002a` or its identical runtime tree at final HEAD.

Files changed, relative to the starting `origin/main` (added/deleted lines):

| File | Line changes |
|---|---:|
| `rfx/core/drives.py` | +48 / −0 |
| `rfx/nonuniform.py` | +4 / −4 |
| `rfx/runners/_distributed_common.py` | +14 / −17 |
| `rfx/runners/distributed_nu.py` | +4 / −1 |
| `rfx/runners/distributed_v2.py` | +4 / −1 |
| `rfx/simulation.py` | +32 / −25 |
| `rfx/subgridding/jit_runner.py` | +19 / −46 |
| `rfx/vmap_sweep.py` | +6 / −12 |
| `tests/contracts/test_grouped_drives.py` | +283 / −0 |
| `validation/research/s31/job.sh` | +28 / −0 |
| `validation/research/s31/job.yaml` | +18 / −0 |
| `validation/research/s31/measure_sources.py` | +68 / −0 |
| `validation/research/s31/evidence/contract-tests.log` | +232 / −0 |
| `validation/research/s31/evidence/manifest.json` | +33 / −0 |
| `validation/research/s31/evidence/mutation-checks-off.log` | +13 / −0 |
| `validation/research/s31/evidence/mutation-loop-restored.log` | +47 / −0 |
| `validation/research/s31/evidence/vessl-369367267680.log` | +53 / −0 |
| `validation/research/s31/evidence/vmap-test.log` | +11 / −0 |
| `validation/research/s31/REPORT.md` | +83 / −0 |

The evidence directory additionally contains the full contract-test log, the vmap log, both mutation logs, the failed-job provider log and its staged-input hash manifest. This report and the evidence files are validation artifacts, not plan/declaration changes.

Commits:

- `e5114ec2`: grouped implementation, initial contract/equivalence tests and timing harness.
- `36d63053`: VESSL bootstrap fix (not submitted).
- `6d57002a`: reuse normalized waveform tables; decay-padding and magnetic coverage.
- The following evidence-only commit records this report and logs; runtime code is unchanged.

No plan or declaration files were edited or found to require changes. No push, PR, remote Git operations or issue comments. All local test/submission/harvest processes started for this task have exited; the VESSL workload is failed and stopped.

Interpretation: lead fills.
