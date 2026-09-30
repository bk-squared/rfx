# #1402 stage 1 — local implementation facts

Base (read only): `b0984322326593cdc2e584af66784fca2cd27d62`, branch
`agent/distributed-lumped-port-s`, worktree
`/Users/byungkwankim/Documents/rfx-worktrees/dist-s11-impl`.

Read only: `rfx-archive/rfx/records/20260930-distributed-s11-stage1/REPORT.md`
and `rfx-archive/rfx/agent-memory/rfx-known-issues.md:1493–1550`.
The latter records distributed staging, exchange, slab-width admission, and
one-cell conducting CPML-face limitations. No ledger entry contradicts the
five owning-cell recording mechanism used here.

## Implementation (read only: final source inspection)

- `rfx/api/_execute.py:4964`: the uniform distributed run calls the existing
  S-matrix scan driver; default frequencies and extraction steps match the
  uniform runner. The main run's fields/probes remain its returned record.
- `rfx/api/_preflight.py:431`: wire/passive refusals retain the distributed
  feature messages. Graded ports retain the `Phase B distributed+NU` message.
  Specialized-family validation remains before this branch.
- `rfx/probes/sparam_driver.py:212`: one distributed scan per drive; all
  receive ports are sampled in that scan. No mutation of the registered
  ports' excite flags or public probes. Distributed scans assemble/stage their
  own materials; the driver retains no second whole-domain material assembly.
- `rfx/probes/sparam_driver.py:366`: five global cells per receive port.
  Backward H neighbours keep their own global index. The runner's existing
  owner mapping (`rfx/runners/distributed_v2.py:960`) and point-probe sampling
  (`rfx/runners/_distributed_common.py:1070`) record them. No exchange added.
- `rfx/probes/sparam_driver.py:393`: host complex128 DFT reduction in blocks of
  256 phase stamps. Frequencies first cast to float32, as in the reference
  scan. Shared phase, voltage and Ampere-loop helpers feed the existing
  `decompose_lumped_s_matrix` call at `rfx/probes/sparam_driver.py:342` and
  voltage-negating `PortVIReplayBundle` at `rfx/probes/sparam_driver.py:351`.
  No second extractor or replay-bundle construction was introduced.
- `rfx/probes/sparam_driver.py:427`: zero reference placeholders select the
  current decomposition convention. They are not measured pre-injection V;
  the current decomposer consumes only their presence.
- `rfx/runners/distributed_v2.py:1409`, `:1436`, `:1514`, `:1527`, `:1544`:
  the new per-drive scans use the reference driver's post-boundary injection
  order and declared electric walls. The existing main-run order remains.
  The initial nine-node PMC line returned zero distributed fields and
  max complex S error 0.999175389183 before this per-drive correction.
- `tests/contracts/path_disposition.py:573`, `:803`: lumped ports carry S;
  conformal S requests retain the conformal refusal.

Factoring from base `b0984322` (read only: git diff):

- `rfx/simulation.py:2459–2460` phase arithmetic moved into
  `rfx/core/dft_utils.py:112::port_dft_phase`; the scan and host replay both
  call it. Float32 step stamps, complex64 phase and dt weighting retained.
- `rfx/probes/probes.py:228–244` loop component/axis choices and arithmetic
  moved into `_ampere_loop_components` / `_ampere_loop_values` at `:234`,
  `:244`; live-field `_ampere_loop` and recording reconstruction call them.
- The voltage expression at base `rfx/probes/probes.py:126` and production
  lumped sampling at base `rfx/simulation.py:2458`, `:2585` call the shared
  `_port_voltage_value` at `rfx/probes/probes.py:249`.
- `half_step_current_phase` remains the existing helper. The original
  `sparam_driver.py:327–346` decomposition/replay block is retained at
  `:340–359`; only the per-drive raw-accumulator provider is selected.

## Numerical checks (ran)

`tests/unit/runners/test_distributed_lumped_port_s.py` compares every complex
entry and every bin, with exact frequency-array equality. One/two ports,
interior/seam placement, Ex/Ey/Ez, default requests, explicit True, explicit
frequencies, explicit extraction steps, and opt-out are covered. A separate
assertion records exactly two drive calls `(0, 10 probes, 320 steps)` and
`(1, 10 probes, 320 steps)` for the full 2×2 matrix and compares replay V/I.

Vacuum fixture: domain 16×6×6 mm, dx 1 mm, two CPML layers, grid 21×11×11.
On two devices the seam is global x=11; the backward H sample is x=10.
Main record: 256 steps. Explicit S record: 320 steps, bins
1/2.5/5/7.5/10 GHz. Default bins: 50 values from 1 to 10 GHz.

Matched fixture: parallel-plate TEM channel, domain 20×1×1 mm, dx 1 mm,
x CPML (four layers), y PMC, z PEC, two 188.365156834-ohm single-cell Ez
ports separated by two cells. On two devices the first port is at global
x=15 (seam), the other at x=13. Minimum |S11|: -11.0305843 dB.
The corrected nine-node candidate at its original 256 steps and default bins
has max complex S error 9.25312918202e-7 and min S11 -27.5647316 dB.
The five-node closed matched line uses 376.730313668-ohm ports and reaches
-67.3232269 dB. These are finite-record values, not convergence claims.

Measured maximum |delta S| over unmutated 2/3/4-device cases:
2.20425323864e-6. Gate: absolute 9e-6, relative 0, 4.08× the measured maximum.
The factor allows float32 streaming/reduction and compiler rounding; no GPU
or multi-host numerical measurement was made here. The 3/4-device cases run
in separate local processes with four virtual CPU devices, never
`jax.distributed`.

Mutations (ran, expected parity assertion fails, then monkeypatch restored):

| Mutation | Max absolute delta S | Result |
|---|---:|---|
| (a) Replace distributed S with zeros | 0.721627006337 | parity RED |
| (b) Keep local H sample indices but force their owner to the port's slab | 0.724454580116 | seam parity RED |
| (c) Replace replay H half-step factor with one | 0.0325529175502 | parity RED |

The gate is not fitted to a mutated result. Each mutation test requires the
specific S-parity assertion to fail.

## Memory and timing (ran)

Recording output added per device, per drive:
`n_steps × 5 × 4 B × n_ports` (20 B/step/port/device), replicated after the
existing post-scan owner reduction. Scans are sequential. Host phasors and
the main run's returned state are outside this recording-output count.

`measure_step_time.py` ran on two virtual CPU devices, 320 steps, two ports,
21×11×11 vacuum CPML grid. The same drive-0 scan was compiled with zero and
ten recording probes, warmed, then timed for nine repetitions in alternating
order. Each repetition synchronized the full compiled return. Compilation,
setup, final full-domain field gathering and host DFT are outside the timer.
No other task-owned pytest process was running during this measurement.

| Recording | Median microseconds/step | Min–max | Argument bytes/device | Output bytes/device | Temporary bytes/device |
|---|---:|---:|---:|---:|---:|
| zero probes | 56.530989 | 55.728774–66.672915 | 90,576 | 64,168 | 121,608 |
| ten probes | 62.494012 | 58.930079–71.861717 | 90,576 | 76,968 | 127,536 |

Measured median change: +5.963023 microseconds/step (+10.548237%). Ranges
overlap. Output difference: 12,800 B/device = 320×5×4×2. Compiler-reported
temporary difference: +5,928 B/device; argument difference: zero. These are
compiled executable accounting and CPU wall times, not allocator peak RSS
or GPU/multi-host measurements. Raw repetitions are in `benchmark.json`.
Python 3.11.2, JAX 0.10.2, NumPy 2.4.6, macOS 26.5.2 arm64.

Command (with the prefix below):
`XLA_FLAGS=--xla_force_host_platform_device_count=2 ... tests/fixtures/distributed_lumped_port_s/measure_step_time.py`.

## Commands and test counts (ran)

Scratch directory created with `mkdir -p .stage1/tmp` inside this worktree.
All Python commands used this interpreter and environment prefix:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/.stage1/tmp PYTHONPATH=$PWD MPLCONFIGDIR=/tmp /Users/byungkwankim/Documents/rfx/.venv/bin/python
```

The initial search was `rg -l 'compute_s_params' tests/ | xargs rg -l 'devices'`.
A separate search covered `1405`, `S-parameter.*distribut`, and
`distributed.*S-parameter`; the old refusal string was also searched.
Every matching test file and every modified test file was run below.
`tests/contracts/boundary_baseline.py`, `tests/contracts/boundary_fields.py`,
and `tests/contracts/path_disposition.py` were also passed to pytest with the
shared-helper files; these support modules define no collected test cases.
`tests/fixtures/lumped_wire_chain_battery/fixture.json` is matching fixture
metadata (read only), not an executable test file.

Each file below: `-m pytest FILE -q`, default repository marker exclusions.

| File | Counts |
|---|---|
| `tests/interop/test_interop_design_document.py` | 224 passed, 1 skipped, 72 warnings in 4.73s |
| `tests/locks/test_runner_split_bit_identity.py` | 79 passed, 13 skipped, 1 warning in 1.48s |
| `tests/unit/api/test_api.py` | 48 passed, 1 skipped, 118 warnings in 33.46s |
| `tests/unit/boundaries/test_boundary_pmc_composition.py` | 3 passed, 5 warnings in 5.14s |
| `tests/unit/boundaries/test_cpml_material_aware.py` | 11 passed, 42 warnings in 34.42s |
| `tests/unit/farfield/test_current_moment_monitor.py` | 105 passed, 12 deselected, 36 warnings in 102.04s (0:01:42) |
| `tests/unit/nonuniform/test_refinement_refused_on_graded_mesh.py` | 27 passed in 12.58s |
| `tests/unit/runners/test_distributed.py` | 46 passed, 40 warnings in 31.00s |
| `tests/unit/runners/test_distributed_admission_refusals.py` | 32 passed, 11 warnings in 7.28s |
| `tests/unit/runners/test_distributed_minimal_exchange.py` | 28 passed, 23 warnings in 72.11s (0:01:12) |
| `tests/unit/runners/test_distributed_run_result_finishing.py` | 5 passed, 7 warnings in 5.94s |
| `tests/unit/runners/test_path_disposition_cells.py` | 887 passed, 10 xfailed, 8 warnings in 166.83s (0:02:46) |
| `tests/unit/runners/test_silent_drop_warnings.py` | 78 passed, 10 warnings in 33.91s |
| `tests/unit/runners/test_silent_routes.py` | 38 passed, 14 warnings in 28.48s |
| `tests/unit/sparams/test_ringdown_run.py` | 39 passed, 82 warnings in 60.62s (0:01:00) |

Fifteen-file totals: {'passed': 1650, 'skipped': 15, 'xfailed': 10, 'deselected': 12}.

New file: `RFX_LOCAL_DISTRIBUTED=1 ... -m pytest tests/unit/runners/test_distributed_lumped_port_s.py -q -s`: 30 passed, 42 warnings, 89.80 s. The 3/4-device subprocesses inherit the Python prefix and set `XLA_FLAGS=--xla_force_host_platform_device_count=4`.

Shared helpers: 47 passed, 16 warnings, 403.36 s. Command (with the prefix):

```sh
... -m pytest tests/unit/ports/test_port_current_boundary_convention.py tests/unit/sparams/test_sparam_driver_matches_eager.py tests/unit/sparams/test_sparam_driver_dump_parity.py tests/unit/sparams/test_port_dump_replay.py tests/unit/ports/test_lumped_two_port_matched_line.py tests/contracts/path_disposition.py tests/contracts/boundary_baseline.py tests/contracts/boundary_fields.py -q
```

Final verification totals: 1,727 passed, 15 skipped, 10 xfailed, 12 deselected.
The skipped and deselected cases were not run. The default pytest exclusions
are `not gpu and not slow and not slow_physics and not docs_consistency`.
The complete numerical output is retained in `numerical-results.txt`.

Development runs (ran, prior to the final verification): two collection
errors from using pytest's reserved parameter name `request`; one immutable
Result assignment error; one initial closed-line parity failure. The final
code uses `mode`, `Result._replace`, and per-drive boundary/injection parity.

`git diff --check`: ran. `scripts/ci/local.sh`: not run.
No push, PR, GitHub comment, remote operation, new repository, alternate
object store, or multi-process JAX initialization was performed.

Conclusion: (leader fills)
