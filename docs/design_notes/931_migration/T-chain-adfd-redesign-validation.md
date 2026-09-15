# 2026-09-08: chain AD/FD redesign implementation validation

Branch `feat/931-chain-adfd`, parent `0815260cf178b199c761ec026a8061bd2a138c36`.
The dated amendment in `../waveguide_chain_battery_predeclaration.md` was written
before the first validation run. This note records implementation checks, not
adjudication of the predicted converged AD/FD accuracy.

The eps stencil is second-order forward on both slab and PEC-short, at the
unchanged theta0 = 0 and h = 0.05. One implementation serves the declared
material family; sigma remains central. The driver reuses its existing x64
baseline for f0, so there are still only two perturbed solves per group.
Full assembled arrays certify every arm, including vacuum outside the window.
Invalid configurations raise before solving; schema 4 validates mandatory
membership and certificates on cache reuse, assembly and replay. Its 14 rows
exclude the lossless PEC-short eps magnitude. Schemas 1--3 retain their recorded
historical adjudications. No committed fixture, physics gate, timestep or
Courant factor was modified; no new xfail was added.

## Build-time numbers

All three rungs (dx = 0.00254, 0.00127, 0.000635 m), for every row below:

| DUT | eps theta arms | global min eps_r on every arm | global Courant ratio on every arm | global min sigma |
|---|---|---:|---:|---:|
| thru (vacuum control) | 0, 0.05, 0.10 | 1 | 0.990000000 | 0 |
| PEC-short | 0, 0.05, 0.10 | 1 | 0.990000000 | 0 |
| slab | 0, 0.05, 0.10 | 1 | 0.990000000 | 0 |

The PEC-short sigma central arms at 0.05, 0.055, 0.045 S/m also have
min eps_r = 1, ratio = 0.99, min sigma = 0 at all rungs. Global min sigma
is zero because material outside the design window is unchanged.
The fixed dt values are 4.842700168608755e-12, 2.4213500843043775e-12,
1.2106750421521888e-12 seconds.

The historical real-arithmetic minus-window ratios at h = 0.05, 0.02,
0.0199, 0.01 reproduce 1.015718569, 1.000051019, 1.000000000,
0.994987437 to the stated digits. Production assembly explicitly stores
materials as float32 even inside an x64 context. The certificate intentionally
reads the actual rounded override; the table test separately checks both that
value and the real-arithmetic diagnostic. No material precision policy changed.

## Verification and exact counts

Final regression command, from the worktree root:

```sh
JAX_PLATFORMS=cpu PYTHONPATH=$PWD timeout 1200 pytest -n 4 \
  tests/unit/geometry/test_waveguide_chain_battery_fd_validity.py \
  tests/unit/autodiff/test_waveguide_chain_battery_fd_redesign.py \
  tests/unit/geometry/test_waveguide_chain_battery_geometry.py \
  tests/oracle/test_waveguide_chain_battery.py \
  tests/oracle/test_waveguide_chain_battery_guide_cell_aperture.py \
  tests/oracle/test_waveguide_chain_battery_v18_close.py \
  --junitxml=.validation-931/final.xml -o junit_logging=all -rA
```

651 passed, 0 failed, 16 existing xfailed, in 27.63 seconds. Slow/GPU physics
checks were excluded by repository defaults. Only one pytest was launched at a
time, after the other observed shared-pod pytest finished.

| Test file | Passed | Failed | Existing xfailed |
|---|---:|---:|---:|
| `tests/unit/geometry/test_waveguide_chain_battery_fd_validity.py` (new) | 29 | 0 | 0 |
| `tests/unit/autodiff/test_waveguide_chain_battery_fd_redesign.py` (new) | 24 | 0 | 0 |
| `tests/oracle/test_waveguide_chain_battery.py` (modified live leg; historical replay run) | 135 | 0 | 16 |
| `tests/unit/geometry/test_waveguide_chain_battery_geometry.py` (unchanged regression) | 15 | 0 | 0 |
| `tests/oracle/test_waveguide_chain_battery_guide_cell_aperture.py` (unchanged regression) | 233 | 0 | 0 |
| `tests/oracle/test_waveguide_chain_battery_v18_close.py` (unchanged regression) | 215 | 0 | 0 |

Every touched Python file passed compilation (6/6). The three support modules
`tests/_waveguide_chain_battery_fixture.py`, `tests/_waveguide_chain_battery_gates.py`,
and `scripts/diagnostics/waveguide_chain_battery_measure.py` have no standalone
pytest cases; the tests above exercise their certificate, estimator, cache,
assembly/replay refusal, and complete driver-stage behavior. The three touched
Markdown files (this note, `T-tests-crossval.md`, and the parent predeclaration)
have no executable tests; their scope, prediction, falsifier and gate statements
were reviewed. `git diff --check` passed. Ruff introduced no new findings:
10 existing E741 findings remain (gate helper 1, driver 2, historical oracle 7),
down from 11 at the parent; both new test modules and the fixture helper are clean.

The first regression run was 642 passed, 8 failed, 16 existing xfailed. All eight
were implementation/test setup defects: four calls to the initially undeclared
thru diagnostic window, and four expectations that x64 context promotes explicitly
float32 assembled materials. The window is now explicitly the short's window
without adding geometry or an AD accuracy leg; the certificate's actual-material
semantics were preserved. Focused recheck: 52 passed before the additional complete
stage/cache integration test; final counts above include it. These were coding
checks, not repeated physics-parameter attempts.

## Short runtime smoke and remaining adjudication

`JAX_PLATFORMS=cpu PYTHONPATH=$PWD timeout 60 python .validation-931/runtime_smoke.py`
completed in 12.4134 seconds of solve-path wall time: coarse PEC-short,
normalize=False, x64, all three eps arms, two periods. S shape was (2, 2, 17),
dtype complex128, all entries finite; both retained scalar objectives produced
three finite samples. Import provenance resolved inside this worktree. The
full baseline S trace, arm certificates and scalar samples are in the local
`.validation-931/runtime-smoke.json`; its stderr is empty. This deliberately
short record is not settled (centre S11 = 0.0315401 + 0.0613617j), and provides
no relative-accuracy or PEC magnitude conclusion. No VESSL job was launched.

The updated full-record live oracle test was not run. The PI must still launch
and adjudicate the re-measurement, including the retained complex eps legs,
unchanged sigma leg, ULP resolution and unchanged 0.05 relative gate. No new
result is pinned by this commit. Local validation logs remain in `.validation-931/`
and are not fixture updates.
