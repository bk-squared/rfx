# S0 mutation evidence

Re-run after deduplicating the findings into shared causes and witness variants,
using the new manifest loader: CPU, JAX 0.10.2, float32, two host CPU devices. Each run used the real builder,
execution/comparison code, and strict expected-failure classification. Mutations
were isolated in subprocess runs with a 180-second timeout per run; none timed
out. Edited files were restored byte for
byte in `finally` blocks. No mutated production code is included in this change.

| Mutation | Target | Result |
|---|---|---|
| m1 | Independently disable `compare`, `_tree`, `_comparison`, dict traversal, and list traversal | All five runs exited 1; the actual matrix live canary rejected each disabled layer. |
| m2 DFT | Add half a step to NU `t_plane`, retaining its DFT helper calls (`rfx/nonuniform.py:3103`) | Exit 1; the DFT finding's relative discrepancy changed from `0.0579118517` to `0.0289589227`; change `0.028952929` exceeded the unchanged `1e-4` bar. |
| m2 wire | Drop the first column from uniform returned `sparam_time_records`, retaining the scan/helper calls (`rfx/simulation.py:3549`) | Exit 1; V/I/V_port comparisons failed. Maximum relative difference `1`; matching relative bar `7.54426282e-7`. |
| m2 sigma | Pass zero sigma to only NU's `update_e_nu` call (`rfx/nonuniform.py:2921`) | Exit 1; probe relative difference `0.00847978886`; absolute difference `0.00160011649` versus bar `1.34110451e-7`. The corrected E-material capture also measured sigma difference `0.5` versus exact bar `0`. |
| m3 | Add a fake row admitted on uniform and NU, with no builder | Exit 2 at collection: `S0 missing builder: ('_s0_fake', '')`. |
| m4 | Admit Kerr on NU after freezing the generated refusal cell | Exit 1; NU no longer refused the active Kerr row. |
| m5 | Stop the eps derivative at only NU's `update_e_nu` call (`rfx/nonuniform.py:2921`) | Exit 1: **objective passed, gradient failed**. Objective relative difference `0`; gradient relative difference `1`, absolute difference `0.00857174117` versus bar `8.57174117e-7`. |

The unmutated DFT cell already fails its cross-path bar. Therefore m2 DFT is
killed by the existing-finding numeric fingerprint check, not by pretending its
baseline is green. Its physics comparison remains a strict expected failure;
no bar was widened. Wire samples are compared channel by channel, so NU's known
missing V_ref cannot mask corruption of the shared V/I/V_port channels.

All eleven mutation runs were killed. The pytest summaries were one failure
for each of the five m1 variants, each m2 variant, and m4;
one collection error for m3; one failure and one pass for m5.

Conclusions: in the S0 PR body (leader)

Long CPML rework, 2026-10-07, Mac, two CPU devices, float32: both 240-step
cells are ordinary passing `realized` / `final_fields` checks. The committed
controls require both exact cell IDs, six SI final components, the impedance
opt-in, and its use by `execute()`. They also preserve main's 12-step invariant
for PR finding cells. The cause and its numeric witnesses are removed entirely.

Local worker-only mutations retained the generated cells, real two-device
solves, and all execution/comparison helpers. `solve()`'s gathered multi-device
Hy was multiplied after the solve, only for the 240-step base scene. No solver
file was edited. Each run was bounded at 180 seconds; all finished and the
worker was restored byte-for-byte in `finally`.

| Control | Uniform Hy ULP | Graded Hy ULP | Result | Wall time |
|---|---:|---:|---|---:|
| Paired peak off (per-family SI peaks) | 86 | 856 | 2 failed, 2 passed | 42.33 s |
| Hy *= 1 + 1e-4 | 76.875 | 5.261719 | 1 failed, 3 passed | 41.19 s |
| Hy *= 1 + 2e-6 | 6.8125 | 3.25 | 4 passed | 40.70 s |
| Hy *= 1 + 1e-3 | 720.375 | 49.87109375 | 2 failed, 2 passed | 39.45 s |

All failures above are numeric `final_fields` failures; realized prerequisites
pass. The 1e-4 control rejects uniform only; graded's small H stays below the
paired-peak bar. The 2e-6 perturbation is below the gate's resolution on both
cells. The added 1e-3 control rejects both. These are trace-equivalence checks,
not relative-H-accuracy bounds. Each run collected four checks and deselected
604. Two initial injection attempts assigned to an immutable result and failed
in the harness; they are excluded from this numeric evidence and were rerun
using `_replace()`.

Command: `RFX_S0_FULL=1 python -m pytest tests/contracts/path_equivalence/test_matrix.py -k "_boundary and 240" -q -o addopts="" -m "not gpu"`.
Raw logs and measurements: `.s0-work/rework-mutation-*`; local scripts:
`.s0-work/run_paired_mutations.py`, `.s0-work/run_larger_mutation.py`.
Both long cells remain full-matrix only under the unchanged PR selection rule
(passing candidates must have 12 steps), adding no PR solves. Initial clean
uncached solve costs were 9.236 s for uniform and 10.315 s for graded,
19.551 s combined on this Mac; these costs vary with load.

After restoring the worker, the serial PR suite passed 412 checks with 9 expected
xfails (445 deselected; 355.45 s wall). Full distributed selection passed 154
checks with 3 expected xfails (451 deselected; 327.31 s wall), reproducing every
long-cell ULP above; its uncached long-cell solve costs were 6.550 s uniform and
8.789 s graded. Inventory contracts passed 70 checks (13.85 s wall); ruff passed.
Full matrix collection is main's 604 IDs plus exactly four long-cell record IDs;
PR matrix collection is unchanged at 163 IDs, with no removed IDs in either set.
