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
