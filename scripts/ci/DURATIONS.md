# .test_durations — provenance

2026-10-20: Removed 12 obsolete workflow/job test node IDs from each duration file
after moving the A6000 and monthly ladder jobs to agent-rules. Remaining prices
are unchanged; the entry counts below reflect this pruning.

Regenerated 2026-10-16 at `edd70b41efe86a5cf09fc15424a0e1efe5dea363` (branch `ci/cached-durations`,
whose base is `main` `2202b0565`, the merge of #1779) from `regen-durations` run
[38828042377](https://github.com/bk-squared/rfx/actions/runs/38828042377) with `lanes=fast`,
measured 2026-10-16 00:11-01:15 UTC on `ubuntu-latest` with Python 3.11.17 / JAX 0.10.2,
pytest 9.1.1, pytest-split 0.11.0 and trimesh 5.1.1 (the `cad` extra). Only the fast lane was
re-measured: each fast job ran with JAX's on-disk compilation cache in one empty directory per job,
as pr-tests' fast suite does since #1779, so the fast prices are what that lane now pays.
**24640 entries in the file**, of which **347 of them carried unchanged** from the
20407-entry file this replaces. All 12 fast jobs succeeded; slow was skipped. The fast (6) pytest
summary reports one failed test,
`tests/locks/test_lock_provenance_gate.py::test_lock_is_not_expired[test_cpml_axis_params_refactor_bit_identical]`
(the lock expired on 2026-10-15); its duration is stored like any other, and the workflow's pytest
command uses `|| true`.

**Fast prices are normalised to the AMD EPYC 7763.** The hosted runners hand out several CPU classes
and the same test takes different times on them. Each fast artifact value is multiplied by its job's
CPU factor, so the whole fast lane is priced on one machine class, the slowest one the runners hand out:

| CPU (lscpu "Model name") | factor | fast jobs |
|---|---:|---|
| AMD EPYC 7763 | 1.000 | 2, 4, 8, 10, 11 |
| AMD EPYC 9V74 | 1.375 (= 1.10 / 0.80) | 1, 9, 12 |
| AMD EPYC 9V45 | 1.429 (= 1.10 / 0.77, rounded) | 3, 6, 7 |
| Intel Xeon Platinum 8573C | 1.375 (assumed, see below) | 5 |

The factors come from run 38774666720 (pull request 1775): actual / predicted shard time per shard
under one price file was 0.81 / 0.79 / 0.80 on the 9V74, 0.84 / 0.71 / 0.75 on the 9V45, 0.85 on a
Xeon 6973P and 1.03 / 1.11 / 1.16 / 1.10 on the 7763 (shard 10 at 1.63 excluded: a known outlier,
cause undetermined). The 8573C was not measured; the nearest measured Intel, the 6973P, would give
1.294. Fast (5) holds 466.9 s raw, so that choice moves at most about 40 s of price. The applied
9V45 factor is the rounded 1.429; the exact quotient would make the lane total 2.0 s smaller.
The slow-lane and highmem entries carried below are raw runner values, not normalised.

Merge order: the file this replaces first, then the 12 normalised fast maps in numeric order; the
last input wins for overlapping nodeids. The fast artifacts contain 24305 distinct nodeids (19531
already in the replaced file, 4774 new). 529 old entries are in no fast artifact and are not
collected at HEAD, and are dropped (262 of them from `tests/oracle/test_lumped_wire_chain_battery.py`,
which no longer exists; the rest are renamed or removed parametrisations). The 347 carried entries
are 333 weekly-selection tests the fast lane does not run (`slow` / `slow_physics`), with the
values of the file this replaces (run 37983083757, or pull request 1745 for 7 rows), and the 14
highmem tests outside both GitHub selections. No new slow or a6000 measurements were supplied.

At the tree above, fast, weekly and highmem collect 24294, 25326 and 15 tests
respectively (25341 distinct), with 0, 700 and 0 missing durations
respectively (700 distinct unmeasured nodeids: 469 from `tests/contracts/path_equivalence/test_matrix.py`,
144 from `tests/contracts/test_run_forward_one_pipeline.py`, 86 from `tests/contracts/test_absorber_curl_decay.py`
and 1 other; no slow run has measured them yet). The not-code contract selection
collects 9947 tests, with 230 missing durations (the 144 and 86 above).
Collection used Python 3.11.2 / JAX 0.10.2 CPU, pytest 9.1.1 and pytest-split 0.11.0 on macOS.
No collection had the CAD extra (trimesh) or Plotly, so the 11 tests of
`tests/unit/geometry/test_mesh_import.py` are not in the local fast count; the runners collect them
(24305 fast tests), and they are priced from the fast artifacts.
Weekly collection sets `RFX_S0_FULL=1`, matching validation.yml.
Counts use `python -m pytest --collect-only -q -p no:cacheprovider`:
fast uses default addopts; weekly adds
`-m "not gpu and not highmem and not docs_consistency"`; highmem adds
`-m "highmem and not gpu"`; contracts adds
`tests/contracts -o addopts="" -m "not gpu and not docs_consistency"`.
The full inventory used for the drop uses `RFX_S0_FULL=1 -o addopts=""` (26194 tests); an old
entry is dropped only if it is in neither that inventory nor a fast artifact. The contract checks the
entry count and carried-count bounds dynamically; it has no fixed entry or carried count to update.

All 12 fast job logs were checked: **0 entries were re-priced from job logs**.
All 24305 test result lines were timed from the previous result line, with the
first test timed from the pytest session-start line, and compared with the raw (not normalised)
artifact value on the same runner. No gap exceeds its artifact value by more than 600 s; the
largest excess is 36.4 s; in every job the largest excess is the first test, whose gap holds the
collection.

The four entries once set by hand now hold runner measurements. Three are slow-lane entries the
fast lane does not run; they are carried with the raw values of run 37983083757. The fourth runs in
the fast lane and was re-measured here: 30.29959572299998 s raw on fast (6), an EPYC 9V45, stored as
43.29812228816697 s after the 1.429 factor.

| nodeid | recorded seconds | source |
|---|---:|---|
| `tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_end_to_end_matches_historical_base` | 1014.1397751879999 | carried |
| `tests/locks/test_coax_chain_battery_drift.py::test_the_coarsest_mesh_still_solves_to_its_stored_s[bead]` | 139.71913598599997 | carried |
| `tests/locks/test_coax_chain_battery_drift.py::test_the_coarsest_mesh_still_solves_to_its_stored_s[thru]` | 142.896389494 | carried |
| `tests/oracle/test_coax_open_end_settles.py::test_the_open_end_is_passive_and_settled` | 43.29812228816697 | fast (6), normalised |

All 24305 refreshed entries are normalised artifact measurements; the 347 others are carried.
**No floor is added.**

PR 1745 (2026-10-14) replaced 8 CPML dispersive entries with 12: 9 measured on the gate node and scaled by
1.12 to the GitHub runner, and the three cpml `run` rows set to 19.5 s. This regeneration replaced the
5 of them the fast lane runs (the three `run` rows and the two `test_cpml_dispersive_pr_snapshots`
rows) with normalised runner measurements. The 7 rows of
`tests/unit/runners/test_distributed_cpml_dispersive_weekly.py` are slow-lane tests and keep their
gate-scaled values until the next slow regeneration.

## The 0.3 s floor, retired 2026-09-18

Entries used to carry a measurement **plus** 0.3 s, on the reasoning that a shard's wall time
exceeds the sum of pytest's own per-test report durations: module imports, first-touch fixture
construction and framework overhead are outside those numbers and fall hardest on a shard holding
thousands of sub-millisecond tests. The 0.3 s came from one reading on the first revision of pull
request 939 — shard 1 held 3426 tests summing to 1149.7 s and spent 1817 s of pytest time, about
0.195 s per test unaccounted for.

That reading no longer describes this lane. The falsifier is the shard the committed file assigns
to group 1, priced three ways. Measured 2026-09-18, when that shard held 3067 tests:

| | shard 1 |
|---|---|
| predicted from fresh measurements, no floor | 4.2 min |
| predicted with 0.3 s added to each of its 3067 tests | 19.5 min |
| **observed**, `fast-suite (1)` on pull request 1108 | **5 min 19 s** |

The floored model is 3.7x over the observation; the unfloored model is 1.1 min under it. One
reading of a noisy runner cannot separate 4.2 from 5.3, but it settles 19.5: shard 1 has come in
at 4.5-5.3 min in every one of the four observed runs of the committed file listed under Balance,
so the floored model is wrong by a factor, not by noise. Whatever per-test overhead the
2026-09-08 runner had, the runner this lane uses now does not have it at anything like 0.195 s,
and adding 0.3 s to 10625 entries adds 53 minutes of fiction spread evenly across the lane.
Evenly is the problem: a constant per test does not model a real cost, it just pulls every shard
toward equal test COUNTS and away from equal time.

So the floor is gone, and step 4 of Regenerating below says not to re-add it. One consequence to
keep in mind:

- The 14 carried entries retain their previous measurements. No floor is added to them or to
  the refreshed artifact values.

An earlier version of this note blamed a pytest-split threshold that supposedly drops short setup
and teardown readings. That was backwards — `STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD` is
600 s and the plugin discards readings ABOVE it. The floor never stood on that claim, and it no
longer stands on the 2026-09-08 measurement either.

## Sources

Where each entry's value comes from. `merge_test_durations.py` takes the last
input's value for an id measured twice; the file this replaces went first, then
the normalised `fast (1)`–`(12)` maps in numeric order.

| source | entries |
|---|---:|
| fast artifacts, normalised to the EPYC 7763 | 24305 |
| log-derived corrections | 0 |
| carried unchanged from the replaced file (333 weekly-only, 14 highmem-only) | 347 |

All 12 downloaded artifacts had the documented layout `durations-fast-N/fast_N.json`.
The following compares each raw artifact sum with its pytest session summary, and gives the
normalised sum that enters the file. Sums are exact decimal sums of the recorded JSON numbers;
no duration is rounded.

| job | CPU | factor | entries / result lines | raw artifact seconds | session seconds | normalised seconds |
|---|---|---:|---:|---:|---:|---:|
| fast (1) | EPYC 9V74 | 1.375 | 2026 / 2026 | 782.8443531500046499485 | 813.63 | 1076.4109855812563967226 |
| fast (2) | EPYC 7763 | 1.000 | 2026 / 2026 | 1132.48188498699721597867 | 1166.15 | 1132.48188498699721597867 |
| fast (3) | EPYC 9V45 | 1.429 | 2026 / 2026 | 863.7884853009989566028 | 892.01 | 1234.3537454951275064038 |
| fast (4) | EPYC 7763 | 1.000 | 2026 / 2026 | 702.3043285830029131017 | 739.20 | 702.3043285830029131017 |
| fast (5) | Xeon Platinum 8573C | 1.375 | 2026 / 2026 | 466.87644878099685135485 | 493.10 | 641.95511707387067356751 |
| fast (6) | EPYC 9V45 | 1.429 | 2026 / 2026 | 1642.87196972400397683687 | 1673.32 | 2347.6640447356016767075 |
| fast (7) | EPYC 9V45 | 1.429 | 2026 / 2026 | 2168.7977847449935470146 | 2194.02 | 3099.2120344005957754755 |
| fast (8) | EPYC 7763 | 1.000 | 2026 / 2026 | 1483.43582508800679286854 | 1517.06 | 1483.43582508800679286854 |
| fast (9) | EPYC 9V74 | 1.375 | 2026 / 2026 | 1127.27398389200189155921 | 1158.21 | 1550.00172785150260440253 |
| fast (10) | EPYC 7763 | 1.000 | 2026 / 2026 | 3744.39418543600485457209 | 3778.59 | 3744.39418543600485457209 |
| fast (11) | EPYC 7763 | 1.000 | 2026 / 2026 | 1064.1641693020015124040 | 1102.02 | 1064.1641693020015124040 |
| fast (12) | EPYC 9V74 | 1.375 | 2019 / 2019 | 3219.3826677290004798055 | 3252.75 | 4426.6511681273756224883 |

Raw total 18398.61608671801364204733 s; normalised total 22503.02921666134354469274 s
(about 31.3 min per group on a 7763 when split 12 ways).
Fast (5) is 5.3 % short: its session exceeds the artifact sum by 26.2 s, of which 25.3 s is the
collection interval between its session-start and `collected` lines (00:11:57.49 to 00:12:22.75 UTC).
Every other job is less than 5 % short, and every job is less than a minute short; each job's
collection interval is 24-37 s. None of the 24305 result-line gaps meets the re-pricing rule.

- Recorded total: 39736.77336757778481140655 s, against 38492.98267838228939039589 s for the file this replaces.

### Carried slow-lane entries: run 37983083757 (regenerated 2026-10-10)

Most of the 333 carried weekly-only entries (all but the 7 rows of pull request 1745) and three of
the four hand-set rows above come from this earlier regeneration, measured 2026-10-09 UTC on `main` (`4a42dbc9`) with slow artifacts merged first and
fast last; its fast values are now replaced. Its record follows unchanged.

All 17 downloaded artifacts had the documented layout:
`durations-slow-N/slow_N.json` and `durations-fast-N/fast_N.json`.
The following compares each raw artifact sum with its pytest session summary.
Sums are exact decimal sums of the recorded JSON numbers; no duration is rounded.

| job | entries / result lines | artifact seconds | session seconds |
|---|---:|---:|---:|
| slow (1) | 4076 | 1892.30791771400947728836 | 1917.92 |
| slow (2) | 4076 | 4816.47788627496861112753 | 4841.76 |
| slow (3) | 4076 | 8352.47736137299563342507 | 8384.81 |
| slow (4) | 4076 | 10550.28045739796571283792 | 10583.91 |
| slow (5) | 4073 | 10474.84668043799913718666 | 10507.50 |
| fast (1) | 1673 | 982.0858886190035106980 | 1013.22 |
| fast (2) | 1673 | 1329.59095399300199812892 | 1360.42 |
| fast (3) | 1673 | 1111.2056761850081721491 | 1143.55 |
| fast (4) | 1673 | 466.3925578120031319358 | 496.77 |
| fast (5) | 1673 | 1914.8531834199998581145 | 1947.39 |
| fast (6) | 1673 | 2990.1831097710139055818 | 3014.58 |
| fast (7) | 1673 | 1224.0627920830008835834 | 1253.86 |
| fast (8) | 1673 | 776.35146230800359434850 | 802.91 |
| fast (9) | 1673 | 1656.0678250429997536884 | 1687.37 |
| fast (10) | 1673 | 3439.17228643799534898566 | 3471.14 |
| fast (11) | 1673 | 1587.4622946870009852813 | 1622.74 |
| fast (12) | 1670 | 3794.4133760550038207565 | 3827.53 |

Fast (4) is more than 5% short: its session exceeds the artifact sum by
30.3774421879968680642 s. Its session-start and collected timestamps are
19:52:09.7494240Z and 19:52:39.2155498Z (2026-10-09), a 29.4661258 s collection
interval. None of its 1673 result-line gaps meets the re-pricing rule.
All other jobs are less than 5% short; every job is less than a minute short.

- **The previous run 36394962310 used log-derived `slow (5)` entries.** With `-v` each test's result line is written when
  the test ends, and the job log stamps every line. An entry is the gap between a test's result
  line and the one before it; the first test is timed from the `collected` line. The gap holds
  setup, call, teardown and pytest's own work between tests. 2233 of the 2361 were also measured
  by a fast shard. Over those the log-derived sum is the lower one: 4019.2 s against 4100.7 s
  for the fast artifacts (-1.99 %).
- **That previous run used one log-derived `slow (2)` entry too.** `--store-durations` does not store a setup or
  teardown phase longer than 600 s (`STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD` in
  pytest-split's plugin). The module fixture of
  `tests/locks/test_patch_edgefed_resonance_harminv.py::test_ringdowns_are_settled` spends its
  time in setup, so the artifact held 0.0012 s; the entry is the 1514.93 s between its result line
  and the one before it in the `slow (2)` log (08:09:42.28 to 08:34:57.22). It was the only test in
  the ten artifacts whose log time exceeds its artifact value by more than 600 s. Before the fix,
  `slow (2)`'s artifact summed to 3590.9 s against a 5127.3 s pytest session; every other shard
  summed to 97.1-99.5 % of its session, and `fast (1)`, the shortest, to 91.6 % (339 s of 370 s).
- The 14 carried are the highmem-marked tests that also carry `slow` or `slow_physics`, so no
  GitHub selection runs them and no GitHub split uses their time. 12 are the a6000 measurements of
  2026-09-16 (VESSL run 369367259335, `-m "highmem and not gpu"`), carried from the file before.
  The 13th, `tests/unit/autodiff/test_coax_two_port_ad.py::test_compute_coaxial_two_port_ad_grad_finite_and_fd_consistent`,
  became highmem in pull request 1358; its 67.46 s is `slow (2)` of run 35613805791
  (2026-09-21), measured before that.
  The 14th, `tests/unit/autodiff/test_coax_two_port_ad.py::test_coax_two_port_eps_scale_unity_matches_concrete_path`,
  is also outside both GitHub selections now; its 21.178872745 s is carried from the replaced file.
- Recorded total: 36728.06249896402403329589 s, against 34424.59722538594211760151 s for the file this replaces.

## The one highmem test the fast lane does run

`tests/unit/ports/test_msl_source_fixture_static.py::test_auto_eps_msl_gradient_matches_fd_mini_referee`
carries `highmem` and nothing else, so pyproject's default `-m 'not gpu and not slow and not
slow_physics and not docs_consistency'` does not deselect it and it runs in a fast shard. Its entry must therefore be a
runner measurement, and it is: 54.40078801575026 s, the 39.56420946600019 s of `fast (9)` of run
38828042377 (an EPYC 9V74) times 1.375. It is the one highmem-marked entry the regeneration
refreshed; the other 14 are carried. When regenerating, do not overwrite this one from the a6000 map.

## .test_durations_weekly — weekly prices (2026-10-19, refreshed 2026-10-20)

**Refresh of 2026-10-20.** Weekly run `39378910883` at main `a1e03dce6`: shard 5 ran 155.2 min of
its 180 (86.2 %). Its 3,844 priced tests ran 98 min against 102 min of prices; 65 tests with no
weekly price (distributed dispersive-CPML files added on 10-14 and 10-19, after the measurement
above) ran 56 min. pytest-split priced each at the selection mean (about 2 s), and
`duration_based_chunks` keeps collection order, so the whole block landed in shard 5. Its pytest
then aborted (exit 134) in a distributed test, so `--store-durations` stored nothing and the budget
step read 0 measured tests. Since then every test phase is also appended to
`durations/weekly_<k>.log` (`tests/_durations_log.py`, `RFX_DURATIONS_LOG`), and
`weekly_shard_budget.py --log` rebuilds the shard's map from it when pytest stored nothing.

Merge order: the 2026-10-19 file below; `s5_unpriced_from_log.json` (shard 5's unpriced tests
only, from GitHub log timestamps, test i's start to test i+1's start — an **approximation**; the
aborted test is left out); then measured `weekly-durations-1`, `-2`, `-3`, `-4`, `-6`, `-7`, `-8`
from run `39378910883`; then `s5_never_run_estimates.json`. Later input wins; retain only the
weekly collection (27,741 tests with `RFX_S0_FULL=1` on this change); round to 0.01 s.
The inputs and a script that rebuilds this file are in rfx-archive `rfx/records/20261020-weekly-prices-refresh/` (commit 625522eb0).
`s5_never_run_estimates.json` holds **estimates, not measurements**, for four shard-5 tests the
abort left unrun: the aborted lossy-Lorentz test and `..._mixed.py` at the median of the measured
Debye and Lorentz siblings (227.06 s), `..._mixed_800_weekly.py` and `..._mixed_1600_weekly.py` at
the median of the four measured 800/1600 siblings (241.66 s). 22 collected tests stay unpriced: 12
added since `a1e03dce6` (this change's own contract tests among them) and 10 never-run shard-5
tests in files with no measured sibling (`test_distributed_dispersive_cpml_paths.py`, `_ports.py`,
`_snapshots.py`). The 36 log ids a space cut are outside the collection.

| Source (merge order) | Input entries | In weekly collection | Final owned entries |
|---|---:|---:|---:|
| `.test_durations_weekly` of 2026-10-19 | 27,370 | 27,309 | 3,954 |
| `s5_unpriced_from_log.json` | 65 | 29 | 29 |
| `weekly_1.json` | 7,418 | 7,415 | 7,415 |
| `weekly_2.json` | 5,491 | 5,484 | 5,484 |
| `weekly_3.json` | 1,888 | 1,888 | 1,888 |
| `weekly_4.json` | 3,447 | 3,447 | 3,447 |
| `weekly_6.json` | 745 | 745 | 745 |
| `weekly_7.json` | 3,319 | 3,319 | 3,319 |
| `weekly_8.json` | 1,436 | 1,434 | 1,434 |
| `s5_never_run_estimates.json` (estimates) | 4 | 4 | 4 |

Predicted test time per group (`duration_based_chunks`, `--splits 8`, the collection above):
113.8, 113.9, 114.9, 113.8, 117.1, 113.8, 113.8, 109.0 min (max 117.1, group 5, which carries
the four estimates). Group 6 holds the 10 unpriced never-run tests at the selection mean (about 2 s
each); their siblings ran minutes each, so group 6 may run long until the per-test log prices them. The 2026-10-19 file on
the same collection predicts max 123.4 min and shard 5 at 109.4, the 56 unpriced minutes not counted.

### 2026-10-19 measurement

The uncached weekly lane uses its own file; the fast lane and gate keep
`.test_durations`. Weekly run `39220415396`, at main `f0e3805d9` on 2026-10-19,
measured the uncached prices. The old fast-lane prices underpredicted measured
weekly work by x1.05–1.48 (leader's per-shard comparison); shard 4 used 9270 s,
above the 85 % budget, and shard 5 was cancelled at the 180-minute limit.

2026-10-19: 27,370 entries, with 87 fast-lane fallback prices. After the 2026-10-20 refresh:
**27,719 entries in the weekly file**, with at most **87 fast-lane fallback prices** (carried from
2026-10-19; this refresh added none).
Values are rounded to 0.01 s and the file keeps only test ids the weekly selection collects
(rfx-lead D:b089a0): 3,290,060 bytes at regeneration, before the pruning noted above.
Weekly collection contains 27,440 tests, of which 58 have no entry in any
input and receive pytest-split's mean-of-recorded-selection estimate. The fast-lane
fallback count refers to retained `.test_durations` entries with no weekly source,
not these entirely unpriced tests.
Collection was measured on `ci/weekly-shards`, based on
`d21f02905ef8594eb49c7039c0c24983e24d89d5`, including this update's contract
and labeler changes, using `~/Documents/rfx/.venv/bin/python` on macOS.

Merge order is `.test_durations`, then `s5_from_log.json`, then measured
`weekly-durations-1`, `-2`, `-3`, `-4`, `-6` from run `39220415396`, in that order.
Use `scripts/ci/merge_test_durations.py` semantics: convert values to floats,
later input wins, sort nodeids, then retain only the current weekly collection.
The shard-5 artifact is empty; its replacement map is an **approximation** from
GitHub log timestamps, test i's start to test i+1's start, including per-test
setup/teardown. Its sum is approximately 177.9 minutes. Weekly measurements are
raw runner seconds, without the fast lane's cached-compile normalisation.

| Source (merge order) | Input entries | In weekly collection | Final owned entries |
|---|---:|---:|---:|
| `.test_durations (fast-lane fallback)` | 24,652 | 24,514 | 87 |
| `s5_from_log.json` | 3,251 | 3,226 | 3,226 |
| `weekly_1.json` | 11,221 | 11,214 | 11,214 |
| `weekly_2.json` | 3,139 | 3,138 | 3,138 |
| `weekly_3.json` | 3,751 | 3,751 | 3,751 |
| `weekly_4.json` | 3,964 | 3,964 | 3,964 |
| `weekly_6.json` | 2,002 | 2,002 | 2,002 |

### Weekly balance: eight groups

pytest-split 0.11.0's default `duration_based_chunks` algorithm was applied to the
collected items in collection order and the weekly map, with `--splits 8`. It keeps each
module's tests together in one group, which is how the weekly prices were measured: a module's
fixture setup and first compile sit in its first test's price. (`least_duration` was tried and
rejected in review: it spreads modules over every group, paying that setup once per group,
and it put all 11,826 tests priced 0.00 s into one group.) Missing prices use the selection
mean. These are predictions of test time, not job wall times; install/import and shard 1's
three-device steps (about 12 min on run 39220415396) are additional.

```sh
RFX_S0_FULL=1 ~/Documents/rfx/.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider \
  -m "not gpu and not highmem and not docs_consistency"
# To reproduce a group, append (k = 1..8):
# --splits 8 --group "$k" --durations-path .test_durations_weekly
```

| Group | Tests | Predicted minutes |
|---|---:|---:|
| 1 | 7,414 | 111.7 |
| 2 | 5,432 | 112.8 |
| 3 | 2,162 | 112.3 |
| 4 | 3,154 | 111.6 |
| 5 | 3,858 | 113.5 |
| 6 | 452 | 111.6 |
| 7 | 2,968 | 116.6 |
| 8 | 2,000 | 102.2 |
| **Max** | | **116.6** |

The maximum meets the <=130-minute acceptance target (group 1 about 124 min with its three-device steps). The job limit remains
180 minutes, its wall-time budget fraction remains 0.85 (153 minutes), and the
headroom contract retains RUNNER_SPREAD=1.22 and SETUP_S=600.

Regenerate the weekly file separately: seed with the previous weekly file (and
fast prices only for previously unpriced tests), then merge fresh weekly maps in
numeric shard order, with any named log approximation before measured artifacts;
prune to a fresh weekly collection. Update these counts and balance. Do not merge
weekly prices back into the fast file. `regen-durations.yml` already has a `slow`
lane: it now uses eight splits, reads and writes `.test_durations_weekly` in each
job, and uses `--clean-durations` so artifacts contain only fresh measurements. Merge
those artifacts into `.test_durations_weekly`; fast regeneration is unchanged.

## Regenerating

Each weekly run uploads `weekly-durations-N` artifacts measured on the weekly
selection; merge them into `.test_durations_weekly` as described above. A shard
using more than 85 % of its 180-minute job limit fails its final budget check:
add a weekly shard or regenerate `.test_durations_weekly`. Unpriced test counts are
reported without failing the check.
pytest-split reads prices from and stores measurements to one file, so the
weekly step seeds that file with `.test_durations_weekly` and the budget check prunes
it to the entries the shard measured before the upload.

The following fast-file procedure predates the weekly-file separation; slow inputs
are fallback prices for fast-file entries, not the weekly output. Keep fresh weekly
measurements in the weekly file using the procedure above.

1. Dispatch `.github/workflows/regen-durations.yml` and download the artifacts. Read each job's CPU from its
   "Runner CPU" step and multiply each FAST job's artifact values by its factor to the EPYC 7763 (table in the
   header; a CPU model with no measured factor is an assumption, named in the header); slow artifacts stay raw.
   Merge the scaled maps. The log-gap check in step 2 compares a gap with the RAW artifact value of its own job.
2. `python scripts/ci/merge_test_durations.py .test_durations <slow shards...> <fast shards...>`.
   Later inputs win: slow first, fast last — the required lane must be priced by its own runs.
   A shard cut before the session ends uploads nothing; price what it finished from its job log
   as in Sources, pass that map last, and say so in the header. Then compare each input's sum, which the script prints, with
   the pytest session time in that shard's job log (the `= ... in Ns =` line). Look into any shard
   more than about 5 % short, or short by more than a minute: a setup or teardown over 600 s is
   dropped by `--store-durations` and is the known cause (see Sources).
   Check every result-line gap against the merged duration (first test: session-start line);
   re-price entries whose gap exceeds the stored value by more than 600 s from the log, because
   pytest-split drops setup/teardown phases above that threshold.
3. Take only the 14 GPU-only entries above (highmem and slow or slow_physics) from an a6000 run of
   the highmem selection. Leave the fast-lane highmem test on its runner value.
4. **Do not add a floor.** Commit the measurements with the log-derived phase-loss corrections
   above. The additive 0.3 s this step used to require was retired on 2026-09-18 for the reason
   above; if a future lane really does carry
   per-test overhead the reports miss, measure it on that lane first and write the measurement
   here before any constant goes back in.
5. Collect every group before committing (`pytest --collect-only --splits N --group k`,
   pytest-split's own `duration_based_chunks`), then read
   the pull request's own shard times afterwards — that reading is the check, not the simulation.
   Re-measure the collection while you are there and quote it with the commit you measured it at:
   it grew 10224 -> 10472 on the fast lane in the three days this file's own refresh took, so a
   simulation quoted without one is unreproducible.
6. Update the entry count and the carried count in the header above.
   `tests/contracts/test_test_durations_provenance.py` compares both against the file.

## Balance

Collected at `edd70b41efe86a5cf09fc15424a0e1efe5dea363` with the new file using
pytest-split 0.11.0's `duration_based_chunks`, by `--collect-only` for all 30 groups
(fast 12, contract 12, weekly 6, the weekly count validation.yml now uses).
These 30 invocations executed no tests. The ordered groups concatenate to
each unsplit selection exactly, with no duplicate nodeids.

Commands, with `k` ranging over every group (Python and optional dependencies as
specified in the header: no trimesh, so fast is 11 tests short of what the runners split):

```sh
RFX_S0_FULL=1 python -m pytest --collect-only -q -p no:cacheprovider \
  -m "not gpu and not highmem and not docs_consistency" --splits 6 --group "$k"
python -m pytest --collect-only -q -p no:cacheprovider --splits 12 --group "$k"
python -m pytest tests/contracts --collect-only -q -p no:cacheprovider \
  -o addopts="" -m "not gpu and not docs_consistency" --splits 12 --group "$k"
```

"recorded seconds" sums only entries present in the new file, using exact decimal arithmetic;
missing tests contribute nothing to it. "plugin estimate" is the `estimated duration` pytest-split
prints for the group, which prices each missing test at the mean recorded duration of that
selection. Neither is an observed shard wall time. Fast and contract prices are EPYC 7763 seconds;
weekly prices mix those with raw slow-lane runner values.

| selection | group | tests | recorded seconds | missing durations | plugin estimate (s) | plugin estimate (min) |
|---|---:|---:|---:|---:|---:|---:|
| fast | 1 | 3337 | 1869.9989689672500477081 | 0 | 1870.00 | 31.17 |
| fast | 2 | 3016 | 1872.11517967313170079997 | 0 | 1872.12 | 31.20 |
| fast | 3 | 4559 | 1882.47922289755499689101 | 0 | 1882.48 | 31.37 |
| fast | 4 | 1464 | 1881.7378517213354175871 | 0 | 1881.74 | 31.36 |
| fast | 5 | 673 | 1872.9373570075793936141 | 0 | 1872.94 | 31.22 |
| fast | 6 | 2408 | 1869.7612360215983255845 | 0 | 1869.76 | 31.16 |
| fast | 7 | 2669 | 1878.33384427451331985657 | 0 | 1878.33 | 31.31 |
| fast | 8 | 1753 | 1888.10883421900201096140 | 0 | 1888.11 | 31.47 |
| fast | 9 | 316 | 1870.14928588300359042599 | 0 | 1870.15 | 31.17 |
| fast | 10 | 2872 | 1872.9235786345018736596 | 0 | 1872.92 | 31.22 |
| fast | 11 | 946 | 1882.3618030662505754337 | 0 | 1882.36 | 31.37 |
| fast | 12 | 281 | 1795.4732799616225247130 | 0 | 1795.47 | 29.92 |
| contract | 1 | 517 | 385.5082150452488793682 | 40 | 405.01 | 6.75 |
| contract | 2 | 1097 | 386.7840225837587624574 | 46 | 409.21 | 6.82 |
| contract | 3 | 667 | 468.7369204392477668790 | 0 | 468.74 | 7.81 |
| contract | 4 | 955 | 406.4091589589959321050 | 0 | 406.41 | 6.77 |
| contract | 5 | 382 | 421.0004838109991865454 | 0 | 421.00 | 7.02 |
| contract | 6 | 1089 | 404.62336448425882793947 | 0 | 404.62 | 6.74 |
| contract | 7 | 534 | 406.2627340818316328452 | 0 | 406.26 | 6.77 |
| contract | 8 | 422 | 404.5874611349582264223 | 0 | 404.59 | 6.74 |
| contract | 9 | 581 | 406.1074957030820292401 | 0 | 406.11 | 6.77 |
| contract | 10 | 745 | 404.3072501819974040918 | 0 | 404.31 | 6.74 |
| contract | 11 | 2733 | 337.5622119696301798717 | 144 | 407.77 | 6.80 |
| contract | 12 | 225 | 305.8099463608740606281 | 0 | 305.81 | 5.10 |
| weekly | 1 | 11044 | 5622.12608048322667164628 | 699 | 6737.90 | 112.30 |
| weekly | 2 | 2350 | 6801.2172818459243789245 | 1 | 6802.81 | 113.38 |
| weekly | 3 | 3380 | 7265.38196927729716777355 | 0 | 7265.38 | 121.09 |
| weekly | 4 | 3950 | 6743.94939840821346609513 | 0 | 6743.95 | 112.40 |
| weekly | 5 | 2799 | 6794.59396187587704937199 | 0 | 6794.59 | 113.24 |
| weekly | 6 | 1803 | 6081.9692376984974571374 | 0 | 6081.97 | 101.37 |

Largest plugin estimate: fast 31.47 min (group 8), contract 7.81 min (group 3),
weekly 121.09 min (group 3). The fast estimates are EPYC 7763 seconds. The check is the pull
request's own fast-suite shard times, not this simulation.

### Previous balance, 2026-10-10

Collected at `3b62b21bef6a16f3c7aae8274a0f3798e6ff3a0d` with the new file using
pytest-split 0.11.0's `duration_based_chunks`, by `--collect-only` for all 29 groups.
These 29 invocations executed no tests. The ordered groups concatenate to
each unsplit selection exactly, with no duplicate nodeids.

Commands, with `k` ranging over every group (Python and optional dependencies as
specified in the header):

```sh
RFX_S0_FULL=1 python -m pytest --collect-only -q -p no:cacheprovider \
  -m "not gpu and not highmem and not docs_consistency" --splits 5 --group "$k"
python -m pytest --collect-only -q -p no:cacheprovider --splits 12 --group "$k"
python -m pytest tests/contracts --collect-only -q -p no:cacheprovider \
  -o addopts="" -m "not gpu and not docs_consistency" --splits 12 --group "$k"
```

The table sums only entries present in the new file, using exact decimal
arithmetic. Missing tests contribute no recorded seconds to this table;
pytest-split uses the mean recorded duration in that selection when assigning
those tests to groups. These sums are recorded test seconds, not observed shard
wall times or estimates including the plugin's missing-duration fallback.

| selection | group | tests | recorded seconds | missing durations |
|---|---:|---:|---:|---:|
| weekly | 1 | 8000 | 6731.15030291301552742902 | 457 |
| weekly | 2 | 3140 | 7463.05064197900535506421 | 0 |
| weekly | 3 | 4137 | 7482.21194214600592788110 | 0 |
| weekly | 4 | 3709 | 7288.09371869699608202306 | 99 |
| weekly | 5 | 1955 | 7337.2141672090023410625 | 8 |
| fast | 1 | 2655 | 1785.65200252400071172775 | 0 |
| fast | 2 | 3198 | 1782.87584452501298138727 | 0 |
| fast | 3 | 1694 | 1782.6513163670030354908 | 0 |
| fast | 4 | 1533 | 1792.0887382479979263766 | 0 |
| fast | 5 | 1232 | 1782.6028205570164612784 | 0 |
| fast | 6 | 3033 | 1789.65390449500426981420 | 0 |
| fast | 7 | 1704 | 1677.8309318019979743143 | 99 |
| fast | 8 | 1023 | 1788.81842801800211609046 | 0 |
| fast | 9 | 1721 | 1784.9404569499945010480 | 0 |
| fast | 10 | 1213 | 1806.4172510680017829841 | 0 |
| fast | 11 | 916 | 1952.2808278520025411029 | 0 |
| fast | 12 | 258 | 1546.0288840080006616371 | 8 |
| contract | 1 | 227 | 315.2948128149998165981 | 0 |
| contract | 2 | 997 | 311.6519956050009904505 | 0 |
| contract | 3 | 143 | 335.7669219510013342763 | 0 |
| contract | 4 | 408 | 312.7734711170010443642 | 0 |
| contract | 5 | 821 | 311.90673305999763415965 | 0 |
| contract | 6 | 92 | 311.234374045999855950 | 0 |
| contract | 7 | 503 | 322.45159430500303044637 | 0 |
| contract | 8 | 480 | 311.4951441430017946798 | 0 |
| contract | 9 | 259 | 312.6788554160000898642 | 0 |
| contract | 10 | 375 | 312.2940423989990675828 | 0 |
| contract | 11 | 1431 | 313.2927033870084870616 | 0 |
| contract | 12 | 316 | 263.0001858480020466514 | 0 |

The four slab-order nodeids are all in weekly group 3:

| nodeid | recorded seconds | weekly group |
|---|---:|---:|
| `tests/unit/nonuniform/test_dual_volume_interface_order.py::test_slab_three_mesh_orders[eps4]` | 593.0778320900001 | 3 |
| `tests/unit/nonuniform/test_dual_volume_interface_order.py::test_slab_three_mesh_orders[lossy]` | 584.498005817 | 3 |
| `tests/unit/nonuniform/test_dual_volume_interface_order.py::test_slab_three_mesh_orders[debye]` | 870.602081947 | 3 |
| `tests/unit/nonuniform/test_dual_volume_interface_order.py::test_slab_three_mesh_orders[lorentz]` | 862.1636578080002 | 3 |

### Previous balance, 2026-10-07

Recomputed on the `bf9af229` tree with the fast-last file from run 37444295688
and its 1417.9070415 s log correction for `test_ringdowns_are_settled`,
using pytest-split 0.11.0's own `pytest --collect-only -q -p no:cacheprovider --splits N --group k`
for all 23 groups (`duration_based_chunks`): fast 17427 tests / 6 and 12 groups,
slow 18129 tests / 5 groups. Every group's ordered nodeids and duration match the
plugin algorithm applied to the unsplit collection; estimates agree to their displayed 0.01 s.
Python 3.11 / JAX 0.10.2 CPU; fast includes trimesh 5.1.1,
slow excludes the CAD extra, and Plotly is absent. Fast uses
`not gpu and not slow and not slow_physics and not docs_consistency`; slow uses
`not gpu and not highmem and not docs_consistency` with `RFX_S0_FULL=1`, matching
validation.yml. There are 149 fast and 590 slow nodeids without stored measurements;
the plugin prices them at the mean of the measured tests in that selection.
Without `RFX_S0_FULL=1`, the slow selection collects 17688 tests; the flag adds
441 path-equivalence cases. With these optional-dependency settings the local
selections overlap on 17415 ids; the 11 CAD tests are collected only in fast,
in addition to its one highmem test.

For the 2026-10-01 refresh, the reviewer found that slow-last pricing gave fast groups 3 and 4 predictions near
40 min but fast-artifact prices near 52.5 min; slow jobs 3 and 4 had fitted speed
factors 0.74 / 0.76 versus 0.93–1.20 for the other nine jobs. On runs 36739385830 and
36766208720, fast-artifact pricing matched old-split group 3 within 1 min in both runs;
across all 12 jobs the observed/price ratio was 0.69–1.22. The table below uses this file's fast-last prices in both lanes; these
are plugin estimates, including mean-duration fallback, not observed CI wall times.

| lane | group | fast-last minutes |
|---|---|---:|
| fast (6-way) | 1 | 56.10 |
| fast (6-way) | 2 | 56.11 |
| fast (6-way) | 3 | 56.29 |
| fast (6-way) | 4 | 56.10 |
| fast (6-way) | 5 | 56.21 |
| fast (6-way) | 6 | 54.93 |
| fast (6-way) | max | 56.29 |
| fast (12-way) | 1 | 28.80 |
| fast (12-way) | 2 | 28.22 |
| fast (12-way) | 3 | 28.22 |
| fast (12-way) | 4 | 28.35 |
| fast (12-way) | 5 | 28.05 |
| fast (12-way) | 6 | 28.09 |
| fast (12-way) | 7 | 28.00 |
| fast (12-way) | 8 | 28.11 |
| fast (12-way) | 9 | 28.23 |
| fast (12-way) | 10 | 28.01 |
| fast (12-way) | 11 | 31.63 |
| fast (12-way) | 12 | 22.04 |
| fast (12-way) | max | 31.63 |
| slow (5-way) | 1 | 117.12 |
| slow (5-way) | 2 | 117.11 |
| slow (5-way) | 3 | 117.80 |
| slow (5-way) | 4 | 119.63 |
| slow (5-way) | 5 | 113.90 |
| slow (5-way) | max | 119.63 |

Since 2026-10-06 (PI decision), the fast lane uses a 12-way split; historical measurements below retain their original split counts.

Earlier balance records follow.

Every number here is anchored, because both of its inputs move. The collection grows with the
tree: measured at c0ee0a7d the fast lane's selection holds 10472 tests and the slow lane's 10683,
where the same measurement three days earlier gave 10224 and 10435. And a shard time is one
reading of a noisy runner. So the simulation below is quoted at one commit, and the lane is
quoted as a range over repeated runs.

Simulated at `cae7d6bf` with `pytest --collect-only --splits N --group k` (the plugin's own
`duration_based_chunks` estimate), with the four `--ignore` flags both workflows pass:

- fast (`--splits 6`), this file: 35.3 / 35.5 / 35.3 / 35.3 / 35.3 / 35.0 min, critical path 35.5.
  12686 tests collected.
- slow (`--splits 5`, `-m "not gpu and not highmem and not docs_consistency"`), this file:
  96.2 / 96.2 / 96.3 / 96.3 / 96.0 min, critical path 96.3 against the 180-minute job limit.
  12929 tests collected. The file this replaces
  simulates 106.2 / 108.6 / 106.9 / 107.6 / 101.7 min at the same commit; on 2026-09-28 (run
  36377113298, `a3490391`) its shards 4 and 5 ran past 120 min.

Earlier readings, kept for the record:

- At `2bad0f51`, the file of 2026-09-21: fast 21.8 / 21.3 / 19.8 / 19.8 / 20.9 / 14.3 min (8646
  tests); slow (`-m "not gpu and not highmem"`) 64.9 / 64.8 / 71.1 / 65.3 / 57.7 min (8859 tests),
  against the 120-minute limit of the time. Before that file the same lane ran 33 / 113 / 52 / 71 /
  70 min on 2026-09-21 (run 35614412972), the 113 being the shard that held the 2247-second test at
  its old 202-second price.

- slow (`--splits 4`, `-m "not gpu and not highmem"`), the file of 2026-09-21: 67.4 / 83.3 / 85.6 /
  96.6 min against the 120-minute cap.

The shard readings below are of the file of 2026-09-21 ("this file" there) and the one before it;
this file's own lane times have not been read yet.

Then the reading step 5 calls the check, which is the lane itself. This file, five runs of this
pull request's fast lane (the tree grew between runs, so the split is not identical across
columns; compare down a column, not across a row):

| group | run 1 | run 2 | run 3 | run 4 | run 5 |
|---|---|---|---|---|---|
| 1 | 20.3 | 27.2 | 20.0 | 27.0 | 15.7 |
| 2 | 31.4 | 28.7 | 28.6 | 31.0 | 23.6 |
| 3 | 29.7 | 29.6 | 23.2 | 22.3 | 28.5 |
| 4 | 28.2 | 30.6 | 19.0 | 19.4 | 30.4 |
| 5 | 29.0 | 29.4 | 15.2 | 26.4 | 28.3 |
| 6 | 11.1 | 20.0 | 22.5 | 22.2 | 22.2 |
| **critical** | **31.4** | **30.6** | **28.6** | **31.0** | **30.4** |
| spread | 2.8x | 1.5x | 1.9x | 1.6x | 1.9x |

The file this replaces, measured the same way: four runs of the fast lane on three branches that
do not touch `.test_durations` (pull requests 1108, 1105 and 1107).

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| group 1 | 5.3 | 4.8 | 4.5 | 4.5 |
| **critical** | **42.4** | **41.6** | **37.7** | **35.8** |
| spread | 8.0x | 8.7x | 8.3x | 7.9x |

Group 1 gets its own row because two claims rest on it: that this file's imbalance is systematic
rather than noise, and that the retired floor's shard-1 prediction of 19.5 min was wrong by a
factor. Both need the four readings to be 4.5-5.3 and not a single one of them.

**The result: critical path 28.6-31.4 min against 35.8-42.4.** Nothing finer is supportable.
Within either block the split is the same file's, so most of the movement is runner noise, and it
reaches 14 minutes on group 5 and 11 on group 1.

The two spreads are not the same kind of number, which is why both are ranges and only one of
them means anything. The old file reads 7.9-8.7x across four runs because its shard 1 lands at
4.5-5.3 min every time: that imbalance is systematic and survives the noise, and it is what this
refresh removes. This file reads 1.5-2.8x across five runs of an almost unchanged split, so its
spread IS the noise -- no run of it says anything the next run does not contradict, and neither
does any claim about which group ran over or under its prediction.

So quote the critical path. Do not quote the simulated 1.2x, which is finer than this lane can
resolve, and do not read a per-group difference on this file as a mechanism. Re-read the critical
path after the next regen, and re-measure the collection while you are there.
