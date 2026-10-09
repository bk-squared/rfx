# .test_durations — provenance

Regenerated 2026-10-10 at `3b62b21bef6a16f3c7aae8274a0f3798e6ff3a0d` from
`regen-durations` run [37983083757](https://github.com/bk-squared/rfx/actions/runs/37983083757),
measured 2026-10-09 UTC (2026-10-09/10 Asia/Seoul) on `main` (`4a42dbc9`),
`ubuntu-latest` with Python 3.11 / JAX 0.10.2.
**20403 entries in the file**, of which **14 of them carried unchanged** from the
17565-entry file this replaces. All 17 jobs succeeded. The slow (5) pytest summary
nevertheless reports one failed test:
`tests/unit/sparams/test_msl_sheet_threading.py::test_o3_mean_s21_drift_lock`;
the workflow's pytest commands use `|| true`.
Slow artifacts are merged first, then fast artifacts; the last input wins for
overlapping nodeids. The artifacts contain 20389 distinct nodeids; 0 measured
entries and 103 old entries are not collected at HEAD and are dropped.
The 14 carried entries are collected highmem tests outside both GitHub selections;
no new a6000 measurements were supplied.

At the tree above, fast, weekly and highmem collect 20180, 20941 and 15 tests
respectively (20967 distinct), with 107, 564 and 0 missing durations
respectively (564 distinct unmeasured nodeids). The not-code contract selection
collects 6052 tests, with 0 missing durations (2283 were missing in the replaced file).
Collection used Python 3.11 / JAX 0.10.2 CPU, pytest 9.1.1 and pytest-split 0.11.0.
Fast collection includes trimesh 5.1.1 supplied in the temporary directory;
weekly and highmem collection have no CAD extra. Plotly is absent.
Weekly collection sets `RFX_S0_FULL=1`, matching validation.yml.
Counts use `python -m pytest --collect-only -q -p no:cacheprovider`:
fast uses default addopts; weekly adds
`-m "not gpu and not highmem and not docs_consistency"`; highmem adds
`-m "highmem and not gpu"`; contracts adds
`tests/contracts -o addopts="" -m "not gpu and not docs_consistency"`.
The full inventory uses `RFX_S0_FULL=1 -o addopts=""` and includes the CAD tests
collected by fast. The contract checks the entry count and carried-count bounds
dynamically; it has no fixed entry or carried count to update.

All 17 job logs were checked: **0 entries were re-priced from job logs**.
All 40450 test result lines were timed from the previous result line, with the
first test timed from the pytest session-start line. No gap exceeds the merged
value by more than 600 s. The formerly corrected
`test_ringdowns_are_settled` now retains its artifact value, 310.3921270119997 s;
its slow (2) log gap is 310.3928750 s, from
2026-10-09T20:19:41.1379318Z to 2026-10-09T20:24:51.5308068Z.

The four entries once set by hand retain raw runner measurements:

| nodeid | recorded seconds |
|---|---:|
| `tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_end_to_end_matches_historical_base` | 1014.1397751879999 |
| `tests/locks/test_coax_chain_battery_drift.py::test_the_coarsest_mesh_still_solves_to_its_stored_s[bead]` | 139.71913598599997 |
| `tests/locks/test_coax_chain_battery_drift.py::test_the_coarsest_mesh_still_solves_to_its_stored_s[thru]` | 142.896389494 |
| `tests/oracle/test_coax_open_end_settles.py::test_the_open_end_is_passive_and_settled` | 52.19136173999999 |

All 20389 refreshed entries retain raw artifact measurements; the 14 highmem-only
entries are carried. **No floor is added.**

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
input's value for an id measured twice; `slow (1)`–`(5)` went first, then
`fast (1)`–`(12)`, in numeric order within each lane.

| source | entries |
|---|---:|
| slow artifacts, for ids no fast source priced | 316 |
| fast artifacts, including overlapping ids | 20073 |
| log-derived corrections | 0 |
| carried unchanged from the replaced file | 14 |

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
runner measurement, and it is: 42.09557991500037 s from `fast (9)` of run 37983083757.
It is the one highmem-marked entry the regeneration refreshed; the other 14 are
carried. When regenerating, do not overwrite this one from the a6000 map.

## Regenerating

1. Dispatch `.github/workflows/regen-durations.yml` and download the artifacts.
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
