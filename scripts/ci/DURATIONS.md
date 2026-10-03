# .test_durations — provenance

Regenerated 2026-10-01 on `main` (`e6151107`) from `regen-durations` run
[36752442156](https://github.com/bk-squared/rfx/actions/runs/36752442156), measured on
`ubuntu-latest` with Python 3.11 / JAX 0.10.2. **15500 entries in the
file**, of which **14 of them carried unchanged** from the 12677-entry file this
replaces. All eleven jobs succeeded. Slow artifacts are merged first, then fast artifacts;
the last input wins for overlapping nodeids. The artifacts contain 15486 distinct
nodeids; 0 measured entries and 41 old entries are not collected at HEAD and are
dropped. The 14 carried entries are collected highmem tests outside both GitHub selections;
no new a6000 measurements were supplied. At the `1c3617d7` tree, the fast, slow and highmem selections
collect 15231, 15474 and 15 tests respectively (15500 distinct), with no
missing durations in any selection. Collection used the specified local Python 3.11 / JAX 0.10.2
CPU environment, with the locally available trimesh package for the fast/highmem collections;
slow collection has no CAD extra, matching validation.yml. Plotly is absent and its optional
module is skipped. These counts come from the artifacts, the replaced file and HEAD collection;
the contract below checks the entry count and carried-count bounds. All eleven job logs were
checked: **1 entry was re-priced from its job log** because pytest-split 0.11.0 drops setup or
teardown phases above its 600 s `STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD`.
`test_ringdowns_are_settled` is 1448.8538443 s from the log, replacing the artifact's
0.000938106 s (the replaced file held 1514.934755 s). No floor is added.

The four entries once set by hand are runner measurements, from the slow
artifacts except the open-end test, which now uses its fast artifact:
`tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_end_to_end_matches_historical_base`
2006.5 s (was 2218.1 s), the two coaxial chain-battery drift locks
`tests/locks/test_coax_chain_battery_drift.py::test_the_coarsest_mesh_still_solves_to_its_stored_s[bead]`
and `[thru]` 176.9 s and 178.5 s (were 171.6 s and 170.6 s), and
`tests/oracle/test_coax_open_end_settles.py::test_the_open_end_is_passive_and_settled` 61.4 s (was
57.1 s).

Of the refreshed entries, 15485 retain raw artifact measurements and 1 uses log-derived pricing;
the 14 highmem-only entries are carried.
**No floor is added** — see below.

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

So the floor is gone, and step 4 of Regenerating below says not to re-add it. Two consequences to
keep in mind:

- The 14 carried entries retain their previous measurements. No floor is added to them or to
  the refreshed artifact values.

An earlier version of this note blamed a pytest-split threshold that supposedly drops short setup
and teardown readings. That was backwards — `STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD` is
600 s and the plugin discards readings ABOVE it. The floor never stood on that claim, and it no
longer stands on the 2026-09-08 measurement either.

## Sources

Where each entry's value comes from. `merge_test_durations.py` takes the last input's value for an
id measured twice; `slow (1)`–`(5)` went first, then `fast (1)`–`(6)`.

| source | entries |
|---|---|
| `slow (1)`–`(5)` artifacts, for ids no fast source priced, excluding the log-corrected entry | 254 |
| `slow (2)` job log, correcting a dropped setup phase | 1 |
| `fast (1)`–`(6)` artifacts, including overlapping ids | 15231 |
| carried unchanged from the replaced file | 14 |

- The artifacts are `pytest --store-durations` output of run 36752442156 on `ubuntu-latest`, the
  runner class the lanes use, with Python 3.11 / JAX 0.10.2.
- All 30705 test result lines in the eleven jobs of run 36752442156 were checked. Each test's
  wall time is the timestamp gap from the previous result line in that job; the first test is
  timed from the pytest session-start line. Exactly one gap exceeds its new stored value by
  more than 600 s: `tests/locks/test_patch_edgefed_resonance_harminv.py::test_ringdowns_are_settled`
  in `slow (2)`, 1448.8538443 s versus 0.0009381059999213903 s stored (1514.934755 s in the
  replaced file). Its duration now uses that gap. Module sums: old 1515.9735370 s; slow-artifact raw 1.8687104 s;
  fast-last raw 1.9359451 s; log-corrected fast-last 1450.7888513 s.
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
- Recorded total: 29019.1 s (8.06 h), against 28509.1 s (7.92 h) for the file this replaces.

## The one highmem test the fast lane does run

`tests/unit/ports/test_msl_source_fixture_static.py::test_auto_eps_msl_gradient_matches_fd_mini_referee`
carries `highmem` and nothing else, so pyproject's default `-m 'not gpu and not slow and not
slow_physics'` does not deselect it and it runs in a fast shard. Its entry must therefore be a
runner measurement, and it is: 47.77 s from `fast (4)` of run 36752442156 (it was 61.22 s from run
36394962310; the a6000 said 28.9 s). It is the one highmem-marked entry the regen refreshed; the
other 14 carried. When regenerating, do not overwrite this one from the a6000 map.

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

Recomputed on the `1c3617d7` tree with the fast-last file and log correction above,
using pytest-split 0.11.0's own `pytest --collect-only --splits N --group k` for all
11 groups (`duration_based_chunks`): fast 15231 tests / 6 groups, slow 15474 tests /
5 groups. Every group's ordered nodeids and summed durations match the reviewer's
`sim.py` cross-check exactly; plugin estimates agree to their displayed 0.01 s.
Python 3.11 / JAX 0.10.2 CPU; fast includes the locally available trimesh package,
slow excludes the CAD extra, and Plotly is absent. Fast uses
`not gpu and not slow and not slow_physics and not docs_consistency`; slow uses
`not gpu and not highmem and not docs_consistency`, matching the workflows.
The file has no missing entries in either selection. With these optional-dependency
settings the local selections overlap on 15219 ids; the 11 CAD tests are collected
only in fast, in addition to its one highmem test.

The reviewer found that slow-last pricing gave fast groups 3 and 4 predictions near
40 min but fast-artifact prices near 52.5 min; slow jobs 3 and 4 had fitted speed
factors 0.74 / 0.76 versus 0.93–1.20 for the other nine jobs. On runs 36739385830 and
36766208720, fast-artifact pricing matched old-split group 3 within 1 min in both runs;
across all 12 jobs the observed/price ratio was 0.69–1.22. The table below uses this file's fast-last prices in both lanes; these
are duration sums, not observed CI wall times.

| lane | group | fast-last minutes |
|---|---|---:|
| fast | 1 | 45.49 |
| fast | 2 | 45.50 |
| fast | 3 | 46.10 |
| fast | 4 | 45.58 |
| fast | 5 | 46.12 |
| fast | 6 | 44.12 |
| fast | max | 46.12 |
| slow | 1 | 95.30 |
| slow | 2 | 95.32 |
| slow | 3 | 95.30 |
| slow | 4 | 95.59 |
| slow | 5 | 94.86 |
| slow | max | 95.59 |

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
