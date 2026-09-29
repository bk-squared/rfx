# .test_durations — provenance

Regenerated 2026-09-28 on `main` (`58ac0d16`, assembled at `cae7d6bf`) from `regen-durations` run
[36394962310](https://github.com/bk-squared/rfx/actions/runs/36394962310). **12677 entries in the
file**, of which **13 of them carried unchanged** from the 8289-entry file this replaces. Ten of that
run's eleven jobs succeeded. `slow (5)` was cut at its 180-minute job limit with 2361 of its 2567
tests finished; pytest-split writes its durations file only when the session ends, so that shard
uploaded none, and its 2361 finished tests are priced from the job log instead (see Sources). The
206 it never reached are all in the fast selection too and take the fast shards' measurements. The
merge drops the 819 committed entries whose tests are not collected at `cae7d6bf`, and one measured
case that `cae7d6bf` renamed. Of the 12943 tests the three collections (fast, slow, highmem) select
on the pod at `cae7d6bf`, 76 have no entry: the 73 cases of
`tests/unit/runners/test_distributed_cpml_admission.py` and 3 of
`tests/locks/test_runner_split_bit_identity.py`, added by `cae7d6bf` after the run's commit;
pytest-split prices them at the file's mean. The counts here are computed from the artifacts, the
job log, the replaced file and those collection lists; the contract below checks the first two
against the file.

The four entries the previous file set by hand are runner measurements now, from the slow
artifacts: `tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_end_to_end_matches_historical_base`
2218.1 s (was 2247 s, read from a weekly run's clock), the two coaxial chain-battery drift locks
`tests/locks/test_coax_chain_battery_drift.py::test_the_coarsest_mesh_still_solves_to_its_stored_s[bead]`
and `[thru]` 171.6 s and 170.6 s (were 179.59 s and 160.90 s from VESSL), and
`tests/oracle/test_coax_open_end_settles.py::test_the_open_end_is_passive_and_settled` 57.1 s (was
69.56 s from VESSL).

Every entry is a raw measurement or, for `slow (5)` and one `slow (2)` entry, a raw log-derived time.
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

- The 57 carried entries predate the 2026-09-18 floor retirement, so each may still hold the old
  additive 0.3 s (at most 17 s in total, 0.09 % of the file). No collected test that has an entry
  is priced from anything but a runner measurement or one of those 57.

An earlier version of this note blamed a pytest-split threshold that supposedly drops short setup
and teardown readings. That was backwards — `STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD` is
600 s and the plugin discards readings ABOVE it. The floor never stood on that claim, and it no
longer stands on the 2026-09-08 measurement either.

## Sources

Where each entry's value comes from. `merge_test_durations.py` takes the last input's value for an
id measured twice; the fast maps went first, then `slow (1)`–`(4)`, then the `slow (5)` log map.

| source | entries |
|---|---|
| `slow (1)`–`(4)` artifacts | 10274 |
| `slow (5)` job log, log-derived | 2361 |
| `slow (2)` job log, log-derived (a setup over 600 s, below) | 1 |
| `fast (1)`–`(6)` artifacts, for ids no slow source priced | 218 |
| carried unchanged from the replaced file | 13 |

- The artifacts are `pytest --store-durations` output of run 36394962310 on `ubuntu-latest`, the
  runner class the lanes use.
- **The `slow (5)` entries are log-derived.** With `-v` each test's result line is written when
  the test ends, and the job log stamps every line. An entry is the gap between a test's result
  line and the one before it; the first test is timed from the `collected` line. The gap holds
  setup, call, teardown and pytest's own work between tests. 2233 of the 2361 were also measured
  by a fast shard. Over those the log-derived sum is the lower one: 4019.2 s against 4100.7 s
  for the fast artifacts (-1.99 %).
- **One `slow (2)` entry is log-derived too.** `--store-durations` does not store a setup or
  teardown phase longer than 600 s (`STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD` in
  pytest-split's plugin). The module fixture of
  `tests/locks/test_patch_edgefed_resonance_harminv.py::test_ringdowns_are_settled` spends its
  time in setup, so the artifact held 0.0012 s; the entry is the 1514.93 s between its result line
  and the one before it in the `slow (2)` log (08:09:42.28 to 08:34:57.22). It was the only test in
  the ten artifacts whose log time exceeds its artifact value by more than 600 s. Before the fix,
  `slow (2)`'s artifact summed to 3590.9 s against a 5127.3 s pytest session; every other shard
  summed to 97.1-99.5 % of its session, and `fast (1)`, the shortest, to 91.6 % (339 s of 370 s).
- The 13 carried are the highmem-marked tests that also carry `slow` or `slow_physics`, so no
  GitHub selection runs them and no GitHub split uses their time. 12 are the a6000 measurements of
  2026-09-16 (VESSL run 369367259335, `-m "highmem and not gpu"`), carried from the file before.
  The 13th, `tests/unit/autodiff/test_coax_two_port_ad.py::test_compute_coaxial_two_port_ad_grad_finite_and_fd_consistent`,
  became highmem in pull request 1358; its 67.46 s is `slow (2)` of run 35613805791
  (2026-09-21), measured before that.
- Recorded total: 29037.0 s (8.07 h), against 18804.0 s (5.22 h) for the file this replaces.

## The one highmem test the fast lane does run

`tests/unit/ports/test_msl_source_fixture_static.py::test_auto_eps_msl_gradient_matches_fd_mini_referee`
carries `highmem` and nothing else, so pyproject's default `-m 'not gpu and not slow and not
slow_physics'` does not deselect it and it runs in a fast shard. Its entry must therefore be a
runner measurement, and it is: 61.22 s from `fast (5)` of run 36394962310 (it was 36.25 s from run
35613805791; the a6000 said 28.9 s). It is the one highmem-marked entry the regen refreshed; the
other 13 carried. When regenerating, do not overwrite this one from the a6000 map.

## Regenerating

1. Dispatch `.github/workflows/regen-durations.yml` and download the artifacts.
2. `python scripts/ci/merge_test_durations.py .test_durations <fast shards...> <slow shards...>`.
   Later inputs win on a nodeid measured twice, so pass the fast maps first and the slow maps
   last; the two selections overlap and their measurements differ. A shard cut before the session
   ends uploads nothing; price what it finished from its job log as in Sources, pass that map
   last, and say so in the header. Then compare each input's sum, which the script prints, with
   the pytest session time in that shard's job log (the `= ... in Ns =` line). Look into any shard
   more than about 5 % short, or short by more than a minute: a setup or teardown over 600 s is
   dropped by `--store-durations` and is the known cause (see Sources).
3. Take only the 13 GPU-only entries above (highmem and slow or slow_physics) from an a6000 run of
   the highmem selection. Leave the fast-lane highmem test on its runner value.
4. **Do not add a floor.** Commit the raw measurements. The additive 0.3 s this step used to
   require was retired on 2026-09-18 for the reason above; if a future lane really does carry
   per-test overhead the reports miss, measure it on that lane first and write the measurement
   here before any constant goes back in.
5. Simulate both splits before committing (pytest-split's own `duration_based_chunks`), then read
   the pull request's own shard times afterwards — that reading is the check, not the simulation.
   Re-measure the collection while you are there and quote it with the commit you measured it at:
   it grew 10224 -> 10472 on the fast lane in the three days this file's own refresh took, so a
   simulation quoted without one is unreproducible.
6. Update the entry count and the carried count in the header above.
   `tests/contracts/test_test_durations_provenance.py` compares both against the file.

## Balance

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
