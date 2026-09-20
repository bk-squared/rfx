# .test_durations — provenance

Regenerated 2026-09-16 on `main` from `regen-durations` run
[35107932217](https://github.com/bk-squared/rfx/actions/runs/35107932217). **10625 entries in the
file**, of which **297 of them carried unchanged** from the 7594-entry file this replaces. Nine of
that run's ten jobs succeeded; `slow (3)` was killed by the runner (the exit-143 class
`validation.yml` documents), so the merge seeded from the committed file and let the fresh
measurements overwrite what they cover — that seeding is what the 297 are.

Every entry is a raw measurement. **No floor is added** — see below.

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

- The 297 carried entries still have the old additive 0.3 s inside them, because nothing
  re-measured those tests. Measured breakdown: 65 are still collected (53 by the slow lane, 12 by
  the a6000 highmem lane) and 232 are nodeids no current selection collects at all — stale ids the
  seed dragged forward. pytest-split drops durations for uncollected ids, so the 232 cost nothing
  but bytes; the 65 are 0.3 s over, each, until a regen that completes re-measures them. Across the
  whole file 246 of 10625 entries are uncollected.
- `tests/contracts/test_test_durations_provenance.py` pins the retirement: it fails if the
  minimum entry ever climbs back above 0.3 s, which is what re-adding the floor would do.

An earlier version of this note blamed a pytest-split threshold that supposedly drops short setup
and teardown readings. That was backwards — `STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD` is
600 s and the plugin discards readings ABOVE it. The floor never stood on that claim, and it no
longer stands on the 2026-09-08 measurement either.

## Sources

- The fresh measurements come from the six `fast` shards and `slow (1)`, `(2)`, `(4)` of run
  35107932217, measured on `ubuntu-latest`, the same runner class the lanes use. `slow (3)` was
  killed; rather than drop the tests only that shard measures, `merge_test_durations.py` seeds
  from the committed file first and lets later inputs win.
- 12 entries measured on the a6000 lane (VESSL run 369367259335, `-m "highmem and not gpu"`,
  13 passed in 254 s) are among the 297 carried. These 12 carry `slow` or `slow_physics` as well
  as `highmem`, so the fast lane deselects them through pyproject's default `-m` expression and
  the weekly lane excludes them by marker. No GitHub split uses their time.
- Recorded total: 20143.7 s (5.60 h), against 19486.5 s (5.41 h) for the file this replaces.

## The one highmem test the fast lane does run

`tests/unit/ports/test_msl_source_fixture_static.py::test_auto_eps_msl_gradient_matches_fd_mini_referee`
carries `highmem` and nothing else, so pyproject's default `-m 'not gpu and not slow and not
slow_physics'` does not deselect it and it runs in a fast shard. Its entry must therefore be a
runner measurement, and it is: 35.295652 s from run 35107932217 (the a6000 says 28.9 s). It is the
one highmem-marked entry the regen refreshed; the other 12 carried. When regenerating, do not
overwrite this one from the a6000 map.

## Regenerating

1. Dispatch `.github/workflows/regen-durations.yml` and download the artifacts.
2. `python scripts/ci/merge_test_durations.py .test_durations <fast shards...> <slow shards...>`.
   Later inputs win on a nodeid measured twice, so pass the fast maps first and the slow maps
   last; the two selections overlap and their measurements differ.
3. Take only the 12 GPU-only entries above from an a6000 run of the highmem selection. Leave the
   fast-lane highmem test on its runner value.
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

Simulated at c0ee0a7d with `pytest_split.algorithms.Algorithms['duration_based_chunks']` over
that collection, with the four `--ignore` flags both workflows pass:

- fast (`--splits 6`), this file: 21.5 / 26.3 / 27.0 / 27.4 / 27.6 / 27.9 min, critical path
  27.9. 10158 of the 10472 collected tests have a recorded duration.
- fast (`--splits 6`), the file this replaces: 35.3 / 37.4 / 37.4 / 37.4 / 37.5 / 39.2 min,
  critical path 39.2 -- flat-looking, and the lane never obeys it, because 3325 of the same
  10472 are absent from that file and pytest-split charges every absent test the average of the
  ones it knows.
- slow (`--splits 4`, `-m "not gpu and not highmem"`), this file: 67.4 / 83.3 / 85.6 / 96.6 min
  against the 120-minute cap.

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
