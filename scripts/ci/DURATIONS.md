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
to group 1, measured three ways:

| | shard 1 |
|---|---|
| predicted from fresh measurements, no floor | 4.2 min |
| predicted with 0.3 s added to each of its 3067 tests | 19.5 min |
| **observed**, `fast-suite (1)` on pull request 1108 | **5 min 19 s** |

The floored model is 3.7x over the observation; the unfloored model is 1.1 min under it. Whatever
per-test overhead the 2026-09-08 runner had, the runner this lane uses now does not have it at
anything like 0.195 s, and adding 0.3 s to 10625 entries adds 53 minutes of fiction spread evenly
across the lane. Evenly is the problem: a constant per test does not model a real cost, it just
pulls every shard toward equal test COUNTS and away from equal time. On this file it raises the
predicted critical path from 27.1 min to 34.5 min while the observed lane runs at the unfloored
prediction.

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
6. Update the entry count and the carried count in the header above.
   `tests/contracts/test_test_durations_provenance.py` compares both against the file.

## Balance

Simulated on this file with `pytest_split.algorithms.Algorithms['duration_based_chunks']` over the
real collection, with the four `--ignore` flags both workflows pass:

- fast (`--splits 6`, 10224 tests): 23.2 / 25.7 / 25.7 / 26.0 / 26.2 / 27.1 min — spread 3.9 min,
  critical path 27.1 min.
- slow (`--splits 4`, `-m "not gpu and not highmem"`, 10435 tests): 69.9 / 81.4 / 83.1 / 90.8 min
  against the 120-minute cap.

Then the reading that step 5 calls the check. This pull request's fast lane ran three times on
this file, which turned out to be the useful part:

| group | tests | predicted | run 1 | run 2 | run 3 |
|---|---|---|---|---|---|
| 1 | 4258 | 26.0 min | 20.3 | 27.2 | 20.0 |
| 2 | 1808 | 25.7 min | 31.4 | 28.7 | 28.6 |
| 3 | 1438 | 25.7 min | 29.7 | 29.6 | 23.2 |
| 4 | 1459 | 27.1 min | 28.2 | 30.6 | 19.0 |
| 5 | 898 | 26.2 min | 29.0 | 29.4 | 15.2 |
| 6 | 363 | 23.2 min | 11.1 | 20.0 | 22.5 |
| | | **critical** | **31.4** | **30.6** | **28.6** |
| | | spread | 2.8x | 1.5x | 1.9x |

The lane the committed file gives, observed on pull request 1108:
5.3 / 33.6 / 34.5 / 24.3 / 34.6 / 42.4 min, critical path 42.4, spread 8.0x.

**The reportable result is the critical path: 28.6 to 31.4 min, against 42.4.** Nothing finer is
supportable from this lane. All three runs used the same split on the same file, so every
difference between them is runner noise, and it reaches 14 minutes on group 5 and 9 on group 1
and group 6. A spread from one run is not a property of the file -- 2.8x, 1.5x and 1.9x are one
configuration read three times -- and neither is any statement about which group runs over or
under its prediction: each of those held in one run and reversed in the next.

The simulated 1.2x is not wrong, it is just finer than this lane can resolve. Do not quote it, do
not quote a single run's spread, and do not read a per-group difference as a mechanism. Re-read
the critical path after the next regen. (Each push produces another reading; these three are
where the range stopped moving, not where the runs stopped.)

For contrast, the file this replaces predicts 35.3 / 36.5 / 36.5 / 36.5 / 36.5 / 37.6 min on the
fast lane — a flat-looking simulation that the lane does not obey, because 3077 of the 10224
collected tests are absent from it and pytest-split charges every absent test the average of the
ones it knows. What that lane actually did on pull request 1108, shards 1 through 6:
5m19s / 33m35s / 34m32s / 24m20s / 34m38s / 42m23s — 8.0x apart, 42 min of wall clock. Reading the
merged pull request's own shard times is step 5 for exactly this reason.
