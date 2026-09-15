# .test_durations — provenance

Regenerated 2026-09-08 at main 1658f12d. 7594 entries, exactly the union of the fast, slow and
highmem selections at that commit. Every entry is a measurement plus a 0.3 s per-test floor.
Nothing is estimated and nothing is carried from an older file.

## The 0.3 s floor, and why it is there

pytest-split records the sum of each test's own setup/call/teardown report durations. A shard's
wall time is larger than the sum of those numbers: module imports, first-touch fixture
construction and the framework's own per-test cost are not in them, and they fall hardest on a
shard holding thousands of sub-millisecond tests. Measured, not assumed: on the first revision
of pull request 939 the fast lane's shard 1 held 3426 tests whose measured durations sum to
1149.7 s, and it spent 1817 s of pytest time — about 0.195 s per test unaccounted for. The floor
is 0.3 s, a round number above that measurement.

An earlier version of this note blamed a pytest-split threshold that supposedly drops short
setup and teardown readings. That is backwards: `STORE_DURATIONS_SETUP_AND_TEARDOWN_THRESHOLD`
is 600 s and the plugin discards readings ABOVE it. The floor stands on the measurement above,
not on that claim.

The floor changes the split itself, not only the predicted numbers: without it the algorithm
puts 3853 of 7389 fast tests in shard 1, with it 2721. Its falsifier is the observed shard
spread on the pull request that introduces the file.

## Sources

- 7582 entries measured on the GitHub runners by `regen-durations` run 34158542674, the first
  pass in which every shard finished. Four earlier passes lost slow shard 2 to a runner kill
  (exit 143); two >16 GB oracle tests sat in its tail and are now marked `highmem` (pull
  requests 941 and 943), which is what let the shard complete.
- 12 entries measured on the a6000 lane (VESSL run 369367259335, `-m "highmem and not gpu"`,
  13 passed in 254 s). These 12 carry `slow` or `slow_physics` as well as `highmem`, so the fast
  lane deselects them through pyproject's default `-m` expression and the weekly lane excludes
  them by marker. No GitHub split uses their time.

## The one highmem test the fast lane does run

`tests/unit/ports/test_msl_source_fixture_static.py::test_auto_eps_msl_gradient_matches_fd_mini_referee`
carries `highmem` and nothing else, so pyproject's default `-m 'not gpu and not slow and not
slow_physics'` does not deselect it and it runs in a fast shard. Its entry must therefore be a
runner measurement, and it is: 45.221316 s measured on the runner (the a6000 says 28.9 s), plus
the floor. When regenerating, do not overwrite this one from the a6000 map.

## Regenerating

1. Dispatch `.github/workflows/regen-durations.yml` and download the artifacts.
2. `python scripts/ci/merge_test_durations.py .test_durations <fast shards...> <slow shards...>`.
   Later inputs win on a nodeid measured twice, so pass the fast maps first and the slow maps
   last; the two selections overlap on 7388 nodeids and their measurements differ.
3. Take only the 12 GPU-only entries above from an a6000 run of the highmem selection. Leave the
   fast-lane highmem test on its runner value.
4. Add the 0.3 s floor once, to every entry.
5. Simulate both splits before committing (pytest-split's own `duration_based_chunks`), then read
   the pull request's own shard times afterwards — that reading is the check, not the simulation.

## Balance

Simulated on this file: fast 25.4 / 26.2 / 26.2 / 26.3 / 26.3 / 26.7 min (spread 1.3 min); slow
63.0 / 80.2 / 84.9 / 92.3 min against the 120-minute cap. Before pull request 939 the fast lane
ran 1 min on one shard and up to 35 min on another.
