# .test_durations — provenance

Regenerated 2026-09-08 at main 1658f12d. Every entry is a measurement plus a 0.3 s per-test
overhead floor: pytest-split stores no setup or teardown below its own threshold, and on the
runners the real cost of a sub-millisecond test is 0.2–0.4 s. Without the floor the split
piled thousands of contract tests into one shard (30 min against 18 for the rest).

Sources (7594 entries, exactly the union of the fast, slow and highmem selections at
that commit — nothing carried, nothing estimated):

- 7582 measured on the GitHub runners by `regen-durations` run 34158542674, the first pass in
  which every shard finished. Four earlier passes lost slow shard 2 to a runner kill (exit 143):
  two >16 GB oracle tests sat in its tail and are now marked `highmem` (pull requests 941 and 943),
  which is what let the shard complete.
- 12 measured on the a6000 lane (VESSL run 369367259335, `-m "highmem and not gpu"`,
  13 passed in 254 s). These tests run only there — the fast lane deselects `slow`, the slow lane
  deselects `highmem` — so a GPU-measured time is the time of the lane that actually runs them and
  never enters a GitHub split.

Regenerate: dispatch `.github/workflows/regen-durations.yml`, download the artifacts, then
`python scripts/ci/merge_test_durations.py .test_durations <shard json files...>`, add the 0.3 s
floor, and take the highmem entries from an a6000 run of that selection.

Measured shard balance on the pull request that introduced this file: fast 6 shards within one
minute of each other; slow 4 shards 63–92 min against the 120-minute cap.
