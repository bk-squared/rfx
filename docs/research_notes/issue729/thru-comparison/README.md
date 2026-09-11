# Current-main versus source-span correction: actual two-drive thru

The unchanged `test_msl_thru_line_passive_gate` fixture ran sequentially in
both variants. Both exit successfully and satisfy its original gates. Raw
S is saved before passivity projection; the comparison reads those raw
matrices. Both drives settle below -100dB, and grid, port definitions,
record parameters and frequency arrays agree between the runs.

The corrected candidate was initially launched from the working tree,
although its temporary output directory was mistakenly named baseline.
Its receipt immediately exposed commit525b8001 with source_dirty=true. It
was never accepted as baseline evidence: source-correction.json records the
actual role, the source diff (lossless source-working.diff.gz) and hashes were preserved, and all three
relevant source hashes were checked unchanged after the run and against
subsequent commit6b9275ed. The directory was renamed to candidate afterward.
The original receipt is preserved rather than rewritten as a clean run.

The baseline is a separate clean f85ed767 worktree. The checked driver
asserts expected commit and clean status before invoking the fixture. It
ran after the candidate completed. The common driver and every result/log
are retained here; no other FDTD run was started in parallel.

Reproduction: from a clean checkout of the desired revision, run driver.py
with --expected-sha <full commit> --out <fresh output directory>, setting
JAX_PLATFORMS=cpu, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. The script
limits its own CPU affinity to eight available cores. The candidate source
can now be reproduced from6b9275ed; that is a code-equivalence receipt, not
a claim that the original WIP run executed that later git commit.

The pair is evidence for this established thru fixture. It does not replace
NU, source-clearing, AD, off-alignment or coax/mixed consumer checks, and does
not close #729 by itself.
