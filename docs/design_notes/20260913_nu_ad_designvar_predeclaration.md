# NU AD-Q second design: withdrawal and re-declaration

## Withdrawal before measurement

This note supersedes the AD6 declaration committed as `73316579`, which
remains in history. **The withdrawal happens before any measurement against
that declaration.** At withdrawal, HEAD is `733165791f417cd57797b767258a3ae0c28eb6d5`;
`git log 73316579..HEAD --oneline` is empty and a repository-wide search for
`"ad6"` JSON keys finds none. The worktree already contains uncommitted AD6
replay tests; those are not measurement results and are preserved.

The PI rejected the following, quoted from the replacement request:

> “a flat 15 % gate is not a quantitative claim, and neither is the existing convention”

> “the directions must be the DESIGN VARIABLES, not abstract vectors”

Accordingly, tied indicators, all-ones and seeded random directions are
withdrawn. The replacement uses layer thickness and permittivity directions
supplied by a generalized E4 fixed-topology stackup map. The primary claim
will be Taylor-remainder order (R0 slope 0.9–1.1; R1 slope 1.8–2.2), with
central FD assessed against three times its own Richardson/float32 error
estimate. A reference bar exceeding 15 % is inconclusive. The old 5 %
dominance / 15 % agreement rule remains a labelled legacy smoke figure;
its constants elsewhere in the repository will not change.

This first commit records only the withdrawal and replacement intent.
The complete map, ladders, data-dependent window algorithm, error-bar
arithmetic, and one-attempt arms will be frozen in a subsequent commit
before any new measurement. No production `rfx/` behavior change is authorized.

## Results

No replacement measurement has run at the time of this withdrawal commit.
