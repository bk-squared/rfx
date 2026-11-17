### Changed — `rfx.subgridding` states its conventions in docstrings instead of code comments (#PR)

- The sign, staggering, norm, penalty and admission conventions that `rfx.subgridding` kept in comments now sit
  in the docstrings of the functions and classes they describe, so `help()` shows them.
- No computation changes: every result of the subgridding lane is the same as before.
