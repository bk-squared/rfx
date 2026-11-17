### Changed — `rfx.subgridding` states its conventions in docstrings instead of code comments (#PR)

- `rfx.subgridding` has no code comments left. Some of the sign, staggering, unit and source conventions they
  held now sit in the docstrings of the functions and classes they describe, at most one sentence each, so
  `help()` shows them; the other comments were removed.
- No computation changes: every result of the subgridding lane is the same as before.
