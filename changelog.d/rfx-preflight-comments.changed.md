### Changed — the `rfx.preflight` package has no code comments; some of its conventions are now docstring sentences (#PR)

- The 21 `rfx.preflight` modules that carried comments, among them `msl`, `waveguide`, `mesh`, `absorber` and `ports`,
  have no comments left other than tool directives. Some conventions the code cannot show now stand as one docstring
  sentence on the function, class or module they belong to, at most one each, so `help()` shows them. Three commented
  literals became private named constants, and the other comments were removed.
- No check, message or computation changes.
