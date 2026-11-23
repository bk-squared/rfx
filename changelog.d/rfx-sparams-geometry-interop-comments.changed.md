### Changed — 15 modules of `rfx.sparams`, `rfx.geometry` and `rfx.interop` have no code comments; some of their conventions are now docstring sentences (#PR)

- The S-parameter stage modules for microstrip, mixed and waveguide ports, three `rfx.geometry` modules and the
  `rfx.interop` design codec and openEMS emitter have no comments left other than tool directives. Some conventions the
  code cannot show now stand as one docstring sentence on the function, class or module they belong to, at most one
  each, where `help()` on that object shows them. Two commented literals became private named constants, and the other comments were
  removed.
- No check, message or computation changes.
