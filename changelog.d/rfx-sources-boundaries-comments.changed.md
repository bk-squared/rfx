### Changed — 49 modules of `rfx.sources`, `rfx.boundaries`, `rfx.ringdown` and five small packages have no code comments; some of their conventions are now docstring sentences (#PR)

- 49 modules of `rfx.sources`, `rfx.boundaries`, `rfx.ringdown`, `rfx.dashboard`, `rfx.config`, `rfx._internal`,
  `rfx.materials` and `rfx.subgridding` have no comments left other than tool directives. Some conventions the code
  cannot show now stand as one docstring sentence on the function, class or module they belong to, at most one each, so
  `help()` shows them. Eight commented literals became private named constants, and the other comments were removed.
- No check, message or computation changes.
