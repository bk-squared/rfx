### Changed — 41 modules of `rfx.measurement` have no code comments; some of their conventions are now docstring sentences (#PR)

- `rfx.measurement` has no comments left other than tool directives, except two modules an open change holds. Some
  conventions the code cannot show now stand as one docstring sentence on the function, class or module they belong
  to, at most one each, where `help()` on that object shows them. Three literals whose comment gave their meaning are
  private named constants. The other comments were removed.
- No check, message or computation changes.
