### Changed — twenty-seven more top-level `rfx` modules have no code comments; some of their conventions are now docstring sentences (#PR)

- Twenty-seven top-level `rfx` modules, among them `rfx.visualize`, `rfx.lumped`, `rfx.floquet`, `rfx.harminv`
  and `rfx.validation`, have no code comments left other than tool directives. Some conventions the code cannot show
  now stand as one docstring sentence on the function, class or module they belong to, at most one each, so `help()`
  shows them. The other comments were removed.
- No computation changes.
