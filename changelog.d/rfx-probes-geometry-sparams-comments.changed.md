### Changed — `rfx.probes`, `rfx.geometry` and `rfx.sparams` modules state their conventions in docstrings instead of code comments (#PR)

- Twenty-seven modules of `rfx.probes`, `rfx.geometry` and `rfx.sparams` have no code comments left other than tool
  directives. A convention the code cannot show is now one docstring sentence on the function, class or module it
  belongs to, so `help()` shows it: for example how a sample on a mesh face is classified, and that the MSL probe
  ladder comes back in increasing x for either port direction. The other comments were removed.
- No computation changes.
