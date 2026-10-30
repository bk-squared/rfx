### Fixed — preflight and the realized-geometry record no longer reuse a stale context after a declaration changes (#PR)

- Their cache key is now built from one registry of every `Simulation` declaration (`rfx/api/_declarations.py`),
  holding declared objects by reference and mesh profiles by value. Before, changing the boundary spec, the
  explicit-boundary flag, `cpml_kappa_max`, a pinned `dt`, `freq_max` or the 2-D mode after a first preflight, or
  editing a profile in place, could return the previous context's grid and materials.
- Results of a run are unchanged; only which preflight checks see the current declarations.
