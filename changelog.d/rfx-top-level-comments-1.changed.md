### Changed — twenty top-level `rfx` modules state their conventions in docstrings instead of code comments (#PR)

- `rfx.auto_config`, `rfx.fidelity`, `rfx.grid`, `rfx.io`, `rfx.eigenmode` and fifteen smaller top-level modules
  have no code comments left. The few conventions the code cannot show (the units of `FeatureInfo` lengths, what
  `FeatureInfo.estimated_Q` holds, the mesh plane of a PMC face, the Touchstone format read when a file names none)
  are now one docstring sentence each, so `help()` shows them; the other comments were removed.
- No computation changes.
