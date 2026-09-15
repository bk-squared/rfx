# E — handoff: `CHANGELOG.md` entries for the examples and top-level scripts

Owner: the Docs group. This is the text to fold into
`## [Unreleased — 2.0.0]`, under whatever BREAKING / Changed headings that
entry ends up with.

### BREAKING — examples and the declarative front end

* Every etched conductor in `examples/` is now declared with
  `add_thin_conductor` on a zero-thickness Box — a SHEET, one node plane, no
  cell — instead of a one- or two-cell `Box(..., material="pec")`. Affected:
  `patch_antenna_demo` (ground, patch), `nonuniform_patch_demo` (ground,
  patch), `ports_and_sparams_101` (microstrip ground, trace),
  `examples/config/microstrip_thru.yaml` (ground, trace). A conductor with
  physical thickness — an imported CAD solid, a plate, a PEC sphere — stays a
  Box and gains its far wall.
* `examples/config/microstrip_thru.yaml` gains a `thin_conductors:` block. A
  config file that declares foil as a `geometry:` PEC box now gets a VOLUME,
  and a box thinner than one cell is refused outright.
* Two shipped tutorials solved a board thicker than the one they declared,
  because the mesh reserved a cell for a foil that the old rule realized as a
  plane: `patch_antenna_demo` solved 1.905 mm against a declared 1.524 mm,
  `nonuniform_patch_demo` 2.0 mm against 1.5 mm. Their recorded frequencies
  and mode lists move.
* `examples/tutorials/ports_and_sparams_101.py` reports microstrip readiness
  from `report.ok` rather than an empty report, matching its waveguide leg.
  Until preflight reads the realized edge set (#931 §6) a sheet-declared
  conductor draws one advisory saying preflight could not see it.

### Fixed

* `examples/tutorials/slab_rt_flux_monitor.py` no longer nudges its slab's
  upper corner down by `dx/2`. The stated reason ("an inclusive-bounds Box")
  named a convention that does not exist; what the nudge actually dodged was a
  float knife edge — `CENTER_X + D_SLAB/2` lands one ulp above the node and
  the slab realizes eleven cells against a ten-cell Fresnel oracle. The
  corners are built from cell indices and the realized extent is asserted.
* `scripts/msl_flux_ratio_dof.py` imported rfx from a hard-coded clone path,
  which is a prefix of every sibling worktree's path, so its "is this
  checkout" guard passed while running a different tree's rfx.

### Changed — top-level scripts

* `scripts/_gallery_v3_patch_figs.py` moves from 1 mm to 0.5 mm cells so both
  faces of the 1.5 mm board are node planes. Its ground and patch were 0.5 mm
  PEC Boxes — half a cell — which §1.5 refuses; what the old rule realized was
  a 2.0 mm board while every label said 1.5 mm. Gallery assets cost ~10x per
  case now.
* `scripts/patch_edgefed_s11_validation.py` moves from dx = 0.197 mm to
  dx = h_sub/4 = 0.19675 mm so the board's faces are node planes, and its
  three foils sit ON those faces instead of one cell away. The realized board
  goes from 788 µm (with the trace sheet buried half a cell inside the
  laminate, and a +59.7 % sheet-cavity electrical thickness) to 787.0 µm =
  declared. Its committed locks move.
* `scripts/precompute_gallery_artifacts.py` reads its stack coordinates back
  from the built graded mesh; declaring them at fixed coordinates realized a
  1.361 mm cavity against a declared 1.5 mm.
