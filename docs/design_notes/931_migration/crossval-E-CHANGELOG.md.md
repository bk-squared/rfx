# Replacement text for `CHANGELOG.md` — group X-E's contribution

Group X-E does not own `CHANGELOG.md`. These are the lines X-E wants inside
the `## [Unreleased — 2.0.0]` entry the docs group is writing, under whatever
BREAKING / Changed headings that entry ends up using. They are additions to
that entry, not a competing entry.

### Under BREAKING

- `validation/crossval/20_msl_phase_referee.py` refuses a pre-#931 rfx
  fixture by name (`_stage_b_layout` raises when `meta` carries no
  `trace_wall_planes_realized`) instead of falling back to the declared
  board, and
  `scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py` refuses
  `--patch-realized-only` on one. Both re-pins are a re-solve: the contract
  changed rfx's realized MSL trace, so the committed arrays are stale
  physics and a metadata patch would label them as current.

### Under Changed

- Every microstrip foil in `validation/research/` and
  `validation/tmtt_paper/` is declared as a sheet (a zero-thickness Box) at
  the interface it sits on, instead of a 1-cell PEC Box that used to realize
  as a single wall plane and now realizes as a slab of metal:
  `thru_feedpost_deembed.build_thru`,
  `thru_feedpost_twoseg_extraction.build_singlepost`,
  `issue770_offdiag_adjudication.build_fix_t`,
  `msl_stub_notch_tuning.build_sim`.
- The beam-steering reflector plate is declared on the simulation
  (`validation/tmtt_paper/beam_steering_superstrate.py`) instead of being
  injected as `forward(pec_mask_override=)`. Preflight now sees it, and its
  realized aperture is the declared 1.5 lambda exactly — the cell write was
  one cell too wide.
- The P-C trace in `validation/research/convergence_floor/fixture.py` and
  `validation/research/multiband_nu/w4r_port_supraconvergence.py` is drawn on
  NODE planes; the half-cell "knife-edge-free" margins are deleted. They were
  a workaround for the old node sampler dropping a face exactly on a node,
  and under centre sampling they shift the body by a cell.
- The rfx MSL phase-referee trace stays a VOLUME by decision, with the
  measurement that forced it recorded in the script: a sheet at the declared
  254 um snaps to the 250 um node on this off-lattice board and ends up
  buried under 50 um of realized laminate.

### Under Fixed (or a "Known issue" line if the fix does not land in 2.0.0)

- `rfx/fidelity.py` audits a PEC VOLUME's declared-vs-realized bounds with
  the shape's NODE sampler (`_entity_mask` calls `entry.shape.mask(grid)`)
  while the solver realizes PEC volumes from cell CENTRES
  (`pec_volume_cell_mask`). Measured on the cv20 board they disagree by
  exactly one cell on both y and z. Consumers that need a conductor's
  realized bounds must read `rfx.boundaries.pec.realized_pec_edge_masks` /
  `realized_wall_planes` (design note section 1.7); the cv20 fixture producer
  was switched to it. **X-E did not fix `fidelity.py` — it is not this
  group's file.**
