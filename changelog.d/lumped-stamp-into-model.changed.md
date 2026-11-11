### Changed — the lumped and wire ports' load stamping into the material arrays is defined in `rfx.model.lumped_stamp` (#PR)

- `setup_lumped_port`, `setup_wire_port`, `stamp_lumped_sigma`, `stamp_lumped_eps`, `port_sigma`, `port_dual_transverse`, `port_d_parallel` and `wire_port_edge_span` keep their old import paths (`rfx.sources.sources`, and `rfx.sources` or `rfx` where exported), which give the same objects.
- No field, S-parameter or gradient changes.
