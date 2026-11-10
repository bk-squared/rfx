### Changed — source modules import their grid and source-spec types from the modules that define them, and the coaxial line's stamping and TEM plane-source builder are defined in `rfx.model.coax_stamp` (#PR)

- `MagneticSourceSpec` is now defined in `rfx._internal._uniform_specs`; `rfx.simulation.MagneticSourceSpec` is the same class.
- `stamp_coaxial_line`, `stamp_coaxial_short_plane`, `stamp_coaxial_annular_resistor`, `build_coaxial_tem_plane_source_specs` and `CoaxialPlaneSourceSpec` keep their old import paths (`rfx.sources.coaxial_port`, `rfx.sources`, `rfx`), which give the same objects.
- No field, S-parameter or gradient changes.
