### Added — lumped RLC elements run on several devices (#PR)

- `add_lumped_rlc(...)` with `run(devices=...)` or `forward(distributed=True)` no longer raises on uniform or graded meshes; each element is solved by the one-device element update on the slab that owns its edge.
- A lone resistor or folded R/C matches one device within 9 float32 ULP per step. An element with an inductor or in series is compared with one device by the accumulated field law and the probe spectra (within 1e-4) in boxes with absorbing boundaries; in a closed box it has no one-device comparison.
- Graded multi-device `forward()` releases declared RLC edges from `pec_occupancy_override` with the same preparation and release warning as one device.
- `rlc_values_override`, elements with PMC faces, and uniform-mesh elements on a PEC face or realized PEC edge stay refused. Uniform multi-device whole-grid overrides stay refused.
- Closed-box TF/SF admission checks every realized RLC edge, including series RL elements without a material stamp, on both multi-device entries.
