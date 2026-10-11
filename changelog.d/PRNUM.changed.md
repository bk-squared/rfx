### Changed

`run()` now references waveguide S11 to the snapped port plane, matching
`compute_waveguide_s_matrix()`, and S21 runs from that plane to the probe.
Above cutoff, magnitudes are unchanged; S11 phase changes by
2·β·ref_offset·dx and S21 phase by β·ref_offset·dx.
Pass `calibration_preset='measured'` to restore the previous `run()` plane.
`compute_waveguide_s_matrix()` now honours that preset's measured plane too.
