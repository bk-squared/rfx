# S0 CPU timings

Full matrix: **876.610 s** wall time; sum of cell timers **852.607 s**. 340 cells (132 equivalence, 208 refusal).

Verification: 469 passed, 152 strict xfailed, zero failures (621 record/control tests).
The run emitted one pytest iterator-parametrization deprecation; collection was
then changed to materialize the same parameters as a tuple. Recollection retained
621 total tests and 537 PR tests with no warning. Supporting controls, admission
disposition, and CI workflow contracts: 137 passed, 3 deselected. The required
ruff E/F/W check passed.

The generator selected **302 PR cells**; the other **38** cells are marked weekly. The selected cells account for 675.485 s of this full, ordered run. This is not a separately measured cold PR-subset wall time.

Environment: macOS-26.5.2-arm64-arm-64bit, Python 3.11.2, JAX 0.10.2, CPU, float32, two host devices. The full wall time is pytest elapsed time, including collection, worker startup and teardown. Cell timers include compilation, realization, observation, comparison, and device synchronization; they exclude worker import/startup. Cached identical declarations are reused across cells. Per-cell time is reported, not enforced.

| Cell | Seconds | Placement | Outcome |
|---|---:|---|---|
| `_freq_max::run_uniform:run_nonuniform:constant:12` | 7.451944 | PR + weekly | finding |
| `_freq_max::run_uniform:run_distributed:constant:12` | 4.111506 | weekly | equivalent |
| `_freq_max::fwd_uniform:fwd_nonuniform:constant:12` | 9.639925 | PR + weekly | finding |
| `_freq_max::run_nonuniform:run_distributed:graded:12` | 8.535167 | weekly | equivalent |
| `_domain::run_uniform:run_nonuniform:constant:12` | 0.002110 | PR + weekly | finding |
| `_domain::run_uniform:run_distributed:constant:12` | 0.001880 | weekly | equivalent |
| `_domain::fwd_uniform:fwd_nonuniform:constant:12` | 0.002042 | PR + weekly | finding |
| `_domain::run_nonuniform:run_distributed:graded:12` | 0.414094 | weekly | equivalent |
| `_dx::run_uniform:run_nonuniform:constant:12` | 0.002165 | PR + weekly | finding |
| `_dx::run_uniform:run_distributed:constant:12` | 0.002221 | weekly | equivalent |
| `_dx::fwd_uniform:fwd_nonuniform:constant:12` | 0.002785 | PR + weekly | finding |
| `_dx::run_nonuniform:run_distributed:graded:12` | 0.004832 | weekly | equivalent |
| `_dt_pin::run_uniform:run_nonuniform:constant:12` | 0.000186 | PR + weekly | refused |
| `_dt_pin::run_uniform:run_distributed:constant:12` | 0.000159 | PR + weekly | refused |
| `_dt_pin::fwd_uniform:fwd_nonuniform:constant:12` | 0.000112 | PR + weekly | refused |
| `_dt_pin::run_nonuniform:run_distributed:graded:12` | 9.063713 | PR + weekly | equivalent |
| `_dt_min_cell::run_uniform:run_nonuniform:constant:12` | 0.000239 | PR + weekly | refused |
| `_dt_min_cell::run_uniform:run_distributed:constant:12` | 0.000176 | PR + weekly | refused |
| `_dt_min_cell::fwd_uniform:fwd_nonuniform:constant:12` | 0.000162 | PR + weekly | refused |
| `_dt_min_cell::run_nonuniform:run_distributed:graded:12` | 9.340477 | PR + weekly | equivalent |
| `_precision::run_uniform:run_nonuniform:constant:12` | 0.001217 | PR + weekly | refused |
| `_precision::run_uniform:run_distributed:constant:12` | 0.001216 | PR + weekly | refused |
| `_precision::fwd_uniform:fwd_nonuniform:constant:12` | 0.011474 | PR + weekly | refused |
| `_adi_cfl_factor::run_uniform:run_nonuniform:constant:12` | 7.566783 | PR + weekly | finding |
| `_adi_cfl_factor::run_uniform:run_distributed:constant:12` | 4.647401 | weekly | equivalent |
| `_adi_cfl_factor::fwd_uniform:fwd_nonuniform:constant:12` | 10.366020 | PR + weekly | finding |
| `_adi_cfl_factor::run_nonuniform:run_distributed:graded:12` | 8.791713 | weekly | equivalent |
| `_stencil_order::run_uniform:run_nonuniform:constant:12` | 0.001233 | PR + weekly | refused |
| `_stencil_order::run_uniform:run_distributed:constant:12` | 0.001062 | PR + weekly | refused |
| `_stencil_order::fwd_uniform:fwd_nonuniform:constant:12` | 0.011436 | PR + weekly | refused |
| `_mode::run_uniform:run_nonuniform:constant:12` | 0.001139 | PR + weekly | refused |
| `_mode::run_uniform:run_distributed:constant:12` | 5.282070 | PR + weekly | finding |
| `_mode::fwd_uniform:fwd_nonuniform:constant:12` | 0.009968 | PR + weekly | refused |
| `_mode::run_nonuniform:run_distributed:graded:12` | 0.002407 | PR + weekly | refused |
| `_materials:eps:run_uniform:run_nonuniform:constant:12` | 0.001991 | PR + weekly | finding |
| `_materials:eps:run_uniform:run_distributed:constant:12` | 0.002017 | weekly | equivalent |
| `_materials:eps:fwd_uniform:fwd_nonuniform:constant:12` | 0.004458 | PR + weekly | finding |
| `_materials:eps:run_nonuniform:run_distributed:graded:12` | 0.413088 | weekly | equivalent |
| `_materials:sigma:run_uniform:run_nonuniform:constant:12` | 8.279412 | PR + weekly | finding |
| `_materials:sigma:run_uniform:run_distributed:constant:12` | 4.864234 | weekly | equivalent |
| `_materials:sigma:fwd_uniform:fwd_nonuniform:constant:12` | 11.432753 | PR + weekly | finding |
| `_materials:sigma:run_nonuniform:run_distributed:graded:12` | 10.171969 | weekly | equivalent |
| `_materials:mu:run_uniform:run_nonuniform:constant:12` | 8.971424 | PR + weekly | finding |
| `_materials:mu:run_uniform:run_distributed:constant:12` | 4.400161 | weekly | equivalent |
| `_materials:mu:fwd_uniform:fwd_nonuniform:constant:12` | 8.818876 | PR + weekly | finding |
| `_materials:mu:run_nonuniform:run_distributed:graded:12` | 7.950116 | weekly | equivalent |
| `_materials:debye:run_uniform:run_nonuniform:constant:12` | 10.899816 | PR + weekly | finding |
| `_materials:debye:run_uniform:run_distributed:constant:12` | 6.295797 | weekly | equivalent |
| `_materials:debye:fwd_uniform:fwd_nonuniform:constant:12` | 13.932938 | PR + weekly | finding |
| `_materials:debye:run_nonuniform:run_distributed:graded:12` | 12.160438 | weekly | equivalent |
| `_materials:lorentz:run_uniform:run_nonuniform:constant:12` | 11.712993 | PR + weekly | finding |
| `_materials:lorentz:run_uniform:run_distributed:constant:12` | 7.210537 | weekly | equivalent |
| `_materials:lorentz:fwd_uniform:fwd_nonuniform:constant:12` | 15.607005 | PR + weekly | finding |
| `_materials:lorentz:run_nonuniform:run_distributed:graded:12` | 14.080137 | weekly | equivalent |
| `_materials:drude:run_uniform:run_nonuniform:constant:12` | 12.443695 | PR + weekly | finding |
| `_materials:drude:run_uniform:run_distributed:constant:12` | 6.891618 | weekly | equivalent |
| `_materials:drude:fwd_uniform:fwd_nonuniform:constant:12` | 15.947584 | PR + weekly | finding |
| `_materials:drude:run_nonuniform:run_distributed:graded:12` | 13.638195 | weekly | equivalent |
| `_materials:kerr:run_uniform:run_nonuniform:constant:12` | 1.807526 | PR + weekly | refused |
| `_materials:kerr:run_uniform:run_distributed:constant:12` | 0.003462 | PR + weekly | refused |
| `_materials:kerr:fwd_uniform:fwd_nonuniform:constant:12` | 0.001578 | PR + weekly | refused |
| `_geometry:pec_volume:run_uniform:run_nonuniform:constant:12` | 8.712676 | PR + weekly | finding |
| `_geometry:pec_volume:run_uniform:run_distributed:constant:12` | 4.888563 | weekly | equivalent |
| `_geometry:pec_volume:fwd_uniform:fwd_nonuniform:constant:12` | 11.105830 | PR + weekly | finding |
| `_geometry:pec_volume:run_nonuniform:run_distributed:graded:12` | 9.867603 | weekly | equivalent |
| `_geometry:pec_sheet:run_uniform:run_nonuniform:constant:12` | 8.755307 | PR + weekly | finding |
| `_geometry:pec_sheet:run_uniform:run_distributed:constant:12` | 1.217478 | PR + weekly | refused |
| `_geometry:pec_sheet:fwd_uniform:fwd_nonuniform:constant:12` | 8.218576 | PR + weekly | finding |
| `_geometry:pec_sheet:run_nonuniform:run_distributed:graded:12` | 1.132442 | PR + weekly | refused |
| `_geometry:pec_wire:run_uniform:run_nonuniform:constant:12` | 5.014371 | PR + weekly | finding |
| `_geometry:pec_wire:run_uniform:run_distributed:constant:12` | 0.657077 | PR + weekly | refused |
| `_geometry:pec_wire:fwd_uniform:fwd_nonuniform:constant:12` | 7.984619 | PR + weekly | finding |
| `_geometry:pec_wire:run_nonuniform:run_distributed:graded:12` | 1.397558 | PR + weekly | refused |
| `_thin_conductors:lossy_sheet:run_uniform:run_nonuniform:constant:12` | 7.455047 | PR + weekly | finding |
| `_thin_conductors:lossy_sheet:run_uniform:run_distributed:constant:12` | 4.527838 | weekly | equivalent |
| `_thin_conductors:lossy_sheet:fwd_uniform:fwd_nonuniform:constant:12` | 10.371746 | PR + weekly | finding |
| `_thin_conductors:lossy_sheet:run_nonuniform:run_distributed:graded:12` | 9.468608 | weekly | equivalent |
| `_thin_conductors:pec_sheet:run_uniform:run_nonuniform:constant:12` | 8.815885 | PR + weekly | finding |
| `_thin_conductors:pec_sheet:run_uniform:run_distributed:constant:12` | 1.046403 | PR + weekly | refused |
| `_thin_conductors:pec_sheet:fwd_uniform:fwd_nonuniform:constant:12` | 9.499830 | PR + weekly | finding |
| `_thin_conductors:pec_sheet:run_nonuniform:run_distributed:graded:12` | 1.483839 | PR + weekly | refused |
| `_thin_conductors:surface_impedance:run_uniform:run_nonuniform:constant:12` | 8.590909 | PR + weekly | finding |
| `_thin_conductors:surface_impedance:run_uniform:run_distributed:constant:12` | 0.073231 | PR + weekly | refused |
| `_thin_conductors:surface_impedance:fwd_uniform:fwd_nonuniform:constant:12` | 12.620387 | PR + weekly | finding |
| `_thin_conductors:surface_impedance:run_nonuniform:run_distributed:graded:12` | 0.068555 | PR + weekly | refused |
| `_pinned_sheets:pec_sheet:run_uniform:run_nonuniform:constant:12` | 8.307099 | PR + weekly | finding |
| `_pinned_sheets:pec_sheet:run_uniform:run_distributed:constant:12` | 1.047894 | PR + weekly | refused |
| `_pinned_sheets:pec_sheet:fwd_uniform:fwd_nonuniform:constant:12` | 9.485543 | PR + weekly | finding |
| `_pinned_sheets:pec_sheet:run_nonuniform:run_distributed:graded:12` | 1.486088 | PR + weekly | refused |
| `_ports:source:run_uniform:run_nonuniform:constant:12` | 0.001803 | PR + weekly | finding |
| `_ports:source:run_uniform:run_distributed:constant:12` | 0.001955 | weekly | equivalent |
| `_ports:source:fwd_uniform:fwd_nonuniform:constant:12` | 0.001948 | PR + weekly | finding |
| `_ports:source:run_nonuniform:run_distributed:graded:12` | 0.003439 | weekly | equivalent |
| `_ports:amplitude_kind:run_uniform:run_nonuniform:constant:12` | 6.611631 | PR + weekly | finding |
| `_ports:amplitude_kind:run_uniform:run_distributed:constant:12` | 4.128674 | weekly | equivalent |
| `_ports:amplitude_kind:fwd_uniform:fwd_nonuniform:constant:12` | 9.813392 | PR + weekly | finding |
| `_ports:amplitude_kind:run_nonuniform:run_distributed:graded:12` | 8.542809 | weekly | equivalent |
| `_ports:lumped_port:run_uniform:run_nonuniform:constant:12` | 8.011463 | PR + weekly | finding |
| `_ports:lumped_port:run_uniform:run_distributed:constant:12` | 4.355971 | PR + weekly | finding |
| `_ports:lumped_port:fwd_uniform:fwd_nonuniform:constant:12` | 1.016803 | PR + weekly | finding |
| `_ports:lumped_port:run_nonuniform:run_distributed:graded:12` | 7.270916 | PR + weekly | finding |
| `_ports:passive_port:run_uniform:run_nonuniform:constant:12` | 7.851050 | PR + weekly | finding |
| `_ports:passive_port:run_uniform:run_distributed:constant:12` | 0.002858 | PR + weekly | refused |
| `_ports:passive_port:fwd_uniform:fwd_nonuniform:constant:12` | 8.828929 | PR + weekly | finding |
| `_ports:passive_port:run_nonuniform:run_distributed:graded:12` | 0.001035 | PR + weekly | refused |
| `_ports:wire_port:run_uniform:run_nonuniform:constant:12` | 6.072436 | PR + weekly | finding |
| `_ports:wire_port:run_uniform:run_nonuniform:constant:36` | 7.975196 | PR + weekly | finding |
| `_ports:wire_port:run_uniform:run_distributed:constant:12` | 0.955951 | PR + weekly | finding |
| `_ports:wire_port:run_uniform:run_distributed:constant:36` | 0.895681 | PR + weekly | finding |
| `_ports:wire_port:fwd_uniform:fwd_nonuniform:constant:12` | 11.044335 | PR + weekly | finding |
| `_ports:wire_port:fwd_uniform:fwd_nonuniform:constant:36` | 11.252461 | PR + weekly | finding |
| `_ports:wire_port:run_nonuniform:run_distributed:graded:12` | 5.941538 | PR + weekly | finding |
| `_ports:wire_port:run_nonuniform:run_distributed:graded:36` | 5.994836 | PR + weekly | finding |
| `_msl_ports:msl_port:run_uniform:run_nonuniform:constant:12` | 9.165557 | PR + weekly | finding |
| `_msl_ports:msl_port:run_uniform:run_distributed:constant:12` | 0.001342 | PR + weekly | refused |
| `_msl_ports:msl_port:fwd_uniform:fwd_nonuniform:constant:12` | 11.161343 | PR + weekly | finding |
| `_msl_ports:msl_port:run_nonuniform:run_distributed:graded:12` | 0.001443 | PR + weekly | refused |
| `_waveguide_ports:waveguide_port:run_uniform:run_nonuniform:constant:12` | 0.001348 | PR + weekly | finding |
| `_waveguide_ports:waveguide_port:run_uniform:run_distributed:constant:12` | 0.000463 | PR + weekly | finding |
| `_waveguide_ports:waveguide_port:fwd_uniform:fwd_nonuniform:constant:12` | 0.000563 | PR + weekly | finding |
| `_waveguide_ports:waveguide_port:run_nonuniform:run_distributed:graded:12` | 0.000604 | PR + weekly | finding |
| `_floquet_ports:floquet_port:run_uniform:run_nonuniform:constant:12` | 0.000255 | PR + weekly | refused |
| `_floquet_ports:floquet_port:run_uniform:run_distributed:constant:12` | 0.001051 | PR + weekly | refused |
| `_floquet_ports:floquet_port:fwd_uniform:fwd_nonuniform:constant:12` | 0.000277 | PR + weekly | refused |
| `_lumped_rlc:R:run_uniform:run_nonuniform:constant:12` | 7.681114 | PR + weekly | finding |
| `_lumped_rlc:R:run_uniform:run_distributed:constant:12` | 0.001208 | PR + weekly | refused |
| `_lumped_rlc:R:fwd_uniform:fwd_nonuniform:constant:12` | 9.995867 | PR + weekly | finding |
| `_lumped_rlc:R:run_nonuniform:run_distributed:graded:12` | 0.001585 | PR + weekly | refused |
| `_lumped_rlc:series_RL:run_uniform:run_nonuniform:constant:12` | 7.458961 | PR + weekly | finding |
| `_lumped_rlc:series_RL:run_uniform:run_distributed:constant:12` | 0.001115 | PR + weekly | refused |
| `_lumped_rlc:series_RL:fwd_uniform:fwd_nonuniform:constant:12` | 9.714297 | PR + weekly | finding |
| `_lumped_rlc:series_RL:run_nonuniform:run_distributed:graded:12` | 0.001560 | PR + weekly | refused |
| `_tfsf:plane_wave:run_uniform:run_nonuniform:constant:12` | 0.000465 | PR + weekly | finding |
| `_tfsf:plane_wave:run_uniform:run_distributed:constant:12` | 0.000625 | PR + weekly | finding |
| `_tfsf:plane_wave:fwd_uniform:fwd_nonuniform:constant:12` | 0.001533 | PR + weekly | finding |
| `_tfsf:plane_wave:run_nonuniform:run_distributed:graded:12` | 0.000662 | PR + weekly | finding |
| `_boundary:cpml:run_uniform:run_nonuniform:constant:12` | 0.001973 | PR + weekly | finding |
| `_boundary:cpml:run_uniform:run_distributed:constant:12` | 0.001845 | weekly | equivalent |
| `_boundary:cpml:fwd_uniform:fwd_nonuniform:constant:12` | 0.001912 | PR + weekly | finding |
| `_boundary:cpml:run_nonuniform:run_distributed:graded:12` | 0.415321 | weekly | equivalent |
| `_boundary:upml:run_uniform:run_nonuniform:constant:12` | 0.001013 | PR + weekly | refused |
| `_boundary:upml:run_uniform:run_distributed:constant:12` | 0.001021 | PR + weekly | refused |
| `_boundary:upml:fwd_uniform:fwd_nonuniform:constant:12` | 0.010698 | PR + weekly | refused |
| `_pec_faces:pec_face:run_uniform:run_nonuniform:constant:12` | 7.417030 | PR + weekly | finding |
| `_pec_faces:pec_face:run_uniform:run_distributed:constant:12` | 3.997926 | weekly | equivalent |
| `_pec_faces:pec_face:fwd_uniform:fwd_nonuniform:constant:12` | 9.474571 | PR + weekly | finding |
| `_pec_faces:pec_face:run_nonuniform:run_distributed:graded:12` | 8.256869 | weekly | equivalent |
| `_boundary_spec:pmc_face:run_uniform:run_nonuniform:constant:12` | 7.261866 | PR + weekly | finding |
| `_boundary_spec:pmc_face:run_uniform:run_distributed:constant:12` | 0.000950 | PR + weekly | refused |
| `_boundary_spec:pmc_face:fwd_uniform:fwd_nonuniform:constant:12` | 9.687567 | PR + weekly | finding |
| `_boundary_spec:pmc_face:run_nonuniform:run_distributed:graded:12` | 0.001041 | PR + weekly | refused |
| `_boundary_spec:conformal:run_uniform:run_nonuniform:constant:12` | 0.001172 | PR + weekly | refused |
| `_boundary_spec:conformal:run_uniform:run_distributed:constant:12` | 0.001205 | PR + weekly | refused |
| `_boundary_spec:absorbing_lid:run_uniform:run_nonuniform:constant:12` | 5.231782 | PR + weekly | finding |
| `_boundary_spec:absorbing_lid:run_uniform:run_distributed:constant:12` | 2.913781 | weekly | equivalent |
| `_boundary_spec:absorbing_lid:fwd_uniform:fwd_nonuniform:constant:12` | 7.109284 | PR + weekly | finding |
| `_boundary_spec:absorbing_lid:run_nonuniform:run_distributed:graded:12` | 4.961208 | weekly | equivalent |
| `_periodic_axes:periodic:run_uniform:run_nonuniform:constant:12` | 1.019036 | PR + weekly | refused |
| `_periodic_axes:periodic:run_uniform:run_distributed:constant:12` | 0.001148 | PR + weekly | refused |
| `_periodic_axes:periodic:fwd_uniform:fwd_nonuniform:constant:12` | 0.009798 | PR + weekly | refused |
| `_cpml_layers:layers:run_uniform:run_nonuniform:constant:12` | 0.001889 | PR + weekly | finding |
| `_cpml_layers:layers:run_uniform:run_distributed:constant:12` | 0.002166 | weekly | equivalent |
| `_cpml_layers:layers:fwd_uniform:fwd_nonuniform:constant:12` | 0.002168 | PR + weekly | finding |
| `_cpml_layers:layers:run_nonuniform:run_distributed:graded:12` | 0.136244 | weekly | equivalent |
| `_cpml_kappa_max:kappa:run_uniform:run_nonuniform:constant:12` | 0.768160 | PR + weekly | refused |
| `_cpml_kappa_max:kappa:run_uniform:run_distributed:constant:12` | 4.691341 | PR + weekly | equivalent |
| `_cpml_kappa_max:kappa:fwd_uniform:fwd_nonuniform:constant:12` | 1.042237 | PR + weekly | refused |
| `_cpml_kappa_max:kappa:run_nonuniform:run_distributed:graded:12` | 0.010915 | PR + weekly | refused |
| `_interface_eps:dual_average:run_uniform:run_nonuniform:constant:12` | 0.120293 | PR + weekly | refused |
| `_interface_eps:dual_average:fwd_uniform:fwd_nonuniform:constant:12` | 0.009885 | PR + weekly | refused |
| `_interface_eps:dual_average:run_nonuniform:run_distributed:graded:12` | 0.000921 | PR + weekly | refused |
| `_dx_profile:graded:run_uniform:run_nonuniform:constant:12` | 0.000999 | PR + weekly | refused |
| `_dx_profile:graded:run_uniform:run_distributed:constant:12` | 0.000759 | PR + weekly | refused |
| `_dx_profile:graded:fwd_uniform:fwd_nonuniform:constant:12` | 0.001031 | PR + weekly | refused |
| `_dx_profile:graded:run_nonuniform:run_distributed:graded:12` | 7.203884 | PR + weekly | equivalent |
| `_dy_profile:graded:run_uniform:run_nonuniform:constant:12` | 0.001053 | PR + weekly | refused |
| `_dy_profile:graded:run_uniform:run_distributed:constant:12` | 0.001336 | PR + weekly | refused |
| `_dy_profile:graded:fwd_uniform:fwd_nonuniform:constant:12` | 0.009130 | PR + weekly | refused |
| `_dy_profile:graded:run_nonuniform:run_distributed:graded:12` | 8.420071 | PR + weekly | equivalent |
| `_dz_profile:graded:run_uniform:run_nonuniform:constant:12` | 0.001186 | PR + weekly | refused |
| `_dz_profile:graded:run_uniform:run_distributed:constant:12` | 0.001081 | PR + weekly | refused |
| `_dz_profile:graded:fwd_uniform:fwd_nonuniform:constant:12` | 0.011283 | PR + weekly | refused |
| `_dz_profile:graded:run_nonuniform:run_distributed:graded:12` | 8.392352 | PR + weekly | equivalent |
| `_probes:probe:run_uniform:run_nonuniform:constant:12` | 0.001929 | PR + weekly | finding |
| `_probes:probe:run_uniform:run_distributed:constant:12` | 0.002006 | weekly | equivalent |
| `_probes:probe:fwd_uniform:fwd_nonuniform:constant:12` | 0.001954 | PR + weekly | finding |
| `_probes:probe:run_nonuniform:run_distributed:graded:12` | 0.418365 | weekly | equivalent |
| `_dft_planes:dft_plane:run_uniform:run_nonuniform:constant:12` | 7.519567 | PR + weekly | finding |
| `_dft_planes:dft_plane:run_uniform:run_nonuniform:constant:36` | 7.757369 | PR + weekly | finding |
| `_dft_planes:dft_plane:run_uniform:run_distributed:constant:12` | 0.009864 | PR + weekly | refused |
| `_dft_planes:dft_plane:fwd_uniform:fwd_nonuniform:constant:12` | 9.733964 | PR + weekly | finding |
| `_dft_planes:dft_plane:fwd_uniform:fwd_nonuniform:constant:36` | 10.319928 | PR + weekly | finding |
| `_dft_planes:dft_plane:run_nonuniform:run_distributed:graded:12` | 0.011383 | PR + weekly | refused |
| `_flux_monitors:flux:run_uniform:run_nonuniform:constant:12` | 7.729922 | PR + weekly | finding |
| `_flux_monitors:flux:run_uniform:run_nonuniform:constant:36` | 7.840237 | PR + weekly | finding |
| `_flux_monitors:flux:run_uniform:run_distributed:constant:12` | 0.010221 | PR + weekly | refused |
| `_flux_monitors:flux:fwd_uniform:fwd_nonuniform:constant:12` | 9.945313 | PR + weekly | finding |
| `_flux_monitors:flux:fwd_uniform:fwd_nonuniform:constant:36` | 10.247500 | PR + weekly | finding |
| `_flux_monitors:flux:run_nonuniform:run_distributed:graded:12` | 0.010395 | PR + weekly | refused |
| `_ntff:ntff_box:run_uniform:run_nonuniform:constant:12` | 7.917338 | PR + weekly | finding |
| `_ntff:ntff_box:run_uniform:run_distributed:constant:12` | 4.943030 | PR + weekly | finding |
| `_ntff:ntff_box:fwd_uniform:fwd_nonuniform:constant:12` | 10.871663 | PR + weekly | finding |
| `_ntff:ntff_box:run_nonuniform:run_distributed:graded:12` | 5.937373 | PR + weekly | finding |
| `_current_moments:block_moments:run_uniform:run_nonuniform:constant:12` | 10.673670 | PR + weekly | finding |
| `_current_moments:block_moments:run_uniform:run_distributed:constant:12` | 0.001317 | PR + weekly | refused |
| `_current_moments:block_moments:fwd_uniform:fwd_nonuniform:constant:12` | 13.554037 | PR + weekly | finding |
| `_current_moments:block_moments:run_nonuniform:run_distributed:graded:12` | 0.001055 | PR + weekly | refused |
| `_dt_pin::run_adi:refusal:constant:0` | 0.000210 | PR + weekly | refused |
| `_dt_pin::fwd_adi:refusal:constant:0` | 0.000102 | PR + weekly | refused |
| `_dt_pin::run_subgridded:refusal:constant:0` | 0.000104 | PR + weekly | refused |
| `_dt_min_cell::run_adi:refusal:constant:0` | 0.000159 | PR + weekly | refused |
| `_dt_min_cell::fwd_adi:refusal:constant:0` | 0.000096 | PR + weekly | refused |
| `_dt_min_cell::run_subgridded:refusal:constant:0` | 0.000075 | PR + weekly | refused |
| `_precision::run_adi:refusal:constant:0` | 0.000801 | PR + weekly | refused |
| `_precision::fwd_adi:refusal:constant:0` | 0.000803 | PR + weekly | refused |
| `_precision::run_subgridded:refusal:constant:0` | 0.000796 | PR + weekly | refused |
| `_solver::run_subgridded:refusal:constant:0` | 0.000670 | PR + weekly | finding |
| `_stencil_order::run_adi:refusal:constant:0` | 0.001258 | PR + weekly | refused |
| `_stencil_order::fwd_adi:refusal:constant:0` | 0.000808 | PR + weekly | refused |
| `_stencil_order::run_subgridded:refusal:constant:0` | 0.000978 | PR + weekly | refused |
| `_mode::run_subgridded:refusal:constant:0` | 0.001227 | PR + weekly | refused |
| `_materials:mu:run_adi:refusal:constant:0` | 0.001295 | PR + weekly | refused |
| `_materials:mu:fwd_adi:refusal:constant:0` | 0.000918 | PR + weekly | refused |
| `_materials:debye:run_adi:refusal:constant:0` | 0.001165 | PR + weekly | refused |
| `_materials:debye:fwd_adi:refusal:constant:0` | 0.000918 | PR + weekly | refused |
| `_materials:debye:run_subgridded:refusal:constant:0` | 0.001198 | PR + weekly | refused |
| `_materials:lorentz:run_adi:refusal:constant:0` | 0.000970 | PR + weekly | refused |
| `_materials:lorentz:fwd_adi:refusal:constant:0` | 0.001260 | PR + weekly | refused |
| `_materials:lorentz:run_subgridded:refusal:constant:0` | 0.001094 | PR + weekly | refused |
| `_materials:drude:run_adi:refusal:constant:0` | 0.000993 | PR + weekly | refused |
| `_materials:drude:fwd_adi:refusal:constant:0` | 0.000999 | PR + weekly | refused |
| `_materials:drude:run_subgridded:refusal:constant:0` | 0.001035 | PR + weekly | refused |
| `_materials:kerr:run_adi:refusal:constant:0` | 0.001028 | PR + weekly | refused |
| `_materials:kerr:fwd_adi:refusal:constant:0` | 0.000934 | PR + weekly | refused |
| `_materials:kerr:run_subgridded:refusal:constant:0` | 0.000984 | PR + weekly | refused |
| `_geometry:pec_volume:run_adi:refusal:constant:0` | 0.001037 | PR + weekly | refused |
| `_geometry:pec_volume:fwd_adi:refusal:constant:0` | 0.001089 | PR + weekly | refused |
| `_geometry:pec_sheet:run_adi:refusal:constant:0` | 0.001126 | PR + weekly | refused |
| `_geometry:pec_sheet:fwd_adi:refusal:constant:0` | 0.000935 | PR + weekly | refused |
| `_geometry:pec_sheet:run_subgridded:refusal:constant:0` | 0.001021 | PR + weekly | refused |
| `_geometry:pec_wire:run_adi:refusal:constant:0` | 0.024604 | PR + weekly | refused |
| `_geometry:pec_wire:fwd_adi:refusal:constant:0` | 0.015188 | PR + weekly | refused |
| `_geometry:pec_wire:run_subgridded:refusal:constant:0` | 0.014099 | PR + weekly | refused |
| `_thin_conductors:lossy_sheet:run_adi:refusal:constant:0` | 0.000821 | PR + weekly | refused |
| `_thin_conductors:lossy_sheet:fwd_adi:refusal:constant:0` | 0.000933 | PR + weekly | refused |
| `_thin_conductors:lossy_sheet:run_subgridded:refusal:constant:0` | 0.000783 | PR + weekly | refused |
| `_thin_conductors:pec_sheet:run_adi:refusal:constant:0` | 0.001204 | PR + weekly | refused |
| `_thin_conductors:pec_sheet:fwd_adi:refusal:constant:0` | 0.000898 | PR + weekly | refused |
| `_thin_conductors:pec_sheet:run_subgridded:refusal:constant:0` | 0.000868 | PR + weekly | refused |
| `_thin_conductors:surface_impedance:run_adi:refusal:constant:0` | 0.068366 | PR + weekly | refused |
| `_thin_conductors:surface_impedance:fwd_adi:refusal:constant:0` | 0.001472 | PR + weekly | refused |
| `_thin_conductors:surface_impedance:run_subgridded:refusal:constant:0` | 0.001139 | PR + weekly | refused |
| `_pinned_sheets:pec_sheet:run_adi:refusal:constant:0` | 0.001016 | PR + weekly | refused |
| `_pinned_sheets:pec_sheet:fwd_adi:refusal:constant:0` | 0.000800 | PR + weekly | refused |
| `_pinned_sheets:pec_sheet:run_subgridded:refusal:constant:0` | 0.000608 | PR + weekly | refused |
| `_ports:amplitude_kind:run_adi:refusal:constant:0` | 0.000636 | PR + weekly | refused |
| `_ports:amplitude_kind:fwd_adi:refusal:constant:0` | 0.000988 | PR + weekly | refused |
| `_ports:lumped_port:run_adi:refusal:constant:0` | 0.000743 | PR + weekly | refused |
| `_ports:lumped_port:fwd_adi:refusal:constant:0` | 0.000877 | PR + weekly | refused |
| `_ports:passive_port:run_adi:refusal:constant:0` | 0.000817 | PR + weekly | refused |
| `_ports:passive_port:fwd_adi:refusal:constant:0` | 0.000870 | PR + weekly | refused |
| `_ports:wire_port:run_adi:refusal:constant:0` | 0.000805 | PR + weekly | refused |
| `_ports:wire_port:fwd_adi:refusal:constant:0` | 0.001013 | PR + weekly | refused |
| `_msl_ports:msl_port:run_adi:refusal:constant:0` | 0.001051 | PR + weekly | refused |
| `_msl_ports:msl_port:fwd_adi:refusal:constant:0` | 0.000903 | PR + weekly | refused |
| `_msl_ports:msl_port:run_subgridded:refusal:constant:0` | 0.001172 | PR + weekly | refused |
| `_waveguide_ports:waveguide_port:run_adi:refusal:constant:0` | 0.000387 | PR + weekly | finding |
| `_waveguide_ports:waveguide_port:fwd_adi:refusal:constant:0` | 0.000533 | PR + weekly | finding |
| `_waveguide_ports:waveguide_port:run_subgridded:refusal:constant:0` | 0.000630 | PR + weekly | finding |
| `_coaxial_ports:coax_port:run_adi:refusal:constant:0` | 0.001029 | PR + weekly | refused |
| `_coaxial_ports:coax_port:fwd_adi:refusal:constant:0` | 0.000893 | PR + weekly | refused |
| `_coaxial_ports:coax_port:run_subgridded:refusal:constant:0` | 0.000771 | PR + weekly | refused |
| `_floquet_ports:floquet_port:run_adi:refusal:constant:0` | 0.000975 | PR + weekly | refused |
| `_floquet_ports:floquet_port:fwd_adi:refusal:constant:0` | 0.001217 | PR + weekly | refused |
| `_floquet_ports:floquet_port:run_subgridded:refusal:constant:0` | 0.000983 | PR + weekly | refused |
| `_floquet_ports:scan_angle:run_adi:refusal:constant:0` | 0.001114 | PR + weekly | refused |
| `_floquet_ports:scan_angle:fwd_adi:refusal:constant:0` | 0.000890 | PR + weekly | refused |
| `_floquet_ports:scan_angle:run_subgridded:refusal:constant:0` | 0.001143 | PR + weekly | refused |
| `_lumped_rlc:R:run_adi:refusal:constant:0` | 0.000916 | PR + weekly | refused |
| `_lumped_rlc:R:fwd_adi:refusal:constant:0` | 0.000726 | PR + weekly | refused |
| `_lumped_rlc:R:run_subgridded:refusal:constant:0` | 0.001313 | PR + weekly | refused |
| `_lumped_rlc:series_RL:run_adi:refusal:constant:0` | 0.000889 | PR + weekly | refused |
| `_lumped_rlc:series_RL:fwd_adi:refusal:constant:0` | 0.001226 | PR + weekly | refused |
| `_lumped_rlc:series_RL:run_subgridded:refusal:constant:0` | 0.000807 | PR + weekly | refused |
| `_tfsf:plane_wave:run_adi:refusal:constant:0` | 0.000640 | PR + weekly | finding |
| `_tfsf:plane_wave:fwd_adi:refusal:constant:0` | 0.000625 | PR + weekly | finding |
| `_tfsf:plane_wave:run_subgridded:refusal:constant:0` | 0.000710 | PR + weekly | finding |
| `_refinement:slab:run_adi:refusal:constant:0` | 0.001464 | PR + weekly | refused |
| `_refinement:slab:fwd_adi:refusal:constant:0` | 0.001029 | PR + weekly | refused |
| `_refinement:slab:run_subgridded:refusal:constant:0` | 0.001071 | PR + weekly | refused |
| `_refinement:relaxed_validation:run_adi:refusal:constant:0` | 0.001045 | PR + weekly | refused |
| `_refinement:relaxed_validation:fwd_adi:refusal:constant:0` | 0.001238 | PR + weekly | refused |
| `_refinement:relaxed_validation:run_subgridded:refusal:constant:0` | 0.001152 | PR + weekly | refused |
| `_boundary:cpml:run_subgridded:refusal:constant:0` | 0.000625 | PR + weekly | refused |
| `_boundary:upml:run_adi:refusal:constant:0` | 0.000988 | PR + weekly | refused |
| `_boundary:upml:fwd_adi:refusal:constant:0` | 0.000819 | PR + weekly | refused |
| `_boundary:upml:run_subgridded:refusal:constant:0` | 0.001143 | PR + weekly | refused |
| `_pec_faces:pec_face:run_adi:refusal:constant:0` | 0.000814 | PR + weekly | refused |
| `_pec_faces:pec_face:fwd_adi:refusal:constant:0` | 0.000907 | PR + weekly | refused |
| `_pec_faces:pec_face:run_subgridded:refusal:constant:0` | 0.000905 | PR + weekly | refused |
| `_boundary_spec:pmc_face:run_adi:refusal:constant:0` | 0.001002 | PR + weekly | refused |
| `_boundary_spec:pmc_face:fwd_adi:refusal:constant:0` | 0.000824 | PR + weekly | refused |
| `_boundary_spec:pmc_face:run_subgridded:refusal:constant:0` | 0.001145 | PR + weekly | refused |
| `_boundary_spec:conformal:run_adi:refusal:constant:0` | 0.000756 | PR + weekly | refused |
| `_boundary_spec:conformal:fwd_adi:refusal:constant:0` | 0.000622 | PR + weekly | refused |
| `_boundary_spec:conformal:run_subgridded:refusal:constant:0` | 0.001045 | PR + weekly | refused |
| `_boundary_spec:conformal_s_matrix:run_adi:refusal:constant:0` | 0.000818 | PR + weekly | refused |
| `_boundary_spec:conformal_s_matrix:fwd_adi:refusal:constant:0` | 0.000851 | PR + weekly | refused |
| `_boundary_spec:conformal_s_matrix:run_subgridded:refusal:constant:0` | 0.002771 | PR + weekly | refused |
| `_boundary_spec:absorbing_lid:run_adi:refusal:constant:0` | 0.000760 | PR + weekly | refused |
| `_boundary_spec:absorbing_lid:fwd_adi:refusal:constant:0` | 0.000823 | PR + weekly | refused |
| `_periodic_axes:periodic:run_adi:refusal:constant:0` | 0.000758 | PR + weekly | refused |
| `_periodic_axes:periodic:fwd_adi:refusal:constant:0` | 0.000817 | PR + weekly | refused |
| `_periodic_axes:periodic:run_subgridded:refusal:constant:0` | 0.000761 | PR + weekly | refused |
| `_cpml_layers:layers:run_subgridded:refusal:constant:0` | 0.000662 | PR + weekly | refused |
| `_cpml_kappa_max:kappa:run_adi:refusal:constant:0` | 0.001017 | PR + weekly | refused |
| `_cpml_kappa_max:kappa:fwd_adi:refusal:constant:0` | 0.001176 | PR + weekly | refused |
| `_cpml_kappa_max:kappa:run_subgridded:refusal:constant:0` | 0.001078 | PR + weekly | refused |
| `_interface_eps:dual_average:run_adi:refusal:constant:0` | 0.000830 | PR + weekly | refused |
| `_interface_eps:dual_average:fwd_adi:refusal:constant:0` | 0.001129 | PR + weekly | refused |
| `_interface_eps:dual_average:run_subgridded:refusal:constant:0` | 0.000793 | PR + weekly | refused |
| `_dx_profile:graded:run_adi:refusal:constant:0` | 0.000811 | PR + weekly | refused |
| `_dx_profile:graded:fwd_adi:refusal:constant:0` | 0.000870 | PR + weekly | refused |
| `_dx_profile:graded:run_subgridded:refusal:constant:0` | 0.001107 | PR + weekly | refused |
| `_dy_profile:graded:run_adi:refusal:constant:0` | 0.000966 | PR + weekly | refused |
| `_dy_profile:graded:fwd_adi:refusal:constant:0` | 0.001124 | PR + weekly | refused |
| `_dy_profile:graded:run_subgridded:refusal:constant:0` | 0.000871 | PR + weekly | refused |
| `_dz_profile:graded:run_adi:refusal:constant:0` | 0.001010 | PR + weekly | refused |
| `_dz_profile:graded:fwd_adi:refusal:constant:0` | 0.000939 | PR + weekly | refused |
| `_dz_profile:graded:run_subgridded:refusal:constant:0` | 0.000928 | PR + weekly | refused |
| `_dft_planes:dft_plane:run_adi:refusal:constant:0` | 0.010315 | PR + weekly | refused |
| `_dft_planes:dft_plane:fwd_adi:refusal:constant:0` | 0.001028 | PR + weekly | refused |
| `_dft_planes:dft_plane:run_subgridded:refusal:constant:0` | 0.001555 | PR + weekly | refused |
| `_flux_monitors:flux:run_adi:refusal:constant:0` | 0.000910 | PR + weekly | refused |
| `_flux_monitors:flux:run_subgridded:refusal:constant:0` | 0.000839 | PR + weekly | refused |
| `_ntff:ntff_box:run_adi:refusal:constant:0` | 0.000793 | PR + weekly | refused |
| `_ntff:ntff_box:fwd_adi:refusal:constant:0` | 0.000741 | PR + weekly | refused |
| `_current_moments:block_moments:run_adi:refusal:constant:0` | 0.001693 | PR + weekly | refused |
| `_current_moments:block_moments:fwd_adi:refusal:constant:0` | 0.000870 | PR + weekly | refused |
| `_current_moments:block_moments:run_subgridded:refusal:constant:0` | 0.000754 | PR + weekly | refused |

Conclusions: 리더가 채움.
