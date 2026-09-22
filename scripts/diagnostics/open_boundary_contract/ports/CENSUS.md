Face order: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi. Bounds: m. Boundary column: declared BoundarySpec.

Entry counts: isolated entry assembly. Combined counts: census.json total_cells, total_edges, total_eps_not_1.

Last interior index: pad_lo; shape[axis] - pad_hi - 1. Absorber indices: [0,pad_lo); [shape-pad_hi,shape).

Declared reach tolerance: domain_length × 1e-9. Affected: declared reach = 1 and absorber conductor cells + edges = 0.

| lane | records | affected records | affected indices |
| --- | --- | --- | --- |
| coax_msl_transition | 1 | 1 | 78 |
| coax_two_port | 3 | 1 | 86 |
| lumped_wire | 12 | 0 |  |
| mixed | 1 | 1 | 81 |
| msl_two_port | 23 | 22 | 28, 29, 60, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 80, 82, 83, 84, 85 |
| none | 40 | 0 |  |
| waveguide | 7 | 0 |  |

| index | source :: builder :: variant | lane / port family | shape | face pads | declared boundaries | nonuniform | completed | affected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | examples/inverse_design/differentiable_s11_design.py::_build_sim::default | waveguide | 96, 19, 9 | 8, 8, 0, 0, 0, 0 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 1 | examples/inverse_design/field_observable_shielding.py::_build_sim::default | none | 77, 23, 23 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 2 | examples/inverse_design/multilayer_ar_coating.py::_build_simulation::default | none | 176, 2, 2 | 10, 10, 0, 0, 0, 0 | cpml, cpml, periodic, periodic, periodic, periodic | 0 | 1 | 0 |
| 3 | examples/inverse_design/progressive_demo.py::sim_factory::dx=1.0mm | none | 37, 37, 37 | 6, 6, 6, 6, 6, 6 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 4 | examples/inverse_design/progressive_demo.py::sim_factory::dx=0.5mm | none | 61, 61, 61 | 6, 6, 6, 6, 6, 6 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 5 | examples/quickstart/hello_world.py::build_simulation::default | none | 11, 11, 11 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 6 | examples/tutorials/adi_solver_demo.py::build_cavity::yee_cfl5 | none | 17, 9, 25 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 7 | examples/tutorials/adi_solver_demo.py::build_cavity::adi_cfl2 | none | 17, 9, 25 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 8 | examples/tutorials/adi_solver_demo.py::build_cavity::adi_cfl5 | none | 17, 9, 25 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 9 | examples/tutorials/antenna_farfield_pattern.py::build_simulation::default | none | 53, 53, 53 | 6, 6, 6, 6, 6, 6 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 10 | examples/tutorials/artifact_report_demo.py::build_demo_simulation::default | lumped_wire | 13, 11, 5 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 11 | examples/tutorials/boundary_spec_demo.py::build_simulation::open box | none | 53, 53, 43 | 16, 16, 16, 16, 16, 16 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 12 | examples/tutorials/boundary_spec_demo.py::build_simulation::ground plane | none | 53, 53, 27 | 16, 16, 16, 16, 0, 16 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 0 |
| 13 | examples/tutorials/boundary_spec_demo.py::build_simulation::closed cavity | none | 21, 21, 11 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 14 | examples/tutorials/boundary_spec_demo.py::build_simulation::periodic cell | none | 21, 21, 43 | 0, 0, 0, 0, 16, 16 | periodic, periodic, periodic, periodic, cpml, cpml | 0 | 1 | 0 |
| 15 | examples/tutorials/materials_and_dispersion.py::make_sim::default | none | 23, 23, 23 | 3, 3, 3, 3, 3, 3 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 16 | examples/tutorials/nonuniform_patch_demo.py::build_simulation::default | none | 97, 92, 86 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 1 | 1 | 0 |
| 17 | examples/tutorials/patch_antenna_demo.py::build_simulation::default | none | 132, 132, 90 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 1 | 1 | 0 |
| 18 | examples/tutorials/ports_and_sparams_101.py::build_generic_port_demo::add_component=False | lumped_wire | 24, 24, 24 | 4, 4, 4, 4, 4, 4 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 19 | examples/tutorials/ports_and_sparams_101.py::build_generic_port_demo::add_component=True | lumped_wire | 24, 24, 24 | 4, 4, 4, 4, 4, 4 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 20 | examples/tutorials/ports_and_sparams_101.py::build_microstrip_ports::default | msl_two_port | 87, 47, 23 | 3, 3, 3, 3, 3, 3 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 21 | examples/tutorials/ports_and_sparams_101.py::build_waveguide_ports::default | waveguide | 59, 21, 11 | 4, 4, 0, 0, 0, 0 | cpml, cpml, pec, pec, pec, pec | 0 | 1 | 0 |
| 22 | examples/tutorials/ports_and_sparams_101.py::build_coaxial_port::default | coax_two_port | 51, 51, 51 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 23 | examples/tutorials/rcs_scattering.py::build_preflight_model::default | none | 58, 58, 58 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 24 | examples/tutorials/resonance_harminv.py::build_cavity::default | none | 33, 17, 49 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 25 | examples/tutorials/run_control_and_fields.py::build_simulation::default | none | 41, 41, 41 | 4, 4, 4, 4, 4, 4 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 26 | examples/tutorials/slab_rt_flux_monitor.py::build_sim::with_slab=False | none | 221, 29, 1 | 10, 10, 10, 10, 0, 0 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 27 | examples/tutorials/slab_rt_flux_monitor.py::build_sim::with_slab=True | none | 221, 29, 1 | 10, 10, 10, 10, 0, 0 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 28 | validation/crossval/06b_msl_notch_filter_uniform.py::_build_sim::default | msl_two_port | 553, 280, 37 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 29 | validation/crossval/07_sheen_lpf.py::build_rfx_sim::default | msl_two_port | 155, 149, 28 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 30 | validation/crossval/11_waveguide_port_wr90.py::_build_sim::empty | waveguide | 287, 24, 12 | 43, 43, 0, 0, 0, 0 | cpml, cpml, pec, pec, pec, pec | 0 | 1 | 0 |
| 31 | validation/crossval/11_waveguide_port_wr90.py::_build_sim::pec_short | waveguide | 287, 24, 12 | 43, 43, 0, 0, 0, 0 | cpml, cpml, pec, pec, pec, pec | 0 | 1 | 0 |
| 32 | validation/crossval/14_rect_cavity_pozar.py::build_cavity::dx=1.0mm | none | 51, 31, 41 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 33 | validation/crossval/14_rect_cavity_pozar.py::build_cavity::dx=0.5mm | none | 101, 61, 81 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 34 | validation/crossval/15_patch_antenna_rt5880.py::build_rfx_sim::default | lumped_wire | 108, 121, 50 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 35 | validation/crossval/19_wr90_iris_filter_aghanim.py::build::gated | waveguide | 577, 91, 5 | 110, 110, 0, 0, 0, 0 | cpml, cpml, pec, pec, pec, pec | 0 | 1 | 0 |
| 36 | validation/crossval/24_nu_rect_cavity_pozar.py::build_cavity::uniform | none | 51, 31, 41 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 37 | validation/crossval/24_nu_rect_cavity_pozar.py::build_cavity::single_band | none | 51, 31, 47 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 38 | validation/crossval/24_nu_rect_cavity_pozar.py::build_cavity::multi_band | none | 51, 31, 50 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 39 | validation/crossval/24_nu_rect_cavity_pozar.py::build_cavity::uniform_fine | none | 101, 61, 81 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 40 | validation/research/convergence_floor/fixture.py::build_sim::uniform_s1.0 | none | 37, 31, 55 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 41 | validation/research/convergence_floor/fixture.py::build_sim::multiband_s1.0 | none | 37, 31, 35 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 42 | validation/research/convergence_floor/fixture.py::build_sim::no_trace_s1.0 | none | 37, 31, 55 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 43 | validation/research/issue683_sampling_order_decision.py::build::nu-matched | lumped_wire | 17, 13, 13 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 44 | validation/research/issue764_wireport_norm_falsifiers.py::build_fix_a::short | lumped_wire | 21, 21, 17 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 45 | validation/research/issue770_offdiag_adjudication.py::build_fix_t::uniform-both | lumped_wire | 81, 57, 29 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 0 |
| 46 | validation/research/issue770_offdiag_adjudication.py::build_fix_t::nu-drive0 | lumped_wire | 81, 57, 29 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 1 | 1 | 0 |
| 47 | validation/research/multiband_nu/w4_supraconvergence.py::build_sim::s1.5_multiband | none | 25, 21, 25 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 48 | validation/research/multiband_nu/w4r_port_supraconvergence.py::build_sim::s1.5_multiband | none | 25, 21, 25 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 49 | validation/research/nu_cavity_gates/nu_cavity_gate_scan.py::build_tm110_sim::xy_committed | none | 57, 51, 11 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 50 | validation/research/nu_cavity_gates/nu_cavity_gate_scan.py::build_tm111_sim::z_committed | none | 41, 36, 53 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 1 | 1 | 0 |
| 51 | validation/research/subgrid/12_subgrid_disjoint_prototype.py::_build_disjoint_simulation::default | none | 21, 21, 13 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 52 | validation/research/subgrid/13_subgrid_material_validation.py::build_vacuum_subgrid::vacuum_guarded | none | 21, 21, 13 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 53 | validation/research/subgrid/13_subgrid_material_validation.py::build_dielectric_subgrid::dielectric_centered | none | 21, 21, 13 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 54 | validation/research/subgrid/13_subgrid_material_validation.py::build_uniform_reference::uniform_reference | none | 21, 21, 13 | 0, 0, 0, 0, 0, 0 | pec, pec, pec, pec, pec, pec | 0 | 1 | 0 |
| 55 | validation/research/thru_feedpost_deembed.py::build_thru::band-pulse | lumped_wire | 81, 57, 29 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 0 |
| 56 | validation/research/thru_feedpost_deembed.py::build_thru::insitu-refplane | lumped_wire | 81, 57, 29 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 0 |
| 57 | validation/research/thru_feedpost_twoseg_extraction.py::build_singlepost::refplane-n10 | lumped_wire | 81, 57, 29 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 0 |
| 58 | validation/tmtt_paper/beam_steering_superstrate.py::build_problem::default | none | 65, 65, 68 | 10, 10, 10, 10, 10, 10 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 59 | validation/tmtt_paper/lumped_port_gradient_check.py::build::default | lumped_wire | 57, 37, 31 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 60 | validation/tmtt_paper/msl_stub_notch_tuning.py::build_sim::f_target | msl_two_port | 279, 180, 19 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 61 | validation/tmtt_paper/waveguide_dielectric_taper.py::build_sim::default | waveguide | 116, 19, 9 | 22, 22, 0, 0, 0, 0 | cpml, cpml, pec, pec, pec, pec | 0 | 1 | 0 |
| 62 | tests/unit/sparams/test_msl_internal_probe_advisories.py::_thru::default | msl_two_port | 77, 57, 33 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 1 |
| 63 | tests/unit/sparams/test_msl_passivity_enforcement.py::_thru::default | msl_two_port | 77, 57, 33 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 1 |
| 64 | tests/unit/sparams/test_msl_plane_primitives_smoke.py::_build_thru_line::default | msl_two_port | 279, 180, 19 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 65 | tests/unit/sparams/test_msl_probe_offset_interval.py::_open_thru_sim::default | none | 118, 150, 37 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 66 | tests/unit/sparams/test_msl_sheet_threading.py::build_msl_thru::default | msl_two_port | 159, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 67 | tests/unit/sparams/test_msl_sheet_threading.py::build_msl_thru::pec_sheet | msl_two_port | 159, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 68 | tests/unit/sparams/test_msl_sparse_dft.py::_thru_sim::x_uniform | msl_two_port | 183, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 69 | tests/unit/sparams/test_msl_sparse_dft.py::_thru_sim::x_nonuniform | msl_two_port | 182, 52, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 1 | 1 | 1 |
| 70 | tests/unit/sparams/test_msl_sparse_dft.py::_thru_sim::y_uniform | msl_two_port | 53, 183, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 71 | tests/unit/sparams/test_msl_sparse_dft.py::_thru_sim::y_nonuniform | msl_two_port | 52, 182, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 1 | 1 | 1 |
| 72 | tests/unit/sparams/test_msl_port_integration.py::test_msl_thru_line_passive_gate::default | msl_two_port | 183, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 73 | tests/unit/sparams/test_msl_port_integration.py::test_msl_thru_line_eigenmode_gate::default | msl_two_port | 183, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 74 | tests/unit/sparams/test_msl_port_integration.py::_run_msl_thru::0.008 | msl_two_port | 159, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 75 | tests/unit/sparams/test_msl_port_integration.py::_run_msl_thru::0.01 | msl_two_port | 183, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 76 | tests/unit/sparams/test_msl_port_integration.py::_run_msl_thru::0.012 | msl_two_port | 206, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 77 | tests/unit/sparams/test_coax_two_port_smatrix.py::_sim::default | coax_two_port | 55, 55, 194 | 16, 16, 16, 16, 16, 16 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 0 |
| 78 | tests/unit/sparams/test_coax_msl_transition.py::_build_coax_msl_transition_sim::default | coax_msl_transition | 67, 51, 56 | 8, 8, 8, 8, 8, 8 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 1 |
| 79 | tests/unit/sparams/test_waveguide_lane_pad_continuation.py::_guide::WR90_slab | waveguide | 77, 19, 9 | 8, 8, 0, 0, 0, 0 | cpml, cpml, pec, pec, pec, pec | 0 | 1 | 0 |
| 80 | scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py::_build_sim::cv20_rfx | msl_two_port | 297, 66, 45 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 81 | tests/unit/sparams/test_mixed_port_sparam.py::test_mixed_probe_fed_msl_plumbing_smoke::default | mixed | 112, 53, 18 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 82 | tests/unit/sparams/test_msl_plane_primitives_parity.py::_build_thru::aligned | msl_two_port | 183, 53, 30 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 83 | tests/unit/sparams/test_msl_plane_primitives_parity.py::_build_thru::bisecting | msl_two_port | 192, 54, 31 | 8, 8, 8, 8, 0, 8 | cpml, cpml, cpml, cpml, pec, cpml | 0 | 1 | 1 |
| 84 | tests/unit/sparams/test_msl_power_normalization.py::_case::unequal | msl_two_port | 45, 21, 13 | 2, 2, 2, 2, 2, 2 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 1 |
| 85 | tests/unit/sparams/test_msl_power_normalization.py::_case::equal | msl_two_port | 45, 21, 13 | 2, 2, 2, 2, 2, 2 | cpml, cpml, cpml, cpml, cpml, cpml | 0 | 1 | 1 |
| 86 | tests/unit/sparams/test_coax_two_port_smatrix.py::_sim::default::generated_port_geometry | coax_two_port | 55, 55, 194 | 16, 16, 16, 16, 16, 16 | cpml, cpml, cpml, cpml, cpml, cpml | NA | 1 | 1 |

| index | entry | material / name | conductor | shape | lo m | hi m | sheets | wires |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | _geometry[0] | substrate | 0 | Box | 0.002, 0.002, 0.0005 | 0.008, 0.008, 0.0015 | 0 | 0 |
| 16 | _geometry[0] | fr4 | 0 | Box | 0.01, 0.01, 0.0164355697522 | 0.07, 0.065, 0.0179355697522 | 0 | 0 |
| 16 | _thin_conductors[0] | ThinConductor(shape=Box(corner_lo=(0.010000000000000002, 0.009999999999999998, 0.01643556975219032), corner_hi=(0.07, 0.065, 0.01643556975219032)), sigma_bulk=58000000.0, thickness=3.5e-05, eps_r=1.0, surface_impedance_f0=None) | 1 | Box | 0.01, 0.01, 0.0164355697522 | 0.07, 0.065, 0.0164355697522 | 1 | 0 |
| 16 | _thin_conductors[1] | ThinConductor(shape=Box(corner_lo=(0.02525, 0.0185, 0.017935569752190322), corner_hi=(0.05475, 0.056499999999999995, 0.017935569752190322)), sigma_bulk=58000000.0, thickness=3.5e-05, eps_r=1.0, surface_impedance_f0=None) | 1 | Box | 0.02525, 0.0185, 0.0179355697522 | 0.05475, 0.0565, 0.0179355697522 | 1 | 0 |
| 17 | _geometry[0] | sub | 0 | Box | 0.085, 0.085, 0.0352854919264 | 0.145, 0.145, 0.0368094919264 | 0 | 0 |
| 17 | _thin_conductors[0] | ThinConductor(shape=Box(corner_lo=(0.085, 0.085, 0.035285491926446665), corner_hi=(0.14500000000000002, 0.14500000000000002, 0.035285491926446665)), sigma_bulk=58000000.0, thickness=3.5e-05, eps_r=1.0, surface_impedance_f0=None) | 1 | Box | 0.085, 0.085, 0.0352854919264 | 0.145, 0.145, 0.0352854919264 | 1 | 0 |
| 17 | _thin_conductors[1] | ThinConductor(shape=Box(corner_lo=(0.099, 0.095, 0.03680949192644666), corner_hi=(0.131, 0.135, 0.03680949192644666)), sigma_bulk=58000000.0, thickness=3.5e-05, eps_r=1.0, surface_impedance_f0=None) | 1 | Box | 0.099, 0.095, 0.0368094919264 | 0.131, 0.135, 0.0368094919264 | 1 | 0 |
| 20 | _geometry[0] | substrate | 0 | Box | 0.00075, 0.00075, 0.00125 | 0.01925, 0.00925, 0.00225 | 0 | 0 |
| 20 | _thin_conductors[0] | ThinConductor(shape=Box(corner_lo=(0.00075, 0.00075, 0.00125), corner_hi=(0.01925, 0.00925, 0.00125)), sigma_bulk=58000000.0, thickness=3.5e-05, eps_r=1.0, surface_impedance_f0=None) | 1 | Box | 0.00075, 0.00075, 0.00125 | 0.01925, 0.00925, 0.00125 | 1 | 0 |
| 20 | _thin_conductors[1] | ThinConductor(shape=Box(corner_lo=(0.00075, 0.0045, 0.00225), corner_hi=(0.01925, 0.0055, 0.00225)), sigma_bulk=58000000.0, thickness=3.5e-05, eps_r=1.0, surface_impedance_f0=None) | 1 | Box | 0.00075, 0.0045, 0.00225 | 0.01925, 0.0055, 0.00225 | 1 | 0 |
| 23 | _geometry[0] | pec | 1 | Sphere | 0.0341, 0.0341, 0.0341 | 0.0659, 0.0659, 0.0659 | 0 | 0 |
| 27 | _geometry[0] | slab | 0 | Box | 0.095, -1, -1 | 0.105, 1, 1 | 0 | 0 |
| 28 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.034, 0.016664, 0.000254 | 0 | 0 |
| 28 | _geometry[1] | pec | 1 | Box | 0, 0.001016, 0.000254 | 0.034, 0.001616, 0.000254 | 1 | 0 |
| 28 | _geometry[2] | pec | 1 | Box | 0.0167, 0.001616, 0.000254 | 0.0173, 0.013616, 0.000254 | 1 | 0 |
| 29 | _geometry[0] | duroid | 0 | Box | 0, 0, 0 | 0.027472, 0.02632, 0.000794 | 0 | 0 |
| 29 | _geometry[1] | pec | 1 | Box | 0, 0.00865, 0.000794 | 0.012466, 0.011063, 0.000794 | 1 | 0 |
| 29 | _geometry[2] | pec | 1 | Box | 0.012466, 0.003, 0.000794 | 0.015006, 0.02332, 0.000794 | 1 | 0 |
| 29 | _geometry[3] | pec | 1 | Box | 0.015006, 0.015257, 0.000794 | 0.027472, 0.01767, 0.000794 | 1 | 0 |
| 31 | _geometry[0] | pec | 1 | Box | 0.145, 0, 0 | 0.147, 0.023, 0.011 | 0 | 0 |
| 34 | _geometry[0] | pec | 1 | Box | 0.008, 0.008, 0.0079375 | 0.064, 0.074, 0.0079375 | 1 | 0 |
| 34 | _geometry[1] | sub | 0 | Box | 0.008, 0.008, 0.0079375 | 0.064, 0.074, 0.0111125 | 0 | 0 |
| 34 | _geometry[2] | pec | 1 | Box | 0.016, 0.016, 0.0111125 | 0.056, 0.066, 0.0111125 | 1 | 0 |
| 35 | _geometry[0] | pec | 1 | Box | 0.01016, -1, -1 | 0.012192, 0.00635, 1 | 0 | 0 |
| 35 | _geometry[1] | pec | 1 | Box | 0.01016, 0.01651, -1 | 0.012192, 1, 1 | 0 | 0 |
| 35 | _geometry[2] | pec | 1 | Box | 0.026416, -1, -1 | 0.028448, 0.008128, 1 | 0 | 0 |
| 35 | _geometry[3] | pec | 1 | Box | 0.026416, 0.014732, -1 | 0.028448, 1, 1 | 0 | 0 |
| 35 | _geometry[4] | pec | 1 | Box | 0.044196, -1, -1 | 0.046228, 0.008382, 1 | 0 | 0 |
| 35 | _geometry[5] | pec | 1 | Box | 0.044196, 0.014478, -1 | 0.046228, 1, 1 | 0 | 0 |
| 35 | _geometry[6] | pec | 1 | Box | 0.061976, -1, -1 | 0.064008, 0.008128, 1 | 0 | 0 |
| 35 | _geometry[7] | pec | 1 | Box | 0.061976, 0.014732, -1 | 0.064008, 1, 1 | 0 | 0 |
| 35 | _geometry[8] | pec | 1 | Box | 0.078232, -1, -1 | 0.080264, 0.00635, 1 | 0 | 0 |
| 35 | _geometry[9] | pec | 1 | Box | 0.078232, 0.01651, -1 | 0.080264, 1, 1 | 0 | 0 |
| 40 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.027, 0.0225, 0.0015 | 0 | 0 |
| 40 | _geometry[1] | upper | 0 | Box | 0, 0, 0.0075 | 0.027, 0.0225, 0.009 | 0 | 0 |
| 40 | _geometry[2] | pec | 1 | Box | 0.00675, 0.009, 0.0015 | 0.02025, 0.0135, 0.003 | 0 | 0 |
| 41 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.027, 0.0225, 0.0015 | 0 | 0 |
| 41 | _geometry[1] | upper | 0 | Box | 0, 0, 0.0075 | 0.027, 0.0225, 0.009 | 0 | 0 |
| 41 | _geometry[2] | pec | 1 | Box | 0.00675, 0.009, 0.0015 | 0.02025, 0.0135, 0.003 | 0 | 0 |
| 42 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.027, 0.0225, 0.0015 | 0 | 0 |
| 42 | _geometry[1] | upper | 0 | Box | 0, 0, 0.0075 | 0.027, 0.0225, 0.009 | 0 | 0 |
| 43 | _geometry[0] | pec | 1 | Box | 0.004, 0.005, 0.003 | 0.012, 0.007, 0.005 | 0 | 0 |
| 43 | _geometry[1] | pec | 1 | Box | 0.004, 0.005, 0.007 | 0.012, 0.007, 0.009 | 0 | 0 |
| 44 | _geometry[0] | pec | 1 | Box | 0.0035, 0.0035, 0.0025 | 0.0065, 0.0065, 0.0035 | 0 | 0 |
| 44 | _geometry[1] | pec | 1 | Box | 0.0035, 0.0035, 0.0045 | 0.0065, 0.0065, 0.0055 | 0 | 0 |
| 44 | _geometry[2] | pec | 1 | Box | 0.0045, 0.005, 0.0035 | 0.005, 0.0055, 0.0045 | 0 | 0 |
| 44 | _geometry[3] | pec | 1 | Box | 0.0055, 0.005, 0.0035 | 0.006, 0.0055, 0.0045 | 0 | 0 |
| 44 | _geometry[4] | pec | 1 | Box | 0.005, 0.0045, 0.0035 | 0.0055, 0.005, 0.0045 | 0 | 0 |
| 44 | _geometry[5] | pec | 1 | Box | 0.005, 0.0055, 0.0035 | 0.0055, 0.006, 0.0045 | 0 | 0 |
| 45 | _geometry[0] | pec | 1 | Box | 0.0075, 0.0075, 0.001 | 0.0245, 0.0125, 0.001 | 1 | 0 |
| 46 | _geometry[0] | pec | 1 | Box | 0.0075, 0.0075, 0.001 | 0.0245, 0.0125, 0.001 | 1 | 0 |
| 47 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.027, 0.0225, 0.0015 | 0 | 0 |
| 47 | _geometry[1] | upper | 0 | Box | 0, 0, 0.0075 | 0.027, 0.0225, 0.009 | 0 | 0 |
| 47 | _geometry[2] | pec | 1 | Box | 0.00675, 0.009, 0.0015 | 0.02025, 0.0135, 0.003 | 0 | 0 |
| 48 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.027, 0.0225, 0.0015 | 0 | 0 |
| 48 | _geometry[1] | upper | 0 | Box | 0, 0, 0.0075 | 0.027, 0.0225, 0.009 | 0 | 0 |
| 48 | _geometry[2] | pec | 1 | Box | 0.00675, 0.009, 0.0015 | 0.02025, 0.0135, 0.003 | 0 | 0 |
| 53 | _geometry[0] | dielectric | 0 | Box | -0.001, -0.001, -0.001 | 0.041, 0.041, 0.025 | 0 | 0 |
| 54 | _geometry[0] | dielectric | 0 | Box | -0.001, -0.001, -0.001 | 0.041, 0.041, 0.025 | 0 | 0 |
| 55 | _geometry[0] | pec | 1 | Box | 0.0075, 0.0075, 0.001 | 0.0245, 0.0125, 0.001 | 1 | 0 |
| 56 | _geometry[0] | pec | 1 | Box | 0.0075, 0.0075, 0.001 | 0.0245, 0.0125, 0.001 | 1 | 0 |
| 57 | _geometry[0] | pec | 1 | Box | 0.0075, 0.0075, 0.001 | 0.0285, 0.0125, 0.001 | 1 | 0 |
| 58 | _geometry[0] | pec | 1 | Box | 0.141568660722, 0.141568660722, 0.149896229 | 0.2248443435, 0.2248443435, 0.149896229 | 1 | 0 |
| 60 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.0332, 0.020696, 0.000254 | 0 | 0 |
| 60 | _geometry[1] | pec | 1 | Box | 0, 0.001524, 0.000254 | 0.0332, 0.002124, 0.000254 | 1 | 0 |
| 62 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.012, 0.008, 0.0008 | 0 | 0 |
| 62 | _geometry[1] | pec | 1 | Box | 0, 0, 0 | 0.012, 0.008, 0 | 1 | 0 |
| 62 | _geometry[2] | pec | 1 | Box | 0, 0.0034, 0.0008 | 0.012, 0.0046, 0.0008 | 1 | 0 |
| 63 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.012, 0.008, 0.0008 | 0 | 0 |
| 63 | _geometry[1] | pec | 1 | Box | 0, 0, 0 | 0.012, 0.008, 0 | 1 | 0 |
| 63 | _geometry[2] | pec | 1 | Box | 0, 0.0034, 0.0008 | 0.012, 0.0046, 0.0008 | 1 | 0 |
| 64 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.0332, 0.020696, 0.000254 | 0 | 0 |
| 64 | _geometry[1] | pec | 1 | Box | 0, 0.001524, 0.000254 | 0.0332, 0.002124, 0.000254 | 1 | 0 |
| 65 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.02, 0.02632, 0.000794 | 0 | 0 |
| 65 | _geometry[1] | pec | 1 | Box | 0.001, 0.0119535, 0.000794 | 0.019, 0.0143665, 0.000794 | 1 | 0 |
| 66 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.012, 0.00297066666667, 0.000254 | 0 | 0 |
| 66 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.012, 0.00178533333333, 0.000254 | 1 | 0 |
| 67 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.012, 0.00297066666667, 0.000254 | 0 | 0 |
| 67 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.012, 0.00178533333333, 0.000254 | 1 | 0 |
| 67 | _thin_conductors[0] | ThinConductor(shape=Box(corner_lo=(0.0045, 0.0007853333333333333, 0.0005503333333333332), corner_hi=(0.0075, 0.0021853333333333334, 0.0005503333333333332)), sigma_bulk=58000000.0, thickness=3.5e-05, eps_r=1.0, surface_impedance_f0=None) | 1 | Box | 0.0045, 0.000785333333333, 0.000550333333333 | 0.0075, 0.00218533333333, 0.000550333333333 | 1 | 0 |
| 68 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.014, 0.00297066666667, 0.000254 | 0 | 0 |
| 68 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.014, 0.00178533333333, 0.000254 | 1 | 0 |
| 69 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.014, 0.00297066666667, 0.000254 | 0 | 0 |
| 69 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.014, 0.00178533333333, 0.000254 | 1 | 0 |
| 70 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.00297066666667, 0.014, 0.000254 | 0 | 0 |
| 70 | _geometry[1] | pec | 1 | Box | 0.00118533333333, 0, 0.000254 | 0.00178533333333, 0.014, 0.000254 | 1 | 0 |
| 71 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.00297066666667, 0.014, 0.000254 | 0 | 0 |
| 71 | _geometry[1] | pec | 1 | Box | 0.00118533333333, 0, 0.000254 | 0.00178533333333, 0.014, 0.000254 | 1 | 0 |
| 72 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.014, 0.00297066666667, 0.000254 | 0 | 0 |
| 72 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.014, 0.00178533333333, 0.000254 | 1 | 0 |
| 73 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.014, 0.00297066666667, 0.000254 | 0 | 0 |
| 73 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.014, 0.00178533333333, 0.000254 | 1 | 0 |
| 74 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.012, 0.00297066666667, 0.000254 | 0 | 0 |
| 74 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.012, 0.00178533333333, 0.000254 | 1 | 0 |
| 75 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.014, 0.00297066666667, 0.000254 | 0 | 0 |
| 75 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.014, 0.00178533333333, 0.000254 | 1 | 0 |
| 76 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.016, 0.00297066666667, 0.000254 | 0 | 0 |
| 76 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.016, 0.00178533333333, 0.000254 | 1 | 0 |
| 78 | _geometry[0] | pec | 1 | Box | 0, 0, 0.00245 | 0.005, 0.0034, 0.00255 | 0 | 0 |
| 78 | _geometry[1] | ptfe | 0 | Cylinder | 0.0006, 0.0013, 0.0024 | 0.0014, 0.0021, 0.0027 | 0 | 0 |
| 78 | _geometry[2] | sub | 0 | Box | 0, 0, 0.00255 | 0.005, 0.0034, 0.00285 | 0 | 0 |
| 78 | _geometry[3] | pec | 1 | Box | 0.001, 0.0014, 0.00285 | 0.005, 0.002, 0.00295 | 0 | 0 |
| 78 | _geometry[4] | pec | 1 | Cylinder | 0.0008, 0.0015, 0.0024 | 0.0012, 0.0019, 0.003 | 0 | 0 |
| 79 | _geometry[0] | slab | 0 | Box | 0.0381, 0, 0 | 0.0762, 0.02286, 0.01016 | 0 | 0 |
| 80 | _geometry[0] | ro4350b | 0 | Box | 0, 0, 0 | 0.014, 0.002416, 0.000254 | 0 | 0 |
| 80 | _geometry[1] | pec | 1 | Box | 0, 0.000908, 0.000254 | 0.014, 0.001508, 0.000304 | 0 | 0 |
| 81 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.008, 0.003, 0.000254 | 0 | 0 |
| 81 | _geometry[1] | pec | 1 | Box | 0, 0.0012, 0.000254 | 0.008, 0.0018, 0.000254 | 1 | 0 |
| 82 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.014, 0.00297066666667, 0.000254 | 0 | 0 |
| 82 | _geometry[1] | pec | 1 | Box | 0, 0.00118533333333, 0.000254 | 0.014, 0.00178533333333, 0.000254 | 1 | 0 |
| 83 | _geometry[0] | sub | 0 | Box | 0, 0, 0 | 0.014, 0.002896, 0.000254 | 0 | 0 |
| 83 | _geometry[1] | pec | 1 | Box | 0, 0.001148, 0.000254 | 0.014, 0.001748, 0.000254 | 1 | 0 |
| 84 | _geometry[0] | substrate | 0 | Box | 0, 0, 0 | 0.009765625, 0.00390625, 0.00048828125 | 0 | 0 |
| 84 | _geometry[1] | pec | 1 | Box | 0, 0, 0 | 0.009765625, 0.00390625, 0 | 1 | 0 |
| 84 | _geometry[2] | pec | 1 | Box | 0, 0.00146484375, 0.00048828125 | 0.0048828125, 0.00244140625, 0.00048828125 | 1 | 0 |
| 84 | _geometry[3] | pec | 1 | Box | 0.0048828125, 0.0009765625, 0.00048828125 | 0.009765625, 0.0029296875, 0.00048828125 | 1 | 0 |
| 85 | _geometry[0] | substrate | 0 | Box | 0, 0, 0 | 0.009765625, 0.00390625, 0.00048828125 | 0 | 0 |
| 85 | _geometry[1] | pec | 1 | Box | 0, 0, 0 | 0.009765625, 0.00390625, 0 | 1 | 0 |
| 85 | _geometry[2] | pec | 1 | Box | 0, 0.00146484375, 0.00048828125 | 0.0048828125, 0.00244140625, 0.00048828125 | 1 | 0 |
| 85 | _geometry[3] | pec | 1 | Box | 0.0048828125, 0.00146484375, 0.00048828125 | 0.009765625, 0.00244140625, 0.00048828125 | 1 | 0 |
| 86 | stamp[0] | pin | 1 | Cylinder | 0.003365, 0.003365, 0.0003747405725 | 0.004635, 0.004635, 0.0603332321725 | NA | NA |
| 86 | stamp[1] | shell | 1 | Cylinder difference | 0.001945, 0.001945, 0.0003747405725 | 0.006055, 0.006055, 0.0603332321725 | NA | NA |
| 86 | stamp[2] | dielectric_annulus | 0 | Cylinder | 0.0023197405725, 0.0023197405725, 0.0003747405725 | 0.0056802594275, 0.0056802594275, 0.0603332321725 | NA | NA |

| index | entry | face | declared reach | interior PEC cells | absorber PEC cells | interior Ex | interior Ey | interior Ez | absorber Ex | absorber Ey | absorber Ez | interior eps≠1 | absorber eps≠1 | interior sigma≥1e6 | absorber sigma≥1e6 | affected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _geometry[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[1] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 16 | _thin_conductors[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _geometry[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[1] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 17 | _thin_conductors[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _geometry[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[1] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 20 | _thin_conductors[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 23 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 23 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 23 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 23 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 23 | _geometry[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 23 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 27 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 27 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 27 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 100 | NA | NA | 0 |
| 27 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 100 | NA | NA | 0 |
| 28 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1120 | 8960 | NA | NA | 0 |
| 28 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1120 | 8960 | NA | NA | 0 |
| 28 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2212 | 17696 | NA | NA | 0 |
| 28 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2212 | 17696 | NA | NA | 0 |
| 28 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[1] | x_lo | 1 | 0 | 0 | 10 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 28 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 28 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[2] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[2] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[2] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[2] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 28 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 596 | 4768 | NA | NA | 0 |
| 29 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 596 | 4768 | NA | NA | 0 |
| 29 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 620 | 4960 | NA | NA | 0 |
| 29 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 620 | 4960 | NA | NA | 0 |
| 29 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[1] | x_lo | 1 | 0 | 0 | 12 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 29 | _geometry[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[2] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[2] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[2] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[2] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[3] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[3] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 29 | _geometry[3] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[3] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 29 | _geometry[3] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 31 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 31 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[1] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[2] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[2] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[2] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[2] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[2] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 34 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[2] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[2] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[3] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[3] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[4] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[4] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[5] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[5] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[6] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[6] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[7] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[7] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[8] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[8] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[9] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 35 | _geometry[9] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 45 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 45 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 45 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 45 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 45 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 46 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 46 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 46 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 46 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 46 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 55 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 55 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 55 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 55 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 55 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 56 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 56 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 56 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 56 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 56 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 57 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 57 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 57 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 57 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 57 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 58 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 58 | _geometry[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 58 | _geometry[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 58 | _geometry[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 58 | _geometry[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 58 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 60 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 360 | 2880 | NA | NA | 0 |
| 60 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 360 | 2880 | NA | NA | 0 |
| 60 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 558 | 4464 | NA | NA | 0 |
| 60 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 558 | 4464 | NA | NA | 0 |
| 60 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 60 | _geometry[1] | x_lo | 1 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 60 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 60 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 60 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 60 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 62 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 684 | 5472 | NA | NA | 0 |
| 62 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 684 | 5472 | NA | NA | 0 |
| 62 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 924 | 7392 | NA | NA | 0 |
| 62 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 924 | 7392 | NA | NA | 0 |
| 62 | _geometry[0] | z_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4389 | 35112 | NA | NA | 0 |
| 62 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 62 | _geometry[1] | x_lo | 1 | 0 | 0 | 41 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 62 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 62 | _geometry[1] | y_lo | 1 | 0 | 0 | 60 | 61 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 62 | _geometry[1] | y_hi | 1 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 62 | _geometry[1] | z_lo | 1 | 0 | 0 | 2460 | 2440 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 62 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 62 | _geometry[2] | x_lo | 1 | 0 | 0 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 62 | _geometry[2] | x_hi | 1 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 62 | _geometry[2] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 62 | _geometry[2] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 62 | _geometry[2] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 62 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 63 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 684 | 5472 | NA | NA | 0 |
| 63 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 684 | 5472 | NA | NA | 0 |
| 63 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 924 | 7392 | NA | NA | 0 |
| 63 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 924 | 7392 | NA | NA | 0 |
| 63 | _geometry[0] | z_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4389 | 35112 | NA | NA | 0 |
| 63 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 63 | _geometry[1] | x_lo | 1 | 0 | 0 | 41 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 63 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 63 | _geometry[1] | y_lo | 1 | 0 | 0 | 60 | 61 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 63 | _geometry[1] | y_hi | 1 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 63 | _geometry[1] | z_lo | 1 | 0 | 0 | 2460 | 2440 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 63 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 63 | _geometry[2] | x_lo | 1 | 0 | 0 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 63 | _geometry[2] | x_hi | 1 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 63 | _geometry[2] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 63 | _geometry[2] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 63 | _geometry[2] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 63 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 64 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 360 | 2880 | NA | NA | 0 |
| 64 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 360 | 2880 | NA | NA | 0 |
| 64 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 558 | 4464 | NA | NA | 0 |
| 64 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 558 | 4464 | NA | NA | 0 |
| 64 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 64 | _geometry[1] | x_lo | 1 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 64 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 64 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 64 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 64 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 65 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1800 | 14400 | NA | NA | 0 |
| 65 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1800 | 14400 | NA | NA | 0 |
| 65 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1416 | 11328 | NA | NA | 0 |
| 65 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1416 | 11328 | NA | NA | 0 |
| 65 | _geometry[0] | z_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17700 | 141600 | NA | NA | 0 |
| 65 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 65 | _geometry[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 65 | _geometry[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 65 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 65 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 65 | _geometry[1] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 65 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 66 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 66 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 66 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 477 | 3816 | NA | NA | 0 |
| 66 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 477 | 3816 | NA | NA | 0 |
| 66 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 66 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 66 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 66 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 66 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 66 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 67 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 67 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 477 | 3816 | NA | NA | 0 |
| 67 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 477 | 3816 | NA | NA | 0 |
| 67 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 67 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 67 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _thin_conductors[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _thin_conductors[0] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _thin_conductors[0] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _thin_conductors[0] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 67 | _thin_conductors[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 68 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 68 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 68 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 68 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 68 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 68 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 68 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 68 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 68 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 68 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 69 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 156 | 1248 | NA | NA | 0 |
| 69 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 156 | 1248 | NA | NA | 0 |
| 69 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 546 | 4368 | NA | NA | 0 |
| 69 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 546 | 4368 | NA | NA | 0 |
| 69 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 69 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 69 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 69 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 69 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 69 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 70 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 70 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 70 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 70 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 70 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 70 | _geometry[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 70 | _geometry[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 70 | _geometry[1] | y_lo | 1 | 0 | 0 | 7 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 70 | _geometry[1] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 70 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 71 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 546 | 4368 | NA | NA | 0 |
| 71 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 546 | 4368 | NA | NA | 0 |
| 71 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 156 | 1248 | NA | NA | 0 |
| 71 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 156 | 1248 | NA | NA | 0 |
| 71 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 71 | _geometry[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 71 | _geometry[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 71 | _geometry[1] | y_lo | 1 | 0 | 0 | 7 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 71 | _geometry[1] | y_hi | 1 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 71 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 72 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 72 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 72 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 72 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 72 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 72 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 72 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 72 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 72 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 72 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 73 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 73 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 73 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 73 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 73 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 73 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 73 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 73 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 73 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 73 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 74 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 74 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 74 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 477 | 3816 | NA | NA | 0 |
| 74 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 477 | 3816 | NA | NA | 0 |
| 74 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 74 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 74 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 74 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 74 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 74 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 75 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 75 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 75 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 75 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 75 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 75 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 75 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 75 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 75 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 75 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 76 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 76 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 76 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 618 | 4944 | NA | NA | 0 |
| 76 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 618 | 4944 | NA | NA | 0 |
| 76 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 76 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 76 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 76 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 76 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 76 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[0] | x_lo | 1 | 34 | 0 | 70 | 68 | 35 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 78 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 68 | 35 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 78 | _geometry[0] | y_lo | 1 | 50 | 0 | 100 | 102 | 51 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 78 | _geometry[0] | y_hi | 1 | 0 | 0 | 100 | 0 | 51 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 78 | _geometry[0] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[1] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[1] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[1] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[2] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 153 | 1224 | NA | NA | 0 |
| 78 | _geometry[2] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 153 | 1224 | NA | NA | 0 |
| 78 | _geometry[2] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 201 | 1608 | NA | NA | 0 |
| 78 | _geometry[2] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 201 | 1608 | NA | NA | 0 |
| 78 | _geometry[2] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[3] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[3] | x_hi | 1 | 0 | 0 | 0 | 12 | 7 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 78 | _geometry[3] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[3] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[3] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[3] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[4] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[4] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[4] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[4] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[4] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 78 | _geometry[4] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 79 | _geometry[0] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 79 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 144 | 1152 | NA | NA | 0 |
| 80 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 396 | 3168 | NA | NA | 0 |
| 80 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 396 | 3168 | NA | NA | 0 |
| 80 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1782 | 14256 | NA | NA | 0 |
| 80 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1782 | 14256 | NA | NA | 0 |
| 80 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 80 | _geometry[1] | x_lo | 1 | 12 | 0 | 26 | 24 | 13 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 80 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 24 | 13 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 80 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 80 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 80 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 81 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 81 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 81 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 336 | 2688 | NA | NA | 0 |
| 81 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 336 | 2688 | NA | NA | 0 |
| 81 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 81 | _geometry[1] | x_lo | 1 | 0 | 0 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 81 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 81 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 81 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 81 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 82 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 82 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 159 | 1272 | NA | NA | 0 |
| 82 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 82 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 4392 | NA | NA | 0 |
| 82 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 82 | _geometry[1] | x_lo | 1 | 0 | 0 | 8 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 82 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 82 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 82 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 82 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 83 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 216 | 1728 | NA | NA | 0 |
| 83 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 216 | 1728 | NA | NA | 0 |
| 83 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 768 | 6144 | NA | NA | 0 |
| 83 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 768 | 6144 | NA | NA | 0 |
| 83 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 83 | _geometry[1] | x_lo | 1 | 0 | 0 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 83 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 83 | _geometry[1] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 83 | _geometry[1] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 83 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 84 | 168 | NA | NA | 0 |
| 84 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 84 | 168 | NA | NA | 0 |
| 84 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 180 | 360 | NA | NA | 0 |
| 84 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 180 | 360 | NA | NA | 0 |
| 84 | _geometry[0] | z_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 945 | 1890 | NA | NA | 0 |
| 84 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[1] | x_lo | 1 | 0 | 0 | 17 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 84 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 84 | _geometry[1] | y_lo | 1 | 0 | 0 | 40 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 84 | _geometry[1] | y_hi | 1 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 84 | _geometry[1] | z_lo | 1 | 0 | 0 | 680 | 656 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 84 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[2] | x_lo | 1 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 84 | _geometry[2] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[2] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[2] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[2] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[3] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[3] | x_hi | 1 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 84 | _geometry[3] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[3] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[3] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 84 | _geometry[3] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[0] | x_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 84 | 168 | NA | NA | 0 |
| 85 | _geometry[0] | x_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 84 | 168 | NA | NA | 0 |
| 85 | _geometry[0] | y_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 180 | 360 | NA | NA | 0 |
| 85 | _geometry[0] | y_hi | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 180 | 360 | NA | NA | 0 |
| 85 | _geometry[0] | z_lo | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 945 | 1890 | NA | NA | 0 |
| 85 | _geometry[0] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[1] | x_lo | 1 | 0 | 0 | 17 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 85 | _geometry[1] | x_hi | 1 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 85 | _geometry[1] | y_lo | 1 | 0 | 0 | 40 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 85 | _geometry[1] | y_hi | 1 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 85 | _geometry[1] | z_lo | 1 | 0 | 0 | 680 | 656 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 85 | _geometry[1] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[2] | x_lo | 1 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 85 | _geometry[2] | x_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[2] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[2] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[2] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[2] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[3] | x_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[3] | x_hi | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 1 |
| 85 | _geometry[3] | y_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[3] | y_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[3] | z_lo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 85 | _geometry[3] | z_hi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 |
| 86 | stamp[0] | x_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[0] | x_hi | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[0] | y_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[0] | y_hi | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[0] | z_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[0] | z_hi | 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 8 | 0 | 1 |
| 86 | stamp[1] | x_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[1] | x_hi | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[1] | y_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[1] | y_hi | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[1] | z_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 |
| 86 | stamp[1] | z_hi | 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 32 | 0 | 1 |
| 86 | stamp[2] | x_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | NA | NA | 0 |
| 86 | stamp[2] | x_hi | 0 | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | NA | NA | 0 |
| 86 | stamp[2] | y_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | NA | NA | 0 |
| 86 | stamp[2] | y_hi | 0 | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | NA | NA | 0 |
| 86 | stamp[2] | z_lo | 0 | NA | NA | NA | NA | NA | NA | NA | NA | 0 | 0 | NA | NA | 0 |
| 86 | stamp[2] | z_hi | 1 | NA | NA | NA | NA | NA | NA | NA | NA | 54 | 0 | NA | NA | 0 |
