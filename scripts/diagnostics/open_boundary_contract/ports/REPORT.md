Audited variants: 62. Additional fixture variants: 24. Generated port-geometry records: 1.

| quantity | count |
| --- | --- |
| census records completed | 87 |
| records meeting the brief affected criterion | 25 |
| baseline/continued S-matrix pairs | 2 |
| time-stepped internal drives | 8 |
| solver edge-mask equality checks | 24 |
| coax time-stepped internal drives | 0 |
| coax continuation assertion exceptions | 1 |
| waveguide affected records | 0 |
| lumped/wire affected records | 0 |
| VESSL submissions | 2 |
| VESSL deletions | 0 |
| git commands | 0 |
| exported source edits | 0 |
| existing-file overwrites | 0 |

## Commands and runs

[commands.md](commands.md). Local Python: `/root/workspace/bk-workspace/rfx/.venv/bin/python`; `JAX_PLATFORMS=cpu`; `-B`.

| run id | yaml | fixture | returned S matrices | log |
| --- | --- | --- | --- | --- |
| 369367262622 | vessl_ports_msl.yaml | 62 | 2 | vessl_369367262622.log |
| 369367262623 | vessl_ports_mixed.yaml | 81 | 2 | vessl_369367262623.log |

## Fixture selection

Rank quantity: cells × committed steps. Equal products: ascending census index. Coax-to-MSL/mixed selection: one shared slot.

| lane | fixture | cells | steps | cells × steps | selected |
| --- | --- | --- | --- | --- | --- |
| msl_two_port | 62 | 144837 | 263 | 38092131 | 1 |
| msl_two_port | 63 | 144837 | 263 | 38092131 | 0 |
| msl_two_port | 69 | 283920 | 3717 | 1055330640 | 0 |
| msl_two_port | 71 | 283920 | 3717 | 1055330640 | 0 |
| msl_two_port | 68 | 290970 | 3717 | 1081535490 | 0 |
| msl_two_port | 70 | 290970 | 3717 | 1081535490 | 0 |
| msl_two_port | 66 | 252810 | 14868 | 3758779080 | 0 |
| msl_two_port | 67 | 252810 | 14868 | 3758779080 | 0 |
| msl_two_port | 74 | 252810 | 14868 | 3758779080 | 0 |
| msl_two_port | 72 | 290970 | 14868 | 4326141960 | 0 |
| msl_two_port | 73 | 290970 | 14868 | 4326141960 | 0 |
| msl_two_port | 75 | 290970 | 14868 | 4326141960 | 0 |
| msl_two_port | 76 | 327540 | 14868 | 4869864720 | 0 |
| msl_two_port | 60 | 954180 | 9178 | 8757464040 | 0 |
| msl_two_port | 80 | 882090 | 25177 | 22208379930 | 0 |
| msl_two_port | 28 | 5729080 | 23600 | 135206288000 | 0 |
| mixed | 81 | 106848 | 4956 | 529538688 | 1 |
| coax_msl_transition | 78 | 191352 | 8000 | 1530816000 | 0 |
| msl_two_port | 29 | 646660 | 7868 | 5087920880 | 0 |
| coax_two_port | 77 | 586850 | 6000 | 3521100000 | 1 |

| fixture | committed calculation / record | FDTD calculation eligible |
| --- | --- | --- |
| 64 | forward(num_periods=15); no public S-matrix call | 0 |
| 82 | manufactured plane accumulators | 0 |
| 83 | manufactured plane accumulators | 0 |
| 84 | manufactured backend; n_steps=1 | 0 |
| 85 | manufactured backend; n_steps=1 | 0 |
| 77 | generated geometry: census record 86; stopped during continuation build | 1 |

## A1–A3

| assumption / measured quantity | numerator / value | denominator / unit |
| --- | --- | --- |
| A1: uniform construction | 48 | 62 |
| A1: non-uniform fallback completed | 14 | 14 |
| A1: completed CPU assemblies | 62 | 62 |
| A1: audited assembly maximum wall time | 4.12905318203 | s |
| A1: audited assembly summed wall time | 107.331929728 | s |
| A2: zero-thickness Box trace continuation fixtures with solver read-back | 2 | 2 |
| A3: selected MSL/mixed public calls returning an S matrix | 4 | 4 |
| A3: selected MSL/mixed public-call refusals | 0 | 4 |
| A3: coax continuation solver calls | 0 | 1 |
| A3: coax-MSL transition selected | 0 | 1 |

A1 fallback: `Simulation._build_nonuniform_grid()` and `Simulation._assemble_materials_nu()`. Initial records: `census_records/`; final records: `census_records_v2/`.

| non-uniform audited index | uniform refusal | fallback completed |
| --- | --- | --- |
| 16 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 17 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 37 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 38 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 40 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 41 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 42 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 43 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 44 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 46 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 47 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 48 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 49 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |
| 50 | uniform grid construction does not support the resolved non-uniform mesh. Use a non-uniform-capable entry point or supply an explicit uniform dx= and resolve all geometry thicknesses. | 1 |

| fixture | variant | entry | sheets | face | absorber volume cells | absorber Ex | absorber Ey | absorber Ez |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 62 | baseline | 2 | 1 | x_lo | 0 | 0 | 0 | 0 |
| 62 | baseline | 2 | 1 | x_hi | 0 | 0 | 0 | 0 |
| 62 | continued | 2 | 1 | x_lo | 0 | 56 | 48 | 0 |
| 62 | continued | 2 | 1 | x_hi | 0 | 49 | 48 | 0 |
| 81 | baseline | 1 | 1 | x_lo | 0 | 0 | 0 | 0 |
| 81 | baseline | 1 | 1 | x_hi | 0 | 0 | 0 | 0 |
| 81 | continued | 1 | 1 | x_lo | 0 | 56 | 48 | 0 |
| 81 | continued | 1 | 1 | x_hi | 0 | 49 | 48 | 0 |

Coax continuation stop, verbatim (`dry_coax/fixture_077/continued/status.json`):

```text
Traceback (most recent call last):
  File "/root/workspace/bk-workspace/.801-measure/ports/coax_run.py", line 97, in execute
    result=sim.compute_coaxial_two_port(n_steps=6000,freqs=mod.BAND)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/bk-workspace/.801-measure/src-main/rfx/sparams/coax.py", line 1127, in compute_coaxial_two_port
    materials, shell_inner = stamp_coaxial_line(
                             ^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/bk-workspace/.801-measure/ports/coax_run.py", line 66, in stamp
    assert np.array_equal(old[:,:,:stop],new[:,:,:stop])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

Coax continuation follow-up solves: 0. Coax YAML submissions: 0.

## FACT record checks

| record | measured value | source / record |
| --- | --- | --- |
| uniform assembly tuple length and count | {7: 70} | census.json |
| non-uniform assembly tuple length and count | {4: 16} | census.json |
| declared PEC/PMC faces with pad > 0 | 0 | census.json |
| audited variant triples | 62 | tests/_example_fidelity_lib.py |
| capture script direct iter_audited_variants definition | 0 | scripts/capture_example_fidelity_snapshot.py |
| capture script lib.iter_audited_variants call | 1 | scripts/capture_example_fidelity_snapshot.py |
| source tree | src-main; df08175c | ../PROVENANCE.txt |

`iter_audited_variants` path: `src-main/tests/_example_fidelity_lib.py`; call site: `src-main/scripts/capture_example_fidelity_snapshot.py`, `lib.iter_audited_variants()`.

## Additional recorded messages

Initial local import, verbatim:

```text
Could not save font_manager cache NO_MUTATION: os.remove ('/root/workspace/bk-workspace/.801-measure/ports/mpl_config/fontlist-v3.11.0.json.matplotlib-lock', -1)
```

VESSL 369367262623 initialization, verbatim excerpt; full text: `vessl_369367262623.log`.

```text
ValueError: bad marshal data (unknown type code)
VESSL CLI not installed.
```

Returned resonance-frequency fields: 0 in each of the 4 S-matrix result objects. Resonance estimates added: 0.

## CENSUS.md

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

## TABLE.md

ΔdB = abs(20 log10(abs(S_cont)) − 20 log10(abs(S_base))). Δphase = abs(arg(S_cont × conj(S_base))), principal angle.

Band: every returned frequency bin. Minimum frequencies: sampled bins; interpolation count 0. Interior local minima: strict left, non-strict right; paired by ascending-frequency index.

Reduction floating-point bits: 64. NA: no value.

| lane | fixture | quantity | baseline | continued | difference / comparison | unit |
| --- | --- | --- | --- | --- | --- | --- |
| msl_two_port | 62 | S11.max_abs_magnitude_db_difference | NA | NA | 14.4335138752 | dB |
| msl_two_port | 62 | S11.mean_abs_magnitude_db_difference | NA | NA | 5.61638719909 | dB |
| msl_two_port | 62 | S11.max_abs_phase_difference_deg | NA | NA | 165.290554724 | deg |
| msl_two_port | 62 | S11.mean_abs_phase_difference_deg | NA | NA | 78.0076277357 | deg |
| msl_two_port | 62 | S11.max_abs_linear_magnitude_difference | NA | NA | 1.63648213021e-09 | 1 |
| msl_two_port | 62 | S11.mean_abs_linear_magnitude_difference | NA | NA | 7.73054642285e-10 | 1 |
| msl_two_port | 62 | S11.db_of_max_abs_linear_magnitude_difference | NA | NA | -175.721774653 | dB |
| msl_two_port | 62 | S11.db_of_mean_abs_linear_magnitude_difference | NA | NA | -182.23579615 | dB |
| msl_two_port | 62 | S12.max_abs_magnitude_db_difference | NA | NA | 0 | dB |
| msl_two_port | 62 | S12.mean_abs_magnitude_db_difference | NA | NA | 0 | dB |
| msl_two_port | 62 | S12.max_abs_phase_difference_deg | NA | NA | 1.16447437843e-06 | deg |
| msl_two_port | 62 | S12.mean_abs_phase_difference_deg | NA | NA | 4.90899654769e-07 | deg |
| msl_two_port | 62 | S12.max_abs_linear_magnitude_difference | NA | NA | 0 | 1 |
| msl_two_port | 62 | S12.mean_abs_linear_magnitude_difference | NA | NA | 0 | 1 |
| msl_two_port | 62 | S12.db_of_max_abs_linear_magnitude_difference | NA | NA | -inf | dB |
| msl_two_port | 62 | S12.db_of_mean_abs_linear_magnitude_difference | NA | NA | -inf | dB |
| msl_two_port | 62 | S21.max_abs_magnitude_db_difference | NA | NA | 0 | dB |
| msl_two_port | 62 | S21.mean_abs_magnitude_db_difference | NA | NA | 0 | dB |
| msl_two_port | 62 | S21.max_abs_phase_difference_deg | NA | NA | 1.02750412008e-06 | deg |
| msl_two_port | 62 | S21.mean_abs_phase_difference_deg | NA | NA | 4.81149711265e-07 | deg |
| msl_two_port | 62 | S21.max_abs_linear_magnitude_difference | NA | NA | 0 | 1 |
| msl_two_port | 62 | S21.mean_abs_linear_magnitude_difference | NA | NA | 0 | 1 |
| msl_two_port | 62 | S21.db_of_max_abs_linear_magnitude_difference | NA | NA | -inf | dB |
| msl_two_port | 62 | S21.db_of_mean_abs_linear_magnitude_difference | NA | NA | -inf | dB |
| msl_two_port | 62 | S22.max_abs_magnitude_db_difference | NA | NA | 14.4335130504 | dB |
| msl_two_port | 62 | S22.mean_abs_magnitude_db_difference | NA | NA | 5.61638729886 | dB |
| msl_two_port | 62 | S22.max_abs_phase_difference_deg | NA | NA | 165.290554759 | deg |
| msl_two_port | 62 | S22.mean_abs_phase_difference_deg | NA | NA | 78.0076259815 | deg |
| msl_two_port | 62 | S22.max_abs_linear_magnitude_difference | NA | NA | 1.63648223991e-09 | 1 |
| msl_two_port | 62 | S22.mean_abs_linear_magnitude_difference | NA | NA | 7.73054606918e-10 | 1 |
| msl_two_port | 62 | S22.db_of_max_abs_linear_magnitude_difference | NA | NA | -175.721774071 | dB |
| msl_two_port | 62 | S22.db_of_mean_abs_linear_magnitude_difference | NA | NA | -182.235796547 | dB |
| msl_two_port | 62 | max_column_power | 0.999984741269 | 0.999984741269 | NA | 1 |
| msl_two_port | 62 | max_column_power.port1 | 0.999984741269 | 0.999984741269 | NA | 1 |
| msl_two_port | 62 | max_column_power.port2 | 0.999984741269 | 0.999984741269 | NA | 1 |
| msl_two_port | 62 | max_abs_S12_minus_S21 | 1.14515438243e-08 | 1.47018668351e-08 | NA | 1 |
| msl_two_port | 62 | S11.global_minimum_frequency | 6571428864 | 2000000000 | -69.5652187463 | Hz; delta % |
| msl_two_port | 62 | S11.interior_local_minimum_count | 2 | 1 | NA | 1 |
| msl_two_port | 62 | S11.interior_local_minimum_frequency[0] | 6571428864 | 11142856704 | 69.5652031637 | Hz; delta % |
| msl_two_port | 62 | S11.interior_local_minimum_frequency[1] | 15714285568 | NA | NA | Hz; delta % |
| msl_two_port | 62 | returned_resonance_frequency_fields | 0 | 0 | NA | 1 |
| msl_two_port | 62 | wall_time | 9.69177904911 | 5.6015251088 | NA | s |
| msl_two_port | 62 | frequency_bins | 8 | 8 | NA | 1 |
| msl_two_port | 62 | frequency_low | 2000000000 | 2000000000 | NA | Hz |
| msl_two_port | 62 | frequency_high | 17999998976 | 17999998976 | NA | Hz |
| msl_two_port | 62 | S_raw.max_column_power | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.Z0[0][0].imag | -416.49822998 | -412.897796631 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][0].real | 10.4243068695 | 11.2374343872 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][1].imag | -188.524642944 | -186.794921875 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][1].real | 10.2894630432 | 11.093378067 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][2].imag | -116.368041992 | -115.179840088 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][2].real | 10.0390796661 | 10.8246517181 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][3].imag | -79.4149932861 | -78.4651412964 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][3].real | 9.64977931976 | 10.4036340714 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][4].imag | -56.0547904968 | -55.2267456055 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][4].real | 9.11094093323 | 9.81524276733 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][5].imag | -39.4304618835 | -38.6717338562 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][5].real | 8.43619632721 | 9.07026386261 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][6].imag | -26.6958217621 | -25.9845695496 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][6].real | 7.67082643509 | 8.21371078491 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][7].imag | -16.4439849854 | -15.7756843567 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[0][7].real | 6.87421989441 | 7.30596113205 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][0].imag | -415.7734375 | -407.169464111 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][0].real | 10.4493150711 | 12.0058193207 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][1].imag | -188.184494019 | -184.040740967 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][1].real | 10.3147506714 | 11.8691482544 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][2].imag | -116.144042969 | -113.285385132 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][2].real | 10.0623006821 | 11.6082525253 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][3].imag | -79.2465209961 | -76.9455337524 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][3].real | 9.67228984833 | 11.1958665848 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][4].imag | -55.9186096191 | -53.8893470764 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][4].real | 9.13219928741 | 10.6103963852 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][5].imag | -39.3152427673 | -37.4194526672 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][5].real | 8.45600700378 | 9.85771083832 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][6].imag | -26.5953655243 | -24.7637252808 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][6].real | 7.68871068954 | 8.97919845581 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][7].imag | -16.354593277 | -14.5569601059 | NA | ohm |
| msl_two_port | 62 | diagnostic.Z0[1][7].real | 6.88980674744 | 8.0340719223 | NA | ohm |
| msl_two_port | 62 | diagnostic.beta[0].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[0].real | 36.5543251038 | 36.5543251038 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[1].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[1].real | 78.330696106 | 78.330696106 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[2].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[2].real | 120.107070923 | 120.107070923 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[3].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[3].real | 161.88343811 | 161.88343811 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[4].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[4].real | 203.659805298 | 203.659805298 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[5].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[5].real | 245.436172485 | 245.436172485 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[6].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[6].real | 287.212554932 | 287.212554932 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[7].imag | 0 | 0 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta[7].real | 328.988891602 | 328.988891602 | NA | rad/m |
| msl_two_port | 62 | diagnostic.beta_railed[0][0] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[0][1] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[0][2] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[0][3] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[0][4] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[0][5] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[0][6] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[0][7] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][0] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][1] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][2] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][3] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][4] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][5] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][6] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.beta_railed[1][7] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[0] | 1.09583287476 | 1.11673342136 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[1] | 1.09665512222 | 1.11802449635 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[2] | 1.09813058018 | 1.120347314 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[3] | 1.10031335312 | 1.12380128819 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[4] | 1.10320242405 | 1.12841657886 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[5] | 1.10668281093 | 1.13405354961 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[6] | 1.1105103186 | 1.14037345819 | NA | 1 |
| msl_two_port | 62 | diagnostic.cond_a[7] | 1.11439649564 | 1.14695012746 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[0] | 2.24229768087e-09 | 1.08968389867e-09 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[1] | 9.60907797776e-10 | 7.99203592194e-10 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[2] | 1.50670365073e-10 | 1.31840893758e-09 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[3] | 2.12811990252e-11 | 1.30065402892e-09 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[4] | 1.17637855013e-09 | 1.09408548887e-09 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[5] | 1.89705318121e-09 | 7.00679958499e-10 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[6] | 1.04085784436e-09 | 1.53940127312e-09 | NA | 1 |
| msl_two_port | 62 | diagnostic.passivity_correction[7] | 9.50124867671e-10 | 2.34605979088e-09 | NA | 1 |
| msl_two_port | 62 | diagnostic.probe_clearance[0].deepest_probe_m | 0.0084 | 0.0084 | NA | m |
| msl_two_port | 62 | diagnostic.probe_clearance[0].first_probe_m | 0.006 | 0.006 | NA | m |
| msl_two_port | 62 | diagnostic.probe_clearance[0].recommended_gap_m | 0.00167593294914 | 0.00167593294914 | NA | m |
| msl_two_port | 62 | diagnostic.probe_clearance[0].rule_frequency_hz | 20000000000 | 20000000000 | NA | 1 |
| msl_two_port | 62 | diagnostic.probe_clearance[1].deepest_probe_m | 0.0036 | 0.0036 | NA | m |
| msl_two_port | 62 | diagnostic.probe_clearance[1].first_probe_m | 0.006 | 0.006 | NA | m |
| msl_two_port | 62 | diagnostic.probe_clearance[1].recommended_gap_m | 0.00167593294914 | 0.00167593294914 | NA | m |
| msl_two_port | 62 | diagnostic.probe_clearance[1].rule_frequency_hz | 20000000000 | 20000000000 | NA | 1 |
| msl_two_port | 62 | diagnostic.reference_impedances[0] | 77.7682160415 | 77.7682160415 | NA | ohm |
| msl_two_port | 62 | diagnostic.reference_impedances[1] | 77.7682160415 | 77.7682160415 | NA | ohm |
| msl_two_port | 62 | diagnostic.reliable[0][0] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[0][1] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[0][2] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[0][3] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[0][4] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[0][5] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[0][6] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[0][7] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][0] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][1] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][2] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][3] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][4] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][5] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][6] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.reliable[1][7] | 1 | 1 | NA | 1 |
| msl_two_port | 62 | diagnostic.settling_db[0] | -0.47337104966 | -0.216971492825 | NA | dB |
| msl_two_port | 62 | diagnostic.settling_db[1] | -0.472422241777 | -0.122038920143 | NA | dB |
| msl_two_port | 62 | returned_field_present.S | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.S_raw | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.Z0 | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.assembly | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.beta | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.beta_railed | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.cond_a | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.freqs | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.passivity_correction | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.port_names | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.probe_clearance | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.reference_impedances | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.reliable | 1 | 1 | NA | 1 |
| msl_two_port | 62 | returned_field_present.settling_db | 1 | 1 | NA | 1 |
| mixed | 81 | S11.max_abs_magnitude_db_difference | NA | NA | 2.85014697693 | dB |
| mixed | 81 | S11.mean_abs_magnitude_db_difference | NA | NA | 2.15041298016 | dB |
| mixed | 81 | S11.max_abs_phase_difference_deg | NA | NA | 9.33706427171 | deg |
| mixed | 81 | S11.mean_abs_phase_difference_deg | NA | NA | 6.97345766112 | deg |
| mixed | 81 | S11.max_abs_linear_magnitude_difference | NA | NA | 0.0919368098883 | 1 |
| mixed | 81 | S11.mean_abs_linear_magnitude_difference | NA | NA | 0.0736445119412 | 1 |
| mixed | 81 | S11.db_of_max_abs_linear_magnitude_difference | NA | NA | -20.7302113978 | dB |
| mixed | 81 | S11.db_of_mean_abs_linear_magnitude_difference | NA | NA | -22.657192233 | dB |
| mixed | 81 | S12.max_abs_magnitude_db_difference | NA | NA | 2.34645098474 | dB |
| mixed | 81 | S12.mean_abs_magnitude_db_difference | NA | NA | 2.27023722587 | dB |
| mixed | 81 | S12.max_abs_phase_difference_deg | NA | NA | 7.68259554989 | deg |
| mixed | 81 | S12.mean_abs_phase_difference_deg | NA | NA | 4.0930109509 | deg |
| mixed | 81 | S12.max_abs_linear_magnitude_difference | NA | NA | 0.212870618907 | 1 |
| mixed | 81 | S12.mean_abs_linear_magnitude_difference | NA | NA | 0.206906354975 | 1 |
| mixed | 81 | S12.db_of_max_abs_linear_magnitude_difference | NA | NA | -13.4376855431 | dB |
| mixed | 81 | S12.db_of_mean_abs_linear_magnitude_difference | NA | NA | -13.6845234019 | dB |
| mixed | 81 | S21.max_abs_magnitude_db_difference | NA | NA | 2.24888287265 | dB |
| mixed | 81 | S21.mean_abs_magnitude_db_difference | NA | NA | 1.66184728031 | dB |
| mixed | 81 | S21.max_abs_phase_difference_deg | NA | NA | 16.1245174239 | deg |
| mixed | 81 | S21.mean_abs_phase_difference_deg | NA | NA | 11.3111953893 | deg |
| mixed | 81 | S21.max_abs_linear_magnitude_difference | NA | NA | 0.188750837889 | 1 |
| mixed | 81 | S21.mean_abs_linear_magnitude_difference | NA | NA | 0.151330005138 | 1 |
| mixed | 81 | S21.db_of_max_abs_linear_magnitude_difference | NA | NA | -14.4822222361 | dB |
| mixed | 81 | S21.db_of_mean_abs_linear_magnitude_difference | NA | NA | -16.4014990636 | dB |
| mixed | 81 | S22.max_abs_magnitude_db_difference | NA | NA | 8.25916976009 | dB |
| mixed | 81 | S22.mean_abs_magnitude_db_difference | NA | NA | 6.92827786027 | dB |
| mixed | 81 | S22.max_abs_phase_difference_deg | NA | NA | 160.566659809 | deg |
| mixed | 81 | S22.mean_abs_phase_difference_deg | NA | NA | 138.691969014 | deg |
| mixed | 81 | S22.max_abs_linear_magnitude_difference | NA | NA | 0.125343797485 | 1 |
| mixed | 81 | S22.mean_abs_linear_magnitude_difference | NA | NA | 0.111277992727 | 1 |
| mixed | 81 | S22.db_of_max_abs_linear_magnitude_difference | NA | NA | -18.0379430362 | dB |
| mixed | 81 | S22.db_of_mean_abs_linear_magnitude_difference | NA | NA | -19.0718143382 | dB |
| mixed | 81 | max_column_power | 0.863076198827 | 0.679447208479 | NA | 1 |
| mixed | 81 | max_column_power.port1 | 0.863076198827 | 0.679447208479 | NA | 1 |
| mixed | 81 | max_column_power.port2 | 0.861332704973 | 0.548455027759 | NA | 1 |
| mixed | 81 | max_abs_S12_minus_S21 | 0.181141834163 | 0.0608341306329 | NA | 1 |
| mixed | 81 | S11.global_minimum_frequency | 1000000000 | 1000000000 | 0 | Hz; delta % |
| mixed | 81 | S11.interior_local_minimum_count | 0 | 0 | NA | 1 |
| mixed | 81 | returned_resonance_frequency_fields | 0 | 0 | NA | 1 |
| mixed | 81 | wall_time | 10.484669462 | 6.33164469 | NA | s |
| mixed | 81 | frequency_bins | 5 | 5 | NA | 1 |
| mixed | 81 | frequency_low | 1000000000 | 1000000000 | NA | Hz |
| mixed | 81 | frequency_high | 4000000000 | 4000000000 | NA | Hz |
| mixed | 81 | S_raw.max_column_power | 1.01864308476 | 0.764861783072 | NA | 1 |
| mixed | 81 | S_wave.max_column_power | 1.98421635208 | 0.831924167971 | NA | 1 |
| mixed | 81 | diagnostic.beta_railed[0][0] | 0 | 0 | NA | 1 |
| mixed | 81 | diagnostic.beta_railed[0][1] | 0 | 0 | NA | 1 |
| mixed | 81 | diagnostic.beta_railed[0][2] | 0 | 1 | NA | 1 |
| mixed | 81 | diagnostic.beta_railed[0][3] | 1 | 1 | NA | 1 |
| mixed | 81 | diagnostic.beta_railed[0][4] | 1 | 1 | NA | 1 |
| mixed | 81 | diagnostic.passivity_correction[0] | 0.136342749 | 0.0802990049124 | NA | 1 |
| mixed | 81 | diagnostic.passivity_correction[1] | 0.131897866726 | 0.0695642381907 | NA | 1 |
| mixed | 81 | diagnostic.passivity_correction[2] | 0.124618306756 | 0.0492519140244 | NA | 1 |
| mixed | 81 | diagnostic.passivity_correction[3] | 0.104260310531 | 0.0049112024717 | NA | 1 |
| mixed | 81 | diagnostic.passivity_correction[4] | 0.0464861430228 | 0 | NA | 1 |
| mixed | 81 | diagnostic.probe_clearance[0].deepest_probe_m | 0.003302 | 0.003302 | NA | m |
| mixed | 81 | diagnostic.probe_clearance[0].first_probe_m | 0.00465666666667 | 0.00465666666667 | NA | m |
| mixed | 81 | diagnostic.probe_clearance[0].recommended_gap_m | 0.00670373179654 | 0.00670373179654 | NA | m |
| mixed | 81 | diagnostic.probe_clearance[0].rule_frequency_hz | 5000000000 | 5000000000 | NA | 1 |
| mixed | 81 | diagnostic.reliable[0][0] | 1 | 1 | NA | 1 |
| mixed | 81 | diagnostic.reliable[0][1] | 1 | 1 | NA | 1 |
| mixed | 81 | diagnostic.reliable[0][2] | 1 | 1 | NA | 1 |
| mixed | 81 | diagnostic.reliable[0][3] | 1 | 1 | NA | 1 |
| mixed | 81 | diagnostic.reliable[0][4] | 1 | 1 | NA | 1 |
| mixed | 81 | diagnostic.s21_power_witness[0][0][0] | 1.33668923378 | 0.750589370728 | NA | 1 |
| mixed | 81 | diagnostic.s21_power_witness[0][0][1] | 1.30744111538 | 0.761813223362 | NA | 1 |
| mixed | 81 | diagnostic.s21_power_witness[0][0][2] | 1.24471020699 | 0.778687059879 | NA | 1 |
| mixed | 81 | diagnostic.s21_power_witness[0][0][3] | 1.11799621582 | 0.791477262974 | NA | 1 |
| mixed | 81 | diagnostic.s21_power_witness[0][0][4] | 0.952034473419 | 0.736132621765 | NA | 1 |
| mixed | 81 | diagnostic.settling_db[0] | -9.42222447939 | -12.2800615154 | NA | dB |
| mixed | 81 | diagnostic.settling_db[1] | -9.95538110374 | -12.5513044105 | NA | dB |
| mixed | 81 | diagnostic.z0_ref[0] | 50 | 50 | NA | ohm |
| mixed | 81 | diagnostic.z0_ref[1] | 47.8947999629 | 47.8947999629 | NA | ohm |
| mixed | 81 | returned_field_present.S | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.S_raw | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.S_wave | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.beta_railed | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.freqs | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.magnitude_channel | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.passivity_correction | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.port_families | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.port_names | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.probe_clearance | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.reliable | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.s21_power_witness | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.settling_db | 1 | 1 | NA | 1 |
| mixed | 81 | returned_field_present.z0_ref | 1 | 1 | NA | 1 |
| coax_two_port | 77 | dry_readback_calls.baseline | 1 | NA | NA | count |
| coax_two_port | 77 | FDTD_steps.baseline | 0 | NA | NA | count |
| coax_two_port | 77 | dry_readback_calls.continued | 0 | NA | NA | count |
| coax_two_port | 77 | FDTD_steps.continued | 0 | NA | NA | count |

Coax continuation, verbatim:

```text
Traceback (most recent call last):
  File "/root/workspace/bk-workspace/.801-measure/ports/coax_run.py", line 97, in execute
    result=sim.compute_coaxial_two_port(n_steps=6000,freqs=mod.BAND)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/bk-workspace/.801-measure/src-main/rfx/sparams/coax.py", line 1127, in compute_coaxial_two_port
    materials, shell_inner = stamp_coaxial_line(
                             ^^^^^^^^^^^^^^^^^^^
  File "/root/workspace/bk-workspace/.801-measure/ports/coax_run.py", line 66, in stamp
    assert np.array_equal(old[:,:,:stop],new[:,:,:stop])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

## Files and sizes

| file | bytes |
| --- | --- |
| ports/CENSUS.md | 74001 |
| ports/REPORT.md | 125579 |
| ports/TABLE.md | 22917 |
| ports/catalog.json | 21710 |
| ports/catalog_v2.json | 23393 |
| ports/census.json | 635667 |
| ports/census_driver.py | 4041 |
| ports/census_records/000.json | 1838 |
| ports/census_records/000.log | 212 |
| ports/census_records/001.json | 3268 |
| ports/census_records/001.log | 208 |
| ports/census_records/002.json | 1875 |
| ports/census_records/002.log | 216 |
| ports/census_records/003.json | 3261 |
| ports/census_records/003.log | 207 |
| ports/census_records/004.json | 3261 |
| ports/census_records/004.log | 208 |
| ports/census_records/005.json | 1072 |
| ports/census_records/005.log | 202 |
| ports/census_records/006.json | 1074 |
| ports/census_records/006.log | 202 |
| ports/census_records/007.json | 1071 |
| ports/census_records/007.log | 199 |
| ports/census_records/008.json | 1074 |
| ports/census_records/008.log | 202 |
| ports/census_records/009.json | 3273 |
| ports/census_records/009.log | 215 |
| ports/census_records/010.json | 1760 |
| ports/census_records/010.log | 215 |
| ports/census_records/011.json | 3271 |
| ports/census_records/011.log | 212 |
| ports/census_records/012.json | 2917 |
| ports/census_records/012.log | 215 |
| ports/census_records/013.json | 1097 |
| ports/census_records/013.log | 215 |
| ports/census_records/014.json | 1855 |
| ports/census_records/014.log | 215 |
| ports/census_records/015.json | 3253 |
| ports/census_records/015.log | 200 |
| ports/census_records/016.json | 1628 |
| ports/census_records/016.log | 2418 |
| ports/census_records/017.json | 1619 |
| ports/census_records/017.log | 2821 |
| ports/census_records/018.json | 3305 |
| ports/census_records/018.log | 232 |
| ports/census_records/019.json | 3302 |
| ports/census_records/019.log | 230 |
| ports/census_records/020.json | 14098 |
| ports/census_records/020.log | 1412 |
| ports/census_records/021.json | 1836 |
| ports/census_records/021.log | 216 |
| ports/census_records/022.json | 1098 |
| ports/census_records/022.log | 218 |
| ports/census_records/023.json | 6784 |
| ports/census_records/023.log | 205 |
| ports/census_records/024.json | 1079 |
| ports/census_records/024.log | 205 |
| ports/census_records/025.json | 3273 |
| ports/census_records/025.log | 214 |
| ports/census_records/026.json | 2546 |
| ports/census_records/026.log | 205 |
| ports/census_records/027.json | 5092 |
| ports/census_records/027.log | 204 |
| ports/census_records/028.json | 12417 |
| ports/census_records/028.log | 217 |
| ports/census_records/029.json | 15083 |
| ports/census_records/029.log | 203 |
| ports/census_records/030.json | 1822 |
| ports/census_records/030.log | 205 |
| ports/census_records/031.json | 3453 |
| ports/census_records/031.log | 209 |
| ports/census_records/032.json | 1086 |
| ports/census_records/032.log | 210 |
| ports/census_records/033.json | 1087 |
| ports/census_records/033.log | 211 |
| ports/census_records/034.json | 13844 |
| ports/census_records/034.log | 213 |
| ports/census_records/035.json | 18210 |
| ports/census_records/035.log | 206 |
| ports/census_records/036.json | 1090 |
| ports/census_records/036.log | 212 |
| ports/census_records/037.json | 1632 |
| ports/census_records/037.log | 1277 |
| ports/census_records/038.json | 1629 |
| ports/census_records/038.log | 1275 |
| ports/census_records/039.json | 1101 |
| ports/census_records/039.log | 218 |
| ports/census_records/040.json | 1632 |
| ports/census_records/040.log | 1277 |
| ports/census_records/041.json | 1636 |
| ports/census_records/041.log | 1279 |
| ports/census_records/042.json | 1636 |
| ports/census_records/042.log | 1280 |
| ports/census_records/043.json | 1633 |
| ports/census_records/043.log | 1277 |
| ports/census_records_v2/000.json | 1955 |
| ports/census_records_v2/000.log | 131 |
| ports/census_records_v2/001.json | 3385 |
| ports/census_records_v2/001.log | 126 |
| ports/census_records_v2/002.json | 1986 |
| ports/census_records_v2/002.log | 125 |
| ports/census_records_v2/003.json | 3372 |
| ports/census_records_v2/003.log | 127 |
| ports/census_records_v2/004.json | 3372 |
| ports/census_records_v2/004.log | 128 |
| ports/census_records_v2/005.json | 1182 |
| ports/census_records_v2/005.log | 125 |
| ports/census_records_v2/006.json | 1185 |
| ports/census_records_v2/006.log | 126 |
| ports/census_records_v2/007.json | 1182 |
| ports/census_records_v2/007.log | 126 |
| ports/census_records_v2/008.json | 1185 |
| ports/census_records_v2/008.log | 126 |
| ports/census_records_v2/009.json | 3384 |
| ports/census_records_v2/009.log | 128 |
| ports/census_records_v2/010.json | 1877 |
| ports/census_records_v2/010.log | 132 |
| ports/census_records_v2/011.json | 3382 |
| ports/census_records_v2/011.log | 129 |
| ports/census_records_v2/012.json | 3028 |
| ports/census_records_v2/012.log | 128 |
| ports/census_records_v2/013.json | 1208 |
| ports/census_records_v2/013.log | 127 |
| ports/census_records_v2/014.json | 1966 |
| ports/census_records_v2/014.log | 128 |
| ports/census_records_v2/015.json | 3371 |
| ports/census_records_v2/015.log | 128 |
| ports/census_records_v2/016.json | 14509 |
| ports/census_records_v2/016.log | 1268 |
| ports/census_records_v2/017.json | 14459 |
| ports/census_records_v2/017.log | 1678 |
| ports/census_records_v2/018.json | 3423 |
| ports/census_records_v2/018.log | 135 |
| ports/census_records_v2/019.json | 3420 |
| ports/census_records_v2/019.log | 135 |
| ports/census_records_v2/020.json | 14214 |
| ports/census_records_v2/020.log | 1326 |
| ports/census_records_v2/021.json | 1953 |
| ports/census_records_v2/021.log | 132 |
| ports/census_records_v2/022.json | 1216 |
| ports/census_records_v2/022.log | 138 |
| ports/census_records_v2/023.json | 6902 |
| ports/census_records_v2/023.log | 129 |
| ports/census_records_v2/024.json | 1190 |
| ports/census_records_v2/024.log | 128 |
| ports/census_records_v2/025.json | 3384 |
| ports/census_records_v2/025.log | 128 |
| ports/census_records_v2/026.json | 2663 |
| ports/census_records_v2/026.log | 126 |
| ports/census_records_v2/027.json | 5208 |
| ports/census_records_v2/027.log | 125 |
| ports/census_records_v2/028.json | 12419 |
| ports/census_records_v2/028.log | 137 |
| ports/census_records_v2/029.json | 15200 |
| ports/census_records_v2/029.log | 136 |
| ports/census_records_v2/030.json | 1939 |
| ports/census_records_v2/030.log | 132 |
| ports/census_records_v2/031.json | 3569 |
| ports/census_records_v2/031.log | 131 |
| ports/census_records_v2/032.json | 1196 |
| ports/census_records_v2/032.log | 127 |
| ports/census_records_v2/033.json | 1198 |
| ports/census_records_v2/033.log | 129 |
| ports/census_records_v2/034.json | 13961 |
| ports/census_records_v2/034.log | 135 |
| ports/census_records_v2/035.json | 18327 |
| ports/census_records_v2/035.log | 133 |
| ports/census_records_v2/036.json | 1200 |
| ports/census_records_v2/036.log | 127 |
| ports/census_records_v2/037.json | 1294 |
| ports/census_records_v2/037.log | 126 |
| ports/census_records_v2/038.json | 1292 |
| ports/census_records_v2/038.log | 126 |
| ports/census_records_v2/039.json | 1210 |
| ports/census_records_v2/039.log | 127 |
| ports/census_records_v2/040.json | 3277 |
| ports/census_records_v2/040.log | 126 |
| ports/census_records_v2/041.json | 3280 |
| ports/census_records_v2/041.log | 125 |
| ports/census_records_v2/042.json | 2613 |
| ports/census_records_v2/042.log | 125 |
| ports/census_records_v2/043.json | 2618 |
| ports/census_records_v2/043.log | 132 |
| ports/census_records_v2/044.json | 5285 |
| ports/census_records_v2/044.log | 132 |
| ports/census_records_v2/045.json | 5963 |
| ports/census_records_v2/045.log | 134 |
| ports/census_records_v2/046.json | 6138 |
| ports/census_records_v2/046.log | 134 |
| ports/census_records_v2/047.json | 3296 |
| ports/census_records_v2/047.log | 125 |
| ports/census_records_v2/048.json | 3309 |
| ports/census_records_v2/048.log | 126 |
| ports/census_records_v2/049.json | 1373 |
| ports/census_records_v2/049.log | 126 |
| ports/census_records_v2/050.json | 1339 |
| ports/census_records_v2/050.log | 126 |
| ports/census_records_v2/051.json | 1162 |
| ports/census_records_v2/051.log | 125 |
| ports/census_records_v2/052.json | 1166 |
| ports/census_records_v2/052.log | 125 |
| ports/census_records_v2/053.json | 1847 |
| ports/census_records_v2/053.log | 125 |
| ports/census_records_v2/054.json | 1840 |
| ports/census_records_v2/054.log | 124 |
| ports/census_records_v2/055.json | 5940 |
| ports/census_records_v2/055.log | 133 |
| ports/census_records_v2/056.json | 5950 |
| ports/census_records_v2/056.log | 133 |
| ports/census_records_v2/057.json | 5976 |
| ports/census_records_v2/057.log | 133 |
| ports/census_records_v2/058.json | 6895 |
| ports/census_records_v2/058.log | 127 |
| ports/census_records_v2/059.json | 3284 |
| ports/census_records_v2/059.log | 132 |
| ports/census_records_v2/060.json | 9237 |
| ports/census_records_v2/060.log | 135 |
| ports/census_records_v2/061.json | 1862 |
| ports/census_records_v2/061.log | 131 |
| ports/census_records_v2/062.json | 14257 |
| ports/census_records_v2/062.log | 806 |
| ports/census_records_v2/063.json | 14429 |
| ports/census_records_v2/063.log | 806 |
| ports/census_records_v2/064.json | 9085 |
| ports/census_records_v2/064.log | 134 |
| ports/census_records_v2/065.json | 10585 |
| ports/census_records_v2/065.log | 127 |
| ports/census_records_v2/066.json | 9641 |
| ports/census_records_v2/066.log | 134 |
| ports/census_records_v2/067.json | 12995 |
| ports/census_records_v2/067.log | 135 |
| ports/census_records_v2/068.json | 9311 |
| ports/census_records_v2/068.log | 134 |
| ports/census_records_v2/069.json | 9514 |
| ports/census_records_v2/069.log | 134 |
| ports/census_records_v2/070.json | 9312 |
| ports/census_records_v2/070.log | 135 |
| ports/census_records_v2/071.json | 9515 |
| ports/census_records_v2/071.log | 135 |
| ports/census_records_v2/072.json | 9342 |
| ports/census_records_v2/072.log | 135 |
| ports/census_records_v2/073.json | 9346 |
| ports/census_records_v2/073.log | 135 |
| ports/census_records_v2/074.json | 9324 |
| ports/census_records_v2/074.log | 135 |
| ports/census_records_v2/075.json | 9322 |
| ports/census_records_v2/075.log | 135 |
| ports/census_records_v2/076.json | 9324 |
| ports/census_records_v2/076.log | 134 |
| ports/census_records_v2/077.json | 3595 |
| ports/census_records_v2/077.log | 136 |
| ports/census_records_v2/078.json | 21343 |
| ports/census_records_v2/078.log | 142 |
| ports/census_records_v2/079.json | 3748 |
| ports/census_records_v2/079.log | 131 |
| ports/census_records_v2/080.json | 9320 |
| ports/census_records_v2/080.log | 134 |
| ports/census_records_v2/081.json | 9442 |
| ports/census_records_v2/081.log | 127 |
| ports/census_records_v2/082.json | 9177 |
| ports/census_records_v2/082.log | 134 |
| ports/census_records_v2/083.json | 9127 |
| ports/census_records_v2/083.log | 807 |
| ports/census_records_v2/084.json | 17533 |
| ports/census_records_v2/084.log | 801 |
| ports/census_records_v2/085.json | 17530 |
| ports/census_records_v2/085.log | 802 |
| ports/census_records_v2/086.json | 5108 |
| ports/census_v2.py | 8060 |
| ports/coax_port_geometry_received.npz | 5905 |
| ports/coax_run.py | 7706 |
| ports/collect_runs.py | 890 |
| ports/commands.md | 20040 |
| ports/comparison.json | 5094 |
| ports/dry_coax/fixture_077/baseline/assembly_before_solve.json | 904 |
| ports/dry_coax/fixture_077/baseline/assembly_before_solve.npz | 5562 |
| ports/dry_coax/fixture_077/baseline/assembly_received_00.json | 567 |
| ports/dry_coax/fixture_077/baseline/assembly_received_00.npz | 6882 |
| ports/dry_coax/fixture_077/baseline/preflight.txt | 51 |
| ports/dry_coax/fixture_077/baseline/run.log | 1073 |
| ports/dry_coax/fixture_077/baseline/settings.json | 205 |
| ports/dry_coax/fixture_077/baseline/status.json | 155 |
| ports/dry_coax/fixture_077/continued/preflight.txt | 51 |
| ports/dry_coax/fixture_077/continued/run.log | 699 |
| ports/dry_coax/fixture_077/continued/settings.json | 205 |
| ports/dry_coax/fixture_077/continued/status.json | 949 |
| ports/dry_coax/fixture_077/provenance.json | 651 |
| ports/dry_readback/fixture_062/baseline/assembly_before_solve.json | 13438 |
| ports/dry_readback/fixture_062/baseline/assembly_before_solve.npz | 5394 |
| ports/dry_readback/fixture_062/baseline/assembly_received_00.json | 2859 |
| ports/dry_readback/fixture_062/baseline/assembly_received_00.npz | 6988 |
| ports/dry_readback/fixture_062/baseline/preflight.txt | 5326 |
| ports/dry_readback/fixture_062/baseline/run.log | 9705 |
| ports/dry_readback/fixture_062/baseline/settings.json | 266 |
| ports/dry_readback/fixture_062/baseline/status.json | 184 |
| ports/dry_readback/fixture_062/continued/assembly_before_solve.json | 14640 |
| ports/dry_readback/fixture_062/continued/assembly_before_solve.npz | 5730 |
| ports/dry_readback/fixture_062/continued/assembly_received_00.json | 2949 |
| ports/dry_readback/fixture_062/continued/assembly_received_00.npz | 7324 |
| ports/dry_readback/fixture_062/continued/preflight.txt | 7888 |
| ports/dry_readback/fixture_062/continued/run.log | 12629 |
| ports/dry_readback/fixture_062/continued/settings.json | 266 |
| ports/dry_readback/fixture_062/continued/status.json | 187 |
| ports/dry_readback/fixture_062/provenance.json | 1798 |
| ports/dry_readback/fixture_081/baseline/assembly_before_solve.json | 8598 |
| ports/dry_readback/fixture_081/baseline/assembly_before_solve.npz | 3429 |
| ports/dry_readback/fixture_081/baseline/assembly_received_00.json | 2549 |
| ports/dry_readback/fixture_081/baseline/assembly_received_00.npz | 4733 |
| ports/dry_readback/fixture_081/baseline/preflight.txt | 11216 |
| ports/dry_readback/fixture_081/baseline/run.log | 3717 |
| ports/dry_readback/fixture_081/baseline/settings.json | 232 |
| ports/dry_readback/fixture_081/baseline/status.json | 178 |
| ports/dry_readback/fixture_081/continued/assembly_before_solve.json | 9098 |
| ports/dry_readback/fixture_081/continued/assembly_before_solve.npz | 3495 |
| ports/dry_readback/fixture_081/continued/assembly_received_00.json | 2553 |
| ports/dry_readback/fixture_081/continued/assembly_received_00.npz | 4799 |
| ports/dry_readback/fixture_081/continued/preflight.txt | 11744 |
| ports/dry_readback/fixture_081/continued/run.log | 3971 |
| ports/dry_readback/fixture_081/continued/settings.json | 232 |
| ports/dry_readback/fixture_081/continued/status.json | 180 |
| ports/dry_readback/fixture_081/provenance.json | 1786 |
| ports/final_report.py | 9073 |
| ports/measure_common.py | 10027 |
| ports/mixed/fixture_081/baseline/assembly_before_solve.json | 8598 |
| ports/mixed/fixture_081/baseline/assembly_before_solve.npz | 3429 |
| ports/mixed/fixture_081/baseline/assembly_received_00.json | 2549 |
| ports/mixed/fixture_081/baseline/assembly_received_00.npz | 4733 |
| ports/mixed/fixture_081/baseline/assembly_received_01.json | 2549 |
| ports/mixed/fixture_081/baseline/assembly_received_01.npz | 4733 |
| ports/mixed/fixture_081/baseline/diagnostics.json | 7567 |
| ports/mixed/fixture_081/baseline/diagnostics.npz | 2571 |
| ports/mixed/fixture_081/baseline/preflight.txt | 11216 |
| ports/mixed/fixture_081/baseline/run.log | 63373 |
| ports/mixed/fixture_081/baseline/s.npz | 581 |
| ports/mixed/fixture_081/baseline/settings.json | 232 |
| ports/mixed/fixture_081/baseline/status.json | 210 |
| ports/mixed/fixture_081/baseline/witness_series_00.npz | 87439 |
| ports/mixed/fixture_081/baseline/witness_series_01.npz | 85667 |
| ports/mixed/fixture_081/continued/assembly_before_solve.json | 9098 |
| ports/mixed/fixture_081/continued/assembly_before_solve.npz | 3495 |
| ports/mixed/fixture_081/continued/assembly_received_00.json | 2553 |
| ports/mixed/fixture_081/continued/assembly_received_00.npz | 4799 |
| ports/mixed/fixture_081/continued/assembly_received_01.json | 2553 |
| ports/mixed/fixture_081/continued/assembly_received_01.npz | 4799 |
| ports/mixed/fixture_081/continued/diagnostics.json | 7516 |
| ports/mixed/fixture_081/continued/diagnostics.npz | 2560 |
| ports/mixed/fixture_081/continued/preflight.txt | 11744 |
| ports/mixed/fixture_081/continued/run.log | 63632 |
| ports/mixed/fixture_081/continued/s.npz | 580 |
| ports/mixed/fixture_081/continued/settings.json | 232 |
| ports/mixed/fixture_081/continued/status.json | 210 |
| ports/mixed/fixture_081/continued/witness_series_00.npz | 87106 |
| ports/mixed/fixture_081/continued/witness_series_01.npz | 86817 |
| ports/mixed/fixture_081/provenance.json | 1787 |
| ports/mpl_config/fontlist-v3.11.0.json | 26961 |
| ports/mpl_config/fontlist-v3.11.0.json.matplotlib-lock | 0 |
| ports/msl_two_port/fixture_062/baseline/assembly_before_solve.json | 13438 |
| ports/msl_two_port/fixture_062/baseline/assembly_before_solve.npz | 5394 |
| ports/msl_two_port/fixture_062/baseline/assembly_received_00.json | 2859 |
| ports/msl_two_port/fixture_062/baseline/assembly_received_00.npz | 6988 |
| ports/msl_two_port/fixture_062/baseline/assembly_received_01.json | 2859 |
| ports/msl_two_port/fixture_062/baseline/assembly_received_01.npz | 6988 |
| ports/msl_two_port/fixture_062/baseline/diagnostics.json | 10610 |
| ports/msl_two_port/fixture_062/baseline/diagnostics.npz | 2905 |
| ports/msl_two_port/fixture_062/baseline/preflight.txt | 5326 |
| ports/msl_two_port/fixture_062/baseline/run.log | 22009 |
| ports/msl_two_port/fixture_062/baseline/s.npz | 635 |
| ports/msl_two_port/fixture_062/baseline/settings.json | 266 |
| ports/msl_two_port/fixture_062/baseline/status.json | 216 |
| ports/msl_two_port/fixture_062/baseline/witness_series_00.npz | 8873 |
| ports/msl_two_port/fixture_062/baseline/witness_series_01.npz | 8871 |
| ports/msl_two_port/fixture_062/continued/assembly_before_solve.json | 14640 |
| ports/msl_two_port/fixture_062/continued/assembly_before_solve.npz | 5730 |
| ports/msl_two_port/fixture_062/continued/assembly_received_00.json | 2949 |
| ports/msl_two_port/fixture_062/continued/assembly_received_00.npz | 7324 |
| ports/msl_two_port/fixture_062/continued/assembly_received_01.json | 2949 |
| ports/msl_two_port/fixture_062/continued/assembly_received_01.npz | 7324 |
| ports/msl_two_port/fixture_062/continued/diagnostics.json | 10606 |
| ports/msl_two_port/fixture_062/continued/diagnostics.npz | 2905 |
| ports/msl_two_port/fixture_062/continued/preflight.txt | 7888 |
| ports/msl_two_port/fixture_062/continued/run.log | 26569 |
| ports/msl_two_port/fixture_062/continued/s.npz | 638 |
| ports/msl_two_port/fixture_062/continued/settings.json | 266 |
| ports/msl_two_port/fixture_062/continued/status.json | 217 |
| ports/msl_two_port/fixture_062/continued/witness_series_00.npz | 8867 |
| ports/msl_two_port/fixture_062/continued/witness_series_01.npz | 8866 |
| ports/msl_two_port/fixture_062/provenance.json | 1799 |
| ports/port_geometry.py | 3511 |
| ports/port_run.py | 12108 |
| ports/reduce_ports.py | 12165 |
| ports/run_id.txt | 70 |
| ports/runs.json | 178 |
| ports/verification.json | 4443 |
| ports/vessl_369367262622.log | 55948 |
| ports/vessl_369367262622_status.txt | 1756 |
| ports/vessl_369367262623.log | 144756 |
| ports/vessl_369367262623_status.txt | 1758 |
| ports/vessl_ports_mixed.yaml | 842 |
| ports/vessl_ports_msl.yaml | 840 |
