"""tests/_jax_version_xfails.py -- strict xfails for tests that fail on a newer jax for a known cause.

The weekly Python 3.12 / newest-jax job (validation.yml, ``latest-jax``) installs jax above the install cap.
Its first run (jax 0.11.2, run 40468718500) found three causes: #2114 is fixed by #2108 and #2115 by #2117, so
neither is listed; the 4 tests of #2116 stay under #2125, the open slowdown issue. Its third run (jax 0.12.0.1,
run 40632569850) found #2167 (a two-device mirror case).
The jax 0.12.1 run 44523333144 found the newest-jax failures tracked in #2896.
Each listed node id gets ``xfail(strict=True, reason="#N ...")`` on the entry's jax version or later and nothing on
older jax, where these tests pass. The fix for an issue deletes that issue's entry: a fixed test still listed turns
red (XPASS strict).

Ids are exact node ids, parameters included, because several parametrized tests fail for some cases only.
"""
import pytest

JAX_XFAILS = {
    2125: (
        (0, 11),
        "the compiled RLC step holds 26 domain copies on jaxlib 0.11.2 and later against 8 on 0.10.2 (measured in #2116)",
        (
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[parallel-graded]",
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[parallel-uniform]",
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[series-graded]",
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[series-uniform]",
        ),
    ),
    2167: (
        (0, 12),
        "C2 mirror bar (200 eps of its own peak over 200 steps) is below one run's float32 rounding; holds only for identically rounding programs",
        (
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pec-two_graded_forward]",
        ),
    ),
    2896: (
        (0, 12, 1),
        "jax 0.12.1 newest-jax lane: cause undetermined; rounding-level rows await the #2166 "
        "bars, the others their lanes",
        (
            "tests/contracts/path_equivalence/test_source_on_occupied_edge.py::test_g1b_uniform_vs_constant_graded[0.0625]",
            "tests/contracts/path_equivalence/test_source_on_occupied_edge.py::test_g1b_uniform_vs_constant_graded[0.25]",
            "tests/contracts/path_equivalence/test_source_on_occupied_edge.py::test_g1b_uniform_vs_constant_graded[0.5]",
            "tests/contracts/path_equivalence/test_source_on_occupied_edge.py::test_g1b_uniform_vs_constant_graded[1.0]",
            "tests/contracts/path_equivalence/test_step_reads_published_keep.py::test_occupancy_probe_records_match_public_paths[graded]",
            "tests/contracts/test_ad_memory_tape_terms.py::test_ad_memory_tape_terms",
            "tests/contracts/test_ad_memory_tape_terms.py::test_closed_box_record_memory_slope[record-chosen]",
            "tests/contracts/test_cpml_writeback_backend.py::test_selected_backend_probe_plain_vs_jit[120-graded]",
            "tests/contracts/test_cpml_writeback_backend.py::test_selected_backend_probe_plain_vs_jit[240-graded]",
            "tests/contracts/test_cpml_writeback_backend.py::test_selected_backend_probe_plain_vs_jit[480-graded]",
            "tests/contracts/test_design_box_constant_ca.py::test_observation_residuals_match_box_estimate[2-combined-9]",
            "tests/contracts/test_design_box_constant_ca.py::test_observation_residuals_match_box_estimate[2-flux-4]",
            "tests/contracts/test_design_box_constant_ca.py::test_observation_residuals_match_box_estimate[2-ntff-4]",
            "tests/contracts/test_design_box_constant_ca.py::test_observation_residuals_match_box_estimate[5-combined-9]",
            "tests/contracts/test_design_box_constant_ca.py::test_observation_residuals_match_box_estimate[5-flux-4]",
            "tests/contracts/test_design_box_constant_ca.py::test_observation_residuals_match_box_estimate[5-ntff-4]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[i-2-pmc-graded_forward]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[i-2-pmc-graded_run]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[i-2-pmc-two_graded_forward]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pec-graded_forward]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pec-graded_run]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pec-two_uniform_run]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pmc-graded_forward]",
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pmc-graded_run]",
            "tests/contracts/test_measurement_plan_accumulators.py::test_b1_bits[full-uniform-decay]",
            "tests/contracts/test_measurement_plan_accumulators.py::test_b1_bits[full-uniform-forward]",
            "tests/contracts/test_measurement_plan_accumulators.py::test_b1_bits[full-uniform-run]",
            "tests/contracts/test_measurement_plan_accumulators.py::test_b1_bits[hann-uniform-forward]",
            "tests/contracts/test_measurement_plan_accumulators.py::test_b1_bits[hann-uniform-run]",
            "tests/contracts/test_multidevice_lumped_rlc.py::test_multidevice_lumped_rlc[pec-S3-400-uniform]",
            "tests/contracts/test_multidevice_sheets_wires.py::test_b2_visible_fields[pec-False-sheet_x_seam]",
            "tests/contracts/test_multidevice_sheets_wires.py::test_b2_visible_fields[pec-False-sheet_z_cross]",
            "tests/contracts/test_multidevice_sheets_wires.py::test_b2_visible_fields[pec-False-sheet_z_edge]",
            "tests/contracts/test_multidevice_sheets_wires.py::test_b2_visible_fields[pec-False-wire_z_seam]",
            "tests/contracts/test_multidevice_wire_radius.py::test_probe_records[graded-pec-R1]",
            "tests/contracts/test_multidevice_wire_radius.py::test_probe_records[graded-pec-R3]",
            "tests/contracts/test_multidevice_wire_radius.py::test_probe_records[graded-pec-R4]",
            "tests/contracts/test_multidevice_wire_radius.py::test_probe_records[uniform-pec-R1]",
            "tests/contracts/test_multidevice_wire_radius.py::test_probe_records[uniform-pec-R3]",
            "tests/contracts/test_multidevice_wire_radius.py::test_probe_records[uniform-pec-R4]",
            "tests/contracts/test_oblique_bloch_material_admission.py::test_supported_probe_record_bytes_unchanged[debye-ez-0-f09bc39f1e0095e1e585cdfe5dd05e4afa593b5c8cad3c5547022f0b93b73efe]",
            "tests/contracts/test_oblique_bloch_material_admission.py::test_supported_probe_record_bytes_unchanged[diagonal-ez-0-a537be2179a2fd5d1f44280478427dea6ab14625e2a7b463df98021cc56c64b8]",
            "tests/contracts/test_oblique_bloch_material_admission.py::test_supported_probe_record_bytes_unchanged[plain-ey-30-5c0b0fa223c360448be858d0298a30f4bb0e6a645b9f947433fea5d850e30f27]",
            "tests/contracts/test_oblique_bloch_material_admission.py::test_supported_probe_record_bytes_unchanged[plain-ez-30-5c0b0fa223c360448be858d0298a30f4bb0e6a645b9f947433fea5d850e30f27]",
            "tests/contracts/test_settling_witness_shared.py::test_two_port_drive_witnesses_tell_driven_from_passive",
            "tests/contracts/test_uniform_distributed_main_s.py::test_main_record_and_s[forward-lumped-pec]",
            "tests/contracts/test_uniform_distributed_main_s.py::test_main_record_and_s[run-lumped-pec]",
            "tests/unit/autodiff/test_design_box_tape.py::test_design_box_on_a_reflecting_wall_is_accepted_and_exact[pmc-graded]",
            "tests/unit/autodiff/test_forward_jit_contract.py::test_jit_with_no_traced_input_equals_the_plain_call[lumped port, cpml/graded]",
        ),
    ),
}


_ENTRIES = {nodeid: (since, f"#{issue} {reason}")
            for issue, (since, reason, ids) in JAX_XFAILS.items() for nodeid in ids}


def reason_for(nodeid, jax_version):
    """The xfail reason for this node id under this jax version (a tuple), or None."""
    since, reason = _ENTRIES.get(nodeid, (None, None))
    if since is None or tuple(jax_version[:len(since)]) < since:
        return None
    return reason


def pytest_collection_modifyitems(config, items):
    import jax

    parts = []
    for part in jax.__version__.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    version = tuple(parts)
    for item in items:
        reason = reason_for(item.nodeid, version)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
