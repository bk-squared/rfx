"""tests/_jax_version_xfails.py -- strict xfails for tests that fail on jax 0.11 for a known cause.

The weekly Python 3.12 / jax 0.11 job (validation.yml, ``latest-jax``) ran red on its first run
(run 40468718500: 160 failing tests, three causes; #2114, 130 of them, is fixed by #2108 and not listed). Each cause has an issue; each listed node id gets
``xfail(strict=True, reason="#N ...")`` on jax 0.11 or later and nothing on older jax, where these tests
pass. The fix for an issue deletes that issue's entry: a fixed test still listed turns red (XPASS strict).

Ids are exact node ids, parameters included, because several parametrized tests fail for some cases only.
"""
import pytest

FROM_VERSION = (0, 11)

JAX_011_XFAILS = {
    2115: (
        "a closed-box plane-wave run() raises: get_opaque_trace_state() asserts outside a trace on jax 0.11",
        (
            "tests/contracts/test_backing_wall_one_rule.py::test_uniform_boundary_inputs_equal[closed_box]",
            "tests/contracts/test_backing_wall_one_rule.py::test_wall_matrix[distributed-uniform-run-closed_box]",
            "tests/contracts/test_backing_wall_one_rule.py::test_wall_matrix[uniform-run-closed_box]",
            "tests/contracts/test_closed_box_incident_carry.py::test_nonpaying_requests_build_no_face_record",
            "tests/contracts/test_closed_box_incident_carry.py::test_only_a_paying_forward_builds_a_face_record",
            "tests/contracts/test_film_one_plane.py::test_c4_tfsf_refusals[closed]",
            "tests/contracts/test_ntff_boundary_admission.py::test_ntff_boundary_table[uniform-closed-run]",
            "tests/contracts/test_ntff_boundary_admission.py::test_ntff_boundary_table[uniform-closed-run_skip]",
            "tests/contracts/test_run_forward_one_pipeline.py::test_assembly[PW-closed-multi-device-uniform]",
            "tests/contracts/test_run_forward_one_pipeline.py::test_assembly[PW-closed-uniform]",
            "tests/contracts/test_run_forward_one_pipeline.py::test_fields[PW-closed-multi-device-uniform]",
            "tests/contracts/test_tfsf_boundary_admission.py::test_closed_box_keeps_finite_scatterer[False]",
            "tests/contracts/test_tfsf_boundary_admission.py::test_closed_box_keeps_finite_scatterer[True]",
            "tests/contracts/test_tfsf_boundary_override.py::test_uniform_entry_points_use_tfsf_boundary_policy[closed-expected3-run]",
            "tests/contracts/test_tfsf_closed_box_transforms.py::test_run_keeps_its_refusal[jit]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-drude]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-kerr]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-pec_sheet]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-pec_volume]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-pec_wire]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-pinned_sheet]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-series_RL]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-surface_impedance]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_closed_box_distributed_injection_plane[run_distributed-thin_pec_sheet]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_other_injection_kinds[drude-box]",
            "tests/contracts/test_tfsf_injection_plane_operands.py::test_other_injection_kinds[pec_sheet-box]",
        ),
    ),
    2116: (
        "the compiled RLC step holds 26 domain copies on jax 0.11 against 8 on jax 0.10.2",
        (
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[parallel-graded]",
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[parallel-uniform]",
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[series-graded]",
            "tests/contracts/test_rlc_adds_no_domain_copy.py::test_rlc_adds_no_domain_copy[series-uniform]",
        ),
    ),
}


_REASONS = {nodeid: f"#{issue} {reason}" for issue, (reason, ids) in JAX_011_XFAILS.items() for nodeid in ids}


def reason_for(nodeid, jax_version):
    """The xfail reason for this node id under this jax version (a tuple), or None."""
    if tuple(jax_version[:2]) < FROM_VERSION:
        return None
    return _REASONS.get(nodeid)


def pytest_collection_modifyitems(config, items):
    import jax

    version = tuple(int(part) for part in jax.__version_info__[:2])
    for item in items:
        reason = reason_for(item.nodeid, version)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
