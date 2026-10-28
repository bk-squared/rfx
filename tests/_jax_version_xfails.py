"""tests/_jax_version_xfails.py -- strict xfails for tests that fail on a newer jax for a known cause.

The weekly Python 3.12 / newest-jax job (validation.yml, ``latest-jax``) installs jax above the install cap.
Its first run (jax 0.11.2, run 40468718500) found three causes: #2114 is fixed by #2108 and #2115 by #2117, so
neither is listed; the 4 tests of #2116 stay under #2125, the open slowdown issue. Its third run (jax 0.12.0.1,
run 40632569850) found two more: #2166 (exact and cross-trace bars) and #2167 (a two-device mirror case).
Each listed node id gets ``xfail(strict=True, reason="#N ...")`` on the entry's jax version or later and nothing on
older jax, where these tests pass. The fix for an issue deletes that issue's entry: a fixed test still listed turns
red (XPASS strict).

Ids are exact node ids, parameters included, because several parametrized tests fail for some cases only.
"""
import pytest

_CURL = "tests/contracts/test_absorber_curl_coefficients.py::test_step_uses_owner_curl_coefficient"
_CURL_PATHS = "tests/contracts/test_absorber_curl_coefficients_paths.py::test_distributed_face_uses_owner_coefficient"

# issue: (first jax (major, minor) that fails, reason, node ids)
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
    2166: (
        (0, 12),
        "on jax 0.12 the CPML curl-coefficient owner check is one float32 step off, cross-trace checks reach 13-72 ULP "
        "and 2-device DFT planes are not bit-equal to one device",
        (
            *(f"{_CURL}[{mesh}-{medium}]" for mesh in ("uniform", "graded")
              for medium in ("plain", "dielectric", "lorentz")),
            *(f"{_CURL_PATHS}[{face}-distributed_{mesh}-{medium}]" for face in ("x_lo", "y_lo")
              for mesh in ("uniform", "graded") for medium in ("plain", "dielectric", "lorentz")),
            "tests/unit/autodiff/test_forward_jit_contract.py::test_the_fallback_probe_record_equals_the_plain_call",
            "tests/unit/autodiff/test_observables_dft_field.py::test_run_distributed_dft_planes_match_native",
            "tests/contracts/test_padded_writeback_cpu_cross_trace.py::test_padded_writeback_cpu_plain_vs_jit[120]",
            "tests/contracts/test_padded_writeback_cpu_cross_trace.py::test_padded_writeback_cpu_plain_vs_jit[240]",
            "tests/contracts/test_soft_source_one_drive.py::test_graded_current_forward_stays_within_nine_ulp[240-2.65-0]",
            "tests/contracts/test_soft_source_one_drive.py::test_graded_current_forward_stays_within_nine_ulp[240-5.3-0]",
            "tests/contracts/test_soft_source_one_drive.py::test_graded_current_forward_stays_within_nine_ulp[240-3.25-0.02]",
            "tests/contracts/test_soft_source_one_drive.py::test_graded_current_forward_stays_within_nine_ulp[240-3.7-0.02]",
        ),
    ),
    2167: (
        (0, 12),
        "on jax 0.12 a two-device graded run at a PEC high face is not the mirror of the low-face run",
        (
            "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pec-two_graded_forward]",
        ),
    ),
}


_ENTRIES = {nodeid: (since, f"#{issue} {reason}")
            for issue, (since, reason, ids) in JAX_XFAILS.items() for nodeid in ids}


def reason_for(nodeid, jax_version):
    """The xfail reason for this node id under this jax version (a tuple), or None."""
    since, reason = _ENTRIES.get(nodeid, (None, None))
    if since is None or tuple(jax_version[:2]) < since:
        return None
    return reason


def pytest_collection_modifyitems(config, items):
    import jax

    version = tuple(int(part) for part in jax.__version_info__[:2])
    for item in items:
        reason = reason_for(item.nodeid, version)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
