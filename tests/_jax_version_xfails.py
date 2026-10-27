"""tests/_jax_version_xfails.py -- strict xfails for tests that fail on jax 0.11 for a known cause.

The weekly Python 3.12 / jax 0.11 job (validation.yml, ``latest-jax``) ran red on its first run
(run 40468718500: 160 failing tests, three causes; #2114, 130 of them, is fixed by #2108 and #2115, 26, by #2117, so neither is listed; the 4 of #2116 stay under #2125, the open slowdown issue). Each cause has an issue; each listed node id gets
``xfail(strict=True, reason="#N ...")`` on jax 0.11 or later and nothing on older jax, where these tests
pass. The fix for an issue deletes that issue's entry: a fixed test still listed turns red (XPASS strict).

Ids are exact node ids, parameters included, because several parametrized tests fail for some cases only.
"""
import pytest

FROM_VERSION = (0, 11)

JAX_011_XFAILS = {
    2125: (
        "the compiled RLC step holds 26 domain copies on jaxlib 0.11.2 and later against 8 on 0.10.2 (measured in #2116)",
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
