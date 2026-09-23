"""Boundary/difference registry; removed sites must also leave the allow-list."""

import json

import pytest

from tests.contracts.boundary_registry import ROOT, compare_registry, scan_source, scan_tree


def test_boundary_sites_registered():
    expected = json.loads((ROOT / "tests/contracts/boundary_registry.json").read_text())
    compare_registry(expected, scan_tree(ROOT))


@pytest.mark.parametrize("expression", [
    "a[1:] - a[:-1]", "a[:, 1:, :] - a[:, :-1, :]",
    "jnp.roll(a, -1, axis=1) - a", "a - _shift_bwd(a, 2)",
    "apply_pmc_faces(state, faces)", "apply_pec_face_shmap(state, mesh, 2, nx)",
    "_apply_fine_pec_axes(state, 'xy')",
    "init_cpml(grid)", "precompute_coeffs(materials, dt, dx)",
])
def test_new_function_is_unlisted(expression):
    rows = scan_source(f"def new_function(a):\n    return {expression}\n", "rfx/new.py")
    assert rows
    with pytest.raises(AssertionError, match="unlisted sites"):
        compare_registry([], rows)


def test_removed_function_is_stale():
    rows = scan_source("def old(a):\n    return a[1:] - a[:-1]\n", "rfx/old.py")
    with pytest.raises(AssertionError, match="removed sites"):
        compare_registry(rows, [])


def test_new_function_calling_existing_neighbour_is_unlisted():
    original = "from rfx.core.yee import _shift_bwd as back\ndef bwd(a):\n    return back(a, 0)\n"
    before = scan_source(original, "rfx/old.py")
    after = scan_source(original + "def new_function(a):\n    return a - bwd(a)\n", "rfx/old.py")
    with pytest.raises(AssertionError, match="unlisted sites"):
        compare_registry(before, after)
