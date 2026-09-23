"""Presence registry, with exact counts in explicitly listed curl functions."""

import json

import pytest

from tests.contracts import boundary_registry
from tests.contracts.boundary_registry import ROOT, compare_registry, scan_source, scan_tree


COUNTED = [dict(file="rfx/old.py", function="old")]


def test_boundary_sites_registered():
    expected = json.loads((ROOT / "tests/contracts/boundary_registry.json").read_text())
    compare_registry(expected["sites"], scan_tree(ROOT), expected["counted_functions"])


@pytest.mark.parametrize("expression", [
    "a[1:] - a[:-1]", "a[:, 1:, :] - a[:, :-1, :]", "a[1:n+1] - a[:n]",
    "jnp.roll(a, -1, axis=1) - a", "a - _shift_bwd(a, 2)",
    "apply_pmc_faces(state, faces)", "apply_pec_face_shmap(state, mesh, 2, nx)",
    "_apply_fine_pec_axes(state, 'xy')", "apply_adi_cpml_2d(state, cpml)",
    "init_cpml(grid)", "precompute_coeffs(materials, dt, dx)",
    "ex.at[:, 0, :].set(0.0)", "hy.at[0].set(0.0)", "ex.at[:, :, -1].multiply(0)",
    "d(hz[1:], hz[:-1], h)", "jnp.subtract(hz[1:], hz[:-1])",
    "jnp.subtract(a, _shift_bwd(a, 0))", "jnp.diff(a, axis=1)",
    "jnp.where(jnp.arange(n)[:, None, None] == 0, 0, ex)",
    "jnp.where(jnp.arange(n)[None, :, None] == n - 1, 0, ex)",
])
def test_new_function_is_unlisted(expression):
    rows = scan_source(f"def new_function(a):\n    return {expression}\n", "rfx/new.py")
    assert rows
    with pytest.raises(AssertionError, match="unlisted sites"):
        compare_registry([], rows)


@pytest.mark.parametrize("expression", [
    "1.0 - G", "z_src - 4", "a.shape[0] - 1", "x[-1] - x[0]",
    "b.corner_hi[0] - b.corner_lo[0]", "a[1:] - b[:-1]",
    "a[2:] - a[:-2]", "a[1:, 1:] - a[:-1, :-1]",
    "jnp.subtract(1.0, G)", "ex.at[0, 0, :].set(0)", "ex.at[1:, :, :].set(0)",
])
def test_scalar_index_and_nonadjacent_arithmetic_is_not_a_site(expression):
    assert scan_source(f"def old(a):\n    return {expression}\n", "rfx/old.py") == []


def test_assigned_scalar_is_not_spatial():
    assert scan_source("def old(a):\n    G = a[0]\n    return 1.0 - G\n", "rfx/old.py") == []


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


@pytest.mark.parametrize("count,expression,label", [
    (1, "(a[1:] - a[:-1]) + (a[2:] - a[1:-1])", "unlisted sites"),
    (2, "a[1:] - a[:-1]", "removed sites"),
])
def test_registered_curl_count_change(count, expression, label):
    expected = [dict(file="rfx/old.py", function="old", kind="difference", count=count)]
    actual = scan_source(f"def old(a):\n    return {expression}\n", "rfx/old.py")
    with pytest.raises(AssertionError, match=label) as exc:
        compare_registry(expected, actual, COUNTED)
    message = str(exc.value)
    assert "rfx/old.py :: old [difference]" in message
    assert "rfx/old.py:2:" in message
    assert "python -m tests.contracts.boundary_registry --update" in message
    assert "main's count stale, run --update on main" in message


def test_existing_noncounted_function_can_add_same_kind_site():
    before = scan_source("def old(a):\n    return a[1:] - a[:-1]\n", "rfx/old.py")
    after = scan_source("def old(a):\n    return (a[1:] - a[:-1]) + (a[2:] - a[1:-1])\n", "rfx/old.py")
    assert len(before) == 1 and len(after) == 2
    compare_registry(before, after)


@pytest.mark.parametrize("original,edited", [
    ("return a[1:] - a[:-1]", "return 2 * (a[2:] - a[1:-1])"),
    ("return apply_pec_faces(state, faces)", "return apply_pec_faces(state, faces=new_faces)"),
])
def test_edit_in_registered_function_without_added_site_passes(original, edited):
    before = scan_source(f"class Step:\n    def run(self, a):\n        {original}\n", "rfx/old.py")
    after = scan_source(f"class Step:\n    def run(self, a):\n\n        {edited}\n", "rfx/old.py")
    assert len(before) == len(after) == 1
    assert before[0]["expression"] != after[0]["expression"]
    assert before[0]["function"] == "Step.run"
    compare_registry(before, after)


def test_update_writes_counts_and_follows_shrink(tmp_path, monkeypatch):
    (tmp_path / "rfx").mkdir()
    (tmp_path / "tests/contracts").mkdir(parents=True)
    source = tmp_path / "rfx/old.py"
    source.write_text("def old(a):\n    return (a[1:] - a[:-1]) + (a[2:] - a[1:-1])\n")
    registry = tmp_path / "tests/contracts/boundary_registry.json"
    registry.write_text(json.dumps(dict(counted_functions=COUNTED, sites=[])))
    monkeypatch.setattr(boundary_registry, "ROOT", tmp_path)
    boundary_registry.main(["--update"])
    assert json.loads(registry.read_text())["sites"] == [
        dict(file="rfx/old.py", function="old", kind="difference", count=2)]
    boundary_registry.main([])
    source.write_text("def old(a):\n    return a[1:] - a[:-1]\n")
    with pytest.raises(AssertionError, match="removed sites"):
        boundary_registry.main([])
    boundary_registry.main(["--update"])
    assert json.loads(registry.read_text())["sites"] == [
        dict(file="rfx/old.py", function="old", kind="difference", count=1)]
    boundary_registry.main([])
