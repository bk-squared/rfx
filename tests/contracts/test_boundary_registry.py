"""Boundary/difference registry; removed sites must also leave the allow-list."""

import json

import pytest

from tests.contracts import boundary_registry
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


@pytest.mark.parametrize("count,expression,label", [
    (1, "(a[1:] - a[:-1]) + (a[2:] - a[:-2])", "unlisted sites"),
    (2, "a[1:] - a[:-1]", "removed sites"),
])
def test_registered_function_count_change(count, expression, label):
    expected = [dict(file="rfx/old.py", function="old", kind="difference", count=count)]
    actual = scan_source(f"def old(a):\n    return {expression}\n", "rfx/old.py")
    with pytest.raises(AssertionError, match=label) as exc:
        compare_registry(expected, actual)
    message = str(exc.value)
    assert "rfx/old.py :: old [difference]" in message
    assert "rfx/old.py:2:" in message
    assert "python -m tests.contracts.boundary_registry --update" in message


@pytest.mark.parametrize("original,edited", [
    ("return a[1:] - a[:-1]", "return 2 * (a[2:] - a[:-2])"),
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
    source.write_text("def old(a):\n    return (a[1:] - a[:-1]) + (a[2:] - a[:-2])\n")
    monkeypatch.setattr(boundary_registry, "ROOT", tmp_path)
    boundary_registry.main(["--update"])
    registry = tmp_path / "tests/contracts/boundary_registry.json"
    assert json.loads(registry.read_text()) == [
        dict(file="rfx/old.py", function="old", kind="difference", count=2)]
    boundary_registry.main([])
    source.write_text("def old(a):\n    return a[1:] - a[:-1]\n")
    with pytest.raises(AssertionError, match="removed sites"):
        boundary_registry.main([])
    boundary_registry.main(["--update"])
    assert json.loads(registry.read_text()) == [
        dict(file="rfx/old.py", function="old", kind="difference", count=1)]
    boundary_registry.main([])
