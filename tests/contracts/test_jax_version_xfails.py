"""The JAX-version hook applies a strict xfail from each entry's version.
"""


from tests import _jax_version_xfails as plugin


def test_the_hook_adds_a_strict_mark_from_the_entry_version(monkeypatch):
    import jax

    class Item:
        def __init__(self, nodeid):
            self.nodeid, self.marks = nodeid, []

        def add_marker(self, mark):
            self.marks.append(mark)

    for issue, (since, _, ids) in plugin.JAX_XFAILS.items():
        listed, other = Item(ids[0]), Item("tests/contracts/test_jax_version_xfails.py::not_listed")
        monkeypatch.setattr(jax, "__version__", ".".join(map(str, (*since, 1))))
        plugin.pytest_collection_modifyitems(None, [listed, other])
        assert [(m.name, m.kwargs["strict"]) for m in listed.marks] == [("xfail", True)], issue
        assert listed.marks[0].kwargs["reason"].startswith(f"#{issue} ")
        assert other.marks == []
        monkeypatch.setattr(jax, "__version__", f"{since[0]}.{since[1] - 1}.2")
        again = Item(ids[0])
        plugin.pytest_collection_modifyitems(None, [again])
        assert again.marks == [], issue


def test_patch_entry_starts_at_the_patch_and_minor_entry_keeps_its_boundary():
    patch_id = "tests/contracts/test_ad_memory_tape_terms.py::test_ad_memory_tape_terms"
    assert plugin.reason_for(patch_id, (0, 12, 1)).startswith("#2896 ")
    assert plugin.reason_for(patch_id, (0, 12, 2)).startswith("#2896 ")
    assert plugin.reason_for(patch_id, (0, 13)).startswith("#2896 ")
    assert plugin.reason_for(patch_id, (0, 12, 0, 1)) is None
    assert plugin.reason_for(patch_id, (0, 12)) is None
    minor_id = "tests/contracts/test_field_index_high_face.py::test_c2_c2b[ii-2-pec-two_graded_forward]"
    assert plugin.reason_for(minor_id, (0, 12)).startswith("#2167 ")
    assert plugin.reason_for(minor_id, (0, 12, 0, 1)).startswith("#2167 ")
    assert plugin.reason_for(minor_id, (0, 11, 9)) is None


def test_hook_reads_leading_integer_parts_of_the_version(monkeypatch):
    import jax

    versions = []
    monkeypatch.setattr(plugin, "reason_for", lambda nodeid, version: versions.append(version))

    class Item:
        nodeid = "unlisted"

    for version in ("0.12.0.1", "0.12.1.dev1", "0.12.1rc1", "0.12.1"):
        monkeypatch.setattr(jax, "__version__", version)
        plugin.pytest_collection_modifyitems(None, [Item()])
    assert versions == [(0, 12, 0, 1), (0, 12, 1), (0, 12), (0, 12, 1)]
