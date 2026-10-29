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
        monkeypatch.setattr(jax, "__version_info__", (*since, 1))
        plugin.pytest_collection_modifyitems(None, [listed, other])
        assert [(m.name, m.kwargs["strict"]) for m in listed.marks] == [("xfail", True)], issue
        assert listed.marks[0].kwargs["reason"].startswith(f"#{issue} ")
        assert other.marks == []
        monkeypatch.setattr(jax, "__version_info__", (since[0], since[1] - 1, 2))
        again = Item(ids[0])
        plugin.pytest_collection_modifyitems(None, [again])
        assert again.marks == [], issue
