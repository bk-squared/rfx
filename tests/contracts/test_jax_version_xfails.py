"""The jax 0.11 xfail list marks only on jax 0.11 and names tests that exist (tests/_jax_version_xfails.py)."""

import ast
from pathlib import Path

from tests import _jax_version_xfails as plugin

ROOT = Path(__file__).resolve().parents[2]


def test_the_plugin_is_loaded_by_the_root_conftest():
    tree = ast.parse((ROOT / "conftest.py").read_text(encoding="utf-8"))
    plugins = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["pytest_plugins"])
    assert "tests._jax_version_xfails" in ast.literal_eval(plugins)


def test_marks_only_from_jax_011():
    issue, (_, ids) = next(iter(plugin.JAX_011_XFAILS.items()))
    nodeid = ids[0]
    assert plugin.reason_for(nodeid, (0, 10, 2)) is None
    assert plugin.reason_for(nodeid, (0, 10)) is None
    assert plugin.reason_for(nodeid, (0, 11, 0)).startswith(f"#{issue} ")
    assert plugin.reason_for(nodeid, (0, 12)).startswith(f"#{issue} ")
    assert plugin.reason_for("tests/unit/autodiff/test_estimate_ad_memory.py::not_listed", (0, 11)) is None


def test_every_listed_id_names_an_existing_test_function():
    missing = []
    defined = {}
    for issue, (reason, ids) in plugin.JAX_011_XFAILS.items():
        assert reason and ids, issue
        for nodeid in ids:
            path, *names = nodeid.split("[", 1)[0].split("::")
            if path not in defined:
                file = ROOT / path
                defined[path] = (
                    {f"{c.name}.{f.name}" for c in ast.walk(ast.parse(file.read_text(encoding="utf-8")))
                     if isinstance(c, ast.ClassDef) for f in c.body if isinstance(f, ast.FunctionDef)}
                    | {f.name for f in ast.parse(file.read_text(encoding="utf-8")).body
                       if isinstance(f, ast.FunctionDef)}
                ) if file.is_file() else set()
            if ".".join(names) not in defined[path]:
                missing.append(nodeid)
    assert not missing, missing


def test_no_id_is_listed_twice():
    ids = [nodeid for _, (_, listed) in plugin.JAX_011_XFAILS.items() for nodeid in listed]
    assert len(ids) == len(set(ids))


def test_the_hook_adds_a_strict_mark_on_jax_011(monkeypatch):
    import jax

    class Item:
        def __init__(self, nodeid):
            self.nodeid, self.marks = nodeid, []

        def add_marker(self, mark):
            self.marks.append(mark)

    issue, (_, ids) = next(iter(plugin.JAX_011_XFAILS.items()))
    listed, other = Item(ids[0]), Item("tests/contracts/test_jax_version_xfails.py::not_listed")
    monkeypatch.setattr(jax, "__version_info__", (0, 11, 2))
    plugin.pytest_collection_modifyitems(None, [listed, other])
    assert [(m.name, m.kwargs["strict"]) for m in listed.marks] == [("xfail", True)]
    assert listed.marks[0].kwargs["reason"].startswith(f"#{issue} ")
    assert other.marks == []
    monkeypatch.setattr(jax, "__version_info__", (0, 10, 2))
    again = Item(ids[0])
    plugin.pytest_collection_modifyitems(None, [again])
    assert again.marks == []
