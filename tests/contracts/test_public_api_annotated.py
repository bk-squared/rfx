"""Annotations cover the live public surface, including re-exports and mixins."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.source_scan


def public_functions():
    """Resolve lazy exports and inherited methods before scanning definitions."""
    import rfx

    exports = [(name, getattr(rfx, name)) for name in rfx.__all__]
    methods = [(name, fn) for name, fn in inspect.getmembers(rfx.Simulation)
               if not name.startswith("_")
               and (inspect.isfunction(fn) or inspect.ismethod(fn))]
    return ([fn for _, fn in exports if inspect.isfunction(fn)]
            + [getattr(fn, "__func__", fn) for _, fn in methods])


def annotation_errors(functions, sources=None):
    """Return file:line diagnostics; optional sources support AST mutations."""
    trees = {} if sources is None else dict(sources)
    errors = []
    for fn in functions:
        fn = inspect.unwrap(fn)
        path = Path(inspect.getsourcefile(fn))
        if path not in trees:
            trees[path] = ast.parse(path.read_text(), filename=str(path))
        node = next(n for n in ast.walk(trees[path])
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == fn.__code__.co_name
                    and min([n.lineno] + [d.lineno for d in n.decorator_list])
                    <= fn.__code__.co_firstlineno <= n.body[0].lineno)
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                node.args.vararg, node.args.kwarg]
        missing = [a.arg for a in args if a is not None
                   and a.arg not in {"self", "cls"} and a.annotation is None]
        if node.returns is None:
            missing.append("return")
        if missing:
            label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            errors.append(f"{label}:{node.lineno}: {node.name}: missing {', '.join(missing)}")
    return errors


def test_public_api_annotated():
    assert not (errors := annotation_errors(public_functions())), "\n".join(errors)


def test_dropped_annotation_is_detected():
    import rfx

    fn = rfx.make_probe
    path = Path(inspect.getsourcefile(fn))
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "make_probe")
    assert node.returns is not None
    node.returns = None
    errors = annotation_errors(public_functions(), {path: tree})
    assert errors == [f"rfx/simulation.py:{node.lineno}: make_probe: missing return"]


def test_new_unannotated_public_function_is_detected(monkeypatch):
    import rfx

    def new_public_function(value):
        return value

    monkeypatch.setattr(rfx, "new_public_function", new_public_function, raising=False)
    monkeypatch.setattr(rfx, "__all__", [*rfx.__all__, "new_public_function"])
    errors = annotation_errors(public_functions())
    assert errors == [
        f"tests/contracts/test_public_api_annotated.py:"
        f"{new_public_function.__code__.co_firstlineno}: "
        "new_public_function: missing value, return"
    ]
