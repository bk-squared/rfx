"""Keep export-only packages merge-friendly without changing their public names."""

import ast
from collections import defaultdict
import io
import json
from pathlib import Path
import tokenize

import pytest


REPO = Path(__file__).resolve().parents[2]
FLOOR = {
    f"rfx/{package}/__init__.py"
    for package in (
        "boundaries", "config", "interop", "interop/emitters", "measurement",
        "model", "runners", "stepping",
    )
}
NAME_SET_EXEMPTIONS = {
    "rfx/boundaries/__init__.py": (
        "Deliberately exports fewer names than it imports; expanding __all__ "
        "would change from rfx.boundaries import *."
    ),
}


def is_docstring(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def is_all_assignment(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__" and isinstance(node.value, ast.List))


def is_lazy_assignment(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_LAZY_EXPORTS" and isinstance(node.value, ast.Dict))


LAZY_GETATTR = 'def __getattr__(name):\n    from importlib import import_module\n\n    if name not in _LAZY_EXPORTS:\n        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n    value = getattr(import_module(_LAZY_EXPORTS[name]), name)\n    globals()[name] = value\n    return value\n'


def is_lazy_getattr(node):
    return (isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
            and ast.dump(node) == ast.dump(ast.parse(LAZY_GETATTR).body[0]))


def discover():
    found = {}
    for path in sorted((REPO / "rfx").rglob("__init__.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        body = tree.body[1:] if tree.body and is_docstring(tree.body[0]) else tree.body
        if (sum(is_all_assignment(node) for node in body) == 1
                and all(isinstance(node, ast.ImportFrom) or is_all_assignment(node)
                        or is_lazy_assignment(node) or is_lazy_getattr(node)
                        for node in body)):
            found[path.relative_to(REPO).as_posix()] = source, tree
    return found


def attached_comments(source, tree):
    """Attach standalone comment lines to the next statement, verbatim."""
    comments = {node: ([], []) for node in tree.body}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line, column = token.start
        standalone = not token.line[:column].strip()
        owner = next((node for node in tree.body
                      if (node.lineno > line if standalone
                          else node.lineno <= line <= node.end_lineno)), None)
        assert owner is not None, f"Unattached comment on line {line}: {token.string}"
        comments[owner][0 if standalone else 1].append(
            token.line.rstrip("\r\n") if standalone else token.string
        )
    return comments


def sort_name(name):
    return name.lower(), name


def render(source, tree, comments):
    """Build canonical text independently from the AST and tokenized comments."""
    lines = []

    def emit(nodes, statement_lines):
        leading = [line for node in nodes for line in comments[node][0]]
        trailing = [line for node in nodes for line in comments[node][1]]
        lines.extend(leading)
        if trailing:
            statement_lines[0] += "  " + "  ".join(trailing)
        lines.extend(statement_lines)

    if tree.body and is_docstring(tree.body[0]):
        docstring = tree.body[0]
        emit([docstring], ast.get_source_segment(source, docstring).splitlines())
        lines.append("")

    imports = defaultdict(list)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imports[node.level, node.module or ""].append(node)
    future = imports.pop((0, "__future__"), [])
    if future:
        assert [alias.name for node in future for alias in node.names] == ["annotations"]
        assert all(alias.asname is None for node in future for alias in node.names)
        emit(future, ["from __future__ import annotations"])
        lines.append("")
    for (level, module), nodes in sorted(imports.items()):
        names = [alias.name + (f" as {alias.asname}" if alias.asname else "")
                 for node in nodes for alias in node.names]
        assert "*" not in names, "Wildcard imports cannot declare explicit export names"
        emit(nodes, [f"from {'.' * level}{module} import ("]
             + [f"    {name}," for name in sorted(set(names), key=sort_name)] + [")"])

    all_node = next(node for node in tree.body if is_all_assignment(node))
    exports = ast.literal_eval(all_node.value)
    assert all(isinstance(name, str) for name in exports), "__all__ must contain strings"
    if lines and lines[-1] != "":
        lines.append("")
    emit([all_node], ["__all__ = ["]
         + [f"    {json.dumps(name, ensure_ascii=False)},"
            for name in sorted(set(exports), key=sort_name)] + ["]"])
    lazy = [node for node in tree.body if is_lazy_assignment(node)]
    getters = [node for node in tree.body if is_lazy_getattr(node)]
    assert len(lazy) == len(getters) <= 1, "Lazy mapping and resolver must occur together"
    if lazy:
        mapping = ast.literal_eval(lazy[0].value)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items())
        assert len(mapping) == len(lazy[0].value.keys), "Duplicate lazy export"
        lines.append("")
        emit(lazy, ["_LAZY_EXPORTS = {"]
             + [f"    {json.dumps(k)}: {json.dumps(mapping[k])},"
                for k in sorted(mapping, key=sort_name)] + ["}"])
        lines.extend(["", ""])
        emit(getters, LAZY_GETATTR.rstrip().splitlines())
    return "\n".join(lines) + "\n"


EXPORT_ONLY = discover()


def test_export_only_discovery_floor():
    assert FLOOR <= EXPORT_ONLY.keys(), f"Missing export-only packages: {FLOOR - EXPORT_ONLY.keys()}"


@pytest.mark.parametrize("path", sorted(EXPORT_ONLY))
def test_export_init_form(path):
    source, tree = EXPORT_ONLY[path]
    comments = attached_comments(source, tree)
    imports = {
        alias.asname or alias.name
        for node in tree.body if isinstance(node, ast.ImportFrom)
        and not (node.level == 0 and node.module == "__future__")
        for alias in node.names
    }
    lazy = [node for node in tree.body if is_lazy_assignment(node)]
    if lazy:
        mapping = ast.literal_eval(lazy[0].value)
        assert not (imports & mapping.keys()), "Lazy/eager names overlap"
        if path == "rfx/stepping/__init__.py":
            assert mapping == {
                "core_step_invariants": "rfx.stepping.uniform_setup",
                "make_core_step": "rfx.stepping.uniform_setup",
                "run_nonuniform": "rfx.stepping.graded_loop",
                "run_nonuniform_until_decay": "rfx.stepping.graded_loop",
                "run_uniform_loop": "rfx.stepping.uniform_loop",
                "run_uniform_until_decay": "rfx.stepping.uniform_loop",
            }
        imports |= mapping.keys()
    all_node = next(node for node in tree.body if is_all_assignment(node))
    exports = set(ast.literal_eval(all_node.value))
    if path in NAME_SET_EXEMPTIONS:
        assert exports <= imports, f"{path}: __all__ contains names not imported"
    else:
        assert imports == exports, (
            f"{path}: import/__all__ name-set mismatch; "
            f"import-only={sorted(imports - exports)}, export-only={sorted(exports - imports)}"
        )
    assert source == render(source, tree, comments), f"{path}: noncanonical export init form"
