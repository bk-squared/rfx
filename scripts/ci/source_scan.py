"""Find source-reading tests and source_scan marks using only the stdlib AST."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ScannedTest:
    name: str
    lineno: int
    hit_line: int | None
    marked: bool


@dataclass(frozen=True)
class SourceScan:
    tests: tuple[ScannedTest, ...]
    module_marked: bool
    stale_marks: tuple[int, ...]


def scan_source(source: str, filename: str = "<test>") -> SourceScan:
    """Resolve source hits, helper/fixture dependencies and marks within a file.

    Names propagate to a fixed point without importing or collecting tests.
    Hit lines point back to the source-reading expression, including when a
    test reaches it through a module constant, helper or fixture.
    """
    tree = ast.parse(source, filename=filename)
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    aliases: dict[str, str] = {}
    rfx_names = {"rfx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                aliases[bound] = alias.name if alias.asname else bound
                if alias.name == "rfx" or alias.name.startswith("rfx."):
                    rfx_names.add(bound)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                if node.module == "inspect" and alias.name == "*":
                    aliases.update((name, f"inspect.{name}") for name in (
                        "getsource", "getsourcefile", "getsourcelines",
                    ))
                bound = alias.asname or alias.name
                aliases[bound] = f"{node.module}.{alias.name}"
                if node.module == "rfx" or node.module.startswith("rfx."):
                    rfx_names.add(bound)

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{qualified(node.value)}.{node.attr}"
        return ""

    def direct_hit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return re.fullmatch(r"rfx(?:/[A-Za-z0-9_./-]*)?", node.value) is not None
        if isinstance(node, ast.Attribute):
            return (node.attr == "__file__" and isinstance(node.value, ast.Name)
                    and node.value.id in rfx_names)
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if name in {"inspect.getsource", "inspect.getsourcefile", "inspect.getsourcelines"}:
                return True
            if name == "importlib.resources.files" and node.args:
                arg = node.args[0]
                return (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        and (arg.value == "rfx" or arg.value.startswith("rfx.")))
        return False

    hit_names: dict[str, int] = {}

    def hit_line(nodes):
        lines = []
        for root in nodes:
            for node in ast.walk(root):
                if direct_hit(node):
                    lines.append(node.lineno)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id in hit_names:
                        lines.append(hit_names[node.id])
        return min(lines, default=None)

    def mark_calls(nodes, name):
        for node in nodes:
            target = node.func if isinstance(node, ast.Call) else node
            if qualified(target) == f"pytest.mark.{name}":
                yield node

    def fixture_lines(node, decorators):
        lines = [hit_names[arg.arg] for arg in (
            *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
        ) if arg.arg in hit_names]
        for mark in mark_calls(decorators, "usefixtures"):
            if isinstance(mark, ast.Call):
                lines.extend(hit_names[arg.value] for arg in mark.args
                             if isinstance(arg, ast.Constant) and arg.value in hit_names)
        return lines

    module_marks = []
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
                value = statement.value
                if value is not None:
                    module_marks.extend(value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value])
    module_marked = bool(list(mark_calls(module_marks, "source_scan")))

    # Only module bindings escape into the shared name table; function locals
    # stay inside their function. Compound module statements can also bind names.
    def bindings(node):
        if isinstance(node, (*functions, ast.ClassDef)):
            return {node.name}
        names = set()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            else:
                names.update(bindings(child))
        return names

    statements = [node for node in tree.body
                  if not (isinstance(node, functions) and node.name.startswith("test_"))]
    while True:
        previous = dict(hit_names)
        for statement in statements:
            lines = [hit_line([statement])]
            if isinstance(statement, functions):
                lines.extend(fixture_lines(statement, statement.decorator_list))
            line = min((line for line in lines if line is not None), default=None)
            if line is not None:
                names = bindings(statement)
                if isinstance(statement, functions):
                    for decorator in statement.decorator_list:
                        if isinstance(decorator, ast.Call) and qualified(decorator.func) == "pytest.fixture":
                            names.update(keyword.value.value for keyword in decorator.keywords
                                         if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
                                         and isinstance(keyword.value.value, str))
                for name in names:
                    hit_names[name] = min(hit_names.get(name, line), line)
        if previous == hit_names:
            break

    tests = []

    def add_test(node, prefix="", class_marks=()):
        decorators = [*module_marks, *class_marks, *node.decorator_list]
        lines = [hit_line(node.body), *fixture_lines(node, decorators)]
        for mark in mark_calls(node.decorator_list, "parametrize"):
            if isinstance(mark, ast.Call):
                lines.append(hit_line([*mark.args, *[kw.value for kw in mark.keywords]]))
        line = min((line for line in lines if line is not None), default=None)
        tests.append(ScannedTest(
            prefix + node.name, node.lineno, line,
            bool(list(mark_calls(decorators, "source_scan"))),
        ))

    for node in tree.body:
        if isinstance(node, functions) and node.name.startswith("test_"):
            add_test(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for method in node.body:
                if isinstance(method, functions) and method.name.startswith("test_"):
                    add_test(method, node.name + "::", node.decorator_list)

    stale = [test.lineno for test in tests if test.marked and test.hit_line is None]
    if module_marked and not tests and hit_line(tree.body) is None:
        stale.extend(mark.lineno for mark in mark_calls(module_marks, "source_scan"))
    return SourceScan(tuple(tests), module_marked, tuple(stale))
