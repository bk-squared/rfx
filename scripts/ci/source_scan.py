"""Find source-reading tests and source_scan marks using only the stdlib AST."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re


STAGE2_EXCLUDED_MARKERS = frozenset({"slow", "slow_physics", "gpu", "docs_consistency"})


@dataclass(frozen=True)
class ScannedTest:
    name: str
    lineno: int
    hit_line: int | None
    reader_line: int | None
    marked: bool
    deselected_markers: tuple[str, ...]


@dataclass(frozen=True)
class SourceScan:
    tests: tuple[ScannedTest, ...]
    module_marked: bool
    stale_marks: tuple[int, ...]


def scan_source(source: str, filename: str = "<test>") -> SourceScan:
    """Resolve candidates and readers independently through names to a fixed point.

    A test needs a mark only when its reachable scope contains both. The two
    origin lines identify the candidate and reader, including through helpers,
    fixtures, parametrization and module bindings. No tests are imported.
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
                        "getfile", "getsource", "getsourcefile", "getsourcelines",
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

    def rfx_argument(node):
        return (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.startswith("rfx"))

    def inspection(node):
        if not isinstance(node, ast.Call):
            return False
        name = qualified(node.func)
        return name == "inspect.getfile" or name.startswith("inspect.getsource")

    def source_handle(node):
        if not isinstance(node, ast.Attribute) or node.attr not in {"__file__", "origin"}:
            return False
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name):
            return root.id in rfx_names
        if isinstance(root, ast.Call):
            return (qualified(root.func) in {
                "__import__", "importlib.import_module", "importlib.util.find_spec",
            } and bool(root.args) and rfx_argument(root.args[0]))
        if isinstance(root, ast.Subscript) and qualified(root.value) == "sys.modules":
            return rfx_argument(root.slice)
        return False

    def candidate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return re.fullmatch(r"rfx(?:/[A-Za-z0-9_./-]*)?", node.value) is not None
        if source_handle(node) or inspection(node):
            return True
        if isinstance(node, ast.Call) and qualified(node.func) == "importlib.resources.files":
            return (bool(node.args) and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and (node.args[0].value == "rfx" or node.args[0].value.startswith("rfx.")))
        return False

    def reader(node):
        if not isinstance(node, ast.Call):
            return False
        name = qualified(node.func)
        return (inspection(node) or name == "os.walk" or name.rsplit(".", 1)[-1] in {
            "read_text", "read_bytes", "open", "rglob", "glob", "iterdir",
            "listdir", "scandir",
        })

    # Each binding carries independent origins, rather than becoming a source
    # hit itself. A consumer can combine a candidate from one helper with a
    # reader from another, but a location check alone remains unmarked.
    name_signals: dict[str, tuple[int | None, int | None]] = {}

    def combine(signals):
        return tuple(min((signal[index] for signal in signals if signal[index] is not None),
                         default=None) for index in (0, 1))

    def signals_in(nodes):
        signals = []
        for root in nodes:
            for node in ast.walk(root):
                signals.append((node.lineno if candidate(node) else None,
                                node.lineno if reader(node) else None))
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id in name_signals:
                        signals.append(name_signals[node.id])
        return combine(signals)

    def mark_calls(nodes, name):
        for node in nodes:
            target = node.func if isinstance(node, ast.Call) else node
            if qualified(target) == f"pytest.mark.{name}":
                yield node

    def fixture_signals(node, decorators):
        signals = [name_signals[arg.arg] for arg in (
            *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
        ) if arg.arg in name_signals]
        for mark in mark_calls(decorators, "usefixtures"):
            if isinstance(mark, ast.Call):
                signals.extend(name_signals[arg.value] for arg in mark.args
                               if isinstance(arg, ast.Constant) and arg.value in name_signals)
        return combine(signals)

    module_marks = []
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
                value = statement.value
                if value is not None:
                    module_marks.extend(value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value])
    module_marked = bool(list(mark_calls(module_marks, "source_scan")))

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
        previous = dict(name_signals)
        for statement in statements:
            signals = [signals_in([statement])]
            if isinstance(statement, functions):
                signals.append(fixture_signals(statement, statement.decorator_list))
            signal = combine(signals)
            if signal != (None, None):
                names = bindings(statement)
                if isinstance(statement, functions):
                    for decorator in statement.decorator_list:
                        if isinstance(decorator, ast.Call) and qualified(decorator.func) == "pytest.fixture":
                            names.update(keyword.value.value for keyword in decorator.keywords
                                         if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
                                         and isinstance(keyword.value.value, str))
                for name in names:
                    name_signals[name] = combine([name_signals.get(name, (None, None)), signal])
        if previous == name_signals:
            break

    tests = []

    def add_test(node, prefix="", class_marks=()):
        decorators = [*module_marks, *class_marks, *node.decorator_list]
        signals = [signals_in(node.body), fixture_signals(node, decorators)]
        for mark in mark_calls(node.decorator_list, "parametrize"):
            if isinstance(mark, ast.Call):
                signals.append(signals_in([*mark.args, *[kw.value for kw in mark.keywords]]))
        candidate_line, reader_line = combine(signals)
        hit_line = candidate_line if reader_line is not None else None
        tests.append(ScannedTest(
            prefix + node.name, node.lineno, hit_line, reader_line,
            bool(list(mark_calls(decorators, "source_scan"))),
            tuple(sorted(name for name in STAGE2_EXCLUDED_MARKERS
                         if list(mark_calls(decorators, name)))),
        ))

    for node in tree.body:
        if isinstance(node, functions) and node.name.startswith("test_"):
            add_test(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for method in node.body:
                if isinstance(method, functions) and method.name.startswith("test_"):
                    add_test(method, node.name + "::", node.decorator_list)

    stale = [test.lineno for test in tests if test.marked and test.hit_line is None]
    if module_marked and not tests:
        candidate_line, reader_line = signals_in(tree.body)
        if candidate_line is None or reader_line is None:
            stale.extend(mark.lineno for mark in mark_calls(module_marks, "source_scan"))
    return SourceScan(tuple(tests), module_marked, tuple(stale))
