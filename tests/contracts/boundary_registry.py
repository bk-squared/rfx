"""AST locations of spatial neighbours, differences and boundary primitives."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path


WALL_PREFIXES = ("apply_pec", "apply_pmc", "apply_cpml", "apply_upml", "init_cpml", "init_upml")
WALL_HELPERS = {"_apply_fine_pec_axes"}
NEIGHBOURS = {"_shift_fwd", "_shift_bwd", "_diff_fwd", "_diff_bwd", "_bwd_h", "_bwd_neighbor"}


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def spatial_operand(node, assigned, neighbours, aliases, visited=frozenset()):
    if isinstance(node, ast.Name) and node.id in assigned and node.id not in visited:
        return spatial_operand(assigned[node.id], assigned, neighbours, aliases, visited | {node.id})
    if isinstance(node, ast.Subscript):
        return True
    if isinstance(node, ast.Call) and aliases.get(call_name(node.func), call_name(node.func)) in neighbours:
        return True
    return any(spatial_operand(child, assigned, neighbours, aliases, visited)
               for child in ast.iter_child_nodes(node))


def scan_source(source, path):
    """Return file, qualified function and kind, with sites for inspection."""
    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
    neighbours = NEIGHBOURS | {"roll", "diff"}
    # Local wrappers such as Debye's bwd() also spell neighbour operations.
    # Record their callers as well as the primitive inside the wrapper.
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    while True:
        added = {node.name for node in functions
                 if any(isinstance(child, ast.Call)
                        and aliases.get(call_name(child.func), call_name(child.func)) in neighbours
                        for child in ast.walk(node))}
        if added <= neighbours:
            break
        neighbours |= added
    rows = []

    class Visitor(ast.NodeVisitor):
        scope = []
        assigned = {}

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            old = self.assigned
            self.assigned = dict(old)
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            self.assigned[target.id] = child.value
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and child.value:
                    self.assigned[child.target.id] = child.value
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()
            self.assigned = old

        visit_AsyncFunctionDef = visit_FunctionDef

        def record(self, node, kind):
            expression = ast.unparse(node)
            rows.append(dict(file=path, function=".".join(self.scope) or "<module>",
                             kind=kind, line=node.lineno, expression=expression))

        def visit_Call(self, node):
            name = aliases.get(call_name(node.func), call_name(node.func))
            if name.lstrip("_").startswith(WALL_PREFIXES) or name in WALL_HELPERS | {"precompute_coeffs"}:
                self.record(node, "boundary")
            elif name in ("roll", "diff"):
                self.record(node, "roll/diff")
            elif name in neighbours:
                self.record(node, "neighbour")
            self.generic_visit(node)

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Sub) and (spatial_operand(node.left, self.assigned, neighbours, aliases)
                                                 or spatial_operand(node.right, self.assigned, neighbours, aliases)):
                self.record(node, "difference")
            self.generic_visit(node)

    Visitor().visit(tree)
    return rows


def scan_tree(root):
    return [row for path in sorted((root / "rfx").rglob("*.py"))
            for row in scan_source(path.read_text(), path.relative_to(root).as_posix())]


def identities(rows):
    """Count sites by file, qualified function and kind, independent of content."""
    counts = Counter()
    for row in rows:
        counts[row["file"], row["function"], row["kind"]] += row.get("count", 1)
    return counts


def registry_rows(rows):
    return [dict(file=file, function=function, kind=kind, count=count)
            for (file, function, kind), count in sorted(identities(rows).items())]


def compare_registry(expected, actual):
    want, got = identities(expected), identities(actual)
    if got == want:
        return
    messages = []
    for label, changes in (("unlisted sites", got - want), ("removed sites", want - got)):
        if not changes:
            continue
        messages.append(f"{label}:")
        for file, function, kind in sorted(changes):
            key = file, function, kind
            messages.append(f"  {file} :: {function} [{kind}]: registered {want[key]}, found {got[key]}")
            for row in actual:
                if (row["file"], row["function"], row["kind"]) == key:
                    messages.append(f"    {file}:{row['line']}: {row['expression']}")
    messages.append("Run python -m tests.contracts.boundary_registry --update to rewrite "
                    "tests/contracts/boundary_registry.json; review the JSON diff.")
    raise AssertionError("\n".join(messages))


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite boundary_registry.json from the current tree")
    args = parser.parse_args(argv)
    path = ROOT / "tests/contracts/boundary_registry.json"
    actual = scan_tree(ROOT)
    if args.update:
        rows = registry_rows(actual)
        path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Registered {len(actual)} sites in {len(rows)} keys: {path.relative_to(ROOT)}")
    else:
        compare_registry(json.loads(path.read_text()), actual)
        print(f"All {len(actual)} sites registered.")


if __name__ == "__main__":
    main()
