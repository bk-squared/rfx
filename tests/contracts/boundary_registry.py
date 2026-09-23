"""AST locations of spatial neighbours, differences and boundary primitives."""

import ast
import hashlib
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
    """Return stable function/expression identities, with lines for inspection."""
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
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()[:16]
            rows.append(dict(file=path, function=".".join(self.scope) or "<module>",
                             kind=kind, line=node.lineno, expression=expression, digest=digest))

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
    """Counts distinguish identical expressions occurring twice in one function."""
    from collections import Counter
    return Counter((r["file"], r["function"], r["kind"], r["digest"]) for r in rows)


def compare_registry(expected, actual):
    want, got = identities(expected), identities(actual)
    assert got == want, f"unlisted sites: {got - want}; removed sites: {want - got}"


ROOT = Path(__file__).resolve().parents[2]
