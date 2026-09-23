"""AST locations of spatial neighbours, offset differences and face writes.

Literal face writes and simple arange equality masks are recognized. Arbitrary
boolean masks, computed index lists, and multiplication by a mask need manual
inspection; no array shape or general data-flow inference is attempted.

Known blind spots: a face write at index -2 (how ``apply_pmc_faces`` zeroes a
hi face through its ghost layout) or at 1 is not a site, only 0 and -1 are;
offset-2 and wider differences (``a[2:] - a[:-2]``) are not sites, though a
fourth-order stencil's offset-1 term still registers its function; shifts built
with pad / concatenate or ``lax.slice_in_dim`` are not recognized; nor are face
indices held in named variables (``lo``, ``hi``) or ``.at[...].mul`` writes.
"""

import argparse
import ast
from collections import Counter
from itertools import combinations
import json
from pathlib import Path


WALL_PREFIXES = ("apply_pec", "apply_pmc", "apply_cpml", "apply_upml", "apply_adi_cpml",
                 "init_cpml", "init_upml")
WALL_HELPERS = {"_apply_fine_pec_axes", "precompute_coeffs"}
NEIGHBOURS = {"_shift_fwd", "_shift_bwd", "_diff_fwd", "_diff_bwd", "_diff_fwd_o", "_diff_bwd_o",
              "_bwd_h", "_bwd_neighbor", "_material_bwd_neighbour"}
ROOT = Path(__file__).resolve().parents[2]


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def same(left, right):
    return ast.dump(left) == ast.dump(right)


def integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = integer(node.operand)
        return -value if value is not None else None
    return None


def bound(node, *, upper=False):
    """Affine slice bound: an origin (start/end/symbol) and integer offset."""
    if node is None:
        return ("end" if upper else "start"), 0
    value = integer(node)
    if value is not None:
        return ("end" if value < 0 else "start"), value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        offset = integer(node.right)
        if offset is not None:
            base, old = bound(node.left, upper=upper)
            return base, old + (offset if isinstance(node.op, ast.Add) else -offset)
    return ast.dump(node), 0


def offset_slices(left, right):
    if not (isinstance(left, ast.Subscript) and isinstance(right, ast.Subscript)
            and same(left.value, right.value)):
        return False
    a = left.slice.elts if isinstance(left.slice, ast.Tuple) else [left.slice]
    b = right.slice.elts if isinstance(right.slice, ast.Tuple) else [right.slice]
    if len(a) != len(b):
        return False
    changed = [(x, y) for x, y in zip(a, b) if not same(x, y)]
    if len(changed) != 1:
        return False
    x, y = changed[0]
    if not (isinstance(x, ast.Slice) and isinstance(y, ast.Slice)):
        return False
    if any(s.step is not None and integer(s.step) != 1 for s in (x, y)):
        return False
    offsets = []
    for upper, p, q in ((False, x.lower, y.lower), (True, x.upper, y.upper)):
        pbase, pi = bound(p, upper=upper)
        qbase, qi = bound(q, upper=upper)
        if pbase != qbase:
            return False
        offsets.append(pi - qi)
    return offsets[0] == offsets[1] and abs(offsets[0]) == 1


def resolve(node, assigned, visited=frozenset()):
    if isinstance(node, ast.Name) and node.id in assigned and node.id not in visited:
        return resolve(assigned[node.id], assigned, visited | {node.id})
    return node


def spatial_difference(left, right, assigned, neighbours, name):
    if offset_slices(left, right):
        return True
    left, right = resolve(left, assigned), resolve(right, assigned)
    if offset_slices(left, right):
        return True
    for shifted, original in ((left, right), (right, left)):
        if (isinstance(shifted, ast.Call) and name(shifted.func) in neighbours
                and shifted.args and same(resolve(shifted.args[0], assigned), original)):
            return True
    return False


def face_index(node):
    indices = node.elts if isinstance(node, ast.Tuple) else [node]
    fixed = [i for i in indices if integer(i) in (0, -1)]
    full = [i for i in indices if (isinstance(i, ast.Slice) and i.lower is None
                                 and i.upper is None and i.step is None)
            or (isinstance(i, ast.Constant) and i.value is Ellipsis)]
    return len(fixed) == 1 and len(fixed) + len(full) == len(indices) and len(indices) <= 3


def face_write(node):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("set", "multiply")):
        return False
    target = node.func.value
    return (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
            and target.value.attr == "at" and face_index(target.slice))


def face_mask(node, assigned, name):
    node = resolve(node, assigned)
    if face_write(node):
        return True
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
        return False
    for index, end in ((node.left, node.comparators[0]), (node.comparators[0], node.left)):
        index = resolve(index, assigned)
        if isinstance(index, ast.Subscript):
            index = resolve(index.value, assigned)
        if not (isinstance(index, ast.Call) and name(index.func) == "arange" and len(index.args) == 1):
            continue
        if integer(end) == 0:
            return True
        if (isinstance(end, ast.BinOp) and isinstance(end.op, ast.Sub)
                and integer(end.right) == 1 and same(end.left, index.args[0])):
            return True
    return False


def scan_source(source, path):
    """Return file, qualified function and kind, with sites for inspection."""
    tree = ast.parse(source)
    aliases = {alias.asname or alias.name: alias.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) for alias in node.names}

    def name(node):
        return aliases.get(call_name(node), call_name(node))

    neighbours = NEIGHBOURS | {"roll"}
    # Only wrappers RETURNING a neighbour qualify; a whole update function
    # containing a neighbour call does not become a neighbour primitive.
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    while True:
        added = {fn.name for fn in functions if any(
            isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
            and name(n.value.func) in neighbours for n in scoped_nodes(fn))}
        if added <= neighbours:
            break
        neighbours |= added
    rows = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.assigned = {}

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            old = self.assigned
            self.assigned = dict(old)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                self.assigned.pop(arg.arg, None)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()
            self.assigned = old

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Assign(self, node):
            self.generic_visit(node)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assigned[target.id] = node.value

        def visit_AnnAssign(self, node):
            self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.value is not None:
                self.assigned[node.target.id] = node.value

        def record(self, node, kind):
            rows.append(dict(file=path, function=".".join(self.scope) or "<module>",
                             kind=kind, line=node.lineno, expression=ast.unparse(node)))

        def visit_Call(self, node):
            called = name(node.func)
            if (called.lstrip("_").startswith(WALL_PREFIXES) or called in WALL_HELPERS
                    or face_write(node) or (called == "where" and node.args
                                           and face_mask(node.args[0], self.assigned, name))):
                self.record(node, "boundary")
            if called in ("roll", "diff"):
                self.record(node, "roll/diff")
            elif called in neighbours:
                self.record(node, "neighbour")
            args = [*node.args, *(kw.value for kw in node.keywords)]
            if called == "subtract" and len(args) >= 2:
                difference = spatial_difference(args[0], args[1], self.assigned, neighbours, name)
            else:
                difference = any(offset_slices(resolve(a, self.assigned), resolve(b, self.assigned))
                                 for a, b in combinations(args, 2))
            if difference:
                self.record(node, "difference")
            self.generic_visit(node)

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Sub) and spatial_difference(
                    node.left, node.right, self.assigned, neighbours, name):
                self.record(node, "difference")
            self.generic_visit(node)

    Visitor().visit(tree)
    return rows


def scoped_nodes(function):
    for child in ast.iter_child_nodes(function):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield child
            yield from scoped_nodes(child)


def scan_tree(root):
    return [row for path in sorted((root / "rfx").rglob("*.py"))
            for row in scan_source(path.read_text(), path.relative_to(root).as_posix())]


def identities(rows):
    counts = Counter()
    for row in rows:
        counts[row["file"], row["function"], row["kind"]] += row.get("count", 1)
    return counts


def registry_rows(rows, counted_functions=()):
    counted = {(r["file"], r["function"]) for r in counted_functions}
    return [dict(file=file, function=function, kind=kind,
                 **({"count": count} if (file, function) in counted else {}))
            for (file, function, kind), count in sorted(identities(rows).items())]


def compare_registry(expected, actual, counted_functions=()):
    want, got = identities(expected), identities(actual)
    counted = {(r["file"], r["function"]) for r in counted_functions}
    changed = {key for key in want.keys() | got.keys()
               if (key not in want or key not in got
                   or (key[:2] in counted and want[key] != got[key]))}
    if not changed:
        return
    messages = []
    for file, function, kind in sorted(changed):
        key = file, function, kind
        label = "unlisted sites" if got[key] > want[key] else "removed sites"
        messages.append(f"{label}: {file} :: {function} [{kind}]: registered {want[key]}, found {got[key]}")
        for row in actual:
            if (row["file"], row["function"], row["kind"]) == key:
                messages.append(f"  {file}:{row['line']}: {row['expression']}")
    messages.append("Run python -m tests.contracts.boundary_registry --update to rewrite "
                    "tests/contracts/boundary_registry.json; review the JSON diff. "
                    "When two merged PRs leave main's count stale, run --update on main.")
    raise AssertionError("\n".join(messages))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite boundary_registry.json from the current tree")
    args = parser.parse_args(argv)
    path = ROOT / "tests/contracts/boundary_registry.json"
    registry = json.loads(path.read_text())
    counted = registry["counted_functions"]
    actual = scan_tree(ROOT)
    if args.update:
        registry["sites"] = registry_rows(actual, counted)
        path.write_text(json.dumps(registry, indent=2) + "\n")
        print(f"Registered {len(actual)} sites in {len(registry['sites'])} keys: {path.relative_to(ROOT)}")
    else:
        compare_registry(registry["sites"], actual, counted)
        print(f"All {len(actual)} sites registered.")


if __name__ == "__main__":
    main()
