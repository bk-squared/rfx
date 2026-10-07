"""Executable declaration reads belong to the boundary resolver, not consumers."""
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
# Pure forwarding is not a depth decision and does not need an exception.
DEFERRED = {}
LEGACY_SITE = ("rfx/boundaries/cpml.py", "face_layers.get(face_name, n)")
# Legacy dual declaration, exactly one site:
# docs/design_notes/20260923_boundary_model_predeclaration.md §7 Addendum 2.
REALIZED_SITE = ("rfx/farfield.py", "faces.get(face, 0 if legacy is None else legacy)")
# NTFF's legacy face_layers passes through realized pads; its scalar is only a fallback.
DECLARED_SITES = {
    ("rfx/grid.py", "record.declared"),  # Public legacy face_layers view preserves declared counts.
    ("rfx/api/_compile.py", "record.declared"),  # Pass declared counts to the grid's resolver.
    ("rfx/rcs.py", "face.declared"),  # Preserve the scalar-only phase-reference admission gate.
}
DECLARATION_MODULES = {"rfx/boundaries/depths.py", "rfx/boundaries/spec.py"}


def _name_parts(node, bindings, seen=frozenset()):
    """Track string construction without executing consumer code."""
    if isinstance(node, ast.Constant):
        return str(node.value) if isinstance(node.value, str) else "?"
    if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
        return _name_parts(bindings[node.id], bindings, seen | {node.id})
    if isinstance(node, ast.JoinedStr):
        return "".join(_name_parts(part, bindings, seen) for part in node.values)
    if isinstance(node, ast.BinOp):
        left = _name_parts(node.left, bindings, seen)
        return left + _name_parts(node.right, bindings, seen)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _name_parts(node.func.value, bindings, seen)
    return "?"


def declaration_reads(source):
    """Follow mapping aliases and constructed attribute names in executable AST."""
    tree = ast.parse(source)
    reads = []

    def scan(scope, inherited=None):
        bindings = dict(inherited or {})
        nodes = []

        def collect(node):
            nodes.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return
            for child in ast.iter_child_nodes(node):
                collect(child)

        for statement in scope.body:
            collect(statement)
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = node.value

        def mapping(node, seen=frozenset()):
            if isinstance(node, ast.Name):
                if "face_layers" in node.id:
                    return True
                return (node.id in bindings and node.id not in seen
                        and mapping(bindings[node.id], seen | {node.id}))
            if isinstance(node, ast.Attribute):
                return node.attr == "face_layers"
            if isinstance(node, ast.BoolOp):
                return any(mapping(value, seen) for value in node.values)
            if isinstance(node, ast.Dict):
                return any(key is None and mapping(value, seen)
                           for key, value in zip(node.keys, node.values))
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "dict":
                    return any(mapping(arg, seen) for arg in node.args)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "copy":
                    return mapping(node.func.value, seen)
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and len(node.args) > 1
                    and _name_parts(node.args[1], bindings) == "face_layers")

        def thickness(node):
            key = _name_parts(node, bindings)
            return key in {"lo_thickness", "hi_thickness", "resolved_lo_thickness", "resolved_hi_thickness"} or (
                "_thickness" in key and any(part in key for part in ("?", "%", "{")))

        def depth_record(node, seen=frozenset()):
            if isinstance(node, ast.Name):
                return (node.id in {"face", "record"}
                        or (node.id in bindings and node.id not in seen
                            and depth_record(bindings[node.id], seen | {node.id})))
            if isinstance(node, ast.Attribute):
                return node.attr == "boundary_depths"
            if isinstance(node, ast.Subscript):
                return depth_record(node.value, seen)
            if isinstance(node, ast.Call):
                return (isinstance(node.func, ast.Name) and node.func.id in {
                    "resolve_face_depths", "grid_face_depths", "simulation_face_depths"})
            return False

        for node in nodes:
            if isinstance(node, (ast.comprehension, ast.For)) and isinstance(node.target, ast.Name):
                if depth_record(node.iter):
                    bindings[node.target.id] = node.iter

        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scan(node, bindings)
                continue
            matched = isinstance(node, ast.Attribute) and thickness(ast.Constant(node.attr))
            if isinstance(node, ast.Attribute) and node.attr == "declared":
                matched |= depth_record(node.value)
            if isinstance(node, ast.Subscript):
                matched |= mapping(node.value) or thickness(node.slice)
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {"get", "items", "values", "__getitem__"}:
                    matched |= mapping(func.value)
                    matched |= bool(node.args and thickness(node.args[0]))
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                if name == "getattr" and len(node.args) > 1:
                    matched |= thickness(node.args[1])
                    matched |= (_name_parts(node.args[1], bindings) == "declared"
                                and depth_record(node.args[0]))
                if name == "attrgetter":
                    matched |= any(thickness(arg) for arg in node.args)
            if matched:
                reads.append((node.lineno, ast.unparse(node)))
    scan(tree)
    return reads


def test_one_source_for_face_depths():
    seen = {}
    legacy = []
    realized = []
    declared = []
    violations = []
    for path in sorted((ROOT / "rfx").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in DECLARATION_MODULES:
            continue
        reads = declaration_reads(path.read_text())
        if relative in DEFERRED:
            seen[relative] = len(reads)
        else:
            for line, expression in reads:
                if (relative, expression) == LEGACY_SITE:
                    legacy.append(line)
                elif (relative, expression) == REALIZED_SITE:
                    realized.append(line)
                elif (relative, expression) in DECLARED_SITES:
                    declared.append((relative, expression))
                else:
                    violations.append(f"{relative}:{line}: {expression}")
    assert not violations, "Declaration-derived face depths outside the record:\n" + "\n".join(violations)
    assert seen == DEFERRED, "PR2 inventory changed; remove stale exceptions and reject new copies"
    assert len(legacy) == 1, "Addendum 2 permits exactly one legacy dual declaration site"
    assert len(realized) == 1, "NTFF permits exactly one realized-pad pass-through site"
    assert len(declared) == len(DECLARED_SITES) and set(declared) == DECLARED_SITES


@pytest.mark.parametrize("expression", [
    "face_layers.get(face, budget)", "grid.face_layers[face]",
    "face.resolved_lo_thickness(n)", "face.resolved_hi_thickness(n)",
    "face.lo_thickness", "face.hi_thickness",
    'getattr(face, f"{side}_thickness")',
    'getattr(face, f"resolved_{side}_thickness")',
    'declaration.get(f"{side}_thickness")',
])
def test_seeded_declaration_read_is_detected(tmp_path, expression):
    seeded = tmp_path / "seeded_consumer.py"
    seeded.write_text(f"def depth():\n    return {expression}\n")
    assert declaration_reads(seeded.read_text()), "disabled structural scan accepted seeded declaration read"


def test_comments_and_docstrings_are_not_reads():
    assert not declaration_reads('''"""face.lo_thickness and face_layers[face]"""
# face.resolved_hi_thickness(n)
value = 0
''')


@pytest.mark.parametrize("source", [
    "fl = grid.face_layers; depth = fl.get(face, budget)",
    "fl = grid.face_layers; other = fl; depth = other[face]",
    "fl = getattr(grid, 'face_layers', {}); depth = fl.get(face)",
    "fl = grid.face_layers; depth = list(fl.items())",
    "fl = grid.face_layers; depth = max(fl.values())",
    "depth = list(grid.face_layers.items())",
    "depth = max(grid.face_layers.values())",
    "b = boundary; depth = b.lo_thickness",
    "b = Boundary(lo='cpml', hi='cpml'); depth = b.resolved_hi_thickness(n)",
    'depth = getattr(boundary, side + "_thickness")',
    'depth = getattr(boundary, "%s_thickness" % side)',
    'key = f"{side}_thickness"; b = boundary; depth = getattr(b, key)',
    'key = side + "_thick" + "ness"; depth = getattr(boundary, key)',
    'key = "{}_thickness".format(side); depth = getattr(boundary, key)',
    'depth = attrgetter(side + "_thickness")(boundary)',
    'key = "%s_thickness" % side; read = operator.attrgetter(key); depth = read(boundary)',
    'fl = dict(grid.face_layers); depth = fl[face]',
    'fl = dict(grid.face_layers); depth = fl.get(face)',
    'fl = grid.face_layers.copy(); depth = fl.get(face)',
    'fl = grid.face_layers.copy(); depth = fl[face]',
    'fl = {**grid.face_layers}; depth = fl[face]',
    'fl = {**grid.face_layers}; depth = fl.get(face)',
    'depth = record.declared',
    'depth = grid.boundary_depths[0].declared',
    'item = grid.boundary_depths[0]; depth = item.declared',
    'item = grid.boundary_depths[0]; depth = getattr(item, "declared")',
    'depth = [item.declared for item in grid.boundary_depths]',
    'records = resolve_face_depths(spec, budget=n); depth = [item.declared for item in records]',
])
def test_seeded_indirect_declaration_read_is_detected(source):
    assert declaration_reads(source), "disabled widened scan accepted indirect declaration read"


def test_realized_record_and_pure_forwarding_are_not_declaration_reads():
    assert not declaration_reads("""
forward(grid.face_layers)
depths = {face.name: face.realized for face in grid.boundary_depths}
depth = depths.get(face)
""")


@pytest.mark.parametrize("path", ["rfx/boundaries/pec.py", "rfx/boundaries/model.py", "rfx/boundaries/cpml.py"])
def test_boundary_consumers_have_no_blanket_exemption(path):
    assert path not in DECLARATION_MODULES
    assert declaration_reads("depth = grid.face_layers.get(face)")


def feature_axis_reads(source):
    """Feature-derived absorbing axes may only be read inside boundaries."""
    tree = ast.parse(source)
    names = {"_waveguide_cpml_axes", "_guide_axes"}
    return [(node.lineno, ast.unparse(node)) for node in ast.walk(tree)
            if (isinstance(node, ast.Attribute) and node.attr in names)
            or (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names)
            or (isinstance(node, ast.ImportFrom) and any(alias.name in names for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "rfx.sources.tfsf"
                and any(alias.name == "tfsf_boundary_flags" for alias in node.names))
            or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant) and node.args[1].value in names)]


def test_feature_axis_readers_belong_to_boundaries():
    violations = []
    for path in (ROOT / "rfx").rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[:2] == ("rfx", "boundaries"):
            continue
        violations.extend(f"{relative}:{line}: {expression}"
                          for line, expression in feature_axis_reads(path.read_text()))
    assert not violations, "Feature axis reads outside boundaries:\n" + "\n".join(violations)


@pytest.mark.parametrize("source", [
    "axes = sim._waveguide_cpml_axes()",
    "axes = getattr(sim, '_waveguide_cpml_axes')()",
    "from rfx.boundaries.features import _guide_axes as axes; axes(sim)",
    "axes = _guide_axes(sim)",
    "from rfx.sources.tfsf import tfsf_boundary_flags; axes = tfsf_boundary_flags(cfg)",
])
def test_seeded_feature_axis_read_is_detected(source):
    assert feature_axis_reads(source), "disabled scan accepted a feature axis reader"
