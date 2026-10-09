"""Material cell overrides have one owner, including design-window writes."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {"eps_override", "sigma_override", "mu_r_override", "design_box",
          "design_eps_override", "design_sigma_override"}
FIELDS = {"eps_r", "sigma", "mu_r", "eps_r_lumped", "sigma_lumped", "mu_r_wire"}


def _scope_override_writes(tree):
    """Track override aliases and detect material constructors/replacements.

    This is a syntactic contract; dynamic rebinding and computed keyword
    dictionaries are outside its scope. New aliases are followed to a fixed
    point, so spelling a local variable differently does not evade the gate.
    """
    tainted = set(INPUTS)
    def uses(node):
        return any(isinstance(n, ast.Name) and n.id in tainted for n in ast.walk(node))
    assignments = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign))]
    while True:
        old = set(tainted)
        for node in assignments:
            # The assembled container is a stage output, not an alias for
            # an override. Following it would count unrelated ADI broadcast
            # and later readers as fresh override writers.
            value = node.value
            call = value.func if isinstance(value, ast.Call) else None
            name = call.attr if isinstance(call, ast.Attribute) else getattr(call, "id", "")
            if name in ("apply_material_overrides", "_replace", "MaterialArrays"):
                continue
            if value is not None and uses(value):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                tainted.update(n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name))
        if old == tainted:
            break
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name in ("_replace", "MaterialArrays") and any(
                kw.arg in FIELDS and uses(kw.value) for kw in node.keywords):
            hits.append(node.lineno)
        # A positional constructor names no field; any override in it counts.
        elif name == "MaterialArrays" and any(uses(arg) for arg in node.args):
            hits.append(node.lineno)
    return hits


def override_writes(tree):
    # Each function owns its local aliases. A same-named variable in a
    # different runner is not a data-flow connection.
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scopes.append(ast.Module(body=[n for n in getattr(tree, "body", [])
                                  if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
                             type_ignores=[]))
    return sorted({line for scope in scopes for line in _scope_override_writes(scope)})


def test_no_override_writes_outside_model():
    hits = {}
    for path in (ROOT / "rfx").rglob("*.py"):
        if path.is_relative_to(ROOT / "rfx/model"):
            continue
        found = override_writes(ast.parse(path.read_text()))
        if found:
            hits[str(path.relative_to(ROOT))] = found
    assert hits == {}, f"cell override writes outside rfx/model/: {hits}"


def test_one_stage_definition_and_all_direct_writer_routes_call_it():
    definitions = []
    for path in (ROOT / "rfx").rglob("*.py"):
        tree = ast.parse(path.read_text())
        definitions.extend(str(path.relative_to(ROOT)) for n in ast.walk(tree)
                           if isinstance(n, ast.FunctionDef) and n.name == "apply_material_overrides")
    assert definitions == ["rfx/model/overrides.py"]
    for file in ("rfx/api/_execute.py", "rfx/runners/nonuniform.py",
                 "rfx/sparams/waveguide.py", "rfx/model/materials.py"):
        tree = ast.parse((ROOT / file).read_text())
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "apply_material_overrides" for n in ast.walk(tree)), file


def test_counter_catches_direct_and_aliased_inputs():
    assert override_writes(ast.parse("m = m._replace(eps_r=eps_override)")) == [1]
    assert override_writes(ast.parse("e = eps_override\nm = m._replace(eps_r=e)")) == [2]
    assert override_writes(ast.parse("e = design_box.eps_r\nm = MaterialArrays(eps_r=e)")) == [2]
    assert override_writes(ast.parse("m = MaterialArrays(eps_override, m.sigma, m.mu_r)")) == [1]
    assert not override_writes(ast.parse("m = apply_material_overrides(m, eps_override=e)"))
