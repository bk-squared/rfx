"""Written-out default grids may only disappear from this allowlist."""

import ast
from collections import Counter
from pathlib import Path


ALLOWLIST = {
    ('rfx/measurement/flux_weights.py', 'build_flux_monitor_cfgs'): 1,
    ('rfx/measurement/plan.py', '_frequencies'): 1,
    ('rfx/measurement/plan.py', 'build_measurement_plan'): 1,
    ('rfx/measurement/setup.py', 'build_planes'): 1,
    ('rfx/preflight/waveguide.py', 'resolve_waveguide_port_freqs'): 1,
    ('rfx/runners/nonuniform.py', 'run_nonuniform_path'): 2,
    ('rfx/runners/subgridded.py', 'run_subgridded_path'): 1,
    ('rfx/runners/uniform.py', 'run_uniform'): 5,
    ('rfx/sparams/coax.py', 'compute_coax_msl_transition'): 1,
    ('rfx/sparams/coax.py', 'compute_coaxial_line_reflection'): 1,
    ('rfx/sparams/coax.py', 'compute_coaxial_two_port'): 1,
    ('rfx/sparams/mixed.py', 'compute_mixed_s_matrix'): 1,
    ('rfx/sparams/msl.py', 'compute_msl_s_matrix'): 1,
    ('rfx/sparams/waveguide.py', '_compute_waveguide_s_matrix_nu'): 2,
    ('rfx/vmap_sweep.py', '_build_full_scan_fn'): 1,
}
PERMITTED = ("rfx/measurement/monitors.py", "default_frequency_grid")


def _frequency(node):
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
    return "freq_max" in name


def _number(node, value):
    return isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value == value


def _lower_edge(node):
    if not isinstance(node, ast.BinOp):
        return False
    if isinstance(node.op, ast.Div):
        return _frequency(node.left) and _number(node.right, 10)
    if isinstance(node.op, ast.Mult):
        return ((_number(node.left, 0.1) and _frequency(node.right))
                or (_frequency(node.left) and _number(node.right, 0.1)))
    return False


def grid_sites(source):
    """Yield (qualname, Call) for every syntactically written default grid."""
    found = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def visit_scope(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_ClassDef = visit_scope
        visit_FunctionDef = visit_scope
        visit_AsyncFunctionDef = visit_scope

        def visit_Call(self, node):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name.endswith("linspace") and node.args and _lower_edge(node.args[0]):
                found.append((".".join(self.scope) or "<module>", node))
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def test_default_frequency_grid_sites():
    root = Path(__file__).resolve().parents[2]
    found = Counter()
    for path in sorted((root / "rfx").rglob("*.py")):
        for qualname, _ in grid_sites(path.read_text()):
            key = (path.relative_to(root).as_posix(), qualname)
            if key != PERMITTED:
                found[key] += 1
    assert dict(found) == ALLOWLIST, (
        "A new written-out default must call rfx.measurement.default_frequency_grid instead; "
        "a removed site must be deleted from the allowlist (the list may only shrink). "
        f"Found: {dict(found)!r}"
    )
