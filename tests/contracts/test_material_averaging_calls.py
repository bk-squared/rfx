"""Down-only call-site budget for material averaging outside its owner.

Counts direct calls, imported aliases and attribute calls of the seven helpers.
Arbitrary runtime rebinding / computed getattr is outside this syntactic gate.
Both directions are checked: removing a call must lower its budget immediately.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPERS = frozenset({"component_e_materials", "component_h_materials",
                     "cell_component_e_materials", "edge_averaged_materials",
                     "edge_mean_components", "cell_owned_component_materials",
                     "permittivity_without_lumped"})
EXCLUDED = {"rfx/core/yee.py", "rfx/stepping/yee.py", "rfx/model/materials.py"}  # the Yee kernels moved to rfx/stepping/yee.py

# May only shrink. With seven helpers the base (5a7e4f28) has 52 sites in
# 27 modules (the original five: 46 in 26), including two nonuniform.py aliases.
ALLOWED_CALLS = {'rfx/adi.py': 3,
 'rfx/current_moments.py': 2,
 # static_remnant_advisory's H average moved here from rfx/simulation.py (decay stop PR).
 'rfx/measurement/decay.py': 1,
 # slab_e_component_materials moved here from rfx/runners/_distributed_common.py
 # (#1966). It has 3 calls, one more than on main: the slab takes the edge-owned
 # lumped stamp out of the cells before the high-wall view (the view replicates
 # cells, not edges). jit_runner.py's 2 calls moved into rfx/model/materials.py.
 'rfx/model/electric_metrics.py': 3,
 'rfx/stepping/slab_update.py': 4,
 'rfx/stepping/slab_graded_legacy_loop.py': 1,
 'rfx/runners/nonuniform.py': 1,
 'rfx/sources/coaxial_port.py': 1,
 'rfx/sources/msl_port.py': 1,
 'rfx/sources/tfsf.py': 1,
 'rfx/sources/tfsf_2d.py': 1,
 'rfx/stepping/waveguide_source.py': 1,
 'rfx/subgridding/disjoint_3d.py': 1,
 'rfx/subgridding/sbp_sat_1d.py': 1,
 'rfx/subgridding/sbp_sat_2d.py': 2,
    "rfx/stepping/tfsf.py": 1,
    "rfx/stepping/tfsf_2d.py": 2,
    "rfx/stepping/tfsf_oblique.py": 1,
}


def averaging_calls(tree):
    aliases = {a.asname or a.name: a.name
               for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               for a in node.names if a.name in HELPERS}
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (aliases.get(func.id, func.id) if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None)
        if name in HELPERS:
            hits.append((node.lineno, ast.unparse(node)))
    return sorted(hits)


def measure():
    counts = {}
    for path in sorted((ROOT / "rfx").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED:
            continue
        hits = averaging_calls(ast.parse(path.read_text(), filename=str(path)))
        if hits:
            counts[relative] = len(hits)
    return counts


def test_no_module_exceeds_its_averaging_budget():
    growth = {p: (n, ALLOWED_CALLS.get(p, 0)) for p, n in measure().items()
              if n > ALLOWED_CALLS.get(p, 0)}
    assert not growth, f"averaging calls grew (actual, allowed): {growth}"


def test_allow_list_is_not_stale():
    actual = measure()
    stale = {p: (n, actual.get(p, 0)) for p, n in ALLOWED_CALLS.items()
             if n > actual.get(p, 0)}
    assert not stale, f"lower/delete stale budgets (allowed, actual): {stale}"
    assert all(n > 0 and (ROOT / p).is_file() for p, n in ALLOWED_CALLS.items())


def test_scan_root_and_aliases():
    assert len(list((ROOT / "rfx").rglob("*.py"))) > 100
    for helper in HELPERS:
        source = (f"from rfx.core.yee import {helper} as alias\n"
                  f"alias(m)\n{helper}(m)\nyee.{helper}(m)\n")
        assert len(averaging_calls(ast.parse(source))) == 3
    assert not averaging_calls(ast.parse(
        "from rfx.core.yee import component_e_materials\n"
        "name = 'component_e_materials'\n"
        "def component_h_materials(m): pass\n"))
