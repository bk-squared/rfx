"""Warning categories and opt-in console output are public contracts."""

import ast
import builtins
import importlib
from pathlib import Path

import pytest

import rfx
from rfx.errors import AccuracyWarning, ConfigurationWarning, RfxWarning

PACKAGE = Path(__file__).resolve().parents[2] / "rfx"

# Explicit output APIs, user-flagged output, and the two design exemptions.
PRINT_SITES = {
    ("cli.py", "_cmd_run"): "explicit output API or user-flagged output",
    ("cli.py", "_cmd_plot"): "explicit output API or user-flagged output",
    ("cli.py", "_cmd_export_geometry"): "explicit output API or user-flagged output",
    ("cli.py", "main"): "explicit output API or user-flagged output",
    ("diagnostics.py", "main"): "explicit output API or user-flagged output",
    ("model/occupancy.py", "select_occupancy_operator"): "design exemption: occupancy selection",
    ("convergence.py", "summary"): "explicit output API or user-flagged output",
    ("differentiable_material_fit.py", "differentiable_material_fit"): "explicit output API or user-flagged output",
    ("fidelity.py", "fidelity_report"): "explicit output API or user-flagged output",
    ("fidelity.py", "_print"): "explicit output API or user-flagged output",
    ("gpu.py", "print_benchmark"): "explicit output API or user-flagged output",
    ("optimize.py", "_run_one"): "explicit output API or user-flagged output",
    ("optimize.py", "forward"): "explicit output API or user-flagged output",
    ("optimize.py", "progressive_optimize"): "explicit output API or user-flagged output",
    ("sparams/_common.py", "_waveguide_s21_phase_residual"): "design exemption: phase residual",
    ("stepping/progress.py", "report"): "explicit output API or user-flagged output",
    ("stepping/progress.py", "note"): "explicit output API or user-flagged output",
    ("topology.py", "topology_optimize"): "explicit output API or user-flagged output",
    ("preflight/_impl.py", "_preflight_impl"): "preflight report rendering",
    ("api/_preflight.py", "preflight_sparameters"): "preflight report rendering",
    ("subgridding/runner.py", "run_subgridded"): "subgridding progress line",
}

CUSTOM_BASES = [
    ("rfx._runtime_floor", "UntestedJAXWarning", ConfigurationWarning),
    ("rfx._adi_notice", "ExperimentalADIWarning", AccuracyWarning),
    ("rfx._grid_metric", "Float32WidthResolutionWarning", AccuracyWarning),
    ("rfx.api._mesh", "AutoMeshWarning", RfxWarning),
    ("rfx.boundaries.features", "WaveguideBoundaryWarning", RfxWarning),
    ("rfx.boundaries.tfsf", "UnjudgedTFSFInvarianceWarning", AccuracyWarning),
    ("rfx.geometry.rasterize_grid", "ExactNodeSpineMissingWarning", AccuracyWarning),
    ("rfx.geometry.smoothing", "CurvedSmoothingWarning", AccuracyWarning),
    ("rfx.model.graded_admission", "DeclaredZWarning", RfxWarning),
    ("rfx.preflight._common", "PreflightWarning", RfxWarning),
    ("rfx.preflight._common", "PreflightErrorWarning", RfxWarning),
    ("rfx.probes.probes", "PreDecisionLumpedDiagonalWarning", AccuracyWarning),
    ("rfx.subgridding._notice", "ExperimentalSubgridWarning", AccuracyWarning),
]


def test_public_warning_mros_and_exports():
    assert RfxWarning.__mro__ == (
        RfxWarning, UserWarning, Warning, Exception, BaseException, object,
    )
    assert AccuracyWarning.__mro__ == (
        AccuracyWarning, RfxWarning, UserWarning, Warning, Exception, BaseException, object,
    )
    assert ConfigurationWarning.__mro__ == (
        ConfigurationWarning, RfxWarning, UserWarning, Warning, Exception, BaseException, object,
    )
    assert rfx.RfxWarning is RfxWarning
    assert rfx.AccuracyWarning is AccuracyWarning
    assert rfx.ConfigurationWarning is ConfigurationWarning
    assert {"RfxWarning", "AccuracyWarning", "ConfigurationWarning"} <= set(rfx.__all__)


@pytest.mark.parametrize("module, name, base", CUSTOM_BASES)
def test_custom_warning_bases(module, name, base):
    cls = getattr(importlib.import_module(module), name)
    assert issubclass(cls, base)
    assert issubclass(cls, UserWarning)
    if name == "PreflightErrorWarning":
        from rfx.preflight._common import PreflightWarning
        assert cls.__bases__ == (PreflightWarning,)
    else:
        assert cls.__bases__ == (base,)


def _module_name(path):
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(tree, module, is_package):
    """Resolve aliases, including local imports and relative package imports."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            package = module if is_package else module.rpartition(".")[0]
            source = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package,
            ) if node.level else node.module
            for alias in node.names:
                names[alias.asname or alias.name] = source + "." + alias.name
    return names


def _resolve(node, names, module):
    if isinstance(node, ast.Attribute):
        return _resolve(node.value, names, module) + "." + node.attr
    if isinstance(node, ast.Name):
        return names.get(node.id, ("builtins." if hasattr(builtins, node.id) else module + ".") + node.id)
    return ""


def _warning_class(qualified):
    module, _, name = qualified.rpartition(".")
    try:
        value = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError):
        return False
    return isinstance(value, type) and (
        value in (DeprecationWarning, FutureWarning, RuntimeWarning)
        or issubclass(value, RfxWarning)
    )


def _instance_class(node, tree, names, module, seen=()):
    """Follow warning constructors, simple assignments, and factory returns."""
    if node is None or id(node) in seen:
        return False
    seen = (*seen, id(node))
    if isinstance(node, ast.Name):
        values = [
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == node.id for t in n.targets)
        ]
        return bool(values) and all(_instance_class(v, tree, names, module, seen) for v in values)
    if not isinstance(node, ast.Call):
        return False
    qualified = _resolve(node.func, names, module)
    if _warning_class(qualified):
        return True
    source, _, name = qualified.rpartition(".")
    if not source.startswith("rfx"):
        return False
    imported = importlib.import_module(source)
    source_path = Path(imported.__file__)
    source_tree = tree if source == module else ast.parse(source_path.read_text())
    source_names = names if source == module else _imports(
        source_tree, source, source_path.name == "__init__.py",
    )
    functions = [n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        return False
    returns = [n.value for n in ast.walk(functions[0]) if isinstance(n, ast.Return)]
    return bool(returns) and all(
        _instance_class(value, source_tree, source_names, source, seen) for value in returns
    )


@pytest.mark.source_scan
def test_warn_sites_use_public_categories():
    violations = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module = _module_name(path)
        names = _imports(tree, module, path.name == "__init__.py")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Also inspect warning-module parameters such as preflight's _w.
            is_warn = (
                isinstance(node.func, ast.Attribute) and node.func.attr == "warn"
            ) or _resolve(node.func, names, module) == "warnings.warn"
            if not is_warn:
                continue
            category = next(
                (k.value for k in node.keywords if k.arg == "category"),
                node.args[1] if len(node.args) > 1 else None,
            )
            valid = (
                _warning_class(_resolve(category, names, module)) if category is not None
                else bool(node.args) and _instance_class(node.args[0], tree, names, module)
            )
            if not valid:
                violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}: {ast.unparse(node)}")
    assert not violations, "Warnings without an approved category:\n" + "\n".join(violations)


@pytest.mark.source_scan
def test_print_sites_are_opt_in():
    violations = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"):
                continue
            owner = node
            while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parents[owner]
            key = (str(path.relative_to(PACKAGE)), getattr(owner, "name", "<module>"))
            if key not in PRINT_SITES:
                violations.append(f"{key[0]}:{node.lineno}: {key[1]}")
    assert not violations, "Console output without an explicit opt-in:\n" + "\n".join(violations)
