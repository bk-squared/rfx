"""Moved import paths warn on every access; package-owned lazy exports do not."""

import ast
from importlib import import_module, util
from pathlib import Path
import subprocess
import sys
import warnings

import pytest


REPO = Path(__file__).resolve().parents[2]
# Literal contract: do not derive expectations from production redirect tables.
MOVED = [
    ('rfx.adjoint', '_plane_slice', 'rfx.stepping.adjoint_scan',
     'rfx.adjoint._plane_slice moved to rfx.stepping.adjoint_scan._plane_slice; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.adjoint', 'design_adjoint_scan', 'rfx.stepping.adjoint_scan',
     'rfx.adjoint.design_adjoint_scan moved to rfx.stepping.adjoint_scan.design_adjoint_scan; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.rcs', 'compute_rcs', 'rfx.measurement.rcs',
     'rfx.rcs.compute_rcs moved to rfx.measurement.rcs.compute_rcs; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.farfield', 'compute_far_field', 'rfx.measurement.farfield',
     'rfx.farfield.compute_far_field moved to rfx.measurement.farfield.compute_far_field; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.farfield', 'compute_far_field_jax', 'rfx.measurement.farfield',
     'rfx.farfield.compute_far_field_jax moved to rfx.measurement.farfield.compute_far_field_jax; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.ringdown', '_emulate_accumulators', 'rfx.stepping.ringdown_emulation',
     'rfx.ringdown._emulate_accumulators moved to rfx.stepping.ringdown_emulation._emulate_accumulators; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.ringdown', 'plain_dft', 'rfx.measurement.ringdown_tail',
     'rfx.ringdown.plain_dft moved to rfx.measurement.ringdown_tail.plain_dft; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.ringdown', 'tail_dft', 'rfx.measurement.ringdown_tail',
     'rfx.ringdown.tail_dft moved to rfx.measurement.ringdown_tail.tail_dft; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.floquet', 'floquet_phase_shift', 'rfx.measurement.floquet',
     'rfx.floquet.floquet_phase_shift moved to rfx.measurement.floquet.floquet_phase_shift; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.floquet', 'inject_floquet_source', 'rfx.stepping.floquet',
     'rfx.floquet.inject_floquet_source moved to rfx.stepping.floquet.inject_floquet_source; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.floquet', 'update_floquet_dft', 'rfx.measurement.floquet',
     'rfx.floquet.update_floquet_dft moved to rfx.measurement.floquet.update_floquet_dft; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.progress', 'scan_with_progress', 'rfx.stepping.progress_scan',
     'rfx.progress.scan_with_progress moved to rfx.stepping.progress_scan.scan_with_progress; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx._ringdown_jax', 'plain_dft', 'rfx.measurement.ringdown_tail_jax',
     'rfx._ringdown_jax.plain_dft moved to rfx.measurement.ringdown_tail_jax.plain_dft; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx._ringdown_jax', 'tail_dft', 'rfx.measurement.ringdown_tail_jax',
     'rfx._ringdown_jax.tail_dft moved to rfx.measurement.ringdown_tail_jax.tail_dft; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.materials.nonlinear', 'apply_kerr_ade', 'rfx.stepping.kerr',
     'rfx.materials.nonlinear.apply_kerr_ade moved to rfx.stepping.kerr.apply_kerr_ade; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.materials.lorentz', 'update_e_lorentz', 'rfx.stepping.poles',
     'rfx.materials.lorentz.update_e_lorentz moved to rfx.stepping.poles.update_e_lorentz; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.materials.debye', 'update_e_debye', 'rfx.stepping.poles',
     'rfx.materials.debye.update_e_debye moved to rfx.stepping.poles.update_e_debye; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_e', 'rfx.stepping.yee',
     'rfx.core.yee.update_e moved to rfx.stepping.yee.update_e; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_e_aniso', 'rfx.stepping.yee',
     'rfx.core.yee.update_e_aniso moved to rfx.stepping.yee.update_e_aniso; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_e_aniso_inv', 'rfx.stepping.yee',
     'rfx.core.yee.update_e_aniso_inv moved to rfx.stepping.yee.update_e_aniso_inv; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_e_box', 'rfx.stepping.yee',
     'rfx.core.yee.update_e_box moved to rfx.stepping.yee.update_e_box; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_e_fast', 'rfx.stepping.yee',
     'rfx.core.yee.update_e_fast moved to rfx.stepping.yee.update_e_fast; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_e_nu', 'rfx.stepping.yee',
     'rfx.core.yee.update_e_nu moved to rfx.stepping.yee.update_e_nu; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_e_nu_aniso', 'rfx.stepping.yee',
     'rfx.core.yee.update_e_nu_aniso moved to rfx.stepping.yee.update_e_nu_aniso; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_h', 'rfx.stepping.yee',
     'rfx.core.yee.update_h moved to rfx.stepping.yee.update_h; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_h_fast', 'rfx.stepping.yee',
     'rfx.core.yee.update_h_fast moved to rfx.stepping.yee.update_h_fast; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.yee', 'update_h_nu', 'rfx.stepping.yee',
     'rfx.core.yee.update_h_nu moved to rfx.stepping.yee.update_h_nu; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core', 'update_e', 'rfx.stepping.yee',
     'rfx.core.update_e moved to rfx.stepping.yee.update_e; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core', 'update_h', 'rfx.stepping.yee',
     'rfx.core.update_h moved to rfx.stepping.yee.update_h; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core', 'update_h_fast', 'rfx.stepping.yee',
     'rfx.core.update_h_fast moved to rfx.stepping.yee.update_h_fast; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core', 'update_e_fast', 'rfx.stepping.yee',
     'rfx.core.update_e_fast moved to rfx.stepping.yee.update_e_fast; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.jax_utils', 'recorded_scan', 'rfx.stepping.scan',
     'rfx.core.jax_utils.recorded_scan moved to rfx.stepping.scan.recorded_scan; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.jax_utils', '_scan_with_opaque_operands', 'rfx.stepping.scan',
     'rfx.core.jax_utils._scan_with_opaque_operands moved to rfx.stepping.scan._scan_with_opaque_operands; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.jax_utils', '_opaque', 'rfx.stepping.scan',
     'rfx.core.jax_utils._opaque moved to rfx.stepping.scan._opaque; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.core.jax_utils', '_optimization_barrier', 'rfx.stepping.scan',
     'rfx.core.jax_utils._optimization_barrier moved to rfx.stepping.scan._optimization_barrier; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.sources.tfsf_2d', 'bloch_phase_tuple', 'rfx.stepping.tfsf_2d',
     'rfx.sources.tfsf_2d.bloch_phase_tuple moved to rfx.stepping.tfsf_2d.bloch_phase_tuple; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.sources.tfsf', 'apply_tfsf_e', 'rfx.stepping.tfsf',
     'rfx.sources.tfsf.apply_tfsf_e moved to rfx.stepping.tfsf.apply_tfsf_e; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.sources.tfsf', 'apply_tfsf_h', 'rfx.stepping.tfsf',
     'rfx.sources.tfsf.apply_tfsf_h moved to rfx.stepping.tfsf.apply_tfsf_h; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.sources.tfsf', 'measure_normal_incident_spectrum', 'rfx.measurement.incident',
     'rfx.sources.tfsf.measure_normal_incident_spectrum moved to rfx.measurement.incident.measure_normal_incident_spectrum; import it from there. The old path will be removed in rfx 3.0.'),
    ('rfx.sources.tfsf_oblique_open', 'measure_incident_spectrum', 'rfx.measurement.incident',
     'rfx.sources.tfsf_oblique_open.measure_incident_spectrum moved to rfx.measurement.incident.measure_incident_spectrum; import it from there. The old path will be removed in rfx 3.0.'),
]
OWNED = [
    ('rfx.stepping', 'core_step_invariants', 'rfx.stepping.uniform_setup'),
    ('rfx.stepping', 'make_core_step', 'rfx.stepping.uniform_setup'),
    ('rfx.stepping', 'run_nonuniform', 'rfx.stepping.graded_loop'),
    ('rfx.stepping', 'run_nonuniform_until_decay', 'rfx.stepping.graded_loop'),
    ('rfx.stepping', 'run_uniform_loop', 'rfx.stepping.uniform_loop'),
    ('rfx.stepping', 'run_uniform_until_decay', 'rfx.stepping.uniform_loop'),
    ('rfx.stepping', 'scan_with_progress', 'rfx.stepping.progress_scan'),
]


@pytest.mark.parametrize("old,name,target,message", MOVED)
def test_moved_name_redirect(old, name, target, message):
    module = import_module(old)
    expected = getattr(import_module(target), name)
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = getattr(module, name)
        assert actual is expected
        assert len(caught) == 1
        assert caught[0].category is DeprecationWarning
        assert str(caught[0].message) == message
        assert caught[0].filename == __file__


@pytest.mark.parametrize("old,name,target", OWNED)
def test_package_owned_lazy_export(old, name, target):
    module = import_module(old)
    expected = getattr(import_module(target), name)
    # Exercise the lazy resolver even if another import already populated it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert module.__getattr__(name) is expected
        assert getattr(module, name) is expected
    assert caught == []


def test_import_rfx_without_deprecation_warnings():
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import rfx"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.source_scan
def test_lazy_export_inventory():
    found = set()
    for path in (REPO / "rfx").rglob("*.py"):
        module = ".".join(path.relative_to(REPO).with_suffix("").parts).removesuffix(".__init__")
        for node in ast.parse(path.read_text()).body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS" for t in node.targets)):
                found.update((module, name, target) for name, target in ast.literal_eval(node.value).items())
    assert found == {(old, name, target) for old, name, target, _ in MOVED} | set(OWNED)


def _old_path_readers(source, package):
    moved = {(old, name) for old, name, _, _ in MOVED}
    tree = ast.parse(source)
    aliases = {}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = (util.resolve_name("." * node.level + (node.module or ""), package)
                      if node.level else node.module or "")
            for alias in node.names:
                aliases[alias.asname or alias.name] = module + "." + alias.name
                if (module, alias.name) in moved:
                    violations.append((node.lineno, module + "." + alias.name))
    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return resolve(node.value) + "." + node.attr
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (resolve(node.value), node.attr) in moved:
            violations.append((node.lineno, resolve(node)))
        if (isinstance(node, ast.Call) and len(node.args) >= 2
                and isinstance(node.func, (ast.Name, ast.Attribute))
                and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
                in {"getattr", "setattr"}
                and isinstance(node.args[1], ast.Constant)
                and (resolve(node.args[0]), node.args[1].value) in moved):
            violations.append((node.lineno, resolve(node.args[0]) + "." + node.args[1].value))
    return violations


@pytest.mark.source_scan
def test_no_deprecated_path_readers():
    violations = []
    for root in ("rfx", "tests"):
        for path in (REPO / root).rglob("*.py"):
            if path == Path(__file__):
                continue
            relative = path.relative_to(REPO)
            package = ".".join(relative.parent.parts)
            violations.extend(f"{relative}:{line}: {name}" for line, name in
                              _old_path_readers(path.read_text(), package))
    assert violations == [], "\n".join(violations)
