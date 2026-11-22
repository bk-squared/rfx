"""Review witnesses with their check removed and their defect restored."""
import os
import subprocess
import sys

import pytest


CASES = {
    'drude_named': ('check_refusal', "rewrite(admission, 'refuse_bore_materials', \": material '{entry.material_name}' has {words}\", '')"),
    'dispersive_outside_bore': ('check_drude_admitted', "rewrite(admission, 'refuse_bore_materials', 'np.asarray(entry.shape.mask(root.grid)) & bore', 'np.asarray(entry.shape.mask(root.grid))')"),
    'reference_substrate': ('check_substrate_admitted', "rewrite(admission, 'refuse_bore_materials', ' & bore', ' & (bore | np.logical_or.reduce(edge_bore))')"),
    'other_excitation': ('check_refusal', "admission.refuse_other_s_drives = lambda *args: None"),
    'bore_material': ('check_refusal', "admission.refuse_bore_materials = lambda *args: None"),
    'mu_override': ('check_refusal', "\nfrom rfx.runners._admission import LANE_GATES\nfor lane in ('run_uniform', 'fwd_uniform'):\n    LANE_GATES[lane][('_coaxial_ports', 'coax_port')].pop('no mu_r_override')"),
    'off_record_steps': ('check_refusal', "admission.refuse_off_record_steps = lambda *args: None"),
    'pin_absorber': ('check_refusal', "rewrite(builder, 'stamp_coax_pin', 'if not low <= tip <= high:', 'if not 0 <= tip < grid.shape[2]:')"),
    'design_box': ('check_refusal', "admission.refuse_design_box = lambda *args: None"),
    'traced_stamp_material': ('check_refusal', "admission.refuse_traced_materials = lambda *args: None"),
    'user_jit': ('check_jit', "\nimport rfx.runners._admission as lanes\noriginal = lanes.admit_forward_entry\ndef refuse_jit(sim, args):\n    original(sim, args)\n    if lanes._under_external_jax_transform():\n        raise RuntimeError('restored external-jit failure')\nlanes.admit_forward_entry = refuse_jit\nimport rfx.api._execute as execute\nexecute.admit_forward_entry = refuse_jit"),
    'refinement_s_request': ('check_refusal', "\nimport rfx.runners._admission as lanes\nlanes.admit_coax_declaration = lambda *args: None\nfrom rfx.api._preflight import _PreflightMixin\nrewrite(_PreflightMixin, '_validate_run_sparameter_request', 'self._refinement is not None and not self._coaxial_ports', 'self._refinement is not None')"),
    'settling_keeps_user_probes': ('check_probe_records', "\nimport rfx.api._settling as settling\nrewrite(settling, 'settling_source_end', 'self._coaxial_ports and reading_port and grid is not None', 'self._coaxial_ports and port_records and grid is not None')\nfrom rfx import Simulation\nSimulation._settling_source_end = settling.settling_source_end"),
    'occupancy_warns': ('check_release', "rewrite(builder, 'assemble_coax_port', '    root = coax_kernel_conductors(', '    if occupancy is not None:\\n        import jax.numpy as jnp\\n        occupancy = jnp.where(jnp.asarray(bore | cells), 0., occupancy)\\n    root = coax_kernel_conductors(')"),
    'plate_at_reference': ('check_plate', "rewrite(builder, '_refuse_front_pec', 'mask[:, :, ref] = False', 'mask[:, :, ref] = front[:, :, ref]')\nrewrite(builder, 'assemble_coax_port', 'for mask in edge_bore[:2]:', 'for mask in edge_bore:')"),
    'bore_report': ('check_report', "\nimport rfx.realized_geometry as unused\nimport importlib\ngeometry = importlib.import_module('rfx.realized_geometry')\nrewrite(geometry, '_ports', 'cleared_bore_cells=tuple(map(tuple, release.cells))', 'cleared_bore_cells=()')"),
    'forward_diagnostic_arrays': ('check_forward_report', "\nimport rfx.model.realization_report as reporter\nrewrite(reporter, 'rebuild_view', \"solved.lane in ('run_uniform', 'fwd_uniform')\", \"solved.lane == 'run_uniform'\")"),
}


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('mode', ['check_off', 'defect_restored'])
def test_review_mutation(case, mode):
    check, defect = CASES[case]
    witness = {'drude_named': 'drude_inside_bore_refused',
               'dispersive_outside_bore': 'drude_outside_bore_admitted'}.get(case, case)
    script = '''
import inspect
import textwrap
import pytest
import tests.contracts.test_coax_review_refusals as contract
import rfx.model.coax_admission as admission
import rfx.model.coax_port as builder

def rewrite(owner, name, old, new):
    fn = getattr(owner, name)
    source = textwrap.dedent(inspect.getsource(fn))
    assert old in source, (name, old)
    scope = dict(fn.__globals__)
    exec(source.replace(old, new), scope)
    setattr(owner, name, scope[name])
'''
    script += f'contract.{check} = lambda *args: None\n' if mode == 'check_off' else defect + '\n'
    script += f"raise SystemExit(pytest.main(['tests/contracts/test_coax_review_refusals.py', '-q', '-o', 'addopts=', '--tb=short', '-k', 'test_{witness}']))\n"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    expected = 0 if mode == 'check_off' else 1
    print(f'{case} {mode}: exit={result.returncode}\n{result.stdout}\n{result.stderr}', file=sys.stderr)
    assert result.returncode == expected
    assert ('passed' if expected == 0 else 'FAILED tests/contracts/test_coax_review_refusals.py::test_' + witness) in result.stdout


@pytest.mark.parametrize('case,module,selection', [
    ('measurement_text', 'tests.contracts.test_measurement_plan_review', 'test_run_calculator_owner_availability and coax'),
    ('jit_bins_text', 'tests.contracts.test_port_bins_traced_under_jit', 'test_port_bins and coax-graded'),
    ('replay_text', 'tests.unit.autodiff.test_ad_memory_replay_tables', 'test_coaxial_spectra_not_reachable_from_forward and graded'),
])
@pytest.mark.parametrize('mode', ['check_off', 'defect_restored'])
def test_refusal_text_mutation(case, module, selection, mode):
    script = f'''
import importlib
import types
import pytest
contract = importlib.import_module({module!r})
def replace_text(code, old, new):
    return code.replace(co_consts=tuple(replace_text(v, old, new) if isinstance(v, types.CodeType)
        else v.replace(old, new) if isinstance(v, str) else v for v in code.co_consts))
'''
    if case == 'jit_bins_text':
        script += "contract._refusal = lambda *args: None\n" if mode == 'check_off' else '''
original = contract._refusal
def removed_text(outcome, exception, leading):
    original(outcome, exception, 'fwd_nonuniform: add_coaxial_port() is supported only on run_uniform and fwd_uniform')
contract._refusal = removed_text
'''
    else:
        name = ('test_run_calculator_owner_availability' if case == 'measurement_text'
                else 'test_coaxial_spectra_not_reachable_from_forward')
        value = '.*' if mode == 'check_off' else 'supported only on run_uniform and fwd_uniform'
        fragments = (['y-lo absorber leaves.*needs at least 1 wall cell', 'add_coaxial_port.*is not carried']
                     if case == 'measurement_text' else ['add_coaxial_port.*is not carried'])
        for fragment in fragments:
            script += f"contract.{name}.__code__ = replace_text(contract.{name}.__code__, {fragment!r}, {value!r})\n"
    path = module.replace('.', '/') + '.py'
    script += f"raise SystemExit(pytest.main([{path!r}, '-q', '-o', 'addopts=', '--tb=short', '-k', {selection!r}]))\n"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=''))
    print(f'{case} {mode}: exit={result.returncode}\n{result.stdout}\n{result.stderr}', file=sys.stderr)
    assert result.returncode == (0 if mode == 'check_off' else 1)
    assert ('passed' if mode == 'check_off' else 'FAILED ' + path) in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", "-o", "addopts="]))
