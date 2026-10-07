"""J-D: same family observations across paths, and no local message bypass."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests._msl_diagnostic_cases import captured_cases, structure


ROOT = Path(__file__).resolve().parents[3]
EXPECTED = {
    'msl.lateral_clearance', 'msl.normal_resolution',
    'msl.source_absorber_clearance', 'msl.source_near_field',
}


EXPECTED_LEGACY = {
    'msl.' + name: 'msl_port_geometry' for name in (
        'probe_clearance_scan_failed', 'probe_clearance_unavailable',
        'reflector_clearance', 'probe_placement_note', 'probe_placement_failed',
        'lateral_clearance', 'normal_resolution', 'trace_face_alignment',
        'conductor_gap_mismatch', 'source_absorber_clearance', 'probe_ladder_clamped',
        'reflector_scan_unavailable', 'reflector_scan_incomplete',
        'probe_in_absorber', 'probe_near_absorber', 'probe_crosses_feed', 'source_near_field',
    )
}
EXPECTED_LEGACY.update({
    'msl.conductor_assembly_unavailable': 'msl_port_conductor_planes',
    'msl.conductor_attachment': 'msl_port_conductor_planes',
    'msl.line_stub_realization': 'line_stub_realization',
    'msl.line_stub_inspection_unavailable': 'line_stub_inspection_unavailable',
    'msl.line_stub_behind_port': 'line_stub_behind_port',
})
EXPECTED_REFUSALS = {
    'msl.probe_placement_failed', 'msl.conductor_assembly_unavailable',
    'msl.conductor_attachment', 'msl.line_stub_realization', 'msl.line_stub_behind_port',
}


def path_observations():
    import jax
    records = {}
    for path in ('uniform', 'graded', 'distributed', 'distributed_nu'):
        sim = structure(path in ('graded', 'distributed_nu'))
        try:
            result = sim.run(n_steps=2, compute_s_params=False,
                             **({'devices': jax.devices()} if path.startswith('distributed') else {}))
            status = 'returns'
        except (ValueError, NotImplementedError) as error:
            result, status = error, 'refuses'
            assert any(d.severity == 'refusal' for d in error.diagnostics)
        records[path] = {'status': status, 'diagnostics': [d.to_dict() for d in result.diagnostics]}
    return records


def test_jd_path_identity():
    # The second CPU is needed to reach the real multi-device admission fence.
    env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=str(ROOT),
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    code = ('import json; from tests.unit.preflight.test_msl_diagnostic_identity import path_observations; '
            'print("JD_RESULT=" + json.dumps(path_observations()))')
    process = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    observations = json.loads(next(line.removeprefix('JD_RESULT=')
                                   for line in process.stdout.splitlines() if line.startswith('JD_RESULT=')))
    signatures = {}
    for path, result in observations.items():
        family = [d for d in result['diagnostics'] if d['code'].startswith('msl.')]
        assert {d['code'] for d in family} == EXPECTED
        assert all(d['path'] is None for d in family)  # Common checks precede dispatch.
        signatures[path] = sorted((d['code'], d['subject'], d['message']) for d in family)
        assert result['status'] == ('refuses' if path.startswith('distributed') else 'returns')
    # This aligned, constant-profile structure needs none of the listed wording variants.
    # A path-specific wording mutation must fail here even if its code survives.
    assert all(signature == signatures['uniform'] for signature in signatures.values())


def test_jd_no_print_and_no_hardcoded_emission_bypass():
    files = ('msl', 'msl_reflector', 'line_stub', 'line_port_coverage')
    for name in files:
        text = (ROOT / 'rfx/preflight' / f'{name}.py').read_text()
        assert 'print(' not in text
        for n in ast.walk(ast.parse(text)):
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
                continue
            if n.func.id in ('PreflightWarning', 'PreflightErrorWarning', 'PreflightConfigError'):
                assert isinstance(n.args[0], ast.Call), (name, n.lineno)
                assert isinstance(n.args[0].func, ast.Name)
                assert n.args[0].func.id in ('msl_diagnostic', 'stub_diagnostic'), (name, n.lineno)
            if n.func.id == 'msl_diagnostic':
                argument = n.args[1]
                assert isinstance(argument, (ast.Call, ast.Name, ast.BinOp)), (name, n.lineno)
                # No retained lookup followed by a caller-authored message substitution.
                if isinstance(argument, ast.Call):
                    assert isinstance(argument.func, ast.Name) and argument.func.id == 'msl_text'


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_byte_identity_and_every_code_reached(graded):
    from rfx.preflight.msl_codes import CODES
    assert {code: entry.legacy_slug for code, entry in CODES.items()} == EXPECTED_LEGACY
    assert {code for code, entry in CODES.items() if entry.severity == 'refusal'} == EXPECTED_REFUSALS
    expected = json.loads((ROOT / 'tests/data/msl_diagnostic_messages.json').read_text())
    found = set()
    for case, messages in captured_cases(graded).items():
        key = ('graded' if graded else 'uniform') + '/' + case
        assert list(map(str, messages)) == expected[key], key
        for message in messages:
            found.add(message.diagnostic.code)
            assert message.diagnostic.message == str(message)
            assert EXPECTED_LEGACY[message.diagnostic.code] == message.code
            assert message.diagnostic.severity == (
                "refusal" if message.diagnostic.code in EXPECTED_REFUSALS else "advisory")
    assert found == set(EXPECTED_LEGACY)


def test_two_simulations_alternating_do_not_retain_findings():
    first, second = structure(name='first'), structure(name='second')
    for sim, name in [(first, 'first'), (second, 'second'), (first, 'first')]:
        result = sim.run(n_steps=2, compute_s_params=False)
        family = [d for d in result.diagnostics if d.code.startswith('msl.')]
        assert {d.code for d in family} == EXPECTED
        assert {d.subject for d in family} == {name}
        clean = sim.run(n_steps=2, compute_s_params=False, skip_preflight=True)
        assert clean.diagnostics == ()


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_every_family_warning_construction_executes(graded, record_property):
    targets = set()
    for name in ('msl', 'msl_reflector', 'line_stub', 'line_port_coverage'):
        path = ROOT / 'rfx/preflight' / (name + '.py')
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ('PreflightWarning', 'PreflightErrorWarning')):
                targets.add((str(path), node.lineno))
    assert len(targets) == 23
    files = {path for path, _ in targets}
    observed = set()
    def trace(frame, event, arg):
        if frame.f_code.co_filename not in files:
            return None
        if event == 'line':
            observed.add((frame.f_code.co_filename, frame.f_lineno))
        return trace
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        captured_cases(graded)
    finally:
        sys.settrace(previous)
    assert targets <= observed, sorted(targets - observed)
    record_property('family_emit_sites', json.dumps(sorted(targets)))
