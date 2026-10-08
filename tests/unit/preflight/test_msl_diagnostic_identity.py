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
    'msl.conductor_attachment', 'msl.line_stub_realization',
}


def path_observations():
    import jax
    records = {}
    for path in ('uniform', 'graded', 'distributed', 'distributed_nu'):
        sim = structure(path in ('graded', 'distributed_nu'))
        try:
            result = sim.run(
                n_steps=2,
                compute_s_params=False,
                **({"devices": jax.devices()} if path.startswith("distributed") else {}),
            )
            status = 'returns'
        except (ValueError, NotImplementedError) as error:
            result, status = error, 'refuses'
            assert any(d.severity == 'refusal' for d in error.diagnostics)
        records[path] = {'status': status, 'diagnostics': [d.to_dict() for d in result.diagnostics]}
    return records


def test_jd_path_identity():
    # The second CPU is needed to reach the real multi-device admission fence.
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        PYTHONPATH=str(ROOT),
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    code = ('import json; from tests.unit.preflight.test_msl_diagnostic_identity import path_observations; '
            'print("JD_RESULT=" + json.dumps(path_observations()))')
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
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


FAMILY_MODULES = (
    "rfx/preflight/msl.py",
    "rfx/preflight/msl_reflector.py",
    "rfx/preflight/line_stub.py",
    "rfx/preflight/line_port_coverage.py",
    "rfx/sparams/_common.py",
    "rfx/sources/msl_port.py",
    "rfx/preflight/_msl_probe_rules.py",
)


def test_jd_no_print_and_no_hardcoded_emission_bypass():
    from rfx.preflight.msl_codes import CODES, TEMPLATES
    # These are the only opaque messages: inspection exceptions or forwarded
    # placement advice. Another code must not use one as a prose escape hatch.
    opaque = {
        'probe_placement_note': 'msl.probe_placement_note',
        'line_stub_realization': 'msl.line_stub_realization',
        'probe_placement_failed': 'msl.probe_placement_failed',
        'probe_clearance_scan_failed': 'msl.probe_clearance_scan_failed',
        'conductor_attachment': 'msl.conductor_attachment',
    }
    from string import Formatter
    for key, template in TEMPLATES.items():
        fields = {field for _, field, _, _ in Formatter().parse(template)}
        if fields & {'detail', 'exc'}:
            assert key in opaque, key
    for name in FAMILY_MODULES:
        text = (ROOT / name).read_text()
        tree = ast.parse(text)
        if name in ('rfx/sparams/_common.py', 'rfx/sources/msl_port.py'):
            selected = {'_resolve_msl_auto_offsets', 'validate_msl_port_geometry'}
            tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id == 'print' for n in ast.walk(tree)), name
        else:
            assert 'print(' not in text
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def raw_text(node):
            """Text written at the site: a literal, an f-string, a join/format, or a sum of them."""
            if isinstance(node, ast.JoinedStr):
                return True
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return any(c.isalnum() for c in node.value)
            if isinstance(node, ast.BinOp):
                return raw_text(node.left) or raw_text(node.right)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'format':
                    return True
                # ', '.join(names) lists data; a separator with words in it is prose.
                if node.func.attr == 'join':
                    return raw_text(node.func.value)
            return False

        def function_of(node):
            while node is not None and not isinstance(node, ast.FunctionDef):
                node = parents.get(node)
            return getattr(node, 'name', None)

        # A site that stops calling the table is as much a bypass as one that
        # appends to it: no raise or warn in the family may carry text written
        # at the site. The one exception is an internal consistency error that
        # is caught and re-rendered by its caller.
        allowed_raw = {('rfx/preflight/msl.py', '_msl_declared_face_geometry')}
        for n in ast.walk(tree):
            site = None
            if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and n.exc.args:
                site = n.exc.args[0]
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'warn' and n.args):
                site = n.args[0]
            if site is not None and raw_text(site):
                assert (name, function_of(n)) in allowed_raw, (
                    f"{name}:{n.lineno} emits text written at the site; route it through the code table")
            # Messages collected in a list and emitted later (the placement notes) are
            # emission too: only table text and data labels may be collected.
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ('append', 'extend', 'insert')):
                for argument in n.args:
                    prose = (isinstance(argument, ast.JoinedStr) and any(
                        isinstance(part, ast.Constant) and len(str(part.value).split()) >= 2
                        for part in argument.values)) or (
                        isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                        and len(argument.value.split()) >= 2)
                    # The reflector scan's reasons for a conductor it could not evaluate are
                    # forwarded text (as on main; they reach the user inside
                    # msl.reflector_scan_incomplete). Nothing else may collect prose.
                    forwarded = (name == 'rfx/preflight/msl_reflector.py'
                                 and isinstance(n.func.value, ast.Name)
                                 and n.func.value.id == 'unevaluated')
                    assert not prose or forwarded, (
                        f"{name}:{n.lineno} collects text written at the site; use the code table")
        for n in ast.walk(tree):
            if isinstance(n, ast.BinOp):
                for table_side, other in ((n.left, n.right), (n.right, n.left)):
                    if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                           and x.func.id == 'msl_text' for x in ast.walk(table_side)):
                        assert not raw_text(other), (name, n.lineno)
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
                continue
            fn = n.func.id
            if fn in ('PreflightWarning', 'PreflightErrorWarning', 'PreflightConfigError'):
                # Other families in shared modules remain outside this migration.
                if name.startswith('rfx/preflight/'):
                    assert isinstance(n.args[0], ast.Call), (name, n.lineno)
                    assert isinstance(n.args[0].func, ast.Name)
                    assert n.args[0].func.id in ('msl_diagnostic', 'stub_diagnostic'), (name, n.lineno)
            if fn in ('msl_diagnostic', 'msl_geometry_error', 'placement_warning'):
                argument = n.args[1] if fn == 'msl_diagnostic' else n.args[0]
                assert isinstance(argument, (ast.Call, ast.Name, ast.BinOp)), (name, n.lineno)
                if isinstance(argument, ast.Call):
                    assert isinstance(argument.func, ast.Name) and argument.func.id == 'msl_text'
            if fn == 'msl_text':
                key = ast.literal_eval(n.args[0])
                outer = parents.get(n)
                while outer is not None:
                    if isinstance(outer, ast.Call) and isinstance(outer.func, ast.Name):
                        emitter = outer.func.id
                        code = (ast.literal_eval(outer.args[0]) if emitter == 'msl_diagnostic' else
                                {'msl_geometry_error': 'msl.conductor_attachment',
                                 'placement_warning': 'msl.probe_placement_note'}.get(emitter))
                        if code is not None:
                            assert key in CODES[code].templates, (name, n.lineno, code, key)
                            if key in opaque:
                                assert code == opaque[key], (name, n.lineno, code, key)
                            break
                    outer = parents.get(outer)


@pytest.mark.parametrize(
    "graded",
    [False, True],
    ids=["uniform", "graded"],
)
def test_byte_identity_and_every_code_reached(graded):
    from rfx.preflight.msl_codes import CODES
    assert {code: entry.legacy_slug for code, entry in CODES.items()} == EXPECTED_LEGACY
    assert {code for code, entry in CODES.items() if entry.severity == 'refusal'} == EXPECTED_REFUSALS
    from tests._msl_diagnostic_golden import GOLDEN as expected
    found = set()
    for case, messages in captured_cases(graded).items():
        key = ('graded' if graded else 'uniform') + '/' + case
        assert list(map(str, messages)) == expected[key], key
        for message in messages:
            found.add(message.diagnostic.code)
            assert message.diagnostic.message == str(message)
            assert EXPECTED_LEGACY[message.diagnostic.code] == message.code
            assert message.diagnostic.severity == (
                "refusal" if message.severity == "error" else "advisory")
    assert found == set(EXPECTED_LEGACY)


def test_two_simulations_alternating_do_not_retain_findings():
    first, second = (
        structure(
            name="first",
        ),
        structure(
            name="second",
        ),
    )
    for sim, name in [(first, 'first'), (second, 'second'), (first, 'first')]:
        result = sim.run(
            n_steps=2,
            compute_s_params=False,
        )
        family = [d for d in result.diagnostics if d.code.startswith('msl.')]
        assert {d.code for d in family} == EXPECTED
        assert {d.subject for d in family} == {name}
        clean = sim.run(
            n_steps=2,
            compute_s_params=False,
            skip_preflight=True,
        )
        assert clean.diagnostics == ()


@pytest.mark.parametrize(
    "graded",
    [False, True],
    ids=["uniform", "graded"],
)
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
