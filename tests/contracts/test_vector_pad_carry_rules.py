"""Every carry entry written by the padded modules has an explicit rule."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.stepping import vector_pad


ROOT = Path(__file__).resolve().parents[2]
RULES = {
    'fdtd': ('padded', ''),
    'cpml': ('padded', ''),
    'debye': ('padded', ''),
    'lorentz': ('padded', ''),
    'ntff': ('pass', ''),
    'current_moments': ('pass', ''),
    'dft_planes': ('pass', ''),
    'flux_monitors': ('pass', ''),
    'waveguide_port_accs': ('pass', ''),
    'wire_sparam_accs': ('pass', ''),
    'lumped_sparam_accs': ('pass', ''),
    'wire_refplane_accs': ('pass', ''),
    'rlc_states': ('pass', ''),
    'dft_time_records': ('pass', ''),
    'sparam_time_records': ('pass', ''),
    'coax_sparam_accs': ('pass', ''),
    'coax_time_records': ('pass', ''),
    'tfsf': ('refused', 'TFSF plane-wave source'),
    'wire_sparams': ('unpadded path',
                     'graded_accumulators: graded_setup.build_graded_setup; slab_ports.setup_ports'),
}


def carry_keys():
    paths = [ROOT / 'rfx/stepping' / name for name in
             ('uniform.py', 'uniform_loop.py', 'vmap_scan.py')]
    paths += sorted((ROOT / 'rfx/measurement').rglob('*.py'))
    trees = [ast.parse(p.read_text()) for p in paths]
    functions = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(node.name, []).append(node)
    found = set()
    for tree in trees:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name) and node.value.id.endswith('carry')
                    and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
                found.add(node.slice.value)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if (any(isinstance(t, ast.Name) and t.id.endswith('carry') for t in targets)
                        and isinstance(node.value, ast.Dict)):
                    found.update(k.value for k in node.value.keys
                                 if isinstance(k, ast.Constant) and isinstance(k.value, str))
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'update' and isinstance(node.func.value, ast.Name)
                    and node.func.value.id.endswith('carry')):
                continue
            for arg in node.args:
                if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)):
                    continue
                definitions = functions.get(arg.func.id, ())
                assert definitions, f'no source for carry update {arg.func.id}'
                for function in definitions:
                    for returned in ast.walk(function):
                        if (isinstance(returned, ast.Return) and isinstance(returned.value, ast.Call)
                                and isinstance(returned.value.func, ast.Name)
                                and returned.value.func.id == 'dict'):
                            assert all(k.arg is not None for k in returned.value.keywords)
                            found.update(k.arg for k in returned.value.keywords)
    return found


@pytest.mark.source_scan
def test_vector_pad_carry_rules():
    found = carry_keys()
    print(f'VECTOR PAD CARRY KEYS {sorted(found)}', file=sys.stderr)
    assert found == RULES.keys(), (found - RULES.keys(), RULES.keys() - found)
    flags = dict(use_upml=False, use_tfsf=True, bloch=None, use_waveguide_ports=False,
                 use_sheet_impedance=False, use_design_occupancy=False,
                 materials=SimpleNamespace(components=SimpleNamespace(tensor=None)))
    refused = vector_pad.uniform_unsupported(flags, stencil_order=2, field_dtype=np.float32)
    for key, (rule, reason) in RULES.items():
        assert rule in ('padded', 'pass', 'refused', 'unpadded path')
        if rule == 'pass':
            assert key in vector_pad._CARRY_PASS, key
        elif rule == 'refused':
            assert reason in refused, (key, reason, refused)
        elif rule == 'padded':
            assert key in ('fdtd', 'cpml', 'debye', 'lorentz')
        else:
            assert reason


from tests.contracts.coax_port_mutations import VECTOR_PAD_MUTATIONS, run_vector_pad_mutation


@pytest.mark.parametrize('mutation', VECTOR_PAD_MUTATIONS, ids=lambda row: row[0])
def test_vector_pad_mutations(mutation):
    run_vector_pad_mutation(mutation)
