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
                     'graded_accumulators: graded_setup._build_nu_scan; slab_ports.setup_ports'),
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


INVARIANT_RULES = {
    'ctx': ('padded', ''),
    'magnetic_factors': ('padded', ''),
    'h_row_factors': ('padded', ''),
    'cpml_inv_eps_r': ('padded', ''),
    'sheet_coeffs': ('padded', ''),
    'e_loss': ('padded', ''),
    'conformal_zero_masks': ('padded', ''),
    'probe_table': ('pass', ''),
    'port_table': ('pass', ''),
    'waveguide_table': ('pass', ''),
    'coax_table': ('pass', ''),
    'coax_freqs': ('pass', ''),
    'owns_x': ('unpadded path',
               'slab_setup.finish and slab_wrap.split; make_uniform_step refuses a padded slab step'),
    'cpml_x': ('unpadded path',
               'slab_setup.build_setup and slab_cpml.publish; make_uniform_step refuses a padded slab step'),
}


def _invariant_name(node):
    return isinstance(node, ast.Name) and node.id.endswith('invariants')


def _string(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def invariant_keys():
    found = set()
    setup = ast.parse((ROOT / 'rfx/stepping/uniform_setup.py').read_text())
    step = ast.parse((ROOT / 'rfx/stepping/uniform.py').read_text())
    for node in ast.walk(setup):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_invariant_name(t) for t in targets) and isinstance(node.value, ast.Dict):
                found.update(k.value for k in node.value.keys if _string(k))
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                and _invariant_name(node.value) and _string(node.slice)):
            found.add(node.slice.value)
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'update' and _invariant_name(node.func.value)):
            continue
        assert all(k.arg is not None for k in node.keywords)
        found.update(k.arg for k in node.keywords)
        for arg in node.args:
            if isinstance(arg, ast.DictComp):
                for generator in arg.generators:
                    assert isinstance(generator.iter, (ast.Tuple, ast.List))
                    for pair in generator.iter.elts:
                        assert isinstance(pair, (ast.Tuple, ast.List)) and _string(pair.elts[0])
                        found.add(pair.elts[0].value)
    for node in ast.walk(step):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and _invariant_name(node.value) and _string(node.slice)):
            found.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'get' and _invariant_name(node.func.value)
                and node.args and _string(node.args[0])):
            found.add(node.args[0].value)
        if (isinstance(node, ast.Compare) and _string(node.left)
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.In)
                and _invariant_name(node.comparators[0])):
            found.add(node.left.value)
    return found


def padded_invariant_keys():
    tree = ast.parse((ROOT / 'rfx/stepping/vector_pad.py').read_text())
    function, = (node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'pad_invariants')
    found = set()
    for node in ast.walk(function):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == 'key' and len(node.ops) == 1):
            continue
        other = node.comparators[0]
        if isinstance(node.ops[0], ast.Eq) and _string(other):
            found.add(other.value)
        elif isinstance(node.ops[0], ast.In) and isinstance(other, (ast.Tuple, ast.List)):
            assert all(_string(item) for item in other.elts)
            found.update(item.value for item in other.elts)
    return found


@pytest.mark.source_scan
def test_vector_pad_invariant_rules():
    found = invariant_keys()
    print(f'VECTOR PAD INVARIANT KEYS {sorted(found)}', file=sys.stderr)
    assert found == INVARIANT_RULES.keys(), (found - INVARIANT_RULES.keys(),
                                          INVARIANT_RULES.keys() - found)
    padded = padded_invariant_keys()
    for key, (rule, reason) in INVARIANT_RULES.items():
        assert rule in ('padded', 'pass', 'unpadded path')
        if rule == 'padded':
            assert key in padded, key
        elif rule == 'pass':
            assert key in vector_pad._INVARIANT_PASS, key
        else:
            assert reason
