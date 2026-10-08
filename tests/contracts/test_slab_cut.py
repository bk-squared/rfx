"""One ownership invariant across quantity policies and placement forms."""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from rfx.stepping.slab import FILL, Slab, cut


# Independently declared boundary conventions; a FILL edit must be reviewed here.
POLICIES = {
    'eps_r': (1., 1.), 'mu_r': (1., 1.), 'sigma': (0., 0.),
    'lumped': (0., 0.), 'state': (0., 0.),
    'pec_edge': (True, False), 'pec_cell': (True, False),
    'pec_occupancy': (0., 0.), 'inv_dx': (0., 1.),
    'inv_dx_h': (0., 0.), 'width': ('edge', 'edge'),
}


def _assert_rows(actual, array, layout, kind):
    """Check each row by global index, without using cut's slicing helpers."""
    actual = np.asarray(actual).reshape(layout.n_devices, layout.nx_local, *array.shape[1:])
    assert actual.dtype == array.dtype
    alignment, physical = POLICIES[kind]
    for rank in range(layout.n_devices):
        for local in range(layout.nx_local):
            index = rank * layout.nx_per_rank + local - layout.ghost_width
            if 0 <= index < layout.nx:
                expected = array[index]
            else:
                value = alignment if layout.nx <= index < layout.nx_padded else physical
                expected = array[0 if index < 0 else -1] if value == 'edge' else np.full(array.shape[1:], value, dtype=array.dtype)
            np.testing.assert_array_equal(actual[rank, local], expected,
                err_msg=f'{kind}: rank={rank} local={local} global={index}')


def test_fill_has_exact_declared_policies():
    assert {kind: tuple(fill) for kind, fill in FILL.items()} == POLICIES


def test_closed_over_concrete_array_can_be_staged_during_a_trace():
    layout = Slab(11, 2)
    mesh = Mesh(np.array(jax.devices('cpu')[:2]), ('x',))
    array = np.arange(66, dtype=np.float32).reshape(11, 2, 3)
    concrete = jnp.asarray(array)
    result = jax.jit(lambda: cut(concrete, layout, 'eps_r', mesh=mesh))()
    _assert_rows(result, array, layout, 'eps_r')


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('nx', [11, 12])
@pytest.mark.parametrize('kind', tuple(POLICIES))
@pytest.mark.parametrize('form', ['host', 'device', 'traced', 'sharded'])
def test_owned_alignment_and_ghost_rows(n_devices, nx, kind, form):
    if len(jax.devices('cpu')) < n_devices:
        pytest.skip(f'requires {n_devices} CPU devices')
    layout = Slab(nx, n_devices)
    mesh = Mesh(np.array(jax.devices('cpu')[:n_devices]), ('x',))
    shape = (nx,) if kind in ('inv_dx', 'inv_dx_h', 'width') else (nx, 2, 3)
    array = (10 + np.arange(np.prod(shape), dtype=np.float32)).reshape(shape)
    assert layout.owned == tuple((r * layout.nx_per_rank, min((r + 1) * layout.nx_per_rank, nx)) for r in range(n_devices))
    assert layout.faces == tuple((r == 0, r == n_devices - 1) for r in range(n_devices))
    if form == 'host':
        result = cut(array, layout, kind, mesh=mesh)
    elif form == 'device':
        result = cut(jnp.asarray(array), layout, kind, mesh=mesh)
    elif form == 'traced':
        # The caller owns the enclosing JIT's result placement.
        result = jax.jit(lambda a: cut(a, layout, kind, mesh=mesh),
                         out_shardings=NamedSharding(mesh, P('x')))(jnp.asarray(array))
    else:
        # Caller-supplied alignment is deliberately wrong; cut must replace it.
        padded = np.pad(array, [(0, layout.pad_x)] + [(0, 0)] * (array.ndim - 1), constant_values=997.)
        placed = jax.device_put(padded, NamedSharding(mesh, P('x')))
        result = cut(placed, layout, kind, mesh=mesh)
    assert result.sharding == NamedSharding(mesh, P('x'))
    _assert_rows(result, array, layout, kind)


@pytest.mark.parametrize('kind', ['pec_edge', 'lumped'])
def test_optional_component_leaves(kind):
    layout = Slab(11, 2)
    dtype = bool if kind == 'pec_edge' else np.float32
    array = np.arange(11 * 6).reshape(11, 2, 3).astype(dtype)
    result = cut((array, None, array), layout, kind, mesh=None)
    assert result[1] is None and cut(None, layout, kind, mesh=None) is None
    _assert_rows(result[0], array, layout, kind)
    _assert_rows(result[2], array, layout, kind)


@pytest.mark.parametrize('location', ['owned', 'interior_ghost', 'physical_ghost', 'alignment'])
def test_comparison_rejects_corrupted_rows(location):
    layout = Slab(11, 3)
    array = np.arange(11 * 6, dtype=np.float32).reshape(11, 2, 3)
    rows = np.asarray(cut(array, layout, 'eps_r', mesh=None)).copy()
    index = {'owned': (1, 2), 'interior_ghost': (1, 0),
             'physical_ghost': (0, 0), 'alignment': (2, 4)}[location]
    rows[index] += 17
    with pytest.raises(AssertionError):
        _assert_rows(rows, array, layout, 'eps_r')


RETIRED = {
    'split_array_x', 'shard_x_slabs', 'stage_concrete_forward_array',
    'stage_forward_array_x_slab', 'stage_sharded_forward_override',
    'shard_pec_mask_x_slab', 'shard_pec_occupancy_x_slab', 'split_1d_with_ghost',
}


def _placement_violations(source, path):
    violations = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == '_split_lumped':
            violations.append((str(path), node.name, node.lineno, 'removed helper'))
        if node.name in RETIRED:
            # Import compatibility may retain only one return delegating to cut.
            body = node.body
            if len(body) != 1 or not isinstance(body[0], ast.Return) or not any(
                    isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'cut'
                    for call in ast.walk(body[0])):
                violations.append((str(path), node.name, node.lineno, 'retired implementation'))
        # Follow local names so renaming lo/hi or nx_per cannot hide a copy.
        if node.name.startswith(('_split_debye', '_split_lorentz', 'split_poles')):
            continue  # Pole packing is step 3b.
        if node.name in ('_apply_cpml_e_distributed', '_apply_cpml_h_distributed'):
            continue  # CPML face views inside updates are outside spatial staging.
        assigned = {}
        def assign(target, value):
            if isinstance(target, ast.Name):
                assigned[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
                for name, expression in zip(target.elts, value.elts):
                    assign(name, expression)
        for assignment in ast.walk(node):
            if isinstance(assignment, ast.Assign):
                for target in assignment.targets:
                    assign(target, assignment.value)
        def dependencies(expression, seen=frozenset()):
            names = set()
            for child in ast.walk(expression):
                if isinstance(child, ast.Attribute):
                    names.add(child.attr)
                if isinstance(child, ast.Name):
                    names.add(child.id)
                    if child.id in assigned and child.id not in seen:
                        names.update(dependencies(assigned[child.id], seen | {child.id}))
            return names
        for sub in (child for child in ast.walk(node) if isinstance(child, ast.Subscript)):
            if (isinstance(sub.value, ast.Call) and isinstance(sub.value.func, ast.Attribute)
                    and sub.value.func.attr in ('devices', 'local_devices')):
                continue  # Selecting devices is not slicing a spatial array.
            if isinstance(sub.slice, ast.Slice) and dependencies(sub.slice) & {
                    'n_devices', 'nx_per', 'nx_per_rank'}:
                violations.append((str(path), node.name, sub.lineno, 'slab slicing'))
    return violations


def test_retired_placements_do_not_reappear():
    root = Path(__file__).resolve().parents[2] / 'rfx'
    found = []
    for directory in ('runners', 'api'):
        for path in sorted((root / directory).rglob('*.py')):
            found.extend(_placement_violations(path.read_text(), path.relative_to(root)))
    assert not found, found


def test_scan_rejects_reintroduced_splitter():
    source = '''def split_array_x(arr, n_devices):
    slabs = []
    for rank in range(n_devices):
        width = arr.shape[0] // n_devices
        x_start = rank * width
        x_end = (rank + 1) * width
        slabs.append(arr[x_start:x_end])
    return slabs
'''
    assert _placement_violations(source, 'runners/reintroduced.py')
    renamed = source.replace('split_array_x', 'new_placement').replace(
        'x_start = rank * width\n        x_end = (rank + 1) * width',
        'x_start, x_end = rank * width, (rank + 1) * width')
    assert _placement_violations(renamed, 'api/reintroduced.py')
    assert _placement_violations('def _split_lumped(arr):\n    return arr\n',
                                 'runners/reintroduced.py')


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('nx', [11, 12])
@pytest.mark.parametrize('kind', ['eps_r', 'sigma', 'pec_occupancy'])
def test_sharded_transpose_returns_neighbour_cotangents(n_devices, nx, kind):
    if len(jax.devices('cpu')) < n_devices:
        pytest.skip(f'requires {n_devices} CPU devices')
    layout = Slab(nx, n_devices)
    mesh = Mesh(np.array(jax.devices('cpu')[:n_devices]), ('x',))
    host = np.arange(nx * 6, dtype=np.float32).reshape(nx, 2, 3)
    padded = np.pad(host, ((0, layout.pad_x), (0, 0), (0, 0)), constant_values=997.)
    design = jax.device_put(padded, NamedSharding(mesh, P('x')))
    weights = np.arange(n_devices * layout.nx_local * 6, dtype=np.float32).reshape(-1, 2, 3)
    actual = jax.grad(lambda a: jnp.sum(cut(a, layout, kind, mesh=mesh) * weights))(design)
    expected = np.zeros_like(padded)
    for rank in range(n_devices):
        for local in range(layout.nx_local):
            index = rank * layout.nx_per_rank + local - 1
            if 0 <= index < nx:
                expected[index] += weights[rank * layout.nx_local + local]
    np.testing.assert_array_equal(actual, expected)


def test_named_material_record_policies():
    from rfx.core.yee import MaterialArrays
    layout = Slab(11, 3)
    array = np.arange(11 * 6, dtype=np.float32).reshape(11, 2, 3)
    materials = MaterialArrays(array, array, array, (array, None, array))
    kinds = dict(eps_r='eps_r', sigma='sigma', mu_r='mu_r',
                 sigma_lumped='lumped', eps_r_lumped='lumped')
    result = cut(materials, layout, kinds, mesh=None)
    assert isinstance(result, MaterialArrays)
    for name in ('eps_r', 'sigma', 'mu_r'):
        _assert_rows(getattr(result, name), array, layout, kinds[name])
    _assert_rows(result.sigma_lumped[0], array, layout, 'lumped')
    assert result.sigma_lumped[1] is result.eps_r_lumped is result.mu_r_wire is result.components is None


def test_device_selection_is_not_spatial_placement():
    assert not _placement_violations(
        'def select(n_devices):\n    return jax.devices()[:n_devices]\n', 'runners/device_selection.py')
