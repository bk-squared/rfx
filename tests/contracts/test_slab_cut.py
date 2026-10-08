"""One ownership invariant across quantity policies and placement forms."""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from rfx.stepping.slab import FILL, Slab, cut, cut_poles
from rfx.core.yee import EPS_0

TEST_DT = 1.7e-12


# Independently declared boundary conventions; a FILL edit must be reviewed here.
POLICIES = {
    'pole_mask': (False, False),
    'debye_ca': (1., 1.), 'debye_cb': ('dt/eps0', 'dt/eps0'),
    'debye_cc': ('1/eps0', '1/eps0'), 'debye_alpha': (0., 0.),
    'debye_beta': (0., 0.), 'debye_state': (0., 0.),
    'lorentz_ca': (1., 1.), 'lorentz_cb': ('dt/eps0', 'dt/eps0'),
    'lorentz_cc': ('1/eps0', '1/eps0'), 'lorentz_a': (0., 0.),
    'lorentz_b': (0., 0.), 'lorentz_c': (0., 0.), 'lorentz_state': (0., 0.),
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
                if value == 'dt/eps0':
                    value = np.float32(TEST_DT) / np.float32(EPS_0)
                elif value == '1/eps0':
                    value = np.float32(1.) / np.float32(EPS_0)
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
        result = cut(array, layout, kind, mesh=mesh, dt=TEST_DT)
    elif form == 'device':
        result = cut(jnp.asarray(array), layout, kind, mesh=mesh, dt=TEST_DT)
    elif form == 'traced':
        # The caller owns the enclosing JIT's result placement.
        result = jax.jit(lambda a: cut(a, layout, kind, mesh=mesh, dt=TEST_DT),
                         out_shardings=NamedSharding(mesh, P('x')))(jnp.asarray(array))
    else:
        # Caller-supplied alignment is deliberately wrong; cut must replace it.
        padded = np.pad(array, [(0, layout.pad_x)] + [(0, 0)] * (array.ndim - 1), constant_values=997.)
        placed = jax.device_put(padded, NamedSharding(mesh, P('x')))
        result = cut(placed, layout, kind, mesh=mesh, dt=TEST_DT)
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
    violations = _positive_placement_violations(source, path)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == '_split_lumped':
            violations.append((str(path), node.name, node.lineno, 'removed helper'))
        if node.name in RETIRED:
            # Import compatibility may retain only one return delegating to cut.
            body = [statement for statement in node.body if not (
                node.name in ('split_array_x', 'shard_x_slabs')
                and isinstance(statement, ast.If)
                and all(isinstance(child, ast.Raise) for child in statement.body)
                and not statement.orelse)]
            if len(body) != 1 or not isinstance(body[0], ast.Return) or not any(
                    isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'cut'
                    for call in ast.walk(body[0])):
                violations.append((str(path), node.name, node.lineno, 'retired implementation'))
        # Follow local names so renaming lo/hi or nx_per cannot hide a copy.
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
            if any(isinstance(child, ast.BinOp) and isinstance(child.op, ast.FloorDiv)
                   and any(isinstance(attr, ast.Attribute) and attr.attr == 'shape'
                           for attr in ast.walk(child)) for child in ast.walk(expression)):
                names.add('shape_divided')
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
            deps = dependencies(sub.slice)
            if ('shape_divided' in deps or (isinstance(sub.slice, ast.Slice) and deps & {
                    'n_devices', 'nx_per', 'nx_per_rank'})):
                violations.append((str(path), node.name, sub.lineno, 'slab slicing'))
    return violations


def test_retired_placements_do_not_reappear():
    root = Path(__file__).resolve().parents[2] / 'rfx'
    found = []
    for directory in ('runners', 'api', 'model'):
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
    actual = jax.grad(lambda a: jnp.sum(cut(a, layout, kind, mesh=mesh, dt=TEST_DT) * weights))(design)
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


# Exact calls and multiplicities, frozen here; a new placement fails the scan.
PLACEMENT_ALLOWLIST = {
    ('runners/_distributed_common.py', 'shard_stacked_psi', 'jax.device_put(merged, shd)'): (1, 'CPML face-depth packing remains absorber-owned (#1535).'),
    ('runners/_distributed_common.py', '_distributed_cpml_state', "jnp.zeros((n_devices * depth, d1, d2), dtype=jnp.float32, device=NamedSharding(mesh, P('x')))"): (1, 'CPML state allocation remains absorber-owned (#1535).'),
    ('runners/_distributed_ntff.py', 'initial', "jax.device_put(jnp.zeros((self.n_devices, self.capacity), self.dtype), NamedSharding(self.mesh, P('x')))"): (1, 'NTFF observer buffer; source/observer packing is step 3c.'),
    ('runners/_distributed_ntff.py', 'assemble', 'jax.device_put(buffer, NamedSharding(self.mesh, P()))'): (1, 'Replicated NTFF output assembly; gathering is step 3d.'),
    ('runners/distributed_nu.py', 'shard_cpml_state_x_slab._shard_psi', 'shard_stacked_psi(arr, shd)'): (1, 'CPML face-depth packing remains absorber-owned (#1535).'),
    ('runners/distributed_nu.py', '_concrete_on_mesh', 'jax.device_put(arr, sharding)'): (1, 'Replicated geometry, CPML profiles, source tables and step indices; callers use P().'),
    ('runners/distributed_nu.py', '_concrete_on_mesh', 'jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])'): (1, 'Replicated geometry, CPML profiles, source tables and step indices; callers use P().'),
    ('runners/distributed_nu.py', 'run_nonuniform_distributed_pec._zeros', 'jnp.zeros((n_devices * nx_local, ny, nz), dtype=jnp.float32, device=shd)'): (1, 'Direct zero field-carry allocation; no whole-domain slicing.'),
    ('runners/distributed_v2.py', '_shard_field_state', 'jax.device_put(state.ex, shd)'): (1, 'Compatibility placement of already laid-out field buffers and replicated step.'),
    ('runners/distributed_v2.py', '_shard_field_state', 'jax.device_put(state.ey, shd)'): (1, 'Compatibility placement of already laid-out field buffers and replicated step.'),
    ('runners/distributed_v2.py', '_shard_field_state', 'jax.device_put(state.ez, shd)'): (1, 'Compatibility placement of already laid-out field buffers and replicated step.'),
    ('runners/distributed_v2.py', '_shard_field_state', 'jax.device_put(state.hx, shd)'): (1, 'Compatibility placement of already laid-out field buffers and replicated step.'),
    ('runners/distributed_v2.py', '_shard_field_state', 'jax.device_put(state.hy, shd)'): (1, 'Compatibility placement of already laid-out field buffers and replicated step.'),
    ('runners/distributed_v2.py', '_shard_field_state', 'jax.device_put(state.hz, shd)'): (1, 'Compatibility placement of already laid-out field buffers and replicated step.'),
    ('runners/distributed_v2.py', '_shard_field_state', 'jax.device_put(state.step, _rep_sharding(mesh))'): (1, 'Compatibility placement of already laid-out field buffers and replicated step.'),
    ('runners/distributed_v2.py', '_shard_materials', 'jax.device_put(materials.eps_r, shd)'): (1, 'Compatibility placement of already laid-out material buffers.'),
    ('runners/distributed_v2.py', '_shard_materials', 'jax.device_put(materials.sigma, shd)'): (1, 'Compatibility placement of already laid-out material buffers.'),
    ('runners/distributed_v2.py', '_shard_materials', 'jax.device_put(materials.mu_r, shd)'): (1, 'Compatibility placement of already laid-out material buffers.'),
    ('runners/distributed_v2.py', '_init_cpml_sharded._shard_psi', 'shard_stacked_psi(arr, shd)'): (1, 'CPML face-depth packing remains absorber-owned (#1535).'),
    ('runners/distributed_v2.py', 'run_distributed', 'jnp.zeros(field_shape, dtype=jnp.float32, device=shd)'): (1, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jax.device_put(jnp.int32(0), rep)'): (1, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jnp.zeros((n_devices, 1, 1), dtype=jnp.float32, device=shd)'): (2, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jnp.zeros((n_devices, 1, 1, 1), dtype=jnp.float32, device=shd)'): (2, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jax.device_put(_dz, shd)'): (2, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jax.device_put(_dz5, shd)'): (6, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jax.device_put(_lz, shd)'): (3, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jax.device_put(_lz5, shd)'): (9, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jax.device_put(jnp.zeros((_total_x, 1, 1), dtype=jnp.float32), shd)'): (1, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
    ('runners/distributed_v2.py', 'run_distributed', 'jax.device_put(src_waveforms, rep)'): (1, 'Direct zero fields, unread absent-pole dummy carries, CPML dummy state, replicated step/source table.'),
}

def _positive_placement_violations(source, path):
    """Every placement must be slab-owned or an exact, counted exception.

    Check all device_put calls, including aliases whose sharding is stored in a
    variable. Replicated placements are recorded too; an opaque sharding
    variable does not hide a placement call from this inventory.
    """
    from collections import Counter
    tree = ast.parse(source)
    violations, seen, owners = [], Counter(), []
    aliases = {}
    array_splits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
                if node.module in ('numpy', 'jax.numpy') and alias.name in ('split', 'array_split'):
                    array_splits.add(alias.asname or alias.name)

    class Placements(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            owners.append(node.name)
            self.generic_visit(node)
            owners.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            raw = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            name = aliases.get(raw, raw)
            placement = (name in ('device_put', 'make_array_from_callback')
                         or name.startswith('shard_stacked')
                         or any(k.arg == 'device' for k in node.keywords))
            if placement:
                key = (str(path), '.'.join(owners), ast.unparse(node))
                seen[key] += 1
                count, reason = PLACEMENT_ALLOWLIST.get(key, (0, 'unlisted placement'))
                if seen[key] > count:
                    violations.append((str(path), '.'.join(owners), node.lineno, reason))
            if (raw in array_splits or (name in ('split', 'array_split')
                    and isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func.value) in ('np', 'jnp', 'numpy', 'jax.numpy'))):
                violations.append((str(path), '.'.join(owners), node.lineno, 'array split outside slab'))
            self.generic_visit(node)

    Placements().visit(tree)
    return violations


@pytest.mark.parametrize('source', [
    '''def copied(arr, n):
    width = arr.shape[0] // n
    return [arr[r * width:(r + 1) * width] for r in range(n)]
''',
    '''def copied(arr, n):
    width = arr.shape[0] // n
    return [arr[np.arange(r * width, (r + 1) * width)] for r in range(n)]
''',
    '''def copied(arr, mesh):
    def callback(index):
        lo, hi, step = index[0].indices(arr.shape[0])
        return arr[lo:hi:step]
    return jax.make_array_from_callback(arr.shape, NamedSharding(mesh, P("x")), callback)
''',
    '''def copied(arr, n):
    return jnp.split(arr, n, axis=0)
''',
], ids=['shape_width', 'arange_fancy', 'callback_indices', 'jnp_split'])
def test_scan_rejects_review_slicers(source):
    violations = _placement_violations(source, 'runners/review_mutation.py')
    assert violations
    print('MUTATION_RED', violations)


def test_scan_rejects_an_extra_call_at_an_allowed_site():
    key = next(k for k in PLACEMENT_ALLOWLIST if k[1] == '_concrete_on_mesh' and 'device_put' in k[2])
    path, owner, expression = key
    count, _ = PLACEMENT_ALLOWLIST[key]
    source = f'def {owner}(arr, sharding):\n' + ('    ' + expression + '\n') * (count + 1)
    assert _positive_placement_violations(source, path)


def test_scan_rejects_aliased_placement():
    assert _positive_placement_violations(
        'from jax import device_put as place\ndef copied(a, s):\n    return place(a, s)\n',
        'model/review_mutation.py')

@pytest.mark.parametrize('devices', [2, 3])
@pytest.mark.parametrize('kind', [k for k in POLICIES if k.startswith(('debye_', 'lorentz_'))])
@pytest.mark.parametrize('placed', [False, True])
def test_pole_pack_keeps_rank_pole_and_x_axes(devices, kind, placed):
    if len(jax.devices('cpu')) < devices:
        pytest.skip(f'requires {devices} CPU devices')
    layout = Slab(11, devices)
    mesh = Mesh(np.array(jax.devices('cpu')[:devices]), ('x',)) if placed else None
    array = np.arange(2 * 11 * 6, dtype=np.float32).reshape(2, 11, 2, 3)
    actual = np.asarray(cut_poles(jnp.asarray(array), layout, kind, mesh=mesh, dt=TEST_DT))
    actual = actual.reshape(devices, 2, layout.nx_local, 2, 3)
    for pole in range(2):
        _assert_rows(actual[:, pole], array[pole], layout, kind)


def test_retained_splitter_contracts():
    from rfx.runners._distributed_common import shard_x_slabs, split_array_x
    from rfx.runners.distributed_nu import split_1d_with_ghost
    with pytest.raises(ValueError, match='3-D'):
        shard_x_slabs(jnp.ones((8, 2)), 2, 4, 1, 0., None)
    with pytest.raises(ValueError, match='divisible'):
        split_array_x(jnp.ones((11, 2, 3)), 3)
    # Float64 remains host data: this does not enable JAX x64.
    for dtype in (np.int16, np.float32, np.float64):
        array = np.arange(8, dtype=dtype)
        actual = split_1d_with_ghost(array, 2, 4, 6, 1, -3)
        assert actual.dtype == array.dtype
        np.testing.assert_array_equal(actual, [[-3, 0, 1, 2, 3, 4], [3, 4, 5, 6, 7, -3]])


@pytest.mark.parametrize('devices', [2, 3])
@pytest.mark.parametrize('axis', [0, 1])
def test_vmap_cut_places_logical_x_and_preserves_transpose(devices, axis):
    if len(jax.devices('cpu')) < devices:
        pytest.skip(f'requires {devices} CPU devices')
    layout = Slab(11, devices)
    mesh = Mesh(np.array(jax.devices('cpu')[:devices]), ('x',))
    batch = jnp.arange(2 * 11 * 6, dtype=jnp.float32).reshape(2, 11, 2, 3)
    inputs = jnp.moveaxis(batch, 0, axis)
    f = lambda a: cut(a, layout, 'eps_r', mesh=mesh)
    actual = jax.vmap(f, in_axes=axis)(inputs)
    assert tuple(actual.sharding.spec)[:2] == (None, 'x')
    expected = jnp.stack([f(a) for a in batch])
    np.testing.assert_array_equal(actual, expected)
    batched_grad = jax.grad(lambda b: jnp.sum(jax.vmap(f, in_axes=axis)(b)**2))(inputs)
    single_grads = jnp.stack([jax.grad(lambda a: jnp.sum(f(a)**2))(a) for a in batch])
    np.testing.assert_array_equal(batched_grad, jnp.moveaxis(single_grads, 0, axis))
