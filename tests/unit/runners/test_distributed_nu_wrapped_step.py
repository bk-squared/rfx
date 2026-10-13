"""W1 slice 1: model identity and cross-trace bars for the graded slab step."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from rfx import Simulation
from rfx.core.yee import EPS_0, MU_0, MaterialArrays
from tests.contracts.path_equivalence.test_one_vs_several_devices_occupancy import BINS_HZ
from rfx.model.electric_metrics import stage_forward_materials
from rfx.nonuniform import _build_nu_scan, make_current_source, run_nonuniform
from rfx.runners.distributed_nu import build_sharded_nu_grid, run_nonuniform_distributed_pec
from rfx.simulation import SourceSpec, ProbeSpec
from rfx.stepping.slab import cut
from rfx.stepping.slab_wrap import select_step

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def scene(row='0.2', on=False, steps=12, kind='field'):
    sim = Simulation(freq_max=15e9, domain=(.022, .016, .012), dx=.001,
        boundary='pec', dy_profile=np.array([.001] + [.0009, .0011] * 7 + [.001]))
    grid = sim._build_nonuniform_grid()
    shape = grid.shape
    assert shape == (23, 17, 13)
    mat = MaterialArrays(jnp.ones(shape), jnp.zeros(shape), jnp.ones(shape))
    occ = None
    if row != 'none':
        if row == 'random':
            cells = np.random.default_rng(202).uniform(0, .85, shape).astype(np.float32)
            cells[0, 2:5, 1:4] = .5
            width = -(-shape[0] // 2)
            for seam in range(width, shape[0], width):
                cells[seam-1:seam+1, 1:4, 2:4] = .75
        else:
            cells = np.zeros(shape, np.float32)
            cells[5:7, 6:8, 4:6] = float(row)
        occ = jnp.asarray(cells)
    node = (6 if on else 3, 7, 5)
    sim.add_source((node[0] * .001, .007, .005), 'ez', amplitude_kind=kind)
    source = make_current_source(grid, node, 'ez',
        sim._ports[0].waveform, steps, mat, amplitude_kind=kind)
    probes = [(10, 7, 5, 'ez'), (14, 9, 6, 'ez'), (*node, 'ez'),
              (7, 7, 5, 'hx'), (7, 7, 5, 'hy'), (21, 7, 5, 'ez')]
    return grid, mat, occ, [source], probes


def single(data, steps):
    grid, mat, occ, sources, probes = data
    result = run_nonuniform(grid, mat, steps, cpml_axes='', pec_occupancy=occ,
        sources=sources, probes=probes, pec_faces={f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi')})
    return result['time_series'], result['state']


def several(data, steps, devices, *, edge_masks=None, **kw):
    grid, mat, occ, sources, probes = data
    selected = jax.devices('cpu')[:devices]
    mesh = Mesh(np.asarray(selected), ('x',))
    sg = build_sharded_nu_grid(grid, devices)
    result = run_nonuniform_distributed_pec(sg, stage_forward_materials(mat, sg.layout, mesh),
        cut(edge_masks, sg.layout, 'pec_edge', mesh=mesh), steps, sources=[SourceSpec(*s) for s in sources], probes=[ProbeSpec(*p) for p in probes],
        n_devices=devices, devices=selected,
        sharded_pec_occupancy=cut(occ, sg.layout, 'pec_occupancy', mesh=mesh), **kw)
    return result['time_series'], result['final_state']


def difference(want, got):
    for name, a, b in [('probes', want[0], got[0])] + [
            (f, getattr(want[1], f), getattr(got[1], f)) for f in FIELDS]:
        a, b = np.asarray(a), np.asarray(b)
        unequal = a.view(np.uint32) != b.view(np.uint32)
        if np.any(unequal):
            cell = tuple(int(i) for i in np.argwhere(unequal)[0])
            return f'{name}{cell}: single={a[cell]!r}, slab={b[cell]!r}'
    return None


def assert_bits(want, got):
    assert difference(want, got) is None, difference(want, got)


def assert_cross_trace_bars(want, got, dt):
    """Reference scales and DFT values come only from the one-device probes."""
    one, several = np.asarray(want[0]), np.asarray(got[0])
    assert one.dtype == several.dtype == np.float32
    assert one.shape == several.shape
    assert np.isfinite(one).all() and np.isfinite(several).all()
    e_columns, h_columns = (0, 1, 2, 5), (3, 4)
    e_peak = np.float32(np.max(np.abs(one[:, e_columns])))
    h_peak = np.float32(np.max(np.abs(one[:, h_columns])))
    z0 = float(np.sqrt(MU_0 / EPS_0))
    h_scale = np.float32(max(float(h_peak), float(e_peak) / z0))
    delta = np.abs(one.astype(np.float64) - several)
    e_units = np.max(delta[:, e_columns]) / float(np.spacing(e_peak))
    h_units = np.max(delta[:, h_columns]) / float(np.spacing(h_scale))
    assert e_units <= 9, f'E {e_units} float32 units > 9'
    assert h_units <= 9, f'H {h_units} paired float32 units > 9'
    kernel = np.exp(-2j * np.pi * np.outer(BINS_HZ, np.arange(len(one)) * dt))
    x_one = kernel @ one.astype(np.float64)
    x_several = kernel @ several.astype(np.float64)
    dft_error = np.max(np.abs(x_several - x_one))
    dft_peak = np.max(np.abs(x_one))
    assert dft_error <= 1e-4 * dft_peak, f'DFT {dft_error} > 1e-4 * {dft_peak}'


def canonical_setup(data, steps, *, edge_masks=None):
    grid, materials, occupancy, sources, probes = data
    return _build_nu_scan(grid, materials, steps, cpml_axes='',
        pec_occupancy=occupancy, pec_edge_masks=edge_masks,
        sources=sources, probes=probes,
        pec_faces={f'{a}_{side}' for a in 'xyz' for side in ('lo', 'hi')})


def assert_same_tree(expected, actual, label, *, owned=None):
    wanted, tree = jax.tree_util.tree_flatten_with_path(expected)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    assert actual_tree == tree, label
    for (path, a), b in zip(wanted, actual_leaves):
        a, b = np.asarray(a), np.asarray(b)
        if owned is not None and a.ndim:
            b = b[owned]
        name = label + jax.tree_util.keystr(path)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name


def check_one_slab_cut(expected, actual, layout):
    """Compare actual runner payload to independently built single-device inputs."""
    owned = slice(layout.ghost_width, layout.ghost_width + layout.nx)
    inv = expected.invariants
    for name in inv:
        if name in ('drives', 'probe_table'):
            continue
        assert_same_tree(inv[name], actual[name], name, owned=owned)
    drives = jax.tree.map(lambda a: a[0], actual['drives']._replace(waves=None))
    nodes = {c: (indices[0] - layout.ghost_width, *indices[1:])
             for c, indices in drives.nodes.items()}
    assert_same_tree(inv['drives'], drives._replace(nodes=nodes), 'drives')
    assert_same_tree(expected.src_waveforms, actual['drives'].waves[:, 0, :], 'waveforms')
    terms, order, ids = jax.tree.map(lambda a: a[0], actual['probe_table'])
    for reference, (indices, channels, weights) in zip(inv['probe_table'].components, terms):
        local_indices = (indices[0] - layout.ghost_width, *indices[1:])
        assert_same_tree((reference.indices, reference.channel_ids, reference.weights),
                         (local_indices, channels, weights), 'probe_table.' + reference.component)
    assert_same_tree((inv['probe_table'].order, inv['probe_table'].segment_ids),
                     (order, ids), 'probe_table.order_and_segments')
    assert all(np.asarray(value).tolist() == [True] for value in actual['owns_x'].values())
    print('CUT_INVARIANTS: all owned rows bit-identical: drives, waveforms, probe_table, '
          'materials (including component materials and edge_keep), masks, occupancy, '
          'inv_dx, inv_dx_h, dx_arr; owns_x=[True, True]')


def observe_one_slab_cut(monkeypatch, data, steps, *, edge_masks=None, stop=False):
    from rfx.runners import _distributed_nu_wrapped as runner
    expected = canonical_setup(data, steps, edge_masks=edge_masks)
    original = runner.split

    def checked(*args, **kwargs):
        actual, carry = original(*args, **kwargs)
        check_one_slab_cut(expected, actual, args[2])
        layout = args[2]
        assert_same_tree(expected.carry_init, carry, 'carry',
                         owned=slice(layout.ghost_width, layout.ghost_width + layout.nx))
        if stop:
            raise BeforeStepping
        return actual, carry

    monkeypatch.setattr(runner, 'split', checked)


class BeforeStepping(Exception):
    """Test-only exit after placement; no step was executed."""


@pytest.mark.parametrize('row', ['none', '0.2', 'random'])
@pytest.mark.parametrize('kind', ['field', 'current'])
def test_one_slab_invariants(row, kind, monkeypatch):
    data = scene(row, True, kind=kind)
    grid, mat, occ, sources, probes = data
    # Nonconstant component materials, actual hard masks, and interleaved
    # components make this more than a vacuum/None-only placement check.
    mat = mat._replace(eps_r=mat.eps_r.at[4:10, 3:9, 2:8].set(2.3),
                       sigma=mat.sigma.at[8:12, 2:6, 3:7].set(.02),
                       mu_r=mat.mu_r.at[2:7, 5:9, 4:7].set(1.7))
    sources = sources + [(14, 6, 5, 'ex', sources[0][4]),
                         (9, 6, 5, 'ez', sources[0][4])]
    masks = tuple(jnp.zeros(grid.shape, bool).at[18:20, 2:4, 2:4].set(True) for _ in range(3))
    data = grid, mat, occ, sources, probes
    observe_one_slab_cut(monkeypatch, data, 12, edge_masks=masks, stop=True)
    with pytest.raises(BeforeStepping):
        several(data, 12, 1, edge_masks=masks)


@pytest.mark.parametrize('row', ['none', '0.2', 'random'])
@pytest.mark.parametrize('kind', ['field', 'current'])
def test_one_slab(row, kind, monkeypatch):
    data = scene(row, True, kind=kind)
    observe_one_slab_cut(monkeypatch, data, 12)
    assert_cross_trace_bars(single(data, 12), several(data, 12, 1), data[0].dt)


@pytest.mark.parametrize('devices', [2, 3])
@pytest.mark.parametrize('steps', [12, 60, 240])
@pytest.mark.parametrize('on', [False, True], ids=['off', 'on'])
@pytest.mark.parametrize('row', ['none', '0.2', '0.5', '0.8', 'random'])
def test_p4_within_cross_trace_bars(row, on, steps, devices):
    data = scene(row, on, steps)
    assert_cross_trace_bars(single(data, steps), several(data, steps, devices), data[0].dt)


@pytest.mark.parametrize('row,on', [('none', False), ('0.2', False), ('random', True)],
                         ids=['none', 'U-0.2', 'D-random'])
def test_two_vs_three_bitwise(row, on):
    """The wrapped two- and three-device probes and all final fields agree."""
    data = scene(row, on, 60)
    two, three = several(data, 60, 2), several(data, 60, 3)
    assert np.array_equal(np.asarray(two[0]), np.asarray(three[0])), 'probes'
    for field in FIELDS:
        assert np.array_equal(np.asarray(getattr(two[1], field)),
                              np.asarray(getattr(three[1], field))), field


@pytest.mark.parametrize('feature', ['cpml', 'debye', 'lorentz', 'ports', 'dft', 'flux', 'ntff', 'pmc', 'periodic_x'])
def test_selector(feature):
    assert select_step() == 'wrapped'
    assert select_step(**{feature: True}) == 'legacy'


@pytest.mark.parametrize('devices', [1, 2, 3])
def test_first_difference(devices):
    """Opt-in increasing-step localization, without retrying equality failures."""
    import json
    import os
    from pathlib import Path
    if not os.environ.get('W1_LOCALIZE'):
        pytest.skip('set W1_LOCALIZE for first-difference report')
    records = []
    for row in ('none', '0.2', '0.5', '0.8', 'random'):
        for on in (False, True):
            for step in range(1, 13):
                data = scene(row, on, step)
                want, got = single(data, step), several(data, step, devices)
                delta = difference(want, got)
                if delta:
                    record = dict(devices=devices, row=row, on=on, step=step, summary=delta)
                    # The accumulated probe can name an earlier sample. Report
                    # the first differing final field cell at this first step.
                    for component in FIELDS:
                        a, b = np.asarray(getattr(want[1], component)), np.asarray(getattr(got[1], component))
                        cells = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
                        if len(cells):
                            cell = tuple(int(i) for i in cells[0])
                            record.update(component=component, cell=cell,
                                single=float(a[cell]), slab=float(b[cell]),
                                single_bits=f'{a.view(np.uint32)[cell]:08x}',
                                slab_bits=f'{b.view(np.uint32)[cell]:08x}')
                            break
                    records.append(record)
                    print('FIRST ' + json.dumps(record), flush=True)
                    break
            else:
                record = dict(devices=devices, row=row, on=on, equal_through=12)
                records.append(record)
                print('FIRST ' + json.dumps(record), flush=True)
    Path(f'.w1s1/localization-{devices}.json').write_text(json.dumps(records, indent=2))


def localized_operands(target, data, previous, current, reference, staged, layout):
    """Raw operands from uninstrumented runs, without exporting intermediates.

    In this scope H is not changed after AFTER_H. Thus final owned H and its
    owned neighbours are the actual H operands consumed by the E update.
    Previous-step outputs supply pre-update fields. Setup capture only observes
    split's inputs/outputs and does not modify the compiled step.
    """
    cell = tuple(target['cell'])
    slab_cell = (layout.owner(cell[0]) * layout.nx_local + layout.local_index(cell[0]),
                 *cell[1:])
    values = {}

    def add(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        values[name] = dict(single=a.item(), slab=b.item(), dtype=str(a.dtype),
                            single_bytes=a.tobytes().hex(), slab_bytes=b.tobytes().hex(),
                            equal=a.dtype == b.dtype and a.tobytes() == b.tobytes())

    def read(state, field, index):
        arr = np.asarray(getattr(state, field))
        if any(i < 0 or i >= n for i, n in zip(index, arr.shape)):
            return np.zeros((), arr.dtype)
        return arr[index]

    for field in FIELDS:
        add('previous.' + field, read(previous[0][1], field, cell), read(previous[1][1], field, cell))
    for field in ('ex', 'ey', 'ez'):
        for axis in range(3):
            shifted = list(cell)
            shifted[axis] += 1
            add(f'previous.{field}.plus_{"xyz"[axis]}',
                read(previous[0][1], field, tuple(shifted)), read(previous[1][1], field, tuple(shifted)))
    for field in ('hx', 'hy', 'hz'):
        add('after_h.' + field, read(current[0][1], field, cell), read(current[1][1], field, cell))
        for axis in range(3):
            shifted = list(cell)
            shifted[axis] -= 1
            add(f'after_h.{field}.minus_{"xyz"[axis]}',
                read(current[0][1], field, tuple(shifted)), read(current[1][1], field, tuple(shifted)))
    inv = reference.invariants
    for name in ('inv_dx', 'inv_dx_h', 'dx_arr'):
        add(name, inv[name][cell[0]], staged[name][slab_cell[0]])
    grid = data[0]
    for name, axis in (('inv_dy', 1), ('inv_dz', 2), ('inv_dy_h', 1), ('inv_dz_h', 2)):
        metric = getattr(grid, name)[cell[axis]]
        add(name, metric, metric)
    add('dt_float64', np.float64(grid.dt), np.float64(grid.dt))
    for name in ('eps_update', 'sigma_update', 'mu_update'):
        for axis in range(3):
            add(f'materials.{name}.{"xyz"[axis]}',
                getattr(inv['materials'].components, name)[axis][cell],
                getattr(staged['materials'].components, name)[axis][slab_cell])
    for axis in range(3):
        keep, cut_keep = inv['materials'].edge_keep, staged['materials'].edge_keep
        add('edge_keep.' + 'xyz'[axis],
            np.float32(1) if keep is None else keep[axis][cell],
            np.float32(1) if cut_keep is None else cut_keep[axis][slab_cell])
        masks, cut_masks = inv['pec_edge_masks'], staged['pec_edge_masks']
        add('pec_mask.' + 'xyz'[axis],
            np.bool_(False) if masks is None else masks[axis][cell],
            np.bool_(False) if cut_masks is None else cut_masks[axis][slab_cell])
    # Resolve matching source rows independently on the two actual tables.
    def drives_at(drives, waveforms, at_cell, rank=None):
        rows = []
        for field, indices in drives.nodes.items():
            indices = tuple(np.asarray(i if rank is None else i[rank]) for i in indices)
            coef = np.asarray(drives.coef[field] if rank is None else drives.coef[field][rank])
            wave_id = np.asarray(drives.wave_id[field] if rank is None else drives.wave_id[field][rank])
            for n, xyz in enumerate(zip(*indices)):
                if tuple(xyz) == at_cell and coef[n] != 0:
                    sample = np.asarray(waveforms)[target['step']-1, wave_id[n]]
                    rows.append(dict(component=field, coefficient=float(coef[n]), sample=float(sample),
                                     coefficient_bytes=coef[n].tobytes().hex(), sample_bytes=sample.tobytes().hex()))
        return rows
    rank = layout.owner(cell[0])
    drive_rows = dict(single=drives_at(inv['drives'], reference.src_waveforms, cell),
        slab=drives_at(staged['drives'], staged['drives'].waves[:, rank, :],
                      (layout.local_index(cell[0]), *cell[1:]), rank=rank))
    return dict(target=target, operands=values, matching_drive_rows=drive_rows,
                all_recorded_operands_equal=all(v['equal'] for v in values.values())
                    and drive_rows['single'] == drive_rows['slab'])


@pytest.mark.parametrize('row', ['none', '0.2', '0.5', '0.8', 'random'])
@pytest.mark.parametrize('on', [False, True], ids=['off', 'on'])
def test_localized_operands(row, on, monkeypatch):
    """One-slab model identity at the first differing field cell of each row."""
    from rfx.runners import _distributed_nu_wrapped as runner
    from rfx.stepping.slab import Slab

    # The recorded first difference is step 3; verify step 2 still agrees.
    step = 3
    data = scene(row, on, step)
    reference = canonical_setup(data, step)
    captures = []
    original_split = runner.split

    def observe(*args, **kwargs):
        output = original_split(*args, **kwargs)
        captures.append(dict(output[0]))
        return output

    monkeypatch.setattr(runner, 'split', observe)
    previous_data = scene(row, on, step - 1)
    previous = single(previous_data, step - 1), several(previous_data, step - 1, 1)
    assert_bits(*previous)
    current = single(data, step), several(data, step, 1)
    layout = Slab(data[0].nx, 1)
    check_one_slab_cut(reference, captures[-1], layout)
    for component in FIELDS:
        a, b = (np.asarray(getattr(result[1], component)) for result in current)
        cells = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
        if len(cells):
            target = dict(row=row, on=on, step=step, component=component,
                          cell=tuple(int(i) for i in cells[0]))
            record = localized_operands(target, data, previous, current,
                                        reference, captures[-1], layout)
            assert record['all_recorded_operands_equal'], record
            break
    else:
        pytest.fail('recorded step-3 first difference was not reproduced')
