"""W1 slices 1–3: model identity and cross-trace bars for the graded slab step."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from rfx import Simulation
from rfx.core.yee import EPS_0, MU_0, MaterialArrays
from tests.contracts.path_equivalence.test_one_vs_several_devices_occupancy import BINS_HZ
from tests._multi_device_scene import host_devices
from rfx.model.electric_metrics import stage_forward_materials
from rfx.stepping.graded_setup import _build_nu_scan
from rfx.nonuniform import make_current_source, run_nonuniform
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
    selected = host_devices(devices)
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


@pytest.mark.parametrize('feature', ['dft', 'flux', 'ntff', 'pmc', 'periodic_x'])
def test_selector(feature):
    assert select_step() == 'wrapped'
    assert select_step(ports=True) == 'wrapped'
    assert select_step(**{feature: True}) == 'legacy'
    with pytest.raises(NotImplementedError, match='ports require the wrapped step'):
        select_step(ports=True, **{feature: True})


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

    # The first differing step follows the source samples (3 with the old
    # default pulse, 4-5 since it starts at zero): find it, then verify the
    # step before still agrees. No difference in 12 steps is the stronger result.
    def differs(n):
        d = scene(row, on, n)
        return bool(difference(single(d, n), several(d, n, 1)))
    step = next((n for n in range(2, 13) if differs(n)), 3)
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
        assert_bits(*current)


@pytest.mark.parametrize('debye,lorentz', [(True, False), (False, True), (True, True)])
def test_pole_selector(debye, lorentz):
    assert select_step(debye=debye, lorentz=lorentz) == 'wrapped'
    assert select_step(debye=debye, lorentz=lorentz, ports=True) == 'wrapped'
    assert select_step(debye=debye, lorentz=lorentz, pmc=True) == 'legacy'
    with pytest.raises(NotImplementedError, match='ports require the wrapped step'):
        select_step(debye=debye, lorentz=lorentz, ports=True, pmc=True)


# Scenes S1 and S2 of rfx-archive records/20261011-debye-one-vs-two (deb_scene.py, copied):
# a Debye body in a PEC box on a 24-cell x axis, S2 alternating 0.9/1.1 mm inner cells.
_S2_PROFILE = np.array([1.0e-3] + [0.9e-3, 1.1e-3] * 11 + [1.0e-3])


def _declared_pole_scene(scene):
    from rfx import Box, GaussianPulse
    from rfx.materials.debye import DebyePole
    prof = {'S1': np.full(24, 1e-3), 'S2': _S2_PROFILE.copy()}[scene]
    sim = Simulation(freq_max=10e9, domain=(.024, .012, .012), dx=.001, boundary='pec', dx_profile=prof)
    sim.add_material('body', eps_r=2.0, debye_poles=[DebyePole(delta_eps=3.0, tau=2e-11)])
    sim.add_material('air', eps_r=1.0)
    sim.add(Box((.006, 0, 0), (.018, .012, .012)), material='body')
    sim.add(Box((.010, 0, 0), (.014, .012, .012)), material='air')
    sim.add_source((.003, .006, .006), 'ez', waveform=GaussianPulse(f0=8e9, bandwidth=.8))
    sim.add_probe((.0085, .006, .006), 'ez')
    sim.add_probe((.021, .005, .006), 'ez')
    return sim, prof


def pole_scene(name):
    """The declared S1/S2 scenes, plus Lorentz and mixed variants of S1."""
    from rfx.materials.lorentz import LorentzPole
    sim, _ = _declared_pole_scene('S1' if name in ('lorentz', 'mixed') else name)
    if name in ('lorentz', 'mixed'):
        from rfx.materials.debye import DebyePole
        extra = dict(debye_poles=[DebyePole(3., 2e-11)]) if name == 'mixed' else {}
        sim.add_material('body', eps_r=4., **extra, lorentz_poles=[LorentzPole(
            omega_0=2*np.pi*8e9, delta=2*np.pi*1e9, kappa=(2*np.pi*8e9)**2)])
    # Keep the scene's two E probes and add H at those positions for the
    # paired H bar; the declared builder has no magnetic probes.
    sim.add_probe((.0085, .006, .006), 'hy')
    sim.add_probe((.021, .005, .006), 'hy')
    return sim


def pole_metrics(one, several, dt):
    """BRIEF2 T3 scales, with E/H columns of the declared pole scenes."""
    one, several = np.asarray(one), np.asarray(several)
    assert one.dtype == several.dtype == np.float32
    assert one.shape == several.shape
    assert np.isfinite(one).all() and np.isfinite(several).all()
    e_peak = np.float32(np.max(np.abs(one[:, :2])))
    h_peak = np.float32(np.max(np.abs(one[:, 2:])))
    h_scale = np.float32(max(float(h_peak), float(e_peak)/np.sqrt(MU_0/EPS_0)))
    delta = np.abs(one.astype(np.float64) - several)
    e = np.max(delta[:, :2], axis=1) / float(np.spacing(e_peak))
    h = np.max(delta[:, 2:], axis=1) / float(np.spacing(h_scale))
    kernel = np.exp(-2j*np.pi*np.outer(np.linspace(2e9, 10e9, 20), np.arange(len(one))*dt))
    expected, actual = kernel @ one.astype(np.float64), kernel @ several.astype(np.float64)
    return e, h, float(np.max(np.abs(actual-expected))/np.max(np.abs(expected)))


@pytest.mark.parametrize('name', ['S1', 'lorentz'])
def test_pole_one_slab_invariants(name, monkeypatch):
    from rfx.stepping import graded_loop as nonuniform
    from rfx.runners import _distributed_nu_wrapped as runner
    from rfx.stepping.slab import Slab, cut_pole_coeffs, cut_poles
    captures = []
    build = nonuniform._build_nu_scan

    def capture(*args, **kwargs):
        setup = build(*args, **kwargs)
        captures.append(setup)
        return setup

    with monkeypatch.context() as patch:
        patch.setattr(nonuniform, '_build_nu_scan', capture)
        pole_scene(name).run(n_steps=2)
    expected = captures[0]
    layout = Slab(expected.carry_init['fdtd'].ex.shape[0], 1)
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]), ('x',))
    captured_cut = []
    original_split = runner.split

    def observe(*args, **kwargs):
        output = original_split(*args, **kwargs)
        captured_cut.append(output)
        raise BeforeStepping

    with monkeypatch.context() as patch:
        patch.setattr(runner, 'split', observe)
        with pytest.raises(BeforeStepping):
            pole_scene(name)._forward_distributed_nonuniform_from_materials(
                n_steps=2, devices=list(mesh.devices), skip_preflight=True)
    actual, carry = captured_cut[0]
    # Spatial leaves and (pole, x, ...) leaves have distinct owned axes.
    spatial = dict(expected.invariants)
    for family in ('debye', 'lorentz'):
        spatial[family + '_coeffs'] = None
    check_one_slab_cut(expected._replace(invariants=spatial),
        {**actual, 'debye_coeffs': None, 'lorentz_coeffs': None}, layout)
    owned = slice(layout.ghost_width, layout.ghost_width + layout.nx)
    for family in ('debye', 'lorentz'):
        coeffs = expected.invariants[family + '_coeffs']
        if coeffs is None:
            continue
        spatial_fields = ('ca', 'cb') if family == 'debye' else ('ca', 'cb', 'cc')
        for field in coeffs._fields:
            value = getattr(actual[family + '_coeffs'], field)
            if field not in spatial_fields:
                value = jax.tree.map(lambda a: a[:, owned], value)
            assert_same_tree(getattr(coeffs, field), value, family + '.' + field,
                             owned=owned if field in spatial_fields else None)
        assert_same_tree(expected.carry_init[family],
                         jax.tree.map(lambda a: a[:, owned], carry[family]), family + '.state')
        # FILL covers both physical ghosts and alignment rows (2/3 slabs).
        from rfx.stepping.slab import vacuum_dispersion_values
        vacuum = vacuum_dispersion_values(coeffs, expected.step_fn_inv.context.dt)
        for devices in (2, 3):
            padded = Slab(layout.nx, devices)
            cut_coeffs = cut_pole_coeffs(coeffs, padded, family,
                                       dt=expected.step_fn_inv.context.dt)
            cut_state = cut_poles(expected.carry_init[family], padded, family + '_state')
            indices = np.arange(padded.nx_local)[None, :] + np.arange(devices)[:, None]*padded.nx_per_rank - 1
            outside = (indices < 0) | (indices >= layout.nx)
            for field in coeffs._fields:
                for a in jax.tree.leaves(getattr(cut_coeffs, field)):
                    a = np.asarray(a)
                    if field not in spatial_fields:
                        a = a.transpose(0, 2, 1, 3, 4)
                    assert np.all(a[outside] == np.asarray(vacuum[field], a.dtype)), field
            for a in jax.tree.leaves(cut_state):
                assert np.all(np.asarray(a).transpose(0, 2, 1, 3, 4)[outside] == 0)
    assert_same_tree(expected.carry_init['fdtd'], carry['fdtd'], 'fdtd', owned=owned)


def pole_law(one, several):
    """Accumulated float32 rounding law (#1860 law C) on the E and H columns.

    Pole media are not bit-stable across compiled forms on one device: the
    Debye box differs from itself under jit by 251 ULP of the peak, and the
    multi-device step's difference is formed inside the compiled step from
    byte-equal operands (rfx-archive rfx/records/20261017-item5-dispersive,
    RESULT.md). A per-step 9-ULP bar is the wrong unit for such a difference;
    at step n it is allowed 1.2e-7 x the running sum of the peak |field| up to n.
    Returns the largest difference/allowance ratio for E and for H.
    """
    one, several = np.asarray(one, np.float64), np.asarray(several, np.float64)
    ratios = []
    for columns in (slice(0, 2), slice(2, None)):
        difference = np.abs(several[:, columns] - one[:, columns]).max(axis=1)
        allowance = 1.2e-7 * np.cumsum(np.abs(one[:, columns]).max(axis=1))
        ratio = np.divide(difference, allowance, out=np.where(difference > 0, np.inf, 0.),
                          where=allowance > 0)
        ratios.append(float(ratio.max()))
    return tuple(ratios)


@pytest.mark.parametrize('name', ['S1', 'S2', 'lorentz'])
def test_pole_two_vs_three_bitwise_and_bars(name):
    one = pole_scene(name).run(n_steps=60)
    two = pole_scene(name).run(n_steps=60, devices=host_devices(2))
    three = pole_scene(name).run(n_steps=60, devices=host_devices(3))
    assert_bits((two.time_series, two.state), (three.time_series, three.state))
    for result in (two, three):
        e, h, dft = pole_metrics(one.time_series, result.time_series, float(one.dt))
        law_e, law_h = pole_law(one.time_series, result.time_series)
        # S2 on the CI host: 10 ULP of the peak at step 57, law ratio ~0.5 (Mac 0.48).
        assert law_e <= 1. and law_h <= 1., (law_e, law_h, np.max(e), np.max(h))
        assert dft <= 1e-4, dft


@pytest.mark.parametrize('mutation', ['skipped_e_exchange', 'debye_tau_0.01%'])
def test_pole_law_catches(mutation, monkeypatch):
    """The law stays a bar: each defect lands far above it (Mac: 1.3e4, 129)."""
    one = pole_scene('S2').run(n_steps=60)
    sim = pole_scene('S2')
    if mutation == 'skipped_e_exchange':
        from rfx.runners import distributed_nu
        monkeypatch.setattr(distributed_nu, '_exchange_e_ghosts_nu', lambda st, *a, **kw: st)
    else:
        from rfx.materials.debye import DebyePole
        sim.add_material('body', eps_r=2.0, debye_poles=[DebyePole(delta_eps=3.0, tau=2e-11 * 1.0001)])
    several = sim.run(n_steps=60, devices=host_devices(2))
    law_e, law_h = pole_law(one.time_series, several.time_series)
    assert law_e > 1. and law_h > 1., (mutation, law_e, law_h)


@pytest.mark.parametrize('name', ['S1', 'lorentz', 'mixed'])
@pytest.mark.parametrize('warmup,checkpoint', [(0, None), (3, 3)])
def test_pole_forward_gradient_reaches_wrap(name, warmup, checkpoint, monkeypatch):
    from rfx.runners import _distributed_nu_wrapped as runner
    calls = []
    original = runner.split

    def observe(*args, **kwargs):
        calls.append(args[0]['debye_coeffs'] is not None or args[0]['lorentz_coeffs'] is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, 'split', observe)
    sim = pole_scene(name)
    grid = sim._build_nonuniform_grid()
    materials, _, _, _ = sim._assemble_materials_nu(grid)
    def loss(eps):
        result = sim.forward(n_steps=12, checkpoint=False, skip_preflight=True,
            eps_override=eps, distributed=True, devices=host_devices(2),
            n_warmup=warmup, checkpoint_every=checkpoint)
        return jnp.sum(result.time_series**2)
    gradient = jax.grad(loss)(materials.eps_r)
    assert np.isfinite(gradient).all()
    assert calls and all(calls)


def cpml_scene(name):
    """Declared scene 2 (copied builder) and the existing #1302 builder."""
    from rfx import Box, GaussianPulse
    if name == 'debye':
        from tests.unit.runners.test_distributed_edge_averaged_materials import _cpml_1302_box
        sim = _cpml_1302_box('debye', 'fwd_graded')
        sim.add_probe((.019, .006, .006), 'hy')
    else:
        sim = Simulation(freq_max=10e9, domain=(.016, .012, .010), dx=.001,
            boundary='cpml', cpml_layers=4,
            dx_profile=np.array([.001] + [.9*.001, 1.1*.001]*7 + [.001]), precision='float32')
        sim.add_material('block', eps_r=4.)
        sim.add(Box((-1., .003, .002), (1., .008, .006)), material='block')
        sim.add_source((.005, .004, .003), 'ez',
                       waveform=GaussianPulse(f0=5e9, bandwidth=.8), amplitude_kind='current')
        for component, position in [('ez', (.007, .005, .004)), ('ez', (.015, .007, .006)),
                                    ('ez', (.016, .005, .004)), ('hy', (.007, .005, .004)),
                                    ('hy', (.016, .007, .006))]:
            sim.add_probe(position, component)
    source = sim._ports[0]
    sim.add_probe(source.position, source.component)
    return sim


def cpml_metrics(one, several, dt, name):
    ecols, hcols = ((0, 2), (1,)) if name == 'debye' else ((0, 1, 2, 5), (3, 4))
    one, several = np.asarray(one), np.asarray(several)
    assert np.isfinite(one).all() and np.isfinite(several).all()
    epeak = np.float32(np.max(np.abs(one[:, ecols])))
    hscale = np.float32(max(float(np.max(np.abs(one[:, hcols]))), float(epeak)/np.sqrt(MU_0/EPS_0)))
    delta = np.abs(one.astype(np.float64)-several)
    e = np.max(delta[:, ecols], axis=1)/float(np.spacing(epeak))
    h = np.max(delta[:, hcols], axis=1)/float(np.spacing(hscale))
    columns = [0] if name == 'debye' else list(range(5))
    kernel = np.exp(-2j*np.pi*np.outer(np.linspace(2e9, 10e9, 20), np.arange(len(one))*dt))
    expected, actual = (kernel @ a[:, columns].astype(np.float64) for a in (one, several))
    return e, h, float(np.max(np.abs(actual-expected))/np.max(np.abs(expected)))


@pytest.mark.parametrize('name', ['scene2'])
def test_cpml_one_slab_invariants(name, monkeypatch):
    from rfx.stepping import graded_loop as nonuniform
    from rfx.runners import _distributed_nu_wrapped as runner
    from rfx.stepping.slab_cpml import apply
    captures = []
    build = nonuniform._build_nu_scan
    def capture(*args, **kwargs):
        setup = build(*args, **kwargs)
        captures.append(setup)
        return setup
    with monkeypatch.context() as patch:
        patch.setattr(nonuniform, '_build_nu_scan', capture)
        cpml_scene(name).forward(n_steps=2, checkpoint=False, skip_preflight=True)
    expected = captures[0]
    inv, initial = expected.invariants, expected.carry_init
    staged_contexts = []
    build_local = runner.build_setup
    def capture_local(*args, **kwargs):
        setup = build_local(*args, **kwargs)
        staged_contexts.append(setup.step_fn_inv.context)
        return setup
    original = runner.split
    def observe(*args, **kwargs):
        actual, carry = original(*args, **kwargs)
        layout = args[2]
        owned = slice(layout.ghost_width, layout.ghost_width + layout.nx)
        for key in inv:
            if key in ('drives', 'probe_table', 'debye_coeffs', 'lorentz_coeffs', 'e_loss',
                       'cpml_inv_eps_r'):
                continue
            assert_same_tree(inv[key], actual[key], key, owned=owned)
        if name == 'scene2':
            # Plain CPML keeps its existing in-loop component arithmetic.
            assert actual['e_loss'] is None and actual['cpml_inv_eps_r'] is None
        else:
            assert_same_tree(inv['cpml_inv_eps_r'], actual['cpml_inv_eps_r'],
                             'cpml_inv_eps_r', owned=owned)
            loss = actual['e_loss']
            # wrap() restores the static relative-epsilon flag per device.
            assert_same_tree(inv['e_loss'], None if loss is None else (*loss, False),
                             'e_loss', owned=owned)
        for family in ('debye', 'lorentz'):
            coeffs = inv[family + '_coeffs']
            if coeffs is None:
                continue
            spatial = ('ca', 'cb') if family == 'debye' else ('ca', 'cb', 'cc')
            for field in coeffs._fields:
                value = getattr(actual[family + '_coeffs'], field)
                if field not in spatial:
                    value = jax.tree.map(lambda a: a[:, owned], value)
                assert_same_tree(getattr(coeffs, field), value, family + '.' + field,
                                 owned=owned if field in spatial else None)
            assert_same_tree(initial[family], jax.tree.map(lambda a: a[:, owned], carry[family]), family)
        # Tables are checked with the existing slice-1 identity helper.
        tables = {key: inv[key] for key in ('drives', 'probe_table')}
        check_one_slab_cut(expected._replace(invariants=tables), actual, layout)
        assert_same_tree(initial['fdtd'], carry['fdtd'], 'fdtd', owned=owned)
        # Native x psi has one-device depth; y/z psi has local x rows.
        for field in initial['cpml']._fields:
            value = getattr(carry['cpml'], field)
            if field.split('_')[-1][0] != 'x':
                value = value[owned]
            assert_same_tree(getattr(initial['cpml'], field), value, 'native.' + field)
        canonical = expected.step_fn_inv.context.cpml_params
        # Observe the exact profiles and restored widths passed to the kernel.
        from dataclasses import replace
        local = replace(staged_contexts[-1], x_face_frame={'x': (1, layout.nx + 1)})
        def inspect(state, profiles, psi, grid, **kwargs):
            assert_same_tree(canonical, profiles, 'native profiles and widths')
            return state, psi
        apply(inspect, local, actual, carry['fdtd'], carry['cpml'])
        raise BeforeStepping
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'build_setup', capture_local)
        patch.setattr(runner, 'split', observe)
        with pytest.raises(BeforeStepping):
            cpml_scene(name)._forward_distributed_nonuniform_from_materials(
                n_steps=2, devices=jax.devices('cpu')[:1], skip_preflight=True)


@pytest.mark.parametrize('name', ['scene2'])
def test_cpml_two_vs_three_bitwise_and_bars(name):
    from tests._distributed_dispersive_cpml import scene, measure, check
    sim = scene('debye') if name == 'debye' else cpml_scene(name)
    outputs = [sim.forward(n_steps=60, checkpoint=False, skip_preflight=True,
        **({} if n == 1 else dict(distributed=True, devices=host_devices(n))))
        for n in (1, 2, 3)]
    # Forward omits gathered fields, so use the run entry for final-field identity.
    from rfx.runners import _distributed_nu_wrapped as runner
    from unittest.mock import patch
    states = []
    original = runner.reduce
    def gather(records, state, layout, **kwargs):
        result = original(records, state, layout, gather=True)
        states.append(result[1])
        return result
    with patch.object(runner, 'reduce', gather):
        for n in (2, 3):
            sim.forward(n_steps=60, checkpoint=False, skip_preflight=True,
                        distributed=True, devices=host_devices(n))
    assert_bits((outputs[1].time_series, states[0]), (outputs[2].time_series, states[1]))
    for result in outputs[1:]:
        if name == 'debye':
            check(measure(outputs[0].time_series, result.time_series, float(result.dt)))
            continue
        e, h, dft = cpml_metrics(outputs[0].time_series, result.time_series, float(result.dt), name)
        assert np.max(e) <= 9, e
        assert np.max(h) <= 9, h
        assert dft <= 1e-4, dft


@pytest.mark.parametrize('debye,lorentz', [(False, False)])
def test_cpml_selector(debye, lorentz):
    assert select_step(cpml=True, debye=debye, lorentz=lorentz) == 'wrapped'
    assert select_step(cpml=True, debye=debye, lorentz=lorentz, ports=True) == 'wrapped'


@pytest.mark.parametrize('name,mode', [('scene2', mode) for mode in ('run', 'grad')])
def test_cpml_entries_reach_wrap(name, mode, monkeypatch):
    from rfx.runners import _distributed_nu_wrapped as runner
    calls = []
    original = runner.split
    def observe(*args, **kwargs):
        calls.append('cpml_x' in args[0] and 'cpml' in args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'split', observe)
    if name == 'lorentz':
        from tests.unit.runners.test_distributed_edge_averaged_materials import _cpml_1302_box
        sim = _cpml_1302_box('lorentz', 'fwd_graded')
    else:
        sim = cpml_scene(name)
    if mode == 'run':
        sim.run(n_steps=12, devices=host_devices(2), skip_preflight=True)
    else:
        materials, _, _, _ = sim._assemble_materials_nu(sim._build_nonuniform_grid())
        def loss(eps, **kw):
            result = sim.forward(n_steps=12, checkpoint=False, skip_preflight=True,
                eps_override=eps, n_warmup=3, checkpoint_every=3, **kw)
            return jnp.sum(result.time_series**2)
        two = np.asarray(jax.grad(lambda e: loss(e, distributed=True, devices=host_devices(2)))(materials.eps_r))
        assert np.isfinite(two).all()
        if name == 'scene2':
            one = np.asarray(jax.grad(loss)(materials.eps_r))
            # A gradient summed over the record: 1e-4 of its peak (cross-trace bar).
            assert np.max(np.abs(two - one)) <= 1e-4 * np.max(np.abs(one))
    assert calls and all(calls)




@pytest.mark.parametrize('devices', [2, 3])
def test_cpml_native_psi_roundtrip(devices):
    """Nonzero, nonconstant psi pins axes, physical owners and padding."""
    from rfx.boundaries.cpml import init_cpml
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_cpml import import_state, export_state
    grid = cpml_scene('scene2')._build_nonuniform_grid()
    _, psi = init_cpml(grid)
    psi = jax.tree.map(lambda a: jnp.arange(a.size, dtype=a.dtype).reshape(a.shape), psi)
    fields = {}
    for name, value in zip(psi._fields, psi):
        if name.split('_')[-1][0] == 'x':
            extra = grid.nx - value.shape[0]
            value = jnp.pad(value, ((0, extra) if name.endswith('lo') else (extra, 0), (0, 0), (0, 0)))
        fields[name] = value
    layout = Slab(grid.nx, devices)
    mesh = Mesh(np.asarray(host_devices(devices)), ('x',))
    local = cut(type(psi)(**fields), layout, 'cpml_psi', mesh=mesh)
    native = export_state(local, grid, layout)
    restored = import_state(native, psi, layout)
    assert_same_tree(psi, restored, 'native psi roundtrip')


def graded_cpml_box(nx=28):
    """Review scene (W1, 2026-10-15): graded x and y, CPML on every face, a lossy
    dielectric and a PEC block, current source. The native CPML call pins the
    first-divergence regression recorded for item 5."""
    from rfx import Box
    dx = np.where(np.arange(nx) % 3 == 0, .8e-3, 1.1e-3)
    dx[[0, -1]] = 1e-3
    dy = np.where(np.arange(12) % 2 == 0, .9e-3, 1.1e-3)
    dy[[0, -1]] = 1e-3
    lx = float(np.sum(dx))
    xs = np.concatenate([[0], np.cumsum(dx)])
    sim = Simulation(freq_max=15e9, domain=(lx, 12e-3, 10e-3), dx=1e-3, boundary='cpml',
                     cpml_layers=4, dx_profile=dx, dy_profile=dy)
    sim.add_material('diel', eps_r=3.3, sigma=0.05)
    sim.add(Box((xs[nx//3 - 2], 2e-3, 2e-3), (xs[nx//3 + 2], 9e-3, 7e-3)), material='diel')
    sim.add(Box((xs[2*nx//3 - 1], 4e-3, 3e-3), (xs[2*nx//3 + 1], 7e-3, 6e-3)), material='pec')
    sim.add_source((xs[nx//3 + 1] + 1e-4, 5.5e-3, 4.5e-3), 'ez', amplitude_kind='current')
    for fx in (0.1, 0.34, 0.5, 0.66, 0.9, 0.99):
        sim.add_probe((lx * fx, 6e-3, 5e-3), 'ez')
    sim.add_probe((lx * 0.5, 6e-3, 5e-3), 'hy')
    return sim


def _probe_ulp(one, several):
    """Largest difference in float32 units of the probe array's own peak."""
    one, several = np.asarray(one), np.asarray(several)
    peak = np.float32(np.max(np.abs(one)))
    return float(np.max(np.abs(one.astype(np.float64) - several)) / np.spacing(peak))


def test_graded_cpml_runs_on_wrapped_equal_values(monkeypatch):
    from rfx.runners import _distributed_nu_wrapped as runner
    calls = []
    original = runner.run_wrapped
    monkeypatch.setattr(runner, 'run_wrapped', lambda *a, **k: calls.append(1) or original(*a, **k))
    sim = graded_cpml_box()
    one = sim.run(n_steps=200, skip_preflight=True).time_series
    two = sim.run(n_steps=200, devices=host_devices(2), skip_preflight=True).time_series
    assert calls == [1]
    one, two = np.asarray(one), np.asarray(two)
    assert one.dtype == two.dtype and one.shape == two.shape
    # Equal values; the sign of an exact zero may differ (-0.0 vs 0.0 seen at nx=28).
    assert np.array_equal(one, two), _probe_ulp(one, two)


def test_graded_cpml_three_devices_equal_values():
    sim = graded_cpml_box()
    one = sim.run(n_steps=200, skip_preflight=True).time_series
    three = sim.run(n_steps=200, devices=host_devices(3), skip_preflight=True).time_series
    assert np.array_equal(np.asarray(one), np.asarray(three)), _probe_ulp(one, three)


def _item6_identity(case, devices, mode='plain', mutate=None):
    """Independent whole setup is the reference; production setup stays local."""
    from tests.contracts.path_equivalence.test_slab_view_producers import SCENES, _cells, _conductors
    from rfx.model.materials import with_components
    from rfx.materials.debye import init_debye
    from rfx.materials.lorentz import init_lorentz
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_wrap import split
    from rfx.stepping.slab_setup import build_setup, finish, stage_dispersion, stage_pec
    from rfx.stepping.slab_view import uniform_slab_rows
    s = SCENES['poles' if case in ('debye', 'lorentz', 'mixed') else 'material_seam']
    cells = _cells(s)
    db = cells.debye_spec if case in ('debye', 'mixed') else None
    lr = cells.lorentz_spec if case in ('lorentz', 'mixed') else None
    grid = s.grid
    layout = Slab(grid.nx, devices)
    mesh = Mesh(np.asarray(host_devices(devices)), ('x',))
    occ = jnp.asarray(s.occupancy) if case == 'occupancy' else None
    sheets, wires = _conductors(s) if case == 'sheet_wire' else ((), ())
    mask = s.pec_mask if case in ('pec', 'sheet_wire') else None
    edges = None if mask is None and not sheets else realized_pec_edge_masks(mask, sheets, wires)
    mats = with_components(cells.materials, grid, periodic=(False,) * 3, debye_spec=db, lorentz_spec=lr)
    debye = None if db is None else init_debye(db[0], mats, grid.dt, mask=db[1])
    lorentz = None if lr is None else init_lorentz(lr[0], mats, grid.dt, mask=lr[1])
    sources = [SourceSpec(3, 2, 2, 'ez', jnp.arange(3, dtype=jnp.float32))]
    probes = [ProbeSpec(4, 2, 2, 'ez')]
    # This is the sole whole-grid build in this test. It does not use any
    # uniform helper or production setup helper to make expected values.
    whole = _build_nu_scan(grid, mats, 3, cpml_axes='', debye=debye, lorentz=lorentz,
        pec_edge_masks=edges, pec_occupancy=occ, sources=[tuple(a) for a in sources],
        probes=[tuple(a) for a in probes], pec_faces={f'{a}_{b}' for a in 'xyz' for b in ('lo', 'hi')})
    expected = split(whole.invariants, whole.carry_init, layout, mesh,
                     waveforms=whole.src_waveforms, dt=grid.dt)
    capture = {}
    def actual(eps):
        material = stage_forward_materials(cells.materials._replace(eps_r=eps), layout, mesh, uniform=True)
        d = stage_dispersion(material, grid.dt, db, grid, layout, mesh, 'debye')
        l = stage_dispersion(material, grid.dt, lr, grid, layout, mesh, 'lorentz')
        setup = build_setup(grid, material, layout, mesh, 3, sources, probes,
            debye=d, lorentz=l, occupancy=cut(occ, layout, 'pec_occupancy', mesh=mesh),
            pec_mask=stage_pec(mask, sheets, wires, grid, layout, mesh))
        result = finish(setup.invariants, setup.carry_init, layout, mesh,
                        waveforms=setup.src_waveforms, dt=grid.dt)
        if mutate == 'ulp':
            m = result[0]['materials']
            e = m.components.eps_update
            e = (e[0].at[1, 2, 2].set(jnp.nextafter(e[0][1, 2, 2], jnp.float32(jnp.inf))), *e[1:])
            result[0]['materials'] = m._replace(components=m.components._replace(eps_update=e))
        capture['result'] = result
        return jnp.sum(result[0]['cpml_inv_eps_r'][0]) if result[0]['cpml_inv_eps_r'] else jnp.sum(eps)
    if mode == 'grad':
        jax.grad(actual)(cells.materials.eps_r)
        result = jax.tree.map(lambda a: getattr(a, 'primal', a), capture['result'])
    else:
        actual(cells.materials.eps_r)
        result = capture['result']
    assert_same_tree(expected, result, f'item6.{case}.{devices}.{mode}')
    return uniform_slab_rows(layout).indices


@pytest.mark.parametrize('case', ('pec', 'debye', 'lorentz', 'mixed', 'occupancy', 'material_seam', 'sheet_wire'))
@pytest.mark.parametrize('devices', (2, 3))
@pytest.mark.parametrize('mode', ('plain', 'grad'))
def test_item6_all_staged_invariants_and_initial_carry(case, devices, mode):
    _item6_identity(case, devices, mode)


def test_item6_owned_ulp_mutation_is_caught():
    with pytest.raises(AssertionError, match='eps_update'):
        _item6_identity('material_seam', 2, mutate='ulp')


def test_item6_missing_extra_runtime_row_is_caught(monkeypatch):
    from rfx.stepping import slab_setup
    original = slab_setup.uniform_from_cut
    def missing(*args, **kwargs):
        a = original(*args, **kwargs)
        return a if a is None or a.ndim == 0 else a.at[:, 0].set(a[:, 1])
    monkeypatch.setattr(slab_setup, 'uniform_from_cut', missing)
    with pytest.raises(AssertionError, match='materials'):
        _item6_identity('material_seam', 2)


def _item6_memory_worker(boundary, devices, mode, mutation=False, case='mixed', route='wrapped'):
    """Called only in a fresh subprocess, before any other scene is built."""
    from tests.unit.runners import test_distributed_nu_forward_staging as staging
    from unittest.mock import patch
    sim_type = staging.Simulation
    with patch.object(staging, 'Simulation', lambda **kw: sim_type(**{**kw, 'boundary': boundary})):
        sim = staging._model(case, large=True)
    grid = sim._build_nonuniform_grid()
    size = int(np.prod(grid.shape))
    eps = jnp.full(grid.shape, 1.5, dtype=jnp.float32)
    scan = jax.lax.scan
    record = []
    held = []
    if mutation:
        from rfx.runners import _distributed_nu_wrapped as runner
        from jax.sharding import NamedSharding, PartitionSpec as P
        build = runner.build_setup
        def whole_gather(grid, materials, layout, mesh, *args, **kwargs):
            a = materials.eps_r
            whole = a[:, 2:-1].reshape((layout.nx_padded,) + a.shape[2:])[:layout.nx]
            held.append(jax.device_put(whole, NamedSharding(mesh, P())))
            return build(grid, materials, layout, mesh, *args, **kwargs)
        runner.build_setup = whole_gather
    def observe(*args, **kwargs):
        if not record:
            per, whole = {}, []
            for a in jax.live_arrays():
                for shard in a.addressable_shards:
                    name = str(shard.device)
                    per[name] = per.get(name, 0) + shard.data.nbytes
                    if shard.data.size >= size and a is not eps:
                        whole.append(dict(device=name, shape=list(a.shape), dtype=str(a.dtype),
                                          shard_shape=list(shard.data.shape), bytes=shard.data.nbytes))
            record.append(dict(bytes=per, whole=whole, grid=list(grid.shape)))
        return scan(*args, **kwargs)
    jax.lax.scan = observe
    if route == 'legacy':
        from rfx.stepping import slab_wrap
        slab_wrap.select_step = lambda **flags: 'legacy'
    def forward(e):
        kw = {} if route == 'one' else dict(distributed=True, devices=host_devices(devices))
        return sim.forward(eps_override=e, n_steps=8, skip_preflight=True, **kw).time_series
    result = forward(eps) if mode == 'plain' else jax.grad(lambda e: jnp.sum(forward(e) ** 2))(eps)
    jax.block_until_ready(result)
    assert len(record) == 1
    return dict(boundary=boundary, devices=devices, mode=mode, mutation=mutation, route=route, **record[0])


def _item6_memory_subprocess(boundary, devices, mode, mutation=False, case='mixed', route='wrapped'):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[3]
    code = ('import json; from tests.unit.runners.test_distributed_nu_wrapped_step import _item6_memory_worker; '
            f'print("ITEM6_JSON " + json.dumps(_item6_memory_worker({boundary!r}, {devices}, {mode!r}, {mutation}, {case!r}, {route!r})))')
    env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=str(root),
               XLA_FLAGS='--xla_force_host_platform_device_count=3', PYTHONDONTWRITEBYTECODE='1')
    p = subprocess.run([sys.executable, '-c', code], cwd=root, env=env, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(next(s[11:] for s in p.stdout.splitlines() if s.startswith('ITEM6_JSON ')))


# Archive slice3/memw.py and memw_pec.py, commit 7cc7deeb6, CPU,
# remeasured 2026-10-14 with three visible devices and explicit selection.
_ITEM6_LEGACY_MAX = {
    ('pec', 2, 'plain'): 1298821, ('pec', 2, 'grad'): 1298833,
    ('pec', 3, 'plain'): 935612, ('pec', 3, 'grad'): 935624,
    # CPML: the same scene without poles (case 'nopole'), legacy forced, 2a2344276,
    # 2026-10-15 (rfx-archive records/20261013-w1-wrapped-vs-legacy/redo-2a2344276).
    ('cpml', 2, 'plain'): 1122543, ('cpml', 2, 'grad'): 1122555,
    ('cpml', 3, 'plain'): 830359, ('cpml', 3, 'grad'): 830367,
}


def _assert_item6_memory(record):
    assert not record['whole'], f"whole-domain arrays: {record['whole']}"
    legacy = _ITEM6_LEGACY_MAX[record['boundary'], record['devices'], record['mode']]
    assert max(record['bytes'].values()) <= 2.0 * legacy, record


@pytest.mark.parametrize('boundary,devices', [('pec', 2), ('pec', 3), ('cpml', 2),
                                                ('cpml', 3)])
@pytest.mark.parametrize('mode', ('plain', 'grad'))
def test_item6_first_scan_memory(boundary, devices, mode):
    if boundary == 'cpml_poles':
        from tests._distributed_dispersive_cpml import record
        rows = {route: _item6_memory_subprocess('cpml', 1 if route == 'one' else devices,
                mode, route=route) for route in ('one', 'legacy', 'wrapped')}
        record(dict(entry='memory', kind='mixed', mode=mode, devices=devices, rows=rows))
        assert not rows['wrapped']['whole'], rows['wrapped']
        for device, live in rows['wrapped']['bytes'].items():
            assert live <= 2 * rows['legacy']['bytes'][device], rows
        return
    # CPML: the staging scene without poles, against legacy on that scene.
    kw = dict(case='nopole') if boundary == 'cpml' else {}
    _assert_item6_memory(_item6_memory_subprocess(boundary, devices, mode, **kw))


@pytest.mark.parametrize('devices', (2, 3))
@pytest.mark.parametrize('mode', ('plain', 'grad'))
def test_item6_whole_gather_memory_mutation_is_caught(devices, mode):
    record = _item6_memory_subprocess('pec', devices, mode, mutation=True)
    with pytest.raises(AssertionError, match='whole-domain arrays'):
        _assert_item6_memory(record)
