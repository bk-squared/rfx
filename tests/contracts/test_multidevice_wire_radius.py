"""#2032 Part A: fixed-radius public solves against public one-device solves.

Bars and scenes are predeclared in the read-only 20261110 wire-radius design.
Set the host device count before Python starts; three devices select only the
predeclared three-device cases. No cut, slab or stamp helper is an oracle.
"""
import functools
import json
import sys
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx._internal import _realized
from rfx._internal._grid_metric import field_index
from rfx.boundaries.spec import BoundarySpec

DEVICES = min(3, len(jax.devices('cpu')))
assert DEVICES >= 2, 'set XLA_FLAGS for at least two host devices before starting Python'
D = .5e-3
RADIUS = .05e-3
HEIGHT = 1.5e-3
FREQS = np.array([8e9, 11.3e9, 14e9])
MESHES = [(m, b) for m in ('uniform', 'graded') for b in ('cpml', 'pec')]


def diagnostic(bar, case, **numbers):
    print('WRAD ' + json.dumps(dict(bar=bar, case=case, devices=DEVICES, **numbers)),
          file=sys.stderr, flush=True)


def definitions(name, seam, pads):
    y, z = pads[1:]
    if name == 'R2':
        return [((seam-2, y+7, z+4), 'ex', .002, True)], [
            (seam, y+7, z+4), (seam, y+8, z+4),
            (seam-5, y+9, z+3), (seam+5, y+5, z+5)]
    r1 = [((seam, y+7, 0), 'ez', HEIGHT, True)]
    r3 = [((seam-5, y+10, 0), 'ez', HEIGHT, True),
          ((seam+5, y+4, 0), 'ez', HEIGHT, True)]
    p1 = [(seam, y+7, 1), (seam-1, y+7, 1),
          (seam-4, y+9, 1), (seam+4, y+5, 1)]
    p3 = [(seam-5, y+10, 1), (seam+5, y+4, 1),
          (seam-1, y+7, 1), (seam+2, y+9, 1)]
    return {'R1': (r1, p1), 'R3': (r3, p3),
            'R4': (r1+r3+[((seam+9, y+12, 0), 'ez', HEIGHT, False)], p1+p3)}[name]


def declared_cells(ports):
    cells = []
    for start, component, extent, radius in ports:
        if not radius:
            continue
        axis = 'xyz'.index(component[1])
        for offset in range(int(round(extent/D))):
            cell = list(start)
            cell[axis] += offset
            cells.append((axis, tuple(cell)))
    return cells


def scene(mesh, boundary, name, *, radius=True, column=None, length=.0175):
    real_nx = int(round(length/D))
    walls = (BoundarySpec(x=boundary, y=boundary, z='pec')
             if name != 'R2' else boundary)
    kwargs = dict(freq_max=30e9, domain=(length, .008, .004 if name == 'R2' else HEIGHT),
                  dx=D, boundary=walls, cpml_layers=8)
    if mesh == 'graded':
        kwargs.update(dx_profile=np.full(real_nx, D), dy_profile=np.full(16, D))
    sim = Simulation(**kwargs)
    grid = sim._build_realized_grid()
    seam = (grid.shape[0]+DEVICES-1)//DEVICES
    ports, probes = definitions(name, seam, grid.axis_pads)
    if mesh == 'graded':
        stencil_x = set()
        for axis, cell in declared_cells(ports):
            stencil_x.add(cell[0])
            if axis != 0:
                stencil_x.update((cell[0]-1, cell[0]+1))
        pad = grid.axis_pads[0]
        candidates = range(pad+3, pad+real_nx-4) if boundary == 'cpml' else range(real_nx-1)
        pair = next(i for i in candidates
                    if all(abs(j-k) >= 3 for j in (i, i+1) for k in stencil_x))
        profile = np.full(real_nx, D)
        profile[pair-pad:pair-pad+2] = [.9*D, 1.1*D]
        kwargs['dx_profile'] = profile
        sim = Simulation(**kwargs)
        grid = sim._build_realized_grid()
        assert (grid.shape[0]+DEVICES-1)//DEVICES == seam
    def position(cell):
        return tuple(float(grid.node_of(a, i)) for a, i in enumerate(cell))
    for k, (cell, component, extent, has_radius) in enumerate(ports):
        point = position(cell)
        assert tuple(map(int, field_index(grid, point, component))) == cell
        sim.add_port(point, component, extent=extent, impedance=50.,
                     radius=RADIUS if radius and has_radius else None,
                     excite=column is None or k == column,
                     waveform=GaussianPulse(f0=11e9, bandwidth=.9))
    component = ports[0][1]
    for cell in probes:
        point = position(cell)
        assert tuple(map(int, field_index(grid, point, component))) == cell
        sim.add_probe(point, component)
    return sim, grid, ports, probes, seam


def solve(sim, grid, *, distributed=False, entry='run', s=False, steps=None, eps=None):
    kwargs = dict(n_steps=int(np.ceil(.65e-9/grid.dt)) if steps is None else steps,
                  skip_preflight=True)
    if distributed:
        kwargs['devices'] = jax.devices('cpu')[:DEVICES]
    if entry == 'run':
        result = sim.run(**kwargs, compute_s_params=s, **({'s_param_freqs': FREQS} if s else {}))
    else:
        result = sim.forward(**kwargs, distributed=distributed, checkpoint=False,
                             **({'port_s11_freqs': FREQS} if s else {}),
                             **({'eps_override': eps} if eps is not None else {}))
    jax.block_until_ready(result.time_series)
    return result


@functools.lru_cache(maxsize=None)
def validity(mesh, boundary, name):
    """B2 is measured before reading this scene's multi-device result."""
    yes = scene(mesh, boundary, name)
    no = scene(mesh, boundary, name, radius=False)
    one, absent = (solve(*v[:2]) for v in (yes, no))
    a, b = np.asarray(one.time_series)[:, 0], np.asarray(absent.time_series)[:, 0]
    peak = float(np.max(np.abs(b)))
    ratio = float(np.max(np.abs(a.astype(float)-b)))/peak
    s_difference = None
    if name == 'R1':
        sa, sb = (np.asarray(solve(*v[:2], s=True).s_params) for v in (yes, no))
        s_difference = float(np.max(np.abs(sa.astype(np.complex128)-sb)))
    diagnostic('B2', f'{mesh}/{boundary}/{name}', field_ratio=ratio,
               s_difference=s_difference, minimum=.01)
    return ratio, s_difference


def require_valid_scene(mesh, boundary, name):
    ratio, delta_s = validity(mesh, boundary, 'R2' if name == 'R2' else 'R1')
    if ratio < .01 or (delta_s is not None and delta_s < .01):
        pytest.skip(f'B2 outside bar: {mesh}/{boundary}/{name}; '
                    f'field_ratio={ratio}, s_difference={delta_s}; frozen scene unchanged')


@pytest.mark.parametrize('mesh,boundary', MESHES if DEVICES == 2 else [])
@pytest.mark.parametrize('name', ['R1', 'R2'])
def test_radius_present(mesh, boundary, name):
    ratio, delta_s = validity(mesh, boundary, name)
    assert ratio >= .01, ('B2', mesh, boundary, name, ratio)
    if delta_s is not None:
        assert delta_s >= .01, ('B2', mesh, boundary, name, delta_s)


FIELD_CASES = ([(m, b, r) for m, b in MESHES for r in ('R1', 'R2', 'R3', 'R4')]
               if DEVICES == 2 else [(m, 'cpml', 'R4') for m in ('uniform', 'graded')])


@pytest.mark.parametrize('mesh,boundary,name', FIELD_CASES)
def test_probe_records(mesh, boundary, name):
    require_valid_scene(mesh, boundary, name)
    label = f'{mesh}/{boundary}/{name}'
    sim, grid, _, channels, _ = scene(mesh, boundary, name)
    many = {}
    failures = []
    for entry in ('run', 'forward'):
        one = solve(sim, grid, entry=entry)
        actual = solve(sim, grid, distributed=True, entry=entry)
        expected, got = np.asarray(one.time_series), np.asarray(actual.time_series)
        assert expected.shape == got.shape == (int(np.ceil(.65e-9/grid.dt)), len(channels))
        assert np.isfinite(expected).all() and np.isfinite(got).all()
        for n, channel in enumerate(channels):
            peak = np.float32(np.max(np.abs(expected[:, n])))
            assert peak > 0
            error = float(np.max(np.abs(got[:, n].astype(float)-expected[:, n])))
            ulps = error/float(np.spacing(peak))
            diagnostic('B1', label, entry=entry, channel=channel, peak=float(peak),
                       error=error, max_ULP=ulps, limit_ULP=9)
            if ulps > 9:
                failures.append(('B1', entry, channel, ulps))
        many[entry] = got
    if DEVICES == 2:
        a, b = many['run'], many['forward']
        unequal = np.argwhere(a != b)
        first = unequal[0].tolist() if len(unequal) else None
        equal = a.tobytes() == b.tobytes()
        diagnostic('B4', label, equal=equal, first_unequal=first,
                   samples=None if first is None else [float(a[tuple(first)]), float(b[tuple(first)])])
        if not equal:
            failures.append(('B4', first))
    assert not failures, failures


def expected_stencil(ports, family):
    expected = set()
    for axis, cell in declared_cells(ports):
        for transverse in set(range(3))-{axis}:
            component = 3-axis-transverse if family == 'mu_r_wire' else axis
            for offset in ((-1, 0) if family == 'mu_r_wire' else (-1, 1)):
                index = list(cell)
                index[transverse] += offset
                expected.add((component, *index))
    return expected


@pytest.mark.parametrize('mesh,boundary', MESHES if DEVICES == 2 else [])
@pytest.mark.parametrize('name', ['R1', 'R2'])
def test_realized_stencil(mesh, boundary, name):
    require_valid_scene(mesh, boundary, name)
    sim, grid, ports, _, width = scene(mesh, boundary, name)
    with _realized.capture() as dump:
        solve(sim, grid, distributed=True, steps=4)
    records = [r for r in dump.records if 'owned_start' in r and 'eps_e' in r]
    assert records
    for family in ('mu_r_wire', 'eps_r_lumped'):
        got, ownership_errors = set(), []
        for record in records:
            start, count = int(record['owned_start']), int(record['owned_count'])
            for component, array in enumerate(getattr(record['materials'], family)):
                if array is None:
                    continue
                for index in np.argwhere(np.asarray(array)[1:1+count] != 0):
                    cell = (int(index[0])+start, *map(int, index[1:]))
                    got.add((component, *cell))
                    if cell[0]//width != start//width:
                        ownership_errors.append((cell, start//width))
        expected = expected_stencil(ports, family)
        diagnostic('B3', f'{mesh}/{boundary}/{name}/{family}', equal=got == expected,
                   actual=sorted(got), expected=sorted(expected),
                   missing=sorted(expected-got), extra=sorted(got-expected),
                   ownership_errors=ownership_errors)
        assert got == expected and not ownership_errors


S_CASES = ([(m, b, r) for m, b in MESHES for r in ('R1', 'R3')]
           + [('uniform', 'cpml', 'R4')] if DEVICES == 2 else [('uniform', 'cpml', 'R4')])


@pytest.mark.parametrize('mesh,boundary,name', S_CASES)
def test_port_s(mesh, boundary, name):
    require_valid_scene(mesh, boundary, name)
    columns = range(2) if mesh == 'graded' and name == 'R3' else (None,)
    failures = []
    for column in columns:
        sim, grid, *_ = scene(mesh, boundary, name, column=column)
        label = f'{mesh}/{boundary}/{name}/column={column}'
        values = []
        for distributed in (False, True):
            try:
                result = solve(sim, grid, distributed=distributed,
                               entry='forward' if column is not None else 'run', s=True)
            except (NotImplementedError, ValueError) as exc:
                if column is None:
                    raise
                diagnostic('B5', label, status='not reachable', distributed=distributed,
                           message=str(exc))
                pytest.skip(f'not reachable: {exc}')
            value = np.asarray(result.s_params)
            values.append(value if column is None else value[:, column, :])
        a, b = values
        assert a.shape == b.shape
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        finite = np.isfinite(a)
        delta = float(np.max(np.abs(a.astype(np.complex128)[finite]-b[finite]), initial=0))
        diagnostic('B5', label, max_delta_s=delta, limit=1e-4, same_nan=same_nan,
                   one_real=a.real.tolist(), one_imag=a.imag.tolist(),
                   many_real=b.real.tolist(), many_imag=b.imag.tolist())
        if not same_nan or not np.isfinite(b[finite]).all() or delta > 1e-4:
            failures.append((label, delta, same_nan))
    assert not failures, failures


@pytest.mark.parametrize('boundary', ['cpml', 'pec'] if DEVICES == 2 else [])
def test_gradient(boundary):
    require_valid_scene('graded', boundary, 'R1')
    sim, grid, *_ = scene('graded', boundary, 'R1')
    values = []
    for distributed in (False, True):
        def objective(eps):
            result = solve(sim, grid, entry='forward', distributed=distributed,
                           steps=160, eps=jnp.ones(grid.shape)*eps)
            return jnp.log(jnp.mean(result.time_series**2))
        value, gradient = jax.value_and_grad(objective)(jnp.array(1.4))
        values.append((float(value), float(gradient)))
    one, many = values[0][1], values[1][1]
    relative = abs(many-one)/abs(one)
    diagnostic('B6', f'graded/{boundary}/R1', one_ad=one, many_ad=many,
               one_value=values[0][0], many_value=values[1][0], relative=relative, limit=1e-4)
    assert np.isfinite(values).all() and relative <= 1e-4


@contextmanager
def published_materials():
    """Observe the actual publisher of the public solve, without a second fold."""
    from rfx.stepping import slab_setup
    original = slab_setup.realize_materials
    records = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append(jax.tree.map(lambda a: np.asarray(a) if hasattr(a, 'shape') else a, result[0]))
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(slab_setup, 'realize_materials', observe)
        yield records


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_published_increments(mesh):
    # 52 nodes at 17.5 mm divide by 2. The next whole-cell length has
    # 53 nodes, so both two and three ranks have alignment rows.
    length_mm = 17.5
    while True:
        length = length_mm / 1000
        sim, grid, *_ = scene(mesh, 'cpml', 'R1', length=length)
        if all(grid.shape[0] % n for n in (2, 3)):
            break
        length_mm += .5
    assert length == .018
    for n in (2, 3):
        assert ((grid.shape[0]+n-1)//n)*n-grid.shape[0] > 0
    with _realized.capture() as one:
        solve(sim, grid, steps=4)
    raw = next(r['materials'] for r in one.records if 'eps_e' in r)
    e = next(r for r in one.records if 'eps_e' in r)
    h = next(r for r in one.records if 'mu_h' in r)
    oracle = {'mu_r_wire': raw.mu_r_wire, 'eps_r_lumped': raw.eps_r_lumped,
              'sigma_lumped': raw.sigma_lumped,
              'components.mu_wire': raw.mu_r_wire, 'components.eps_lumped': raw.eps_r_lumped,
              'components.sigma_lumped': raw.sigma_lumped,
              'components.mu_update': h['mu_h'], 'components.eps_update': e['eps_e'],
              'components.sigma_update': e['sigma_e']}
    with _realized.capture(), published_materials() as published:
        solve(sim, grid, distributed=True, steps=4)
    assert published, 'public multi-device solve did not publish realized materials'
    width = (grid.shape[0]+DEVICES-1)//DEVICES
    failures = []
    for materials in published:
        for name, expected in oracle.items():
            value = materials
            for part in name.split('.'):
                value = getattr(value, part)
            for axis, (got, ref) in enumerate(zip(value, expected, strict=True)):
                if got is None or ref is None:
                    assert got is None and ref is None
                    continue
                slabs = np.asarray(got).reshape((DEVICES, width+2, *grid.shape[1:]))
                fill = 1 if name in ('components.mu_update', 'components.eps_update') else 0
                owned_equal, outside_equal, outside_count = True, True, 0
                for rank in range(DEVICES):
                    rows = rank*width+np.arange(width+2)-1
                    outside = (rows < 0) | (rows >= grid.shape[0])
                    outside_count += int(np.count_nonzero(outside))
                    outside_equal &= bool(np.all(slabs[rank][outside] == fill))
                    count = min(width, grid.shape[0]-rank*width)
                    owned_equal &= np.array_equal(slabs[rank, 1:1+count], ref[rank*width:rank*width+count])
                diagnostic('P1b', f'{mesh}/cpml/{name}/{axis}', owned_equal=bool(owned_equal),
                           outside_equal=bool(outside_equal), outside_rows=outside_count, fill=fill,
                           nx=grid.shape[0], alignment=width*DEVICES-grid.shape[0])
                if not owned_equal or not outside_equal:
                    failures.append((name, axis, owned_equal, outside_equal))
    assert not failures, failures
