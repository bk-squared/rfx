"""The #1070 declared-span rule belongs to the solve object on every lane."""
import math

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.rasterize_grid import PadFillShortfall
from rfx.model.conductors import realized_conductors
from tests.unit.geometry.test_declared_span_reaches_padded_face import (
    _rig, _size_by_plain_ceil, DX, H, EPS_R, CPML,
)


def _graded_rig(monkeypatch):
    domain = (58 * H, 43 * H, 16 * H)
    profiles = {}
    for axis, length in zip('xyz', domain):
        widths = np.full(math.ceil(length / DX), DX)
        widths[3:5] = (.9 * DX, 1.1 * DX)
        profiles[f'd{axis}_profile'] = widths
    # Restore one excess grid cell, equivalent to the uniform plain-ceil bug,
    # without declaring that excess cell as part of the physical domain.
    oversized = profiles['dx_profile'].copy()
    profiles['dx_profile'] = oversized[:-1]
    sim = Simulation(freq_max=10e9, domain=domain, dx=DX,
                     boundary='cpml', cpml_layers=CPML, **profiles)
    domain = sim._unresolved_domain
    from rfx.nonuniform import make_nonuniform_grid
    grid = make_nonuniform_grid(domain[:2], profiles['dz_profile'], DX, CPML,
                               dx_profile=oversized, dy_profile=profiles['dy_profile'])
    monkeypatch.setattr(sim, '_build_nonuniform_grid', lambda: grid)
    sim.add_material('ro4003c', eps_r=EPS_R)
    sim.add(Box((0, 0, 6 * H), (domain[0], domain[1], 8 * H)), material='ro4003c')
    return sim


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('method', ['run', 'forward'])
@pytest.mark.parametrize('skip_preflight', [False, True])
def test_solve_refuses_shortfall(monkeypatch, graded, distributed, method, skip_preflight):
    if method == 'forward' and distributed and not graded:
        pytest.skip('Uniform distributed forward is not implemented by the API')
    _size_by_plain_ceil(monkeypatch)
    sim = _graded_rig(monkeypatch) if graded else _rig(10)
    kwargs = dict(n_steps=1, skip_preflight=skip_preflight)
    if distributed:
        assert len(jax.devices()) >= 2
        kwargs['devices'] = jax.devices()[:2]
        if method == 'forward':
            kwargs['distributed'] = True
    if method == 'run':
        kwargs['compute_s_params'] = False
    with pytest.raises(PadFillShortfall, match="ro4003c.*x-hi.*2 interior nodes short"):
        getattr(sim, method)(**kwargs)


@pytest.mark.parametrize('graded', [False, True])
def test_audits_report_same_shortfall(monkeypatch, graded):
    _size_by_plain_ceil(monkeypatch)
    sim = _graded_rig(monkeypatch) if graded else _rig(10)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    root = realized_conductors(sim, grid, nonuniform=graded, mode='audit')
    row, = root.pad_fill_findings
    assert (row['face'], row['empty_interior_nodes']) == ('x-hi', 2)
    mask = root.geometry_masks[id(sim._geometry[0])]
    profile = np.asarray(mask).any(axis=(1, 2))[CPML:-CPML]
    assert len(profile) - 1 - np.flatnonzero(profile)[-1] == 2
    assert np.all(np.asarray(root.materials.eps_r)[-CPML:] == 1)
    widths = np.asarray(grid.cells(0))[CPML:-CPML-1]
    expected = np.interp(sim._unresolved_domain[0], np.r_[0, np.cumsum(widths)],
                         np.arange(len(widths) + 1))
    assert row['declared_cells'] == pytest.approx(expected)
    report = sim.preflight()
    assert sum(i.code == 'declared-span-short-of-padded-face' for i in report) == 1
    record = sim.realized_geometry()
    assert len(record.pad_fill_findings) == 1
    fidelity = sim.fidelity_report(print_report=False)
    hits = [f for item in fidelity for f in item.get('findings', [])
            if f.get('kind') == 'declared-span-short-of-padded-face']
    assert len(hits) == 1


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('method', ['run', 'forward'])
def test_exact_face_runs(graded, distributed, method):
    if method == 'forward' and distributed and not graded:
        pytest.skip('Uniform distributed forward is not implemented by the API')
    # Small exact domain; the documented half-open one-node gap is repaired.
    profiles = {}
    if graded:
        for axis, n in zip('xyz', (12, 8, 8)):
            p = np.full(n, DX)
            p[3:5] = (.9 * DX, 1.1 * DX)
            profiles[f'd{axis}_profile'] = p
    sim = Simulation(freq_max=10e9, domain=(12*DX, 8*DX, 8*DX), dx=DX,
                     boundary='cpml', cpml_layers=2, **profiles)
    sim.add_material('slab', eps_r=EPS_R)
    sim.add(Box((0, 0, 2*DX), (12*DX, 8*DX, 4*DX)), material='slab')
    kw = dict(n_steps=1, skip_preflight=True)
    if distributed:
        assert len(jax.devices()) >= 2
        kw['devices'] = jax.devices()[:2]
        if method == 'forward':
            kw['distributed'] = True
    if method == 'run':
        kw['compute_s_params'] = False
    getattr(sim, method)(**kw)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    root = realized_conductors(sim, grid, nonuniform=graded)
    assert not root.pad_fill_findings
    assert np.asarray(root.materials.eps_r)[-1, 5, 5] == pytest.approx(EPS_R)


@pytest.mark.parametrize("graded", [False, True])
def test_builder_checks_once(monkeypatch, graded):
    import rfx.geometry.rasterize_grid as raster
    import rfx.model.pad_fill as pad_fill
    calls = []
    original = raster.assert_declared_span_is_filled
    def spy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(raster, 'assert_declared_span_is_filled', spy)
    monkeypatch.setattr(pad_fill, 'assert_declared_span_is_filled', spy)
    import rfx.model.materials as materials_module
    monkeypatch.setattr(materials_module, 'assert_declared_span_is_filled', spy)
    sim = _graded_rig(monkeypatch) if graded else _rig(10)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    realized_conductors(sim, grid, nonuniform=graded, mode='audit')
    assert calls == ['ro4003c']


def test_declared_cells_uses_local_widths():
    from types import SimpleNamespace
    from rfx.geometry.rasterize_grid import assert_declared_span_is_filled
    widths = np.array([.2, .2, .4, .6, .8, 1., 1., 1.])
    grid = SimpleNamespace(pad_x_lo=1, pad_x_hi=1, pad_y_lo=0, pad_y_hi=0,
                           pad_z_lo=0, pad_z_hi=0, cells=lambda axis: widths)
    mask = np.zeros((8, 3, 3), dtype=bool)
    mask[1:3] = True
    rows = []
    assert_declared_span_is_filled('graded', Box((0, 0, 0), (.8, 1, 1)),
                                   mask, grid, (.8, 1, 1), record=rows)
    row, = rows
    assert row['empty_interior_nodes'] == 4
    assert row['declared_cells'] == pytest.approx(2 + (.8 - .6) / .6)
