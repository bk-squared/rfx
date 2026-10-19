"""Graded DFT planes keep the one-device per-cell arithmetic."""
import pytest
from tests.unit.runners._graded_records_scene import model, coordinate, pair, measure, record, mutation_result, with_bars


def add_planes(sim, grid, seam):
    for axis, index in ((0, seam - 1), (0, seam), (1, grid.shape[1] // 2), (2, grid.shape[2] // 2)):
        for component in ('ez', 'hy'):
            sim.add_dft_plane_probe(axis='xyz'[axis], coordinate=coordinate(grid, axis, index),
                component=component, freqs=[3e9, 7e9], name=f'{axis}/{index}/{component}')


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_planes(boundary, n_devices, entry):
    sim, grid, seam = model(boundary, n_devices)
    add_planes(sim, grid, seam)
    one, many = pair(sim, n_devices, entry)
    values = {key: measure(many.dft_planes[key].accumulator, p.accumulator)
              for key, p in one.dft_planes.items()}
    # CPML wrapped step: compile-form rounding, operands byte-equal (rfx-archive DESIGN.md 4g amendment 1).
    with_bars(values, 'relative_peak' if boundary == 'cpml' else 'array_equal')
    record('planes', dict(boundary=boundary, devices=n_devices, entry=entry, shape=grid.shape), values)
    assert max(row['peak'] for row in values.values()) > 0
    assert all(row['passed'] for row in values.values()), ('plane bar', values)


@pytest.mark.parametrize('mutation', ['owned_rows', 'skip'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_plane_mutations(mutation, boundary, monkeypatch):
    from rfx.runners import _distributed_records as records
    if mutation == 'owned_rows':
        original = records._cut
        monkeypatch.setattr(records, '_cut', lambda axis, index, region, start, end, nx_per:
            original(axis, index, region, start + (axis != 0), end, nx_per))
    else:
        monkeypatch.setattr(records.PlaneRecords, 'update', lambda self, buffer, *a, **kw: buffer)
    with pytest.raises(AssertionError, match='plane bar') as failure:
        test_planes(boundary, 3, 'run')
    mutation_result('planes', f'{mutation}/{boundary}', failure.value)
