"""Centre-sampled material cells make dual_average equal on live E edges."""
import numpy as np
import pytest

from rfx import Box
from rfx.core.yee import edge_averaged_materials
from rfx.geometry.rasterize_grid import unpadded_high_walls
from rfx.model.materials import assemble_cells, electric_cell_sizes
from rfx.runners.nonuniform import assemble_interface_eps_nu
from tests.unit.runners import test_path_disposition_cells as paths


def _scene(kind, on):
    widths = np.array([1, 1, 1, 1, 1, .5, 1.5, .5, 1.5, 1, 1, 1]) if kind == 'graded' else np.ones(12)
    sim = paths._base('run_nonuniform', source=None,
                      interface_eps='dual_average' if on else 'sampled',
                      dx_profile=widths * 1e-3)
    lo, hi = ((5.3, 7.7) if kind == 'graded' else (5.5, 7.5))
    paths._block(sim, lo=(lo, 3, 3), hi=(hi, 9, 9), eps_r=4.)
    if kind == 'pec':
        sim.add(Box(paths.mm(3.5, 3, 3), paths.mm(5.5, 9, 9)), material='pec')
    # Source and the existing x=8 mm probe are both outside the PEC drawing.
    sim.add_source(paths.mm(6, 6, 6), 'ez', waveform=paths.WAVEFORM)
    return sim, widths, (lo, hi)


@pytest.mark.parametrize('kind', ['constant', 'graded', 'pec'])
def test_component_eps_equal_on_live_edges(kind):
    sim, x_widths, (lo, hi) = _scene(kind, True)
    grid = sim._build_nonuniform_grid()
    materials = assemble_cells(sim, grid)[0]
    wall = unpadded_high_walls(grid)
    sampled = edge_averaged_materials(materials.eps_r, materials.sigma,
                                     high_wall=wall, cell_sizes=electric_cell_sizes(grid))[0]
    dual = assemble_interface_eps_nu(sim, grid, materials, high_wall=wall)
    # Independent drawing, cell centres, and four-cell edge ownership, in mm.
    widths = (np.r_[x_widths, x_widths[-1]], np.ones(13), np.ones(13))
    centres = [np.r_[0., np.cumsum(w[:-1])] + w / 2 for w in widths]
    x, y, z = np.meshgrid(*centres, indexing='ij')
    dielectric = ((x >= lo - 1e-12) & (x < hi - 1e-12)
                  & (y >= 3) & (y < 9) & (z >= 3) & (z < 9))
    eps = np.where(dielectric, 4., 1.)
    pec = ((x >= 3.5 - 1e-12) & (x < 5.5 - 1e-12)
           & (y >= 3) & (y < 9) & (z >= 3) & (z < 9)) if kind == 'pec' else np.zeros_like(dielectric)
    for component, (a, b) in enumerate(zip(sampled, dual)):
        expected, blocked = eps.copy(), pec.copy()
        for axis in range(3):
            if axis == component:
                continue
            lower = np.maximum(np.arange(13) - 1, 0)
            shape = [1, 1, 1]
            shape[axis] = 13
            w = widths[axis]
            fraction = (w / (w + w[lower])).reshape(shape)
            below = np.take(expected, lower, axis=axis)
            expected = below + (expected - below) * fraction
            blocked |= np.take(blocked, lower, axis=axis)
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a[~blocked], expected.astype(a.dtype)[~blocked])
        np.testing.assert_array_equal(b[~blocked], a[~blocked])
        print(f'{kind} component {component}: live max difference={np.max(abs(a[~blocked]-b[~blocked]))}')


@pytest.mark.parametrize('lane', ['run_nonuniform', 'fwd_nonuniform'])
def test_pec_adjacent_probe_record_equals_default(lane):
    records = []
    for on in (False, True):
        sim, _, _ = _scene('pec', on)
        records.append(paths._probe(paths._run(sim, lane, paths.Feature(lambda *args: sim))))
    peak = np.float32(max(np.max(abs(v)) for v in records))
    assert peak > 0, 'the probe outside PEC must receive a live signal'
    tolerance = 9 * float(np.spacing(peak))
    difference = float(np.max(abs(records[0] - records[1])))
    assert difference <= tolerance
    print(f'{lane}: peak={peak!r}, max difference={difference!r}, nine float32 ULP={tolerance!r}')
