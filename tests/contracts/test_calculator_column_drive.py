"""#2715 B1–B3: calculator columns survive passive run declarations (A1/A2)."""

from dataclasses import replace
import json
import time

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid, coords_from_uniform_grid
from tests.unit.sparams.test_msl_internal_probe_advisories import _thru


H = .00127
FREQS = np.arange(70, 80) * 1e8


def _guide(mesh, scene, amplitude):
    profile = np.r_[np.full(32, H), np.full(40, .8 * H), np.full(32, H)]
    sim = Simulation(
        freq_max=7.9e9, domain=(.12192, .02286, .01016), dx=H,
        boundary=dict(x='cpml', y='pec', z='pec'), cpml_layers=8,
        **({'dx_profile': profile} if mesh == 'graded' else {}))
    sim.add_material('dielectric', eps_r=2.2)
    sim.add(Box((.06096, 0., 0.),
                (.12192 if scene == 'S1' else .08128, .02286, .01016)),
            material='dielectric')
    for name, x, direction, amp in (
        ('p1', .02032, '+x', 1.), ('p2', .10160, '-x', amplitude),
    ):
        sim.add_waveguide_port(x, direction=direction, amplitude=amp, name=name,
                               f0=7.45e9, freqs=FREQS, ref_offset=2, probe_offset=3)
    return sim


@pytest.mark.parametrize('mesh,scene,normalize', [
    ('uniform', 'S1', False), ('uniform', 'S2', True), ('graded', 'S2', True),
    ('uniform', 'S2', 'flux'), ('graded', 'S2', 'flux'),
])
def test_b1_b2_waveguide(mesh, scene, normalize):
    started = time.monotonic()
    matrices = {}
    for amplitude in (1., 0., 2.):
        sim = _guide(mesh, scene, amplitude)
        matrices[amplitude] = np.asarray(sim.compute_waveguide_s_matrix(
            n_steps=1024, normalize=normalize, skip_preflight=True).s_params)
    grid = sim._build_nonuniform_grid(cpml_axes='x') if mesh == 'graded' else sim._build_grid()
    coords = (coords_from_nonuniform_grid(grid) if mesh == 'graded'
              else coords_from_uniform_grid(grid))
    faces = [.06096, .12192 if scene == 'S1' else .08128]
    realized = [float(coords.x[np.argmin(np.abs(np.asarray(coords.x) - x))]) for x in faces]
    print('COLUMN_DRIVE ' + json.dumps(dict(
        case=f'{mesh}-{scene}', steps=1024, shape=list(grid.shape),
        block_faces_m=realized, wall_s=time.monotonic() - started)), flush=True)
    assert np.iscomplexobj(matrices[1.]) and np.isfinite(matrices[1.]).all()
    assert np.array_equal(matrices[0.], matrices[1.]), 'B1/B2 zero declaration equals all-driven S bit for bit'
    assert np.max(np.abs(matrices[2.] - matrices[1.])) <= 1e-6 * np.max(np.abs(matrices[1.])), 'B1/B2 doubled declaration rounding bar'


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_b3_msl(mesh):
    started = time.monotonic()
    matrices = {}
    for excite in (True, False):
        sim = _thru()
        if mesh == 'graded':
            sim._dx_profile = np.full(60, 2e-4)
        sim._msl_ports[1] = replace(sim._msl_ports[1], excite=excite)
        matrices[excite] = np.asarray(sim.compute_msl_s_matrix(
            n_steps=3934, freqs=np.linspace(2e9, 18e9, 9)).S)
    print('COLUMN_DRIVE ' + json.dumps(dict(
        case=f'MSL-{mesh}', steps=3934,
        s12_abs=np.abs(matrices[False][0, 1]).tolist(),
        wall_s=time.monotonic() - started)), flush=True)
    assert np.iscomplexobj(matrices[True]) and np.isfinite(matrices[True]).all()
    assert np.array_equal(matrices[False], matrices[True]), 'B3 passive declaration equals all-driven S bit for bit'
    assert (np.abs(matrices[False][0, 1]) > .9).all(), 'B3 measured |S12| > 0.9 at every bin'
