"""Hand arithmetic, same-process main control, and closed-box energy for the floor."""
import subprocess
from unittest.mock import patch

import numpy as np

from rfx import GaussianPulse, Simulation, Sphere
from rfx.geometry.conformal_cells import compute_conformal_fractions, conformal_b_floor
from rfx.grid import Grid


def test_hand_faces():
    for normal in range(3):
        u, v = [a for a in range(3) if a != normal]
        lengths = [np.zeros((3, 3, 3)) for _ in range(3)]
        areas = [np.ones((3, 3, 3)) for _ in range(3)]
        base = (1, 1, 1)
        hi_u, hi_v = list(base), list(base)
        hi_u[u], hi_v[v] = 2, 2
        lengths[u][base], lengths[v][base] = .3, .2
        lengths[u][tuple(hi_v)], lengths[v][tuple(hi_u)] = 1., .5
        areas[normal][base] = .1
        areas[normal][0, 0, 0] = 0.
        original = [a.copy() for a in lengths]
        for q, effective in ((.5, .25), (.25, .1)):
            actual = conformal_b_floor(lengths, areas, q)
            expected = [a.copy() for a in areas]
            expected[normal][base] = effective
            for got, want in zip(actual, expected):
                np.testing.assert_array_equal(got, want)
        # At q=.25 the floor bound is .0625; a .1 face remains .1.
        areas[normal][base] = .01
        assert conformal_b_floor(lengths, areas, .25)[normal][base] == .0625
        for got, want in zip(lengths, original):
            np.testing.assert_array_equal(got, want)


def test_default_operands_and_dt_match_main_in_process():
    old = {}
    for name in ('grid', 'geometry/conformal_cells'):
        source = subprocess.check_output(
            ['git', 'show', f'fa8e0eaf5:rfx/{name}.py'], text=True)
        namespace = {'__name__': 'rfx.' + name.replace('/', '.')}
        exec(compile(source, f'main:{name}', 'exec'), namespace)
        old[name] = namespace
    sphere = Sphere((.0061, .0058, .0057), .0023)
    for mode in ('3d', '2d_tmz', '2d_tez'):
        kwargs = dict(freq_max=20e9, domain=(.012,) * 3, dx=.001,
                      cpml_layers=0, mode=mode)
        grid, before = Grid(**kwargs), old['grid']['Grid'](**kwargs)
        assert np.float64(grid.dt).tobytes() == np.float64(before.dt).tobytes()
        shapes = [sphere if mode == '3d' else Sphere((.0061, .0058, 0.), .0023)]
        current = compute_conformal_fractions(grid, shapes)
        previous = old['geometry/conformal_cells']['compute_conformal_fractions'](before, shapes)
        for arrays, baseline in zip(current, previous):
            for got, want in zip(arrays, baseline):
                np.testing.assert_array_equal(np.asarray(got).view(np.uint32),
                                              np.asarray(want).view(np.uint32))


def test_uniform_modes_use_realized_pin_and_raw_edges():
    from rfx.model import tensor_operands
    for mode in ('3d', '2d_tmz', '2d_tez'):
        for q in (.5, .25):
            ordinary = .001 / (299792458. * np.sqrt(3. if mode == '3d' else 2.)) * .99
            sim = Simulation(20e9, (.012,) * 3, dx=.001, boundary='pec', mode=mode,
                             dt=q * ordinary)
            sphere = Sphere((.0061, .0058, .0057 if mode == '3d' else 0.), .0023)
            sim.add(sphere, material='pec')
            component = 'ex' if mode == '2d_tez' else 'ez'
            sim.add_source((.003, .004, .003 if mode == '3d' else 0.), component,
                           amplitude_kind='field')
            captured = []
            real = tensor_operands.form_tensor_operands
            def capture(*args, **kwargs):
                values = real(*args, **kwargs)
                captured.append(values)
                return values
            with patch.object(tensor_operands, 'form_tensor_operands', capture):
                result = sim.run(n_steps=4, conformal_pec='cut_cell_floor',
                                 compute_s_params=False, skip_preflight=True)
            assert result.dt == q * ordinary
            grid = sim._build_grid()
            raw = compute_conformal_fractions(grid, [sphere], clamp=False)[0]
            for got, want in zip(captured[0][2], raw):
                np.testing.assert_array_equal(got, want)
            assert all(np.isfinite(np.asarray(getattr(result.state, c))).all()
                       for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))


def test_closed_box_energy():
    dx, length = .00075, .03
    sphere = Sphere((.01512, .01493, .01507), .009)
    def run(n):
        sim = Simulation(12e9, (length,) * 3, dx=dx, boundary='pec',
                         dt=.5 * Grid.courant_dt(dx))
        sim.add(sphere, material='pec')
        sim.add_source((.004, .0052, .0233), 'ez', amplitude_kind='current',
                       waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        result = sim.run(n_steps=n, conformal_pec='cut_cell_floor',
                         compute_s_params=False, skip_preflight=True)
        fields = {c: np.asarray(getattr(result.state, c), dtype=np.float64)
                  for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')}
        assert all(np.isfinite(a).all() for a in fields.values())
        return sim._build_grid(), fields
    # The energy weights come from the actual E/H operands; the expected
    # conservation bound is independent of the geometry/floor calculation.
    energies = []
    for n in (1000, 2000):
        grid, a = run(n)
        _, b = run(n + 1)
        lengths, areas = compute_conformal_fractions(grid, [sphere], conformal_pec='cut_cell_floor')
        eps = [np.where(v > 0, 1. / np.where(v > 0, v, 1.), 0.)
               for v in map(lambda x: np.asarray(x, np.float64), lengths)]
        mu = [np.where((s > 0) & (s < 1), s, 1.)
              for s in map(lambda x: np.asarray(x, np.float64), areas)]
        we = 8.854187817e-12 * sum(np.sum(e * a[c]**2) for e, c in zip(eps, ('ex', 'ey', 'ez')))
        wh = 4e-7 * np.pi * sum(np.sum(m * a[c] * b[c]) for m, c in zip(mu, ('hx', 'hy', 'hz')))
        energies.append(we + wh)
    assert energies[0] > 0
    assert abs(energies[1] / energies[0] - 1.) < 1e-5
