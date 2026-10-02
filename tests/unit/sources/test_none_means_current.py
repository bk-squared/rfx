"""Declaration defaults and ADI current units (#1373)."""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._source_semantics import resolve_amplitude_kind


def simulation(**kwargs):
    return Simulation(freq_max=1e10, domain=(.01, .01, .01), dx=.001,
                      boundary='pec', **kwargs)


def test_declaration_resolves_once_and_warns_once():
    sim = simulation()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        sim.add_source((.005, .005, .005))
        sim.add_source((.006, .005, .005), amplitude_kind=None)
    assert [p.amplitude_kind for p in sim._ports] == ['current', 'current']
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert "now means 'current'" in str(caught[0].message)
    assert 'on every path' in str(caught[0].message)
    assert resolve_amplitude_kind('field') == 'field'
    assert resolve_amplitude_kind('current') == 'current'
    with pytest.raises(ValueError):
        sim.add_source((.005, .005, .005), amplitude_kind='bad')
    assert len(sim._ports) == 2


def test_config_interop_and_polarized_declarations_resolve():
    from rfx.config.loader import simulation_from_dict
    from rfx.interop import design_to_dict, simulation_from_design

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        sim = simulation_from_dict({
            'frequency': {'freq_max': 1e10}, 'domain': [.01, .01, .01],
            'dx': .001, 'boundary': 'pec',
            'sources': [{'type': 'source', 'position': [.005, .005, .005]}],
        })
        assert sim._ports[0].amplitude_kind == 'current'
        document = design_to_dict(sim)
        document['excitations']['soft_sources'][0]['amplitude_kind'] = None
        restored = simulation_from_design(document)
        assert restored._ports[0].amplitude_kind == 'current'
        assert document['excitations']['soft_sources'][0]['amplitude_kind'] is None
        polarized = simulation()
        polarized.add_polarized_source((.005, .005, .005), polarization='slant45')
        assert [p.amplitude_kind for p in polarized._ports] == ['current', 'current']


@pytest.mark.parametrize('mode', ['3d', '2d_tmz'])
def test_adi_current_agrees_with_yee_as_dt_decreases(mode):
    residuals = []
    for factor in (.05, .025):
        traces = {}
        for solver in ('yee', 'adi'):
            sim = Simulation(freq_max=1e10, domain=(.01, .01, .01 if mode == '3d' else .001),
                             dx=.001, boundary='pec', mode=mode, solver=solver,
                             adi_cfl_factor=1)
            build = sim._build_grid
            def grid():
                g = build()
                g.dt *= factor
                return g
            sim._build_grid = grid
            z = .005 if mode == '3d' else 0
            sim.add_source((.005, .005, z), 'ez',
                           waveform=lambda t: jnp.sin(t / 2e-11)**2,
                           amplitude_kind='current')
            sim.add_probe((.006, .005, z), 'ez')
            result = sim.run(n_steps=round(8 / factor), compute_s_params=False,
                             skip_preflight=True)
            traces[solver] = np.asarray(result.time_series)
        residual = np.max(np.abs(traces['adi'] - traces['yee'])) / np.max(np.abs(traces['yee']))
        residuals.append(residual)
    # Same dt/epsilon drive in vacuum. ADI injects before its split step;
    # Yee injects after E. The time-stagger/splitting residual shrinks with dt.
    assert residuals[-1] < .005, residuals
    assert residuals[-1] < .6 * residuals[0], residuals


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("kind", [None, "current"])
@pytest.mark.parametrize("sigma", [5., 50.])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_adi_refuses_current_in_lossy_source_cell(mode, kind, sigma, entry):
    from rfx import Box

    sim = Simulation(freq_max=1e10, domain=(.01, .01, .01 if mode == "3d" else .001),
                     dx=.001, boundary="pec", mode=mode, solver="adi")
    sim.add_material("loss", eps_r=4., sigma=sigma)
    sim.add(Box((.003, .003, 0.), (.008, .008, .009)), material="loss")
    sim.add_source((.005, .005, .005 if mode == "3d" else 0.),
                   amplitude_kind=kind)
    with pytest.raises(ValueError, match=(
            "amplitude unverified in a lossy cell on ADI; "
            "declare it in a lossless cell, or use the Yee solver")):
        getattr(sim, entry)(n_steps=2, skip_preflight=True)


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("dx", [.001, .002, .003])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_adi_cell_volume_lookup_bit_identity(mode, dx, boundary):
    sim = Simulation(freq_max=1e10, domain=(.03, .03, .03 if mode == "3d" else dx),
                     dx=dx, boundary=boundary, mode=mode, solver="adi")
    grid = sim._build_grid()
    old = np.float64(grid.dx**3)
    for i in (0, grid.shape[0] // 2, grid.shape[0] - 1):
        new = np.float64(float(grid.cells("x")[i])**3)
        assert old.tobytes() == new.tobytes()
