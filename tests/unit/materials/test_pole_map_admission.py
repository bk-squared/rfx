"""Per-pattern map corners and named refusals."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, LorentzPoleMap, Simulation, drude_pole_map
from rfx.core.yee import init_materials
from rfx.materials.lorentz import init_lorentz, lorentz_pole, lorentz_pole_coeffs
from rfx.model.pole_stability import judge_prebuilt_poles, pole_set_extremes
from tests.unit.materials.test_pole_map_coefficients import rig, WP
from tests.unit.materials.test_pole_map_runs import record


def test_p4_gridless_and_prebuilt():
    grid, xyz, mask = rig()
    pole = LorentzPoleMap(0., lambda x, y, z: 1e9 * (1 + x), WP**2)
    with pytest.raises(ValueError, match='LorentzPoleMap'):
        init_lorentz([pole], init_materials(grid.shape), grid.dt, mask=mask, high_wall=(True,) * 3)
    with pytest.raises(ValueError, match='LorentzPoleMap'):
        lorentz_pole_coeffs([pole], grid.dt, grid.shape, [None])
    coeffs, state = init_lorentz([pole], init_materials(grid.shape), grid.dt, mask=mask, grid=grid)
    with pytest.raises(ValueError, match='LorentzPoleMap'):
        judge_prebuilt_poles(grid, (coeffs, state))


@pytest.mark.parametrize('site', ['codec', 'design', 'material_fit', 'differentiable_material_fit', 'subgridded'])
def test_p4_named_refusals(site):
    pole = drude_pole_map(lambda x, y, z: WP + 0*x, 2e9)
    sim = Simulation(freq_max=6e9, domain=(.006,) * 3, dx=.0005, boundary='pec')
    sim.add_material('map', eps_r=1., lorentz_poles=[pole])
    sim.add(Box((.001,) * 3, (.005,) * 3), material='map')
    with pytest.raises((ValueError, NotImplementedError), match='LorentzPoleMap'):
        if site == 'codec':
            from rfx.interop._materials import material_to_dict
            material_to_dict(sim._materials['map'])
        elif site == 'design':
            from rfx.interop import design_to_dict
            design_to_dict(sim)
        elif site == 'material_fit':
            from rfx.material_fit import eval_lorentz
            eval_lorentz(np.array([3e9]), 1., [pole])
        elif site == 'differentiable_material_fit':
            from rfx.differentiable_material_fit import _poles_to_params
            _poles_to_params(1., [], [pole])
        else:
            from rfx.runners._admission import admit
            admit(sim, 'run_subgridded')


@pytest.mark.parametrize('field', ['delta', 'kappa'])
@pytest.mark.parametrize('bad', [-1., np.inf, np.nan])
def test_invalid_occupied_values_name_position(field, bad):
    grid, xyz, mask = rig(True)
    values = dict(omega_0=0., delta=1e9, kappa=WP**2)
    values[field] = lambda x, y, z: np.where(x < .002, bad, 1.)
    with pytest.raises(ValueError, match=rf'LorentzPoleMap {field} at .* m is'):
        init_lorentz([LorentzPoleMap(**values)], init_materials(grid.shape), grid.dt, mask=mask, grid=grid)


def _admission_scene(patterns=False, factor=.99):
    sim = Simulation(freq_max=6e9, domain=(.012,) * 3, dx=.0005, cpml_layers=8)
    grid = sim._build_grid()
    # Drude z=-1: kappa dt**2 + 4(c dt)**2 sum(1/d_a**2) = 4 eps_inf.
    budget = 299792458.**2 * sum(1 / float(np.min(grid.cells(a)))**2 for a in range(3))
    bound = (4 - 4 * grid.dt**2 * budget) / grid.dt**2
    high, low = (2. * bound, .5 * bound) if patterns else (factor * bound, .8 * bound)
    pole = LorentzPoleMap(0., 1e9, lambda x, y, z: np.where(x < .006, high, low))
    if patterns:
        sim.add_material('A', eps_r=10., lorentz_poles=[pole, lorentz_pole(.3, 2*np.pi*5e9, 1e9)])
        sim.add_material('B', eps_r=1., lorentz_poles=[pole])
        sim.add(Box((.004, .004, .004), (.006, .008, .008)), material='A')
        sim.add(Box((.006, .004, .004), (.008, .008, .008)), material='B')
    else:
        sim.add_material('map', eps_r=1., lorentz_poles=[pole])
        sim.add(Box((.004,) * 3, (.008,) * 3), material='map')
    sim.add_source((.00285, .00535, .00485), 'ez', waveform=GaussianPulse(f0=3e9))
    sim.add_probe((.00935, .00685, .00585), 'ez')
    return sim, grid, float(bound)


def test_p5b_corner_admission():
    stable, grid, bound = _admission_scene()
    result = stable.run(n_steps=2000, compute_s_params=False, skip_preflight=True)
    assert np.all(np.isfinite(result.time_series))
    unstable, _, _ = _admission_scene(factor=1.01)
    with pytest.raises(ValueError, match='LorentzPoleMap corner') as error:
        unstable.run(n_steps=2000, compute_s_params=False, skip_preflight=True)
    text = str(error.value)
    assert 'kappa=' in text and 'delta=' in text and 'largest stable step' in text
    assert '(0.00425, 0.00425, 0.00425)' in text
    record('P5b', dict(bound=bound, stable_kappa=.99*bound, refused_kappa=1.01*bound, message=text))


def test_p5c_per_pattern_corner():
    sim, grid, bound = _admission_scene(patterns=True)
    materials, _, spec, *_ = sim._assemble_materials(grid)
    records = pole_set_extremes(materials, spec)
    observed = sorted((float(r.eps_low), float(r.poles[0].kappa)) for r in records)
    assert observed == [(1., .5 * bound), (10., 2. * bound)]
    assert observed[1][1] > bound
    result = sim.run(n_steps=2000, compute_s_params=False, skip_preflight=True)
    assert np.all(np.isfinite(result.time_series))
    record('P5c', dict(bound=bound, corners=observed))
