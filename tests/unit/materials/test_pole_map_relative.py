"""Stage B declaration, concrete coefficients, SI admission and summary pins."""
import itertools

import jax
import numpy as np
import pytest

from rfx import Box, LorentzPoleMap, RelativeStrength, Simulation, drude_pole_map, lorentz_pole_map
from rfx.artifacts import _material_summary
from rfx.materials.pole_map import sample_spec
from tests.unit.materials.test_pole_map_coefficients import EPS0, WP, hand_mean, init, rig
from tests.unit.materials.test_pole_map_runs import record


def test_b1_declarations():
    grid, xyz, mask = rig()
    for value in (1.4, lambda x, y, z: 1 + 40*x):
        direct = LorentzPoleMap(0., 1e9, RelativeStrength(float(WP**2), value))
        drude = drude_pole_map(float(WP), 2e9, density=value)
        lorentz = lorentz_pole_map(value, float(WP), 1e9, relative=True)
        expected = np.broadcast_to(value(*xyz) if callable(value) else value, grid.shape)
        for pole in (direct, drude, lorentz):
            assert isinstance(pole.kappa, RelativeStrength)
            assert pole.kappa.reference == float(WP**2)
            assert pole.kappa.value is value
            sampled = sample_spec(([pole], [mask]), grid, grid.shape)[0][0]
            assert isinstance(sampled.kappa, RelativeStrength)
            assert sampled.kappa.reference == float(WP**2)
            np.testing.assert_allclose(sampled.kappa.value[mask], expected[mask], rtol=0, atol=0)
    with pytest.raises(ValueError, match='drude_pole_map'):
        lorentz_pole_map(1., 0., 1e9, relative=True)
    record('B1', dict(declarations=6, reference=float(WP**2), zero_resonance_refused=True))


def test_b5_artifact_summary():
    summaries = []
    for kappa in (WP**2, RelativeStrength(float(WP**2), 1.)):
        sim = Simulation(freq_max=6e9, domain=(.006,)*3, dx=.0005)
        sim.add_material('map', eps_r=1., lorentz_poles=[LorentzPoleMap(0., 1e9, kappa)])
        summaries.append(_material_summary(sim._materials['map']))
    for summary in summaries:
        assert summary['lorentz_poles'] == dict(present=True, count=1)
        assert summary['debye_poles'] == dict(present=False, count=0)
    assert summaries[0] == summaries[1]
    record('B5-artifact', dict(summaries=summaries))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_b6_5_relative_coefficients(graded, monkeypatch):
    import rfx.materials.lorentz as lorentz
    grid, xyz, mask = rig(graded)
    x, y, z = xyz
    value = np.broadcast_to(1 + 40*x + 20*y, grid.shape)
    delta = np.broadcast_to(np.where(x < .0025, 1e9, 2e9) + 1e10*y, grid.shape)
    pole = LorentzPoleMap(0., lambda x,y,z: np.where(x < .0025, 1e9, 2e9) + 1e10*y,
                         RelativeStrength(float(WP**2), lambda x,y,z: 1 + 40*x + 20*y))
    captured = []
    core = lorentz.pole_update_coeffs
    def observe(omega, damping, strength, dt, field_dtype=None):
        if np.ndim(damping) == 3:
            captured.append(np.asarray(damping))
        return core(omega, damping, strength, dt, field_dtype)
    monkeypatch.setattr(lorentz, 'pole_update_coeffs', observe)
    coeffs, _ = init(pole, grid, mask)
    assert len(captured) == 3
    worst = dict(delta_ulp=0., damping_spacings=0, c_spacings=0)
    for comp, cell in itertools.product(range(3), itertools.product(range(3, 10), repeat=3)):
        ve = hand_mean(mask*value, grid, comp, cell)
        de = hand_mean(mask*value*delta, grid, comp, cell) / ve
        want_delta = np.float32(de)
        d = np.float64(np.float32(captured[comp][cell]))
        ulp = abs(d - float(want_delta)) / abs(float(np.spacing(want_delta)))
        worst['delta_ulp'] = max(worst['delta_ulp'], ulp)
        assert ulp <= 2
        t = np.float64(np.float32(grid.dt))
        wanted = (np.float32(2*d*t/(1+d*t)),
                  np.float32(EPS0*float(WP**2)*ve*float(grid.dt)**2/(1+de*float(grid.dt))))
        actual = (np.float32(np.asarray(coeffs.damping[comp])[(0,)+cell]),
                  np.float32(np.asarray(coeffs.c[comp])[(0,)+cell]))
        for label, got, want, bar in zip(('damping_spacings', 'c_spacings'), actual, wanted, (4, 12)):
            assert np.signbit(got) == np.signbit(want)
            spacings = abs(int(got.view(np.uint32))-int(want.view(np.uint32)))
            worst[label] = max(worst[label], spacings)
            assert spacings <= bar, (comp, cell, label, spacings)
    record(f'B6.5-{graded}', worst)


def test_b6_6_stability_si():
    from tests.unit.materials.test_pole_map_admission import _admission_scene
    from rfx.model.pole_stability import courant_budget, largest_stable_step, pole_set_extremes
    admitted = []
    for relative in (False, True):
        sim, grid, bound = _admission_scene()
        if relative:
            pole = sim._materials['map'].lorentz_poles[0]
            reference = float(bound / 2)
            sim._materials['map'].lorentz_poles[:] = [LorentzPoleMap(0., pole.delta,
                RelativeStrength(reference, lambda x,y,z: np.where(x < .006, .99*bound/reference, .8*bound/reference)))]
            assert .99*bound/reference != 1
        materials, _, spec, *_ = sim._assemble_materials(grid)
        records = pole_set_extremes(materials, spec)
        assert len(records) == 1
        record_ = records[0]
        assert all(isinstance(v, (float, np.floating)) for p in record_.poles for v in p)
        budget = courant_budget(grid).local_axes(record_.occupied)
        # Bracket at the vacuum CFL limit so the result is the admitted pole
        # bound, not the unchanged supplied grid.dt on two already-stable arms.
        upper = 1 / np.sqrt(budget)
        step = largest_stable_step(record_.eps_low, record_.sigma_low, record_.poles, upper, budget)
        assert step is not None
        if not relative:
            assert step < upper
        admitted.append(step)
    difference = abs(admitted[0]-admitted[1])/abs(admitted[0])
    record('B6.6', dict(si_strength_bound=bound, admitted_steps=admitted, bracket=upper,
           relative_difference=difference, largest_value=1.98))
    assert difference <= 1e-12


def test_b6_7_named_refusals():
    from rfx.runners._admission import admit
    pole = drude_pole_map(float(WP), 2e9, density=lambda x,y,z: 1+40*x)
    sim = Simulation(freq_max=6e9, domain=(.006,)*3, dx=.0005, boundary='pec')
    sim.add_material('map', eps_r=1., lorentz_poles=[pole])
    sim.add(Box((.001,)*3, (.005,)*3), material='map')
    messages = {}
    for lane in ('run_subgridded', 'material_fit', 'run_distributed', 'fwd_distributed',
                 'run_distributed_nu', 'fwd_distributed_nu'):
        with pytest.raises((ValueError, NotImplementedError), match='LorentzPoleMap') as error:
            admit(sim, lane)
        messages[lane] = str(error.value)
    record('B6.7', messages)
