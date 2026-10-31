"""Default conformal operands and the diagnostic material gradient through forward."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry import cell_table


STEPS = 64


def scene(eps=2.5, *, conformal=True):
    sim = Simulation(20e9, (.012,) * 3, dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x='pec', y=Boundary('pec', 'pec', conformal=conformal), z='pec'))
    sim.add_material('host', eps_r=eps)
    sim.add(Box((.0013, .0012, .0011), (.0107, .0108, .0106)), material='host')
    sim.add(Sphere((.0061, .0058, .0057), .0023), material='pec')
    sim.add_source((.003, .004, .003), 'ez', amplitude_kind='current')
    sim.add_probe((.008, .004, .003), 'ez')
    return sim


def objective(eps):
    result = scene(eps).forward(n_steps=STEPS, skip_preflight=True)
    return jnp.sum(jnp.abs(result.time_series) ** 2)


def test_material_gradient_and_fresh_model_fd(monkeypatch):
    original = cell_table.build_cell_table
    calls = []

    def counted(grid, shapes):
        from rfx.geometry.smoothing import _shape_parameters
        assert not any(isinstance(value, jax.core.Tracer)
                       for shape in shapes for value in _shape_parameters(shape))
        calls.append(1)
        return original(grid, shapes)

    monkeypatch.setattr(cell_table, 'build_cell_table', counted)
    ad = float(jax.grad(objective)(2.5))
    assert len(calls) == 1
    values = []
    for eps in (2.525, 2.475, 2.5125, 2.4875):
        before = len(calls)
        values.append(float(objective(eps)))
        assert len(calls) == before + 1
    fd = (values[0] - values[1]) / .05
    half = (values[2] - values[3]) / .025
    witness = abs(fd - half) / max(abs(fd), abs(half))
    relative = abs(ad - fd) / max(abs(ad), abs(fd))
    print(json.dumps(dict(AD=ad, FD_h=fd, FD_half=half, fd_relative=witness,
                          ad_relative=relative, table_calls=len(calls))))
    assert witness <= 2e-3, f'FD not converged: AD={ad}, FD(h)={fd}, FD(h/2)={half}, relative={witness}'
    assert relative <= 1e-2, f'AD={ad}, FD(h)={fd}, FD(h/2)={half}, relative={relative}'


def test_operands_match_run_bytes(monkeypatch):
    from rfx import simulation
    original = simulation.run
    operands = []

    def captured(grid, materials, *args, **kwargs):
        operands.append((materials.components.mu_update, kwargs['aniso_eps'], kwargs['conformal_weights']))
        return original(grid, materials, *args, **kwargs)

    monkeypatch.setattr(simulation, 'run', captured)
    ran = scene().run(n_steps=STEPS, skip_preflight=True, compute_s_params=False)
    forward = scene().forward(n_steps=STEPS, skip_preflight=True)
    assert len(operands) == 2
    for a, b in zip(jax.tree.leaves(operands[0]), jax.tree.leaves(operands[1]), strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    np.testing.assert_array_equal(ran.time_series, forward.time_series)
    assert np.max(np.abs(forward.time_series)) > 0


@pytest.mark.parametrize('transform', ('grad', 'jit'))
@pytest.mark.parametrize('route', ('forward', 'driver', 'run'))
@pytest.mark.parametrize('shape', ('sphere_radius', 'sphere_center', 'box'))
def test_traced_pec_refused_before_table_or_steps(monkeypatch, route, shape, transform):
    from tests.unit.runners.test_silent_routes import _mixed, _sim, _scan_call

    def fail(*args, **kwargs):
        pytest.fail('traced PEC geometry reached table formation or stepping')

    monkeypatch.setattr(cell_table, 'build_cell_table', fail)
    monkeypatch.setattr(jax.lax, 'scan', fail)

    def loss(value):
        sim = _mixed(True) if route == 'mixed' else _sim(True, ports='lumped' if route in ('driver', 'run') else None)
        body = (Sphere((.0061, .0062, .0064), value) if shape == 'sphere_radius'
                else Sphere((value, .0062, .0064), .0012) if shape == 'sphere_center'
                else Box((value, .002, .002), (.009, .006, .006)))
        sim.add(body, material='pec')
        if route == 'run':
            sim.run(n_steps=1, skip_preflight=True, compute_s_params=True)
        else:
            _scan_call(route, sim)
        return jnp.array(0.)

    from rfx.geometry.rasterize_grid import TracedShapeDimensionError
    with pytest.raises(TracedShapeDimensionError, match='conformal PEC.*traced shape parameters'):
        getattr(jax, transform)(loss)(.0012 if shape == 'sphere_radius' else .004)


@pytest.mark.parametrize('route', ('optimize', 'topology'))
def test_optimization_refused_at_entry(monkeypatch, route):
    from tests.unit.runners.test_silent_drop_warnings import _optimize
    from tests.unit.runners.test_silent_routes import _scan_call

    def fail(*args, **kwargs):
        pytest.fail('optimization reached table formation or stepping')

    monkeypatch.setattr(cell_table, 'build_cell_table', fail)
    monkeypatch.setattr(jax.lax, 'scan', fail)
    with pytest.raises(NotImplementedError, match='conformal'):
        if route == 'optimize':
            _optimize(scene(), skip_preflight=True)
        else:
            _scan_call('topology', scene())


def test_conformal_pec_mask_override_refused(monkeypatch):
    mask = jnp.zeros(scene()._build_grid().shape, dtype=bool).at[5:8, 2:6, 2:5].set(True)

    def fail(*args, **kwargs):
        pytest.fail('conformal mask override reached time stepping')

    with monkeypatch.context() as guarded:
        guarded.setattr(jax.lax, 'scan', fail)
        with pytest.raises(NotImplementedError, match='pec_mask_override.*conformal.*conformal_pec=False.*PEC shapes'):
            scene().forward(n_steps=STEPS, skip_preflight=True, pec_mask_override=mask)
    result = scene(conformal=False).forward(n_steps=STEPS, skip_preflight=True, pec_mask_override=mask)
    assert np.all(np.isfinite(result.time_series))


def test_conformal_sheet_refusal_matches_run(monkeypatch):
    from tests.unit.runners.test_silent_routes import _sim, _scan_call

    def fail(*args, **kwargs):
        pytest.fail('conformal surface-impedance sheet reached time stepping')

    monkeypatch.setattr(jax.lax, 'scan', fail)
    messages = []
    for route in ('run', 'forward', 'driver'):
        sim = _sim(True, ports='lumped')
        sim.add_thin_conductor(Box((.001, .001, .010), (.011, .011, .010)),
                               surface_impedance_f0=10e9)
        with pytest.raises(ValueError, match='surface_impedance_f0.*conformal_pec') as caught:
            if route == 'driver':
                _scan_call(route, sim)
            else:
                getattr(sim, route)(n_steps=8, skip_preflight=True)
        messages.append(str(caught.value))
    assert messages[0] == messages[1] == messages[2]


def test_conformal_mixed_refused_before_steps(monkeypatch):
    from tests.unit.runners.test_silent_routes import _mixed

    def fail(*args, **kwargs):
        pytest.fail('conformal mixed S-matrix reached time stepping')

    monkeypatch.setattr(jax.lax, 'scan', fail)
    sim = _mixed(True)
    sim.add(Sphere((.0061, .0058, .0057), .0023), material='pec')
    with pytest.raises(NotImplementedError, match=r'compute_mixed_s_matrix\(\) refuses conformal PEC: no comparator yet'):
        sim.compute_mixed_s_matrix(n_steps=8, skip_preflight=True)
