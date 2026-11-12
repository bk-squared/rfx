"""Default conformal operands and the diagnostic material gradient through forward."""

import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry import cell_table


STEPS = 64


def scene(eps=2.5, *, conformal=True, **options):
    sim = Simulation(20e9, (.012,) * 3, dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x='pec', y=Boundary('pec', 'pec', conformal=conformal), z='pec'),
                     **options)
    sim.add_material('host', eps_r=eps)
    sim.add(Box((.0013, .0012, .0011), (.0107, .0108, .0106)), material='host')
    sim.add(Sphere((.0061, .0058, .0057), .0023), material='pec')
    sim.add_source((.003, .004, .003), 'ez', amplitude_kind='current')
    sim.add_probe((.008, .004, .003), 'ez')
    return sim


def s_scan_scene():
    """G1-S: off-node-plane PEC sphere in a dielectric host, with one port."""
    sim = Simulation(20e9, (.012,) * 3, dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x='pec', y=Boundary('pec', 'pec', conformal=True), z='pec'),
                     snap='declared')
    sim.add_material('host', eps_r=2.5)
    sim.add(Box((.0013, .0012, .0011), (.0107, .0108, .0106)), material='host')
    sim.add(Sphere((.0061, .0058, .0057), .0023), material='pec')
    sim.add_port((.003, .004, .003), 'ez', impedance=50.)
    return sim


def test_s_scan_steps_run_operands(monkeypatch):
    from rfx import simulation
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan

    original = simulation.run
    operands = []

    def captured(grid, materials, *args, **kwargs):
        values = (materials.components.mu_update, kwargs['aniso_eps'], kwargs['conformal_weights'])
        operands.append(jax.tree.map(lambda a: np.asarray(a).copy(), values))
        return original(grid, materials, *args, **kwargs)

    monkeypatch.setattr(simulation, 'run', captured)
    s_scan_scene().run(n_steps=512, skip_preflight=True, compute_s_params=False)
    compute_lumped_wire_s_matrix_via_scan(
        s_scan_scene(), np.linspace(6e9, 18e9, 7), n_steps=512, skip_preflight=True)
    assert len(operands) == 2
    left, right = (jax.tree.leaves(value) for value in operands)
    print(json.dumps(dict(run_arrays=len(left), scan_arrays=len(right),
                          structure_equal=jax.tree.structure(operands[0]) == jax.tree.structure(operands[1]))))
    assert jax.tree.structure(operands[0]) == jax.tree.structure(operands[1])
    assert len(left) == len(right) == 9
    for a, b in zip(left, right, strict=True):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert a.tobytes() == b.tobytes()


def test_s_scan_moves_s_from_staircase():
    """G1-S is not settled; read only conformal-minus-staircase at a fixed record.

    This checks the difference at N=512, not the S values themselves.
    """
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan

    freqs = np.linspace(6e9, 18e9, 7)
    spectra = []
    for conformal in (None, False):
        s, bins = compute_lumped_wire_s_matrix_via_scan(
            s_scan_scene(), freqs, n_steps=512, skip_preflight=True, conformal_pec=conformal)
        np.testing.assert_array_equal(bins, freqs)
        spectra.append(np.asarray(s)[0, 0].copy())
    # D_k(main)/10 from rfx-archive rfx/records/20261112-test-audit-220fixes/gaps/DESIGN.md.
    # All seven measured D_k exceed 1e-3; no bin is excluded.
    floor = np.array([2.390819e-4, 4.062380e-4, 3.0621594e-3, 8.380125e-4,
                      8.620271e-4, 3.0284235e-3, 2.90690035e-2])
    difference = np.abs(spectra[0] - spectra[1])
    print(json.dumps(dict(n_steps=512, frequencies_hz=freqs.tolist(),
                          D_k=difference.tolist(), floor_k=floor.tolist())))
    assert np.all(difference >= floor), f'D_k={difference}, floor_k={floor}'


def objective(eps):
    result = scene(eps).forward(n_steps=STEPS, skip_preflight=True)
    return jnp.sum(jnp.abs(result.time_series) ** 2)


def test_material_gradient_and_fresh_model_fd(monkeypatch):
    original = cell_table.build_cell_table
    calls = []

    def counted(grid, shapes, **kwargs):
        from rfx.geometry.smoothing import _shape_parameters
        assert not any(isinstance(value, jax.core.Tracer)
                       for shape in shapes for value in _shape_parameters(shape))
        calls.append(1)
        return original(grid, shapes, **kwargs)

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
    from rfx.model import conformal_rules
    original = simulation.run
    original_fractions = conformal_rules.conformal_fractions_from_table
    operands = []
    fractions = []

    def formed(*args, **kwargs):
        result = original_fractions(*args, **kwargs)
        fractions.append(result)
        return result

    def captured(grid, materials, *args, **kwargs):
        assert len(fractions) == len(operands) + 1
        operands.append((materials.components.mu_update, kwargs['aniso_eps'], kwargs['conformal_weights']))
        return original(grid, materials, *args, **kwargs)

    monkeypatch.setattr(conformal_rules, 'conformal_fractions_from_table', formed)
    monkeypatch.setattr(simulation, 'run', captured)
    ran = scene().run(n_steps=STEPS, skip_preflight=True, compute_s_params=False)
    forward = scene().forward(n_steps=STEPS, skip_preflight=True)
    assert len(operands) == 2
    assert len(fractions) == 2
    left, right = (jax.tree.leaves((operands[i], fractions[i])) for i in range(2))
    for a, b in zip(left, right, strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    np.testing.assert_array_equal(ran.time_series, forward.time_series)
    assert np.max(np.abs(forward.time_series)) > 0
    print(json.dumps(dict(operand_arrays=len(left), unequal_bytes=0,
                          formations=len(fractions), probe_max=float(np.max(np.abs(forward.time_series))))))


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
    name = r'optimize\(\) refuses conformal' if route == 'optimize' else r'(?s)topology_optimize\(\).*conformal'
    with pytest.raises(NotImplementedError, match=name):
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
    # Staircase control deliberately rasterizes the fractional-diameter sphere.
    result = scene(conformal=False, snap="declared").forward(
        n_steps=STEPS, skip_preflight=True, pec_mask_override=mask)
    assert np.all(np.isfinite(result.time_series))


@pytest.mark.parametrize('fixture', ('sphere', 'box'))
def test_strict_conformal_realization_warnings_match_run(fixture):
    realized = []
    for route in ('run', 'forward'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sim = scene() if fixture == 'sphere' else scan_scene(declared=True)
            assert sim._snap == 'strict'
            options = {'compute_s_params': False} if route == 'run' else {}
            getattr(sim, route)(n_steps=8, skip_preflight=True, **options)
        realized.append([(type(w.message), str(w.message)) for w in caught
                         if getattr(w.message, 'code', None) == 'realization'])
    if fixture == 'sphere':
        assert realized[0]
    assert realized[0] == realized[1]


@pytest.mark.parametrize('override', ('pec_occupancy_override', 'design_occupancy_override'))
@pytest.mark.parametrize('model', ('edge_keep', 'kottke'))
def test_conformal_occupancy_override_refused_before_steps(monkeypatch, override, model):
    sim = scene(pec_occupancy_model=model)
    options = ({'design_box': ((.002,) * 3, (.004,) * 3)}
               if override == 'design_occupancy_override' else {})
    shape = (3,) * 3 if options else sim._build_grid().shape
    options[override] = jnp.full(shape, .5)

    def fail(*args, **kwargs):
        pytest.fail('conformal occupancy override reached time stepping')

    monkeypatch.setattr(jax.lax, 'scan', fail)
    with pytest.raises(NotImplementedError, match=override + r'.*conformal PEC.*Boundary'):
        sim.forward(n_steps=8, skip_preflight=True, **options)


def scan_scene(*, declared):
    sim = Simulation(20e9, (.012,) * 3, dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x='pec', y=Boundary('pec', 'pec', conformal=declared), z='pec'))
    sim.add(Box((.0063, .0052, .0043), (.0097, .0088, .0086)), material='pec')
    sim.add_port((.003, .004, .003), 'ez', impedance=50.)
    return sim


@pytest.mark.parametrize('declared', (False, True))
def test_multi_device_conformal_scan_refused_before_steps(monkeypatch, declared):
    from rfx.probes import sparam_driver

    def fail(*args, **kwargs):
        pytest.fail('conformal devices= scan reached distributed accumulation or stepping')

    monkeypatch.setattr(sparam_driver, '_distributed_lumped_accumulators', fail)
    monkeypatch.setattr(jax.lax, 'scan', fail)
    with pytest.raises(NotImplementedError, match=r'compute_lumped_wire_s_matrix_via_scan\(devices=.*conformal PEC.*single device.*conformal_pec=False'):
        sparam_driver.compute_lumped_wire_s_matrix_via_scan(
            scan_scene(declared=declared), [10e9], n_steps=8,
            conformal_pec=None if declared else True, devices=[jax.devices()[0]] * 2)


@pytest.mark.parametrize('declared', (False, True))
def test_strict_conformal_scan_admitted(declared):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan

    sim = scan_scene(declared=declared)
    assert sim._snap == 'strict'
    result, _ = compute_lumped_wire_s_matrix_via_scan(
        sim, [10e9], n_steps=8, conformal_pec=None if declared else True)
    assert np.all(np.isfinite(result))


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
