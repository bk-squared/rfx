"""Pole-map time-domain pins on the predeclared open-domain scene."""
import json
import os
from pathlib import Path
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, LorentzPole, LorentzPoleMap, Simulation, drude_pole, drude_pole_map
from tests.unit.materials.test_pole_map_stability_monotone import DOMAIN, DX, GRADED_X
from tests._x64_compat import enable_x64

WP1, WP2 = 2 * np.pi * np.array([3e9, 2e9])
LO = (.01235, .00835, .00835)
HI = (.02765, .02165, .01765)
BINS = np.linspace(1e9, 6e9, 21)


def record(name, values):
    print(name, json.dumps(values))
    if folder := os.environ.get('POLE_MAP_RESULTS'):
        Path(folder, name + '.json').write_text(json.dumps(values, indent=2) + '\n')


def scene(graded, kind, *, mapped, callable_constant=True, precision='float32'):
    sim = Simulation(freq_max=6e9, domain=DOMAIN, dx=DX, cpml_layers=8,
                     precision=precision, **(dict(dx_profile=GRADED_X) if graded else {}))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    assert grid.shape == (97, 77, 69)
    nodes = []
    centres = []
    for a in range(3):
        d = np.asarray(grid.cells(a), dtype=float)
        n = np.r_[0., np.cumsum(d[:-1])]
        n -= n[8]
        assert n[-9] == pytest.approx(DOMAIN[a])
        nodes.append(n)
        centres.append(n + d / 2)
    split_index = int(np.argmin(abs(nodes[0] - .020)))
    split = float(nodes[0][split_index])
    left = float(nodes[0][split_index - 1]) if kind == 'P2b' else split
    gamma1, gamma2 = (1e9, 4e9) if kind == 'P2b' else (2e9, 4e9) if kind == 'P6' else (2e9, 2e9)
    if kind == 'P1':
        pole = drude_pole_map((lambda x, y, z: WP1 + 0*x) if callable_constant else WP1,
                              (lambda x, y, z: 2e9 + 0*y) if callable_constant else 2e9) if mapped else drude_pole(WP1, 2e9)
        sim.add_material('plasma', eps_r=1., lorentz_poles=[pole])
        sim.add(Box(LO, HI), material='plasma')
        boxes = [(LO, HI)]
    else:
        boxes = [(LO, (left, HI[1], HI[2])), ((split, LO[1], LO[2]), HI)]
        shared = LorentzPoleMap(0., lambda x, y, z: np.where(x < split, gamma1 / 2, gamma2 / 2),
                                lambda x, y, z: np.where(x < split, WP1**2, WP2**2))
        for i, ((lo, hi), wp, gamma) in enumerate(zip(boxes, (WP1, WP2), (gamma1, gamma2))):
            sim.add_material(f'p{i}', eps_r=1., lorentz_poles=[shared if mapped else drude_pole(wp, gamma)])
            sim.add(Box(lo, hi), material=f'p{i}')
    extents = []
    for lo, hi in boxes:
        selected = [np.flatnonzero((c >= l) & (c < h)) for c, l, h in zip(centres, lo, hi)]
        assert all(len(s) > 0 for s in selected)
        extents.append(dict(counts=[len(s) for s in selected],
                            low=[float(n[s[0]]) for n, s in zip(nodes, selected)],
                            high=[float(n[s[-1] + 1]) for n, s in zip(nodes, selected)]))
    if not graded:
        assert extents[0]['counts'][1:] == [26, 18]
        if kind == 'P1':
            assert extents[0]['counts'] == [30, 26, 18]
        else:
            assert split == pytest.approx(.020)
            assert left == pytest.approx(.0195 if kind == 'P2b' else .020)
    sim.add_source((.00985, .01235, .01085), 'ez', waveform=GaussianPulse(f0=3e9))
    for position in ((.03035, .01785, .01435), (.03035, .00935, .01585), (.02185, .01935, .00935)):
        sim.add_probe(position, 'ez')
    return sim, grid, dict(shape=list(grid.shape), boxes=extents, split=split, left=left, dt=float(grid.dt))


class _Captured(Exception):
    pass


def capture_scan(sim, steps, monkeypatch, *, forward=False, perturb=False, perturbation=None):
    """Capture the actual public lane's scan; pair two scans without storing full histories."""
    import rfx.stepping.uniform_loop as uniform
    import rfx.stepping.graded_loop as graded
    import rfx.materials.lorentz as lorentz
    scans = []
    def capture(body, carry, xs, *args, **kwargs):
        scans.append((body, carry, xs))
        raise _Captured
    original = lorentz.lorentz_e_coeffs
    def perturbed(*args, **kwargs):
        coeffs = original(*args, **kwargs)
        c = tuple(v.at[0].set(jnp.where(v[0] > 0, jnp.nextafter(v[0], jnp.float32(jnp.inf)), v[0]))
                  for v in coeffs.c)
        if perturbation is not None:
            for old, new in zip(coeffs.c, c):
                positive = np.asarray(old[0]) > 0
                step = np.asarray(new[0] - old[0])[positive]
                if step.size:
                    perturbation.append(dict(dtype=str(old.dtype), min_step=float(step.min()),
                                             max_step=float(step.max())))
        return coeffs._replace(c=c)
    with monkeypatch.context() as patch:
        patch.setattr(uniform, 'recorded_scan', capture)
        patch.setattr(graded, 'recorded_scan', capture)
        if perturb:
            patch.setattr(lorentz, 'lorentz_e_coeffs', perturbed)
        with pytest.raises(_Captured):
            if forward:
                sim.forward(n_steps=steps, checkpoint=False, skip_preflight=True)
            else:
                sim.run(n_steps=steps, checkpoint=False, compute_s_params=False, skip_preflight=True)
    assert len(scans) == 1
    return scans[0]


def assert_float64_scan(scan):
    """Check both sides of the actual body, including every pole-state leaf."""
    body, initial, xs = scan
    def check(carry):
        for value in (*carry['fdtd'][:3], *jax.tree.leaves(carry['lorentz'])):
            assert value.dtype == jnp.float64, value.dtype
    check(initial)
    def checked(carry, x):
        check(carry)
        carry, out = body(carry, x)
        check(carry)
        return carry, out
    jax.block_until_ready(jax.jit(lambda: jax.lax.scan(
        checked, initial, jax.tree.map(lambda v: v[:1], xs)))())


def spectral_errors(reference, actual, dt):
    kernel = np.exp(-2j * np.pi * BINS[:, None] * np.arange(len(reference))[None, :] * dt) * dt
    expected, got = kernel @ reference, kernel @ actual
    active = abs(expected) > .01 * abs(expected).max(axis=0, keepdims=True)
    errors = np.where(active, abs(got - expected) / np.maximum(abs(expected), np.finfo(float).tiny), 0.)
    return dict(worst=float(errors.max()), errors=errors.tolist(), active=active.tolist())


def paired_arrays(reference, actual):
    f, ci, xs = reference
    g, di, ys = actual
    def step(carry, inputs):
        c, d = carry
        x, y = inputs
        c, out_c = f(c, x)
        d, out_d = g(d, y)
        ce, de = c['fdtd'], d['fdtd']
        errors = jnp.stack([jnp.max(jnp.abs(a-b)) for a, b in zip(ce[:3], de[:3])])
        scales = jnp.stack([jnp.max(jnp.abs(v)) for v in ce[:3]])
        return (c, d), (jnp.max(errors), jnp.max(scales), out_c, out_d)
    return jax.lax.scan(step, (ci, di), (xs, ys))[1]


def paired(reference, actual):
    difference, scale, probes, candidate = jax.jit(lambda: paired_arrays(reference, actual))()
    difference, scale = np.asarray(difference, float), np.asarray(scale, float)
    allowance = 1.2e-7 * np.cumsum(scale)
    ratios = np.divide(difference, allowance, out=np.where(difference > 0, np.inf, 0.), where=allowance > 0)
    def unpack(value):
        return np.asarray(value[0] if isinstance(value, tuple) else value, float)
    return ratios, difference, unpack(probes), unpack(candidate)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('callables', [False, True], ids=['numbers', 'callables'])
def test_p1_fields(monkeypatch, graded, callables):
    scalar, grid, meta = scene(graded, 'P1', mapped=False)
    mapped, _, _ = scene(graded, 'P1', mapped=True, callable_constant=callables)
    _, difference, reference, actual = paired(capture_scan(scalar, 400, monkeypatch),
                                             capture_scan(mapped, 400, monkeypatch))
    record(f'P1-{graded}-{callables}', dict(**meta, max_field_difference=float(difference.max()),
                                         probes_equal=bool(np.array_equal(reference, actual))))
    np.testing.assert_array_equal(difference, 0.)
    np.testing.assert_array_equal(reference, actual)


@pytest.mark.parametrize('graded,precision', [
    pytest.param(False, 'float32', id='uniform-float32'),
    pytest.param(True, 'float32', id='graded-float32'),
    pytest.param(False, 'float64', id='uniform-float64'),
    pytest.param(True, 'float64', id='graded-float64', marks=pytest.mark.xfail(
        strict=True, raises=NotImplementedError, reason='graded float64 needs rfx#2538')),
])
@pytest.mark.parametrize('pin', ['P2a', 'P2b'])
def test_p2_staircase(monkeypatch, graded, pin, precision):
    with enable_x64() if precision == 'float64' else nullcontext():
        try:
            _p2_staircase(monkeypatch, graded, pin, precision)
        except NotImplementedError as error:
            assert graded and precision == 'float64' and 'precision' in str(error), str(error)
            record(f'{pin}-graded-float64-refusal', dict(message=str(error)))
            raise


def _p2_staircase(monkeypatch, graded, pin, precision):
    ref, grid, meta = scene(graded, pin, mapped=False, precision=precision)
    baseline = capture_scan(ref, 2000, monkeypatch)
    perturbation = []
    control = capture_scan(ref, 2000, monkeypatch, perturb=True, perturbation=perturbation)
    if precision == 'float64':
        assert_float64_scan(baseline)
        assert_float64_scan(control)
    ratio, _, reference, control_probes = paired(baseline, control)
    control_ratio = float(ratio.max())
    control_spectrum = spectral_errors(reference, control_probes, grid.dt)
    record(f'{pin}-{graded}-{precision}-control', dict(**meta, control_ratio=control_ratio,
           spectrum=control_spectrum, perturbation=perturbation))
    if precision == 'float64':
        assert control_spectrum['worst'] <= 1e-10, f'cannot judge {pin}: {control_spectrum}'
    else:
        assert control_ratio <= .1, f'cannot judge {pin}: control worst ratio {control_ratio} > 0.1'
    mapped, _, _ = scene(graded, pin, mapped=True, precision=precision)
    candidate = capture_scan(mapped, 2000, monkeypatch)
    if precision == 'float64':
        assert_float64_scan(candidate)
    ratio, _, reference, actual = paired(baseline, candidate)
    spectrum = spectral_errors(reference, actual, grid.dt)
    record(f'{pin}-{graded}-{precision}', dict(**meta, control_ratio=control_ratio,
           worst_ratio=float(ratio.max()), spectrum=spectrum))
    if precision == 'float64':
        assert spectrum['worst'] <= 1e-9, spectrum
    else:
        assert ratio.max() <= 1.


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p6_run_forward(monkeypatch, graded):
    sim, _, meta = scene(graded, 'P6', mapped=True)
    ratio, difference, reference, actual = paired(capture_scan(sim, 2000, monkeypatch),
                                                 capture_scan(sim, 2000, monkeypatch, forward=True))
    record(f'P6-{graded}', dict(**meta, max_difference=float(difference.max()),
                               worst_ratio=float(ratio.max())))
    np.testing.assert_array_equal(difference, 0.)
    np.testing.assert_array_equal(reference, actual)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p6_traced_forward(monkeypatch, graded):
    sim, _, meta = scene(graded, 'P6', mapped=True)
    reference = capture_scan(sim, 2000, monkeypatch)
    def measured(scale):
        traced, _, _ = scene(graded, 'P6', mapped=True)
        maps = {}
        for material in traced._materials.values():
            if material.lorentz_poles:
                original = material.lorentz_poles[0]
                if id(original) not in maps:
                    profile = original.kappa
                    maps[id(original)] = LorentzPoleMap(original.omega_0, original.delta,
                                                        lambda x, y, z: scale * profile(x, y, z))
                material.lorentz_poles[:] = [maps[id(original)]]
        difference, magnitude, _, _ = paired_arrays(reference, capture_scan(traced, 2000, monkeypatch, forward=True))
        return difference, magnitude
    difference, magnitude = jax.jit(measured)(jnp.float32(1.))
    allowance = 1.2e-7 * np.cumsum(np.asarray(magnitude, float))
    difference = np.asarray(difference, float)
    ratio = np.divide(difference, allowance, out=np.where(difference > 0, np.inf, 0.), where=allowance > 0)
    record(f'P6-traced-{graded}', dict(**meta, worst_ratio=float(ratio.max())))
    assert ratio.max() <= 1.


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p2c_time_difference(monkeypatch, graded):
    ref, _, meta = scene(graded, 'P6', mapped=False)
    sim, _, _ = scene(graded, 'P6', mapped=True)
    _, difference, reference, actual = paired(capture_scan(ref, 2000, monkeypatch),
                                             capture_scan(sim, 2000, monkeypatch))
    record(f'P2c-time-{graded}', dict(**meta, max_field_difference=float(difference.max()),
           max_probe_difference=float(np.max(abs(reference - actual)))))


def zero_scene(graded, all_zero, scale=1., without=False):
    sim, grid, meta = scene(graded, 'P1', mapped=True)
    pole = LorentzPoleMap(0., lambda x, y, z: 1e9 * (1 + 20*y),
        lambda x, y, z: scale * jnp.where((x < .020) | all_zero, 0., WP1**2))
    sim._materials['plasma'].lorentz_poles[:] = [] if without else [pole]
    return sim, grid, meta


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p2c_prime_fields(monkeypatch, graded):
    all_zero = False
    sim, grid, meta = zero_scene(graded, all_zero)
    base = capture_scan(sim, 400, monkeypatch)
    # The interior of the zero-strength band excludes its mixed interface edge.
    widths = np.asarray(grid.cells(0), float)
    nodes = np.r_[0., np.cumsum(widths[:-1])]
    nodes -= nodes[8]
    zero_x = jnp.asarray((nodes + widths < .020) | all_zero)[None, :, None, None]
    def measure(scan):
        body, initial, xs = scan
        def step(carry, x):
            carry, out = body(carry, x)
            magnitude = jnp.max(jnp.stack([jnp.max(jnp.where(zero_x, abs(v), 0.))
                                         for v in carry['lorentz'][:3]]))
            return carry, (magnitude, out)
        return jax.lax.scan(step, initial, xs)[1]
    polarization, _ = jax.jit(lambda: measure(base))()
    np.testing.assert_array_equal(polarization, 0.)
    def traced(scale):
        candidate, _, _ = zero_scene(graded, all_zero, scale)
        scan = capture_scan(candidate, 400, monkeypatch, forward=True)
        polarization, probes = measure(scan)
        traces = probes[0] if isinstance(probes, tuple) else probes
        return jnp.sum(traces**2), jnp.max(polarization)
    # One scalar direction needs no reverse-mode tape of the full domain.
    energy, gradient, traced_p = jax.jit(lambda scale: jax.jvp(
        traced, (scale,), (jnp.ones_like(scale),), has_aux=True))(jnp.float32(1.))
    assert np.isfinite(energy) and np.isfinite(gradient)
    assert traced_p == 0.
    record(f'P2c-prime-{graded}-{all_zero}', dict(**meta, max_p=float(polarization.max()),
           traced_max_p=float(traced_p), probe_energy=float(energy), probe_gradient=float(gradient)))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('forward', [False, True], ids=['concrete', 'traced'])
def test_p2c_prime_all_zero(monkeypatch, graded, forward):
    """D5: scalar reduction bit equality and the independent no-pole law."""
    import rfx.materials.lorentz as lorentz
    from rfx.geometry.rasterize_grid import (
        centres_from_nonuniform_grid, centres_from_uniform_grid,
    )

    scalar, grid, meta = zero_scene(graded, True)
    centres = (centres_from_nonuniform_grid if graded else centres_from_uniform_grid)(grid)
    first_y = np.asarray(centres.y)
    first_y = first_y[(first_y >= LO[1]) & (first_y < HI[1])][0]
    scalar_delta = float(1e9 * (1 + 20 * first_y))
    scalar._materials['plasma'].lorentz_poles[:] = [LorentzPole(0., scalar_delta, 0.)]
    vacuum, _, _ = zero_scene(graded, True, without=True)
    scalar_scan = capture_scan(scalar, 400, monkeypatch, forward=forward)
    vacuum_scan = capture_scan(vacuum, 400, monkeypatch, forward=forward)
    original_builder = lorentz.pole_update_coeffs

    def capture_candidate(scale):
        finite = []
        def checked_builder(*args, **kwargs):
            result = original_builder(*args, **kwargs)
            finite.extend(jnp.all(jnp.isfinite(v)) for v in result)
            return result
        candidate, _, _ = zero_scene(graded, True, scale)
        with monkeypatch.context() as patch:
            patch.setattr(lorentz, 'pole_update_coeffs', checked_builder)
            scan = capture_scan(candidate, 400, monkeypatch, forward=forward)
        return scan, jnp.all(jnp.stack(finite)) if finite else jnp.asarray(True)

    def measure(scan, finite):
        body, initial, xs = scan
        def step(carry, x):
            carry, out = body(carry, x)
            values = [jnp.max(abs(v)) for v in carry['lorentz'][:3]] if carry['lorentz'] is not None else []
            return carry, jnp.max(jnp.stack(values)) if values else jnp.asarray(0.)
        polarization = jax.lax.scan(step, initial, xs)[1]
        return (paired_arrays(scalar_scan, scan), paired_arrays(vacuum_scan, scan),
                jnp.max(polarization), finite)

    if forward:
        measured = jax.jit(lambda scale: measure(*capture_candidate(scale)))(jnp.float32(1.))
    else:
        scan, finite = capture_candidate(1.)
        measured = jax.jit(lambda: measure(scan, finite))()
    equal, law, max_p, finite = jax.device_get(measured)
    difference, _, probes, actual = equal
    def unpack(value):
        return np.asarray(value[0] if isinstance(value, tuple) else value)
    probes, actual = unpack(probes), unpack(actual)
    law_difference, scale = (np.asarray(v, float) for v in law[:2])
    allowance = 1.2e-7 * np.cumsum(scale)
    ratios = np.divide(law_difference, allowance, out=np.where(law_difference > 0, np.inf, 0.),
                       where=allowance > 0)
    scalar_ratios, scalar_difference, _, _ = paired(vacuum_scan, scalar_scan)
    values = dict(**meta, scalar_delta=scalar_delta, coefficients_finite=bool(finite),
                  max_p=float(max_p), scalar_max_field_difference=float(np.max(difference)),
                  scalar_max_probe_difference=float(np.max(abs(probes - actual))),
                  probes_bit_equal=bool(probes.tobytes() == actual.tobytes()),
                  worst_ratio=float(ratios.max()), scalar_control_ratio=float(scalar_ratios.max()),
                  no_pole_max_field_difference=float(law_difference.max()),
                  scalar_no_pole_max_field_difference=float(scalar_difference.max()))
    record(f'P2c-prime-all-{graded}-{forward}', values)
    np.testing.assert_array_equal(difference, 0.)
    assert probes.tobytes() == actual.tobytes(), values
    assert finite and max_p == 0., values
    assert ratios.max() <= 1., values
