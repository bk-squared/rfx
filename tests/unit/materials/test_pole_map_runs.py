"""Pole-map time-domain pins on the predeclared open-domain scene."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, LorentzPoleMap, Simulation, drude_pole, drude_pole_map
from tests.unit.materials.test_pole_map_stability_monotone import DOMAIN, DX, GRADED_X

WP1, WP2 = 2 * np.pi * np.array([3e9, 2e9])
LO = (.01235, .00835, .00835)
HI = (.02765, .02165, .01765)
BINS = np.linspace(1e9, 6e9, 21)


def record(name, values):
    print(name, json.dumps(values))
    if folder := os.environ.get('POLE_MAP_RESULTS'):
        Path(folder, name + '.json').write_text(json.dumps(values, indent=2) + '\n')


def scene(graded, kind, *, mapped, callable_constant=True):
    sim = Simulation(freq_max=6e9, domain=DOMAIN, dx=DX, cpml_layers=8,
                     **(dict(dx_profile=GRADED_X) if graded else {}))
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


def capture_scan(sim, steps, monkeypatch, *, forward=False, perturb=False):
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


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('pin', ['P2a', 'P2b'])
def test_p2_staircase(monkeypatch, graded, pin):
    ref, grid, meta = scene(graded, pin, mapped=False)
    baseline = capture_scan(ref, 2000, monkeypatch)
    control = capture_scan(ref, 2000, monkeypatch, perturb=True)
    ratio, _, reference, control_probes = paired(baseline, control)
    control_ratio = float(ratio.max())
    record(f'{pin}-{graded}-control', dict(**meta, control_ratio=control_ratio))
    assert control_ratio <= .1, f'cannot judge {pin}: control worst ratio {control_ratio} > 0.1'
    mapped, _, _ = scene(graded, pin, mapped=True)
    ratio, _, reference, actual = paired(baseline, capture_scan(mapped, 2000, monkeypatch))
    kernel = np.exp(-2j * np.pi * BINS[:, None] * np.arange(2000)[None, :] * grid.dt) * grid.dt
    expected, got = kernel @ reference, kernel @ actual
    active = abs(expected) > .01 * abs(expected).max(axis=0, keepdims=True)
    errors = np.where(active, abs(got - expected) / np.maximum(abs(expected), np.finfo(float).tiny), 0.)
    bin_index, probe = np.unravel_index(errors.argmax(), errors.shape)
    record(f'{pin}-{graded}', dict(**meta, control_ratio=control_ratio, worst_ratio=float(ratio.max()),
                                   worst_spectral_error=float(errors.max()), bin_hz=float(BINS[bin_index]),
                                   probe=int(probe), active_bins=int(active.sum())))
    assert ratio.max() <= 1.
    assert errors.max() <= 1e-4


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
