"""Closed Box overlaps, cavity wall inversion, and a normal-run resonance pin."""

import itertools
import json
import math
import time
import warnings
from dataclasses import replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.geometry.cell_table import build_cell_table
from rfx.geometry.csg import Box, Sphere
from rfx.grid import Grid
from rfx.model import conformal_rules
from rfx.nonuniform import make_nonuniform_grid
from rfx.ringdown import identify
from rfx.sources.sources import ModulatedGaussian


C = 299792458.
D = .001
MODES = ('ex', 'ey', 'ez')


def _along(values, axis):
    shape = [1, 1, 1]
    shape[axis] = -1
    return np.asarray(values).reshape(shape)


def _lattice(grid):
    starts, widths = [], []
    for axis in range(3):
        cells = np.asarray(grid.cells(axis), dtype=float)
        nodes = np.array([grid.node_of(axis, i) for i in range(grid.shape[axis])], dtype=float)
        periodic = 'xyz'[axis] in grid.periodic_axes
        end = cells[-1 if periodic or len(cells) == 1 else -2]
        line = np.r_[nodes[0] - cells[0], nodes, nodes[-1] + end, nodes[-1] + 2 * end]
        if getattr(grid, 'is_2d', False) and axis == 2:
            line[:] = nodes[0]
        elif periodic and not np.all(cells == cells[0]):
            line[0] = nodes[0] - end
            line[-1] = nodes[-1] + end + cells[0]
        widths.append(np.diff(line))
        if periodic:
            line = line % float(grid.domain[axis])
            if not np.all(cells == cells[0]):
                line[0] = line[-3]
                line[-2:] = line[1:3]
        starts.append(line[:-1])
    return starts, widths


def _hand_arrays(grid, boxes):
    """Inclusion-exclusion of clipped intervals and rectangles, independently of SDFs."""
    starts, widths = _lattice(grid)
    stops = [a + b for a, b in zip(starts, widths)]
    shape = tuple(len(a) for a in starts)
    images = []
    tol = 1e-6 * min(float(np.min(grid.cells(a))) for a in range(3))
    for box in boxes:
        for shifts in itertools.product(*(range(-2, 3) if a in grid.periodic_axes else (0,) for a in 'xyz')):
            bounds = np.array([box.corner_lo, box.corner_hi], dtype=float)
            for axis in range(3):
                if 'xyz'[axis] in grid.periodic_axes:
                    bounds[:, axis] += shifts[axis] * float(grid.domain[axis])
                nodes = np.r_[starts[axis], stops[axis]]
                for side in range(2):
                    index = np.argmin(abs(nodes - bounds[side, axis]))
                    if abs(nodes[index] - bounds[side, axis]) <= tol:
                        bounds[side, axis] = nodes[index]
            if all(bounds[1, a] >= min(starts[a]) and bounds[0, a] <= max(stops[a]) for a in range(3)):
                images.append(bounds)

    def fraction(varying):
        varying = [a for a in varying if np.any(widths[a] > 0)]
        covered = np.zeros(shape)
        for count in range(1, len(images) + 1):
            for group in itertools.combinations(images, count):
                lo = np.max([b[0] for b in group], axis=0)
                hi = np.min([b[1] for b in group], axis=0)
                if np.any(hi < lo):
                    continue
                weight = 1.
                for a in range(3):
                    if a in varying:
                        factor = np.maximum(0., np.minimum(stops[a], hi[a]) - np.maximum(starts[a], lo[a])) / widths[a]
                    else:
                        factor = (lo[a] <= starts[a]) & (starts[a] <= hi[a])
                    weight = weight * _along(factor, a)
                covered += (-1.) ** (count + 1) * weight
        return np.clip(1. - covered, 0., 1.)

    return (tuple(fraction([a]) for a in range(3)),
            tuple(fraction([b for b in range(3) if b != a]) for a in range(3)))


def _scene(part, f):
    walls = np.array([.017, .012, .010])
    if part == 'B':
        return tuple(walls), [Box((.006, .004, .003), tuple(np.array([.009, .006, .005]) + f * D))], walls
    walls += np.array([1., 0., 1.]) * f * D
    boxes = []
    for axis in range(3):
        lo = [-.01] * 3
        lo[axis] = walls[axis]
        boxes.append(Box(tuple(lo), (.03,) * 3))
    return (.019, .014, .012), boxes, walls


def _uniform(domain=(.010, .009, .008), periodic='', lane=False):
    return Grid(24e9, domain, dx=D, cpml_layers=0, periodic_axes=periodic,
                **({'mode': '2d_tmz'} if lane else {}))


def _graded(periodic=False):
    profiles = [D * np.array(p) for p in (
        [1., 1., .75, 1.25, .75, 1.25, 1., 1.],
        [1., 1., 1.25, .75, 1.25, .75, 1., 1.],
        [1., 1., .75, 1.25, 1., 1., 1., 1.])]
    return make_nonuniform_grid(tuple(float(p.sum()) for p in profiles[:2]), profiles[2], D,
                                cpml_layers=0, dx_profile=profiles[0], dy_profile=profiles[1],
                                periodic_axes='x' if periodic else '',
                                domain=tuple(float(p.sum()) for p in profiles))


@pytest.mark.parametrize('scene', ('a-half', 'a-quarter', 'b', 'c', 'd', 'd-prime', 'e', 'f', 'h', 'h-periodic'))
def test_t1_exact_overlap(scene, record_property):
    begin = time.perf_counter()
    trailing = scene == 'f'
    if scene in ('a-half', 'a-quarter', 'f', 'b'):
        domain, boxes, _ = _scene('A' if scene == 'b' else 'B', .25 if scene == 'a-quarter' else .5)
        grid = _uniform(domain)
    elif scene == 'c':
        grid = _uniform()
        boxes = [Box((.00125, .00225, .00125), (.00675, .00675, .00575)),
                 Box((.00325, .00125, .00225), (.00825, .00725, .00475))]
    elif scene in ('d', 'd-prime'):
        grid = _uniform(periodic='x')
        boxes = [Box((.00925 if scene == 'd' else .00025, .00225, .00225),
                     (.0115 if scene == 'd' else .0035, .0055, .0055))]
    elif scene == 'e':
        grid = _uniform((.010, .009, .001), lane=True)
        boxes = [Box((.00225, .00225, -.001), (.0065, .0055, .001))]
    else:
        grid = _graded(scene == 'h-periodic')
        boxes = [Box((.00025 if scene == 'h-periodic' else .00225, .00225, .00225),
                     (.0055, .0055, .0055))]
    with jax.enable_x64():
        table = build_cell_table(grid, boxes, include_trailing_node=trailing)
        expected = _hand_arrays(grid, boxes)
    errors = [float(np.max(abs(actual - hand))) for group, hands in
              zip((table.lengths, table.areas), expected) for actual, hand in zip(group, hands)]
    largest = max(errors)
    elapsed = time.perf_counter() - begin
    record_property('max_abs_table_minus_hand', largest)
    record_property('wall_seconds', elapsed)
    print(json.dumps(dict(scene=scene, max_abs_table_minus_hand=largest, wall_seconds=elapsed)))
    assert largest <= 1e-12, (scene, largest, errors)


def test_t1_other_body_reach(record_property):
    begin = time.perf_counter()
    grid = _uniform((.018,) * 3)
    sphere = Sphere((.004,) * 3, .0023)
    box = Box((.01225,) * 3, (.0155,) * 3)
    starts, widths = _lattice(grid)
    nodes = [np.r_[start, start[-1] + width[-1]] for start, width in zip(starts, widths)]
    xyz = np.meshgrid(*nodes, indexing='ij', sparse=True)
    sphere_distance = np.sqrt(sum((coord - center) ** 2 for coord, center in zip(xyz, sphere.center))) - sphere.radius
    center = (np.array(box.corner_lo) + box.corner_hi) / 2
    half = (np.array(box.corner_hi) - box.corner_lo) / 2
    q = [abs(coord - c) - h for coord, c, h in zip(xyz, center, half)]
    box_distance = np.sqrt(sum(np.maximum(t, 0.) ** 2 for t in q)) + np.minimum(np.maximum.reduce(np.broadcast_arrays(*q)), 0.)
    with jax.enable_x64():
        mixed = build_cell_table(grid, [box, sphere])
        alone = build_cell_table(grid, [sphere])
        hand = _hand_arrays(grid, [box])
    errors, reach_errors, bitwise, reached = [], [], [], 0
    for kind, values, sphere_values, expected in zip(('edge', 'face'), (mixed.lengths, mixed.areas),
                                                    (alone.lengths, alone.areas), hand):
        for axis in range(3):
            varying = [axis] if kind == 'edge' else [a for a in range(3) if a != axis]
            diagonal = np.sqrt(sum(_along(widths[a], a) ** 2 for a in varying))
            corners = []
            for moves in itertools.product((0, 1), repeat=len(varying)):
                corner = [slice(None, -1)] * 3
                for a, move in zip(varying, moves):
                    if move:
                        corner[a] = slice(1, None)
                corners.append(tuple(corner))
            reach = np.logical_or.reduce([sphere_distance[c] <= diagonal + 1e-6 * D for c in corners])
            reached += int(reach.sum())
            assert all(np.all(box_distance[c][reach] > sphere_distance[c][reach]) for c in corners)
            bitwise.append(np.array_equal(values[axis][reach].view(np.uint64),
                                          sphere_values[axis][reach].view(np.uint64)))
            reach_errors.append(float(np.max(abs(values[axis][reach] - sphere_values[axis][reach]))))
            errors.append(float(np.max(abs(values[axis][~reach] - expected[axis][~reach]))))
    largest = max(errors + reach_errors)
    elapsed = time.perf_counter() - begin
    record_property('max_abs_table_minus_hand', largest)
    record_property('wall_seconds', elapsed)
    print(json.dumps(dict(reached=reached, max_abs_table_minus_hand=largest, wall_seconds=elapsed)))
    assert reached > 0
    assert all(bitwise), reach_errors
    assert largest <= 1e-12, largest


class _FinitePulse(ModulatedGaussian):
    def __call__(self, t):
        return jnp.where(t < self.t0 + 6 * self.tau, super().__call__(t), 0.)


def _pole(series, dt, target, window):
    start, n = math.ceil(2.5e-9 / dt), len(series)
    model = identify(series, dt, start, n, freq_max=24e9)
    residues = abs(model.c[:, 0])
    floor = float(residues.max(initial=0.)) * 1e-3
    candidates = sorted([float(s.imag / (2 * math.pi)) for s, r in zip(model.s, residues)
                         if r >= floor and abs(s.imag / (2 * math.pi) - target) <= window * target],
                        key=lambda f: abs(f - target))
    assert candidates, (target, window)
    assert not any(abs(f - candidates[0]) <= 1 / ((n - start) * dt) for f in candidates[1:]), candidates
    return candidates[0]


def _run(part, f, exact=False, linked=None):
    domain, boxes, walls = _scene(part, f)
    sim = Simulation(freq_max=24e9, domain=domain, dx=D, boundary='pec', snap='declared')
    for box in boxes:
        sim.add(box, material='pec')
    targets = [C / 2 * math.sqrt(sum(1 / walls[b] ** 2 for b in range(3) if b != a)) for a in range(3)]
    for mode, target in zip(MODES, targets):
        sim.add_source((.0033, .0027, .0021), mode, amplitude_kind='current',
                       waveform=_FinitePulse(f0=target, bandwidth=.2, cutoff=5.))
        sim.add_probe((.0134, .0096, .0063), mode)
    grid = sim._build_grid()
    dt = float(grid.dt)
    n = math.ceil(5e-9 / dt)
    original = conformal_rules.conformal_fractions_from_table

    def substituted(table, run_grid, mode, **kwargs):
        if exact:
            lengths, areas = _hand_arrays(run_grid, boxes)
            table = replace(table, lengths=lengths, areas=areas)
        return original(table, run_grid, mode, **kwargs)

    with warnings.catch_warnings(), patch.object(conformal_rules, 'conformal_fractions_from_table', substituted):
        warnings.simplefilter('ignore')
        result = sim.run(n_steps=n, compute_s_params=False, conformal_pec=True)
    samples = np.asarray(result.time_series).reshape(n, 3)
    axes = range(3) if part == 'A' else (1,)
    frequencies = [_pole(samples[:, a], dt, targets[a] if linked is None else linked,
                         .06 if linked is None else .03) for a in axes]
    return frequencies, dt, walls


def test_t2_closed_form_walls(record_property):
    begin = time.perf_counter()
    frequencies, dt, walls = _run('A', .5)
    s = np.array([math.sin(math.pi * f * dt) ** 2 / (C * dt) ** 2 for f in frequencies])
    k = (s.sum() - 2 * s) / 2
    lengths = math.pi * D / (2 * np.arcsin(D * np.sqrt(k)))
    delta = (lengths - walls) / D
    elapsed = time.perf_counter() - begin
    record_property('delta_cells', delta.tolist())
    record_property('wall_seconds', elapsed)
    print(json.dumps(dict(delta_cells=delta.tolist(), wall_seconds=elapsed)))
    assert np.all(abs(delta) <= .01), delta


def test_t3_normal_run_equivalent_offset(record_property):
    begin = time.perf_counter()
    offsets = np.array([.125, .25, .5])
    exact = np.array([_run('B', f, exact=True)[0][0] for f in offsets])
    beta, alpha = np.linalg.lstsq(np.column_stack((offsets, np.ones(3))), exact, rcond=None)[0]
    plain = _run('B', .5, linked=exact[-1])[0][0]
    peak = float(max(abs(exact - (alpha + beta * offsets))))
    delta = float((plain - exact[-1]) / beta)
    elapsed = time.perf_counter() - begin
    record_property('delta_eq_cells', delta)
    record_property('wall_seconds', elapsed)
    print(json.dumps(dict(exact_hz=exact.tolist(), plain_hz=plain, beta_hz_per_cell=beta,
                          residual_max_hz=peak, delta_eq_cells=delta, wall_seconds=elapsed)))
    assert peak <= .2 * abs(beta) * .375, (peak, beta)
    assert abs(delta) <= .05, delta
