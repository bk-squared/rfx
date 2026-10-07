"""A cell-folded DC sheet realizes sigma*t after the E-edge volume mean."""
import json

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation


H = 1e-3
THICKNESS = 35e-6
SIGMA = 1 / (377 * THICKNESS)


def sheet_fixture(normal, placement, lane, present):
    profile = H * np.array([1] * 10 + [.9, 1.1] + [1] * 12)
    if lane == 'uniform':
        profile = np.full(24, H)
    k = {'equal': 6, 'rise': 11, 'fall': 10}[placement]
    plane = float(np.sum(profile[:k]))
    domain = [12 * H] * 3
    domain[normal] = float(profile.sum())
    kwargs = {} if lane == 'uniform' else {f'd{"xyz"[normal]}_profile': profile}
    sim = Simulation(freq_max=10e9, domain=tuple(domain), dx=H,
                     boundary='pec', snap='declared', **kwargs)
    # Vacuum control: the DC fold replaces its occupied cell layer.
    lo, hi = [3 * H] * 3, [9 * H] * 3
    lo[normal] = hi[normal] = plane
    if present:
        sim.add_thin_conductor(Box(tuple(lo), tuple(hi)),
                               sigma_bulk=SIGMA, thickness=THICKNESS)
    probe = [6 * H] * 3
    sim.add_probe(tuple(probe), 'e' + 'xyz'[(normal + 1) % 3])
    return sim


def measured_ratios(monkeypatch, normal, placement, lane):
    import rfx.model.materials as owner
    original = owner.realize_components
    captured = []

    def observe(cells, grid, **kwargs):
        result = original(cells, grid, **kwargs)
        captured.append((grid, result))
        return result

    monkeypatch.setattr(owner, 'realize_components', observe)
    lines = []
    for present in (False, True):
        captured.clear()
        sim = sheet_fixture(normal, placement, lane, present)
        if lane == 'forward':
            sim.forward(n_steps=2, checkpoint=False, skip_preflight=True)
        else:
            sim.run(n_steps=2, compute_s_params=False, skip_preflight=True)
        assert captured, 'entry point did not realize component materials'
        grid, components = captured[-1]
        index = [6, 6, 6]
        index[normal] = slice(None)
        tangents = [axis for axis in range(3) if axis != normal]
        lines.append([np.asarray(components.sigma_update[axis])[tuple(index)]
                      for axis in tangents])
    # Independent NumPy dual widths, from the primal cell array. No material
    # or dual-spacing product helper enters the conductance reference.
    widths = np.asarray(grid.cells(normal), dtype=float)
    dual = np.r_[widths[0], (widths[:-1] + widths[1:]) / 2]
    delta = np.asarray(lines[1], dtype=float) - np.asarray(lines[0], dtype=float)
    ratios = np.sum(delta * dual, axis=1) / (SIGMA * THICKNESS)
    return ratios, dict(lane=lane, normal=normal, placement=placement,
                        ratios=ratios.tolist(),
                        planes=np.flatnonzero(np.any(delta != 0, axis=0)).tolist())


CASES = [(lane, placement) for lane in ('run', 'forward')
         for placement in ('equal', 'rise', 'fall')] + [('uniform', 'equal')]


@pytest.mark.parametrize('normal', (0, 1, 2))
@pytest.mark.parametrize('lane,placement', CASES)
def test_dc_conductance_at_both_tangential_edges(monkeypatch, normal, lane, placement):
    ratios, record = measured_ratios(monkeypatch, normal, placement, lane)
    print(json.dumps(record))
    np.testing.assert_allclose(ratios, np.ones(2), rtol=4e-7, atol=0)


def test_dc_fold_thickness_gradient_uses_occupied_primal_width():
    import jax
    from dataclasses import replace
    from rfx.model.materials import assemble_cells
    sim = sheet_fixture(2, 'rise', 'run', True)
    grid = sim._build_nonuniform_grid()
    conductor = sim._thin_conductors[0]

    def cell_sum(thickness):
        sim._thin_conductors[0] = replace(conductor, thickness=thickness)
        return jnp.sum(assemble_cells(sim, grid)[0].sigma)

    got = float(jax.grad(cell_sum)(THICKNESS))
    # Half-open footprint [3,9) x [3,9) occupies 6*6 cells, width 1.1h.
    expected = 36 * SIGMA / (1.1 * H)
    np.testing.assert_allclose(got, expected, rtol=2e-7)
