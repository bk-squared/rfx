"""Independent cubic damping profiles and zero-loss full-line arithmetic."""
from dataclasses import replace
import math

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import EPS_0, MU_0, FDTDState
from rfx.sources import tfsf_discrete_plane_wave as plane


@pytest.mark.parametrize("angles", ((0, 0), (45, 0), (30, 20)))
def test_cubic_loss_at_component_positions(angles):
    dx, dt = .001, 1.9065748695310057e-12
    cfg, _ = plane.init_plane_wave(41, dx, dt, ny=41, nz=41, cpml_layers=8,
                                  polarization="s", theta_deg=angles[0], phi_deg=angles[1],
                                  n_steps=1000)
    norm = math.sqrt(sum(v * v for v in cfg.m))
    length = 64 * math.ceil(norm)
    sigma_max = 16 * math.log(10) / (math.sqrt(MU_0 / EPS_0) * length * dx / norm)
    offsets = np.array(((.5, 0, 0), (0, .5, 0), (0, 0, .5),
                        (0, .5, .5), (.5, 0, .5), (.5, .5, 0)))
    for c, offset in enumerate(offsets):
        fraction = float(offset @ cfg.m) % 1
        position = np.arange(length) + fraction
        rho = np.concatenate((1 - position / length, position / length))
        sigma = sigma_max * rho**3
        expected = np.asarray(np.exp(-sigma * dt / EPS_0), np.float32)
        actual = np.asarray(cfg.line_damping[c])
        assert actual.dtype == np.float32
        assert actual.shape == (cfg.n_nodes,)
        np.testing.assert_array_equal(actual[length:-length], np.float32(1))
        ends = np.concatenate((actual[:length], actual[-length:]))
        np.testing.assert_array_max_ulp(ends, expected, maxulp=2)

    for electric in range(3):
        for magnetic in range(3, 6):
            if cfg.fractions[electric] == cfg.fractions[magnetic]:
                np.testing.assert_array_equal(
                    np.asarray(cfg.line_damping[electric]).view(np.uint32),
                    np.asarray(cfg.line_damping[magnetic]).view(np.uint32))


@pytest.mark.parametrize("angles", ((0, 0), (30, 20)))
def test_zero_loss_factors_and_full_line_update(angles):
    cfg, state = plane.init_plane_wave(41, .001, 1.9065748695310057e-12,
        ny=41, nz=41, cpml_layers=8, polarization="s", theta_deg=angles[0],
        phi_deg=angles[1], n_steps=1000, _zero_absorber=True)
    for array in cfg.line_damping:
        expected = np.ones(cfg.n_nodes, np.float32)
        np.testing.assert_array_equal(np.asarray(array).view(np.uint32), expected.view(np.uint32))
    rng = np.random.default_rng(2732)
    state = FDTDState(*(jnp.asarray(rng.normal(size=cfg.n_nodes), jnp.float32)
                        for _ in range(6)), state.step)
    cfg = replace(cfg, src_amp=0.)
    offsets = np.array(((.5, 0, 0), (0, .5, 0), (0, 0, .5),
                        (0, .5, .5), (.5, 0, .5), (.5, .5, 0)))
    fractions = (offsets @ cfg.m) % 1
    for magnetic, update in ((True, plane.update_line_h), (False, plane.update_line_e)):
        actual = update(cfg, state, cfg.dx_1d, cfg.dt)
        expected = list(state[:6])
        for axis, (first, a, second, b) in enumerate(((2, 1, 1, 2), (0, 2, 2, 0), (1, 0, 0, 1))):
            target = axis + (3 if magnetic else 0)
            source_offset = 0 if magnetic else 3
            def difference(source, derivative):
                offset = fractions[target] - fractions[source]
                half = cfg.m[derivative] / 2
                plus = np.arange(cfg.n_nodes) + int(offset + half)
                minus = np.arange(cfg.n_nodes) + int(offset - half)
                # Explicit zero extension, including negative indices.
                def read(indices):
                    return jnp.where((indices >= 0) & (indices < cfg.n_nodes),
                                     state[source][np.clip(indices, 0, cfg.n_nodes - 1)], 0.)
                return read(plus) - read(minus)
            curl = (difference(first + source_offset, a) / cfg.dx_1d
                    - difference(second + source_offset, b) / cfg.dx_1d)
            if magnetic:
                expected[target] = state[target] - cfg.ch * curl
            else:
                expected[target] = cfg.ca * state[target] + cfg.cb * curl
        for a, b in zip(actual[:6], expected):
            np.testing.assert_array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))
