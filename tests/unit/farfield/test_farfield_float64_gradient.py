"""A float64 far-field gradient: central FD against AD falls as h^2 (#1567).

A PEC scatterer and a dielectric block in CPML; the parameter scales the
block's permittivity. This is a float64 sanity check of the far-field
gradient, not the pin of the transform's precision: the parameter moves no
angle, frequency or face position, so a float32 transform (a fixed rounded
linear map) passes it too. The pin is the traced-vs-NumPy far-field row in
tests/contracts/test_precision_realized.py.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._precision import FLOAT64_MODE_ERROR
from rfx.measurement.farfield import compute_far_field_jax
from rfx.sources.sources import GaussianPulse


def fixture(precision='float64'):
    sim = Simulation(freq_max=40e9, domain=(.006, .006, .006), dx=.0005,
                     boundary='cpml', cpml_layers=1, precision=precision)
    # Admission requires positive absorber depth on every face; one is the
    # smallest accepted count. This solid PEC cube has a closed surface.
    sim.add(Box((.0015, .002, .0025), (.0025, .003, .0035)), material='pec')
    sim.add_source((.0025, .003, .002), 'ez', amplitude_kind='current',
                   waveform=GaussianPulse(f0=30e9, bandwidth=.8))
    sim.add_ntff_box((.001, .001, .001), (.005, .005, .005), freqs=[30e9])
    return sim


def far_field_function(precision='float64'):
    sim = fixture(precision)
    grid = sim._build_grid()
    shape = grid.shape
    # A 1 mm dielectric block beside the PEC cube, at (3.5, 2.5, 3) mm.
    pad = grid.cpml_layers
    block = tuple(slice(pad + lo, pad + hi) for lo, hi in ((7, 9), (5, 7), (6, 8)))
    mask = jnp.zeros(shape, dtype=precision).at[block].set(1)
    eps0 = jnp.ones(shape, dtype=precision) + 1.25 * mask

    def fields(a):
        result = sim.forward(eps_override=eps0 * (1 + a * mask),
                             n_steps=128, skip_preflight=True)
        ff = compute_far_field_jax(result.ntff_data, result.ntff_box,
                                   result.grid, [1.1], [.7])
        return ff.E_theta, ff.E_phi

    return fields, shape


def test_float64_far_field_gradient_fd():
    start = perf_counter()
    try:
        with jax.enable_x64():
            fields, shape = far_field_function()

            @jax.jit
            def quantity(a):
                e_theta, _ = fields(a)
                return jnp.abs(e_theta[0, 0, 0]) ** 2

            value, ad = jax.jvp(quantity, (jnp.asarray(0.),), (jnp.asarray(1.),))
            ad = float(ad)
            assert np.isfinite(ad) and ad != 0
            gaps = {}
            for h in (1e-2, 1e-3, 1e-4):
                q_plus, q_minus = float(quantity(h)), float(quantity(-h))
                fd = (q_plus - q_minus) / (2 * h)
                gaps[h] = abs(fd - ad) / abs(ad)
                rounding = (np.finfo(np.float64).eps * max(abs(q_plus), abs(q_minus))
                            / (2 * h * abs(ad)))
                print(f'h={h:g} FD={fd:.17g} AD={ad:.17g} gap={gaps[h]:.17g} '
                      f'rounding_relative={rounding:.17g} '
                      f'rounding_limit={10 * rounding:.17g}')
            print(f'grid={shape} steps=128 Q={float(value):.17g} '
                  f'wall_seconds={perf_counter() - start:.6f}')
            assert min(gaps.values()) <= 1e-6, gaps
            assert gaps[1e-3] <= gaps[1e-2] / 30, gaps
    finally:
        jax.clear_caches()


def test_float64_far_field_setup_refuses_without_x64():
    with jax.enable_x64(False):
        sim = fixture()
        with pytest.raises(ValueError) as caught:
            sim.forward(n_steps=128, skip_preflight=True)
        assert str(caught.value) == FLOAT64_MODE_ERROR
