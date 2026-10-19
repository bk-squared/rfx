"""The NTFF declaration preserves the default grid and explicit array identity."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


@pytest.mark.parametrize("freq_max", [10e9, 3.7e9])
@pytest.mark.parametrize("n_freqs", [50, 7])
def test_default_frequency_grid(freq_max, n_freqs):
    # Minimal construction from tests/unit/farfield/test_ntff_sweep.py.
    sim = Simulation(freq_max=freq_max, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                     cpml_layers=4)
    lo, hi = (0.002, 0.002, 0.002), (0.008, 0.008, 0.003)
    if n_freqs == 50:
        sim.add_ntff_box(lo, hi)
    else:
        sim.add_ntff_box(lo, hi, n_freqs=n_freqs)
    expected = np.asarray(jnp.linspace(freq_max / 10, freq_max, n_freqs))
    actual = np.asarray(sim._ntff[2])
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)
    assert actual.tobytes() == expected.tobytes()


def test_explicit_frequency_identity():
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                     cpml_layers=4)
    freqs = np.array([1.1e9, 2.9e9], dtype=np.float64)
    sim.add_ntff_box((0.002, 0.002, 0.002), (0.008, 0.008, 0.003), freqs=freqs)
    assert sim._ntff[2] is freqs


@pytest.mark.parametrize('form', ['eager', 'compile_time', 'jit'])
def test_default_grid_independent_of_outer_compile_form(form):
    import jax
    from rfx.measurement.monitors import default_frequency_grid, _eager_frequency_grid
    expected = np.asarray(jnp.linspace(1.5e9, 15e9, 50))
    _eager_frequency_grid.cache_clear()
    if form == 'compile_time':
        with jax.ensure_compile_time_eval():
            actual = default_frequency_grid(15e9, 50)
    elif form == 'jit':
        actual = jax.jit(lambda: default_frequency_grid(15e9, 50))()
    else:
        actual = default_frequency_grid(15e9, 50)
    assert np.asarray(actual).tobytes() == expected.tobytes()


def test_plane_default_bins_remain_concrete_inside_a_trace():
    import jax
    from rfx.measurement.setup import build_planes
    sim = Simulation(freq_max=15e9, domain=(.0153, .0087, .0101), dx=.001, boundary='pec')
    sim.add_dft_plane_probe(axis='y', coordinate=.0031)
    grid = sim._build_grid()

    def build():
        bins = build_planes(sim, grid, 4)[0].freqs
        # Concrete bins preserve the existing fixed-point phase path.
        assert not isinstance(bins, jax.core.Tracer)
        return bins

    expected = np.asarray(jnp.linspace(1.5e9, 15e9, 50))
    np.testing.assert_array_equal(jax.jit(build)(), expected)
