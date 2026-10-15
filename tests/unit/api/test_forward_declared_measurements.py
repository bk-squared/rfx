"""Declared forward measurements compared with a fresh run declaration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import flux_spectrum
from tests.contracts.run_forward_census import build_o_flux, build_p_waveguide


def _assert_same_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("path", ["uniform", "graded"])
def test_forward_declared_flux_matches_run(path):
    expected = build_o_flux(path).run(n_steps=150, skip_preflight=True)
    actual = build_o_flux(path).forward(n_steps=150, skip_preflight=True, checkpoint=False)
    assert actual.flux_monitors is not None
    assert list(actual.flux_monitors) == list(expected.flux_monitors)
    for name, monitor in expected.flux_monitors.items():
        for field in ("e1_dft", "e2_dft", "h1_dft", "h2_dft"):
            _assert_same_bits(getattr(actual.flux_monitors[name], field), getattr(monitor, field))


@pytest.mark.parametrize("path", ["uniform", "graded"])
def test_forward_declared_waveguide_matches_run(path):
    expected = build_p_waveguide(path).run(n_steps=150, skip_preflight=True)
    actual = build_p_waveguide(path).forward(n_steps=150, skip_preflight=True, checkpoint=False)
    assert actual.waveguide_ports is not None
    assert list(actual.waveguide_ports) == list(expected.waveguide_ports)
    assert actual.waveguide_sparams is not None
    assert list(actual.waveguide_sparams) == list(expected.waveguide_sparams)
    for name, sparams in expected.waveguide_sparams.items():
        for field in ("s11", "s21"):
            _assert_same_bits(getattr(actual.waveguide_sparams[name], field), getattr(sparams, field))


@pytest.mark.parametrize("path", ["uniform", "graded"])
def test_forward_declared_flux_material_gradient(path):
    sim = build_o_flux(path)
    grid = sim._build_nonuniform_grid() if path == "graded" else sim._build_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss(eps_override):
        result = sim.forward(eps_override=eps_override, n_steps=150,
                             skip_preflight=True, checkpoint=False)
        return jnp.sum(jnp.abs(flux_spectrum(result.flux_monitors["census"])))

    gradient = np.asarray(jax.grad(loss)(eps))
    assert np.isfinite(gradient).all()
    assert np.any(gradient != 0)
