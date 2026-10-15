"""compute_waveguide_s_matrix: eps_override together with subpixel smoothing is refused.

The smoothed permittivity tensor is built from the declared shapes and is the
only permittivity the E update reads, so the override was dropped: with a
slab's permittivity doubled through eps_override the S-matrix came back
identical (difference exactly 0; 1.30 without smoothing). Tracker #1524.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Box, Simulation

A, B, LX, DX = 22.86e-3, 10.16e-3, 60e-3, 2e-3
FREQS = np.linspace(8.5e9, 11.5e9, 3)


def _sim():
    sim = Simulation(freq_max=12e9, domain=(LX, A, B), dx=DX, boundary="cpml", cpml_layers=8)
    sim.add_material("slab", eps_r=2.0)
    sim.add(Box((0.026, 0, 0), (0.035, A, B)), material="slab")
    for direction, x in (("+x", 0.01), ("-x", LX - 0.01)):
        sim.add_waveguide_port(direction=direction, x_position=x, y_range=(0., A),
                               z_range=(0., B), n_modes=1, freqs=FREQS)
    return sim


@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
def test_eps_override_with_smoothing_is_refused_before_the_solve(smoothing):
    sim = _sim()
    override = jnp.ones(sim._build_grid().shape, dtype=jnp.float32) * 3.0
    with pytest.raises(NotImplementedError) as info:
        sim.compute_waveguide_s_matrix(n_steps=50, eps_override=override,
                                       subpixel_smoothing=smoothing)
    text = str(info.value)
    assert "subpixel_smoothing" in text and "eps_override would not reach the solve" in text


def test_the_refusal_comes_before_the_override_is_used():
    """A wrong-shaped override is refused for the combination, not for its shape."""
    sim = _sim()
    with pytest.raises(NotImplementedError, match="eps_override would not reach the solve"):
        sim.compute_waveguide_s_matrix(n_steps=50, eps_override=jnp.ones((2, 2, 2)),
                                       subpixel_smoothing=True)


@pytest.mark.parametrize("off", [False, None])
def test_smoothing_switched_off_by_any_falsy_value_is_not_refused(off):
    import warnings
    sim = _sim()
    ones = jnp.ones(sim._build_grid().shape, dtype=jnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.compute_waveguide_s_matrix(n_steps=20, eps_override=ones * 3.0,
                                                subpixel_smoothing=off)
    assert np.all(np.isfinite(np.asarray(result.s_params)))


@pytest.mark.parametrize("kwargs", ["override", "smoothing", "sigma_and_smoothing"])
def test_each_input_alone_is_still_admitted(kwargs):
    """Only the combination is refused; sigma_override is read by the tensor kernels."""
    import warnings
    sim = _sim()
    ones = jnp.ones(sim._build_grid().shape, dtype=jnp.float32)
    call = {"override": dict(eps_override=ones * 3.0),
            "smoothing": dict(subpixel_smoothing=True),
            "sigma_and_smoothing": dict(sigma_override=ones * 0.01,
                                        subpixel_smoothing=True)}[kwargs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.compute_waveguide_s_matrix(n_steps=20, **call)
    assert np.all(np.isfinite(np.asarray(result.s_params)))
