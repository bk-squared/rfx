"""On the uniform lanes a declared soft source's drive depends on its kind only (run()/forward() census row D2).

run() on a closed boundary built the 'current' drive as (Cb/dV)*w and forward() as
(1/dV)*(Cb*w): the same increment, rounded differently, and in a lossless PEC box the
1-ULP drive difference grew to tens of ULP of the field. The one product keeps Cb a
factor of multiplies only: XLA compiles a traced Cb divided by a constant dV as a
multiply by the reciprocal under jit/vmap, so Cb/dV rounds differently per trace.
The expected tables here are formed with numpy, not with the helper.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.model.source_coefficients import source_table
from rfx.sources.port_drive import soft_source

NATIVES = ('raw', 'cb')
DV = 1e-12
# Cb = dt/eps for 100 permittivities, dt of a 1 mm cell; Cb/dV under jit differs
# from eager Cb/dV for about a third of them.
CBS = (np.float32(1.9e-12) / (np.float32(8.854e-12) *
                              np.linspace(1.0, 5.95, 100, dtype=np.float32)))


def _samples():
    t = np.arange(150, dtype=np.float32) * np.float32(9.6e-13)
    return np.asarray(GaussianPulse(f0=10e9, bandwidth=0.5)(t), dtype=np.float32)


@pytest.mark.parametrize('native', NATIVES)
def test_current_drive_is_one_product_for_every_helper(native):
    w = _samples()
    cb = np.float32(0.0216839)
    got = np.asarray(source_table(w, 'current', native=native, coefficient=cb, volume=DV))
    np.testing.assert_array_equal(got, np.float32(1.0 / DV) * (cb * w))


@pytest.mark.parametrize('native', NATIVES + ('cb_over_dv',))
def test_field_drive_is_the_waveform_for_every_helper(native):
    w = _samples()
    got = np.asarray(source_table(w, 'field', native=native, coefficient=np.float32(3.0)))
    np.testing.assert_array_equal(got, w)


@pytest.mark.parametrize('native', NATIVES)
def test_current_drive_is_the_same_under_jit_and_vmap(native):
    w = jnp.asarray(_samples())

    def table(cb):
        return source_table(w, 'current', native=native, coefficient=cb, volume=DV)

    eager = np.stack([np.asarray(table(jnp.float32(cb))) for cb in CBS])
    np.testing.assert_array_equal(np.asarray(jax.jit(jax.vmap(table))(jnp.asarray(CBS))), eager)
    np.testing.assert_array_equal(
        np.stack([np.asarray(jax.jit(table)(jnp.float32(cb))) for cb in CBS[:10]]), eager[:10])


@pytest.mark.parametrize('kind', ('current', None))
def test_graded_drive_keeps_the_slab_product(kind):
    # The graded single-device table and the multi-device slabs (which supply
    # Cb/dV already resolved) form the same product, so the graded helper keeps
    # (Cb/dV)*w; changing it alone splits single- from multi-device graded runs.
    w = _samples()
    cb = np.float32(0.0216839)
    got = np.asarray(source_table(w, kind, native='cb_over_dv', coefficient=cb, volume=DV))
    np.testing.assert_array_equal(got, (cb / np.float32(DV)) * w)


def test_forward_with_traced_permittivity_is_the_same_under_jit():
    sim = Simulation(freq_max=10e9, domain=(.016, .016, .012), dx=.001, boundary='pec')
    sim.add_source((.008, .008, .004), 'ez', waveform=GaussianPulse(f0=5e9), amplitude_kind='current')
    sim.add_probe((.005, .006, .004), 'ez')
    eps = jnp.full(sim._build_grid().shape, 1.05, jnp.float32)

    def probe(eps):
        return sim.forward(n_steps=60, eps_override=eps, skip_preflight=True,
                           checkpoint=False).time_series

    np.testing.assert_array_equal(np.asarray(jax.jit(probe)(eps)), np.asarray(probe(eps)))


def test_soft_source_refuses_an_unresolved_kind():
    sim = Simulation(freq_max=10e9, domain=(.016, .016, .012), dx=.001, boundary='pec')
    grid = sim._build_grid()
    with pytest.raises(ValueError, match="amplitude_kind 'field' or 'current'"):
        soft_source(grid, (.008, .008, .004), 'ez', GaussianPulse(f0=5e9), 10,
                    materials=None, amplitude_kind=None)
