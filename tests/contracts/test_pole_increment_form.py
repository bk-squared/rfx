"""B1: Drude equilibrium and vacuum fill in the production pole applier."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.materials.lorentz import LorentzCoeffs, LorentzPole, lorentz_p_component, lorentz_pole_coeffs
from rfx.stepping.slab import Slab, cut_pole_coeffs

DT = np.float64(.5e-3 * .99 / (299792458.0 * np.sqrt(3)))
WP = np.float64(2 * np.pi * 3e9)
NU = np.float64(2.3e8)
GAMMAS = ([0.] + [NU * np.exp(x) for x in (0, -.0005, .0005, -.001, .001, -.002, .002)]
          + [1.000e7, 1.063e7, 1.126e7])


def coefficients(w0, gamma, shape=(256,), fractions=None):
    terms = lorentz_pole_coeffs([LorentzPole(w0, gamma / 2, WP**2)], DT, shape, [fractions])
    one = tuple(jnp.ones(shape) for _ in range(3))
    return LorentzCoeffs(one, one, *terms, one)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("gamma", GAMMAS)
def test_drude_stationary(gamma, compiled):
    coeffs = coefficients(np.float64(0), gamma)
    assert np.all(np.asarray(coeffs.restoring) == 0), "Drude restoring must be exactly zero"
    p = jnp.asarray((np.random.default_rng(0).standard_normal(256) * 1e-12).astype(np.float32))[None]
    v = jnp.zeros_like(p)
    apply = jax.jit(lorentz_p_component, static_argnums=1) if compiled else lorentz_p_component
    p_new, v_new = apply(coeffs, 2, jnp.zeros(256), p, v)
    assert np.asarray(p_new).tobytes() == np.asarray(p).tobytes()
    assert np.asarray(v_new).tobytes() == np.asarray(v).tobytes()


@pytest.mark.parametrize("compiled", [False, True])
def test_unreached_and_slab_fill_zero_polarization(compiled):
    shape = (5, 2, 2)
    zero = tuple(jnp.zeros(shape) for _ in range(3))
    coeffs = coefficients(WP, NU, shape, zero)
    cut = cut_pole_coeffs(coeffs, Slab(5, 2, 1), "lorentz", dt=DT)
    apply = jax.jit(lorentz_p_component, static_argnums=1) if compiled else lorentz_p_component
    for fields in (coeffs, jax.tree.map(lambda x: x[0], cut)):
        assert np.all(np.asarray(fields.damping) == 1)
        assert np.all(np.asarray(fields.restoring) == 1)
        rng = np.random.default_rng(7)
        p = jnp.asarray(rng.standard_normal(fields.damping.shape), dtype=jnp.float32)
        v = jnp.asarray(rng.standard_normal(fields.damping.shape), dtype=jnp.float32)
        e = jnp.ones(fields.damping.shape[1:])
        p_new, _ = apply(fields, 2, e, p, v)
        assert np.asarray(p_new).tobytes() == np.zeros(p.shape, np.float32).tobytes()
