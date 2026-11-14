"""Electric updates commute with the physical-field envelope transformation."""
import json
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials, init_state
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import drude_pole, init_lorentz, lorentz_pole
from rfx.stepping.uniform_setup import _update_e_with_optional_dispersion

BRANCHES = ("update_e", "update_e_aniso", "update_e_aniso_inv", "debye", "lorentz", "drude", "mixed")


@lru_cache(None)
def operands(axis, branch):
    rng = np.random.default_rng(703)
    shape = tuple(16 if a == axis else 6 for a in range(3))
    periodic = tuple(a == axis for a in range(3))
    dt, dx = 1e-12, .002
    material = init_materials(shape)._replace(
        eps_r=jnp.asarray(rng.uniform(1.5, 4., shape), dtype=jnp.float32),
        sigma=jnp.asarray(rng.uniform(0., .01, shape), dtype=jnp.float32))
    eps = tuple(jnp.asarray(rng.uniform(1.5, 4., shape), dtype=jnp.float32) for _ in range(3))

    def random_complex(shape, scale):
        return jnp.asarray(scale*(rng.normal(size=shape)+1j*rng.normal(size=shape)), dtype=jnp.complex64)

    state = init_state(shape, field_dtype=jnp.complex64)
    state = state._replace(**{name: random_complex(shape, 1. if name[0] == "e" else .003)
                              for name in ("ex", "ey", "ez", "hx", "hy", "hz")})
    options = dict(periodic=periodic, high_wall=(False, False, False))
    if branch == "update_e_aniso":
        options["aniso_eps"] = eps
    elif branch == "update_e_aniso_inv":
        options["aniso_inv_eps"] = tuple(1/e for e in eps)
    mask = jnp.asarray(rng.random(shape) > .4)
    if branch in ("debye", "mixed"):
        coeffs, carry = init_debye(
            [DebyePole(float(rng.uniform(.5, 2.)), float(rng.uniform(2e-11, 6e-11)))],
            material, dt, mask=mask, periodic=periodic, high_wall=(False, False, False))
        options["debye"] = coeffs, jax.tree.map(lambda a: random_complex(a.shape, 1e-11), carry)
    if branch in ("lorentz", "drude", "mixed"):
        pole = (drude_pole(2*np.pi*6e9, 2*np.pi*.5e9) if branch == "drude" else
                lorentz_pole(float(rng.uniform(.5, 2.)), 2*np.pi*8e9, 2*np.pi*.5e9))
        coeffs, carry = init_lorentz([pole], material, dt, mask=mask,
                                    periodic=periodic, high_wall=(False, False, False))
        options["lorentz"] = coeffs, jax.tree.map(lambda a: random_complex(a.shape, 1e-11), carry)
    return state, material, dt, dx, options


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("pol,axis", [("ez", 1), ("ey", 2)])
@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize("phase_kind", ["concrete", "traced"])
def test_electric_envelope_identity(branch, pol, axis, m, phase_kind):
    state, material, dt, dx, options = operands(axis, branch)
    phase = tuple(complex(np.exp(-2j*np.pi*m/16)) if a == axis else 1.+0j for a in range(3))
    shape = [1, 1, 1]
    shape[axis] = 16
    mapping = jnp.asarray(np.exp(-2j*np.pi*m*np.arange(16)/16).reshape(shape), dtype=jnp.complex64)
    physical = state._replace(**{name: getattr(state, name)*mapping
                                for name in ("ex", "ey", "ez", "hx", "hy", "hz")})
    physical_options = dict(options)
    for name in ("debye", "lorentz"):
        if name in options:
            coeffs, carry = options[name]
            physical_options[name] = coeffs, jax.tree.map(lambda a: a*mapping, carry)
    # The reference calls the same material branch on physical fields with no phase.
    expected = _update_e_with_optional_dispersion(physical, material, dt, dx, bloch=None, **physical_options)
    if phase_kind == "concrete":
        actual = _update_e_with_optional_dispersion(state, material, dt, dx, bloch=phase, **options)
    else:
        actual = jax.jit(lambda p: _update_e_with_optional_dispersion(
            state, material, dt, dx, bloch_operand=p, **options))(tuple(jnp.complex64(p) for p in phase))
    arrays = [(name, getattr(actual[0], name), getattr(expected[0], name)) for name in ("ex", "ey", "ez")]
    for name, got, ref in zip(("debye", "lorentz"), actual[1:], expected[1:]):
        if got is not None:
            arrays.extend((f"{name}.{field}", a, b) for field, a, b in zip(got._fields, got, ref))
    errors = {}
    for name, got, ref in arrays:
        mapped, target = np.asarray(got*mapping), np.asarray(ref)
        peak = np.float32(np.max(np.abs(target)))
        ulp = float(np.spacing(peak))
        error = float(np.max(np.abs(mapped-target)))
        errors[name] = error/ulp
        assert error <= 64*ulp, (branch, pol, m, phase_kind, name, error, peak, error/ulp)
    print("envelope_identity " + json.dumps(dict(branch=branch, pol=pol, m=m, phase=phase_kind, ulps=errors)))
