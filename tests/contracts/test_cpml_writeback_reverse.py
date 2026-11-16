"""Pin ordered CPML write-back values, JVPs, and slab-local reverse work.

A CPML correction's reverse pass touches only its slab: the field cotangent
passes through unchanged, and each correction receives its owned slab slice.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries import _cpml_slab


def _equations(jaxpr):
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    for eqn in jaxpr.eqns:
        yield eqn
        for value in eqn.params.values():
            for sub in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(sub, "eqns") or hasattr(getattr(sub, "jaxpr", None), "eqns"):
                    yield from _equations(sub)


@pytest.fixture
def slab_case(monkeypatch):
    monkeypatch.setattr(_cpml_slab, "writeback_form", lambda: "slab")
    shape = (11, 9, 7)
    windows = ((slice(2, 6), slice(None), slice(None)),
               (slice(None), slice(3, 7), slice(None)))
    rng = np.random.default_rng(2198)
    field = rng.standard_normal(shape).astype(np.float32)
    values = tuple(rng.standard_normal(field[sl].shape).astype(np.float32)
                   for sl in windows)
    owner = np.indices(values[0].shape).sum(axis=0) % 3 == 0
    # The overlap also witnesses float32 rounding in production term order.
    field[3, 4, 2] = np.float32(2**24)
    values[0][1, 4, 2] = np.float32(-(2**24))
    values[1][3, 1, 2] = np.float32(1)
    owner[1, 4, 2] = True
    primals = (field, *values)
    tangents = tuple(rng.standard_normal(v.shape).astype(np.float32) for v in primals)
    ct = (np.arange(1, field.size + 1, dtype=np.float32).reshape(shape)
          / np.float32(128))

    def apply(field, x_value, y_value):
        return _cpml_slab.apply_ordered(field, (
            (windows[0], x_value, jnp.asarray(owner)),
            (windows[1], y_value),
        ))

    return apply, windows, owner, primals, tangents, ct


def test_reverse_has_no_domain_writeback(slab_case):
    apply, _, _, primals, _, ct = slab_case
    _, pullback = jax.vjp(apply, *map(jnp.asarray, primals))
    equations = tuple(_equations(jax.make_jaxpr(pullback)(jnp.asarray(ct))))
    assert not any(eqn.primitive.name == "pad" for eqn in equations)
    domain_writes = [eqn for eqn in equations
                     if eqn.primitive.name == "dynamic_update_slice"
                     and any(getattr(var.aval, "shape", None) == primals[0].shape
                             for var in eqn.outvars)]
    assert not domain_writes, domain_writes


def test_reverse_equals_owned_slab_slices(slab_case):
    apply, windows, owner, primals, _, ct = slab_case
    _, pullback = jax.vjp(apply, *map(jnp.asarray, primals))
    d_field, d_x, d_y = pullback(jnp.asarray(ct))
    np.testing.assert_array_equal(d_field, ct)
    np.testing.assert_array_equal(d_x, np.where(owner, ct[windows[0]], np.float32(0)))
    np.testing.assert_array_equal(d_y, ct[windows[1]])


def test_forward_equals_ordered_float32_adds(slab_case):
    apply, windows, owner, primals, _, _ = slab_case
    expected = primals[0].copy()
    for sl, value, mask in zip(windows, primals[1:], (owner, None)):
        updated = (expected[sl] + value).astype(np.float32)
        expected[sl] = updated if mask is None else np.where(mask, updated, expected[sl])
    actual = apply(*map(jnp.asarray, primals))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert actual[3, 4, 2] == np.float32(1)


def test_jvp_equals_owned_slab_adds(slab_case):
    apply, windows, owner, primals, tangents, _ = slab_case
    _, actual = jax.jvp(apply, tuple(map(jnp.asarray, primals)),
                        tuple(map(jnp.asarray, tangents)))
    expected = tangents[0].copy()
    for sl, d_value, mask in zip(windows, tangents[1:], (owner, None)):
        d_term = d_value.astype(np.float32)
        if mask is not None:
            d_term = np.where(mask, d_term, np.float32(0))
        expected[sl] = expected[sl] + d_term
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
