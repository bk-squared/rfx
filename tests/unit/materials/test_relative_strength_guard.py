"""T2: opt-in SI underflow warning and an unchanged primal/transpose/JVP."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import AccuracyWarning
from rfx.materials import lorentz
from rfx.materials.relative_strength import RelativeStrength
from tests.unit.materials.test_relative_strength_gradient import KAPPA, amplitude, objective


def _read(fn, transform, p, expect):
    context = pytest.warns(AccuracyWarning, match="underflowed to zero") if expect else warnings.catch_warnings(record=True)
    with context as caught:
        warnings.simplefilter("always")
        value, fwd = jax.jvp(fn, (p,), (jnp.ones_like(p),))
        rev = getattr(jax, transform)(fn)(p)
        jax.block_until_ready((value, rev, fwd))
        jax.effects_barrier()
        error = abs(float(rev) - float(fwd)) / abs(float(fwd))
        count = sum("underflowed to zero" in str(w.message) for w in caught)
        print(f"T2 read rev={float(rev):.12e} fwd={float(fwd):.12e} relative={error:.9e} warnings={count}")
    underflow = [w for w in caught if "underflowed to zero" in str(w.message)]
    assert len(underflow) == int(expect)
    assert all(w.category is AccuracyWarning and w.message.count == 1 for w in underflow)
    return tuple(np.asarray(x) for x in (value, rev, fwd))


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("transform", ["grad", "jacrev"])
@pytest.mark.parametrize("flush", [False, True], ids=["a1", "underflow"])
def test_si_guard(lane, transform, flush, monkeypatch):
    a = amplitude(lane) if flush else 1.
    p = jnp.log(jnp.float32(KAPPA))
    monkeypatch.setenv("RFX_POLE_UNDERFLOW_CHECK", "1")
    on = _read(objective(lane, a, relative=False), transform, p, flush)
    monkeypatch.delenv("RFX_POLE_UNDERFLOW_CHECK", raising=False)
    off = _read(objective(lane, a, relative=False), transform, p, False)
    for actual, expected in zip(on, off):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    assert float(on[2]) != 0
    assert (float(on[1]) == 0) == flush
    error = abs(float(on[1]) - float(on[2])) / abs(float(on[2]))
    print(f"T2 {lane} {transform} a={a:.12e}: J={float(on[0]):.12e} "
          f"rev={float(on[1]):.12e} fwd={float(on[2]):.12e} relative={error:.9e} "
          f"warnings_on={int(flush)} warnings_off=0 bitwise=True")


def test_scalar_calls_module_guard(monkeypatch):
    calls = []
    original = lorentz.pole_underflow_guard

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lorentz, "pole_underflow_guard", counted)
    monkeypatch.setenv("RFX_POLE_UNDERFLOW_CHECK", "1")
    jax.jvp(lambda k: lorentz.pole_update_coeffs(0., 2e9, k, 1e-12)[2],
            (jnp.float32(KAPPA),), (jnp.float32(1.),))
    assert len(calls) == 1
    calls.clear()
    jax.jvp(lambda q: lorentz.pole_update_coeffs(0., 2e9, RelativeStrength(KAPPA, q), 1e-12)[2],
            (jnp.float32(1.),), (jnp.float32(1.),))
    assert calls == []
    monkeypatch.setenv("RFX_POLE_UNDERFLOW_CHECK", "0")
    jax.jvp(lambda k: lorentz.pole_update_coeffs(0., 2e9, k, 1e-12)[2],
            (jnp.float32(KAPPA),), (jnp.float32(1.),))
    assert calls == []


def test_batched_guard_zero_count_does_not_warn(monkeypatch):
    monkeypatch.setenv("RFX_POLE_UNDERFLOW_CHECK", "1")
    def fn(k, scale):
        return lorentz.pole_update_coeffs(0., 2e9, k, 1e-12)[2] * scale
    with pytest.warns(AccuracyWarning, match="underflowed to zero") as caught:
        result = jax.vmap(jax.grad(fn), in_axes=(None, 0))(
            jnp.float32(KAPPA), jnp.asarray([1., 1e-7]))
        jax.block_until_ready(result)
        jax.effects_barrier()
    assert len(caught) == 1 and caught[0].message.count == 1
    assert result[0] != 0 and result[1] == 0
