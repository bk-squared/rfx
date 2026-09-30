"""tests/_x64_compat.enable_x64 takes the upstream argument on every JAX the lanes run.

JAX 0.6.2 (``jax.experimental``) and JAX >= 0.8 (top-level ``jax``) both accept
``enable_x64(new_val)``. The shim's own fallback used to take no argument, so a test
written as ``with enable_x64(True):`` passed on the 0.6.2 lane and raised TypeError on
0.10.2 (PR #1378's precision test). This pins the signature and the scoping.
"""
import jax

from tests._x64_compat import enable_x64


def test_enable_x64_accepts_new_val_and_restores():
    before = bool(jax.config.read("jax_enable_x64"))
    with enable_x64(True):
        assert jax.numpy.zeros(()).dtype == jax.numpy.float64
        with enable_x64(False):
            assert jax.numpy.zeros(()).dtype == jax.numpy.float32
        assert jax.numpy.zeros(()).dtype == jax.numpy.float64
    assert bool(jax.config.read("jax_enable_x64")) == before


def test_enable_x64_default_is_true():
    with enable_x64():
        assert jax.numpy.zeros(()).dtype == jax.numpy.float64
