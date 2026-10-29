"""run()'s fence asks one question: is the caller inside a jax transformation?

The answer comes from jax's public trace state, compared with the state of a thread no
transformation has entered. On jax 0.11 the earlier route (reading the state inside
take_current_trace) raised AssertionError outside every transformation, so an eager
closed-box plane-wave run() could not start (#2115). The weekly latest-jax job runs this
file on jax 0.11; the PR lane runs it on jax 0.10.2.
"""

import jax
import jax.numpy as jnp
import pytest

from rfx.runners._admission import _under_external_jax_transform


def _seen_inside(transform):
    seen = []

    def body(x):
        seen.append(_under_external_jax_transform())
        return x * 2.0

    transform(body)
    return seen[0]


TRANSFORMS = {
    "jit": lambda f: jax.jit(f)(1.0),
    "grad": lambda f: jax.grad(f)(1.0),
    "vmap": lambda f: jax.vmap(f)(jnp.ones(3)),
    "jvp": lambda f: jax.jvp(f, (1.0,), (1.0,)),
    "eval_shape": lambda f: jax.eval_shape(f, 1.0),
    "jit_of_grad": lambda f: jax.jit(jax.grad(f))(1.0),
}


def test_compile_time_eval_is_eager():
    with jax.ensure_compile_time_eval():
        assert _under_external_jax_transform() is False


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_every_transform_is_seen(name):
    assert _seen_inside(TRANSFORMS[name]) is True


def test_eager_after_a_transform_is_eager_again():
    _seen_inside(TRANSFORMS["jit"])
    assert _under_external_jax_transform() is False
