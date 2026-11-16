"""run()'s fence asks one question: is the caller inside a jax transformation?

The answer comes from jax's public trace state, compared with the state of a thread no
transformation has entered. On jax 0.11 the earlier route (reading the state inside
take_current_trace) raised AssertionError outside every transformation, so an eager
closed-box plane-wave run() could not start (#2115). The weekly latest-jax job runs this
file on jax 0.11; the PR lane runs it on jax 0.10.2.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.runners._admission import _under_external_jax_transform
from rfx.sources.sources import CustomWaveform


def _sim(kind):
    sim = Simulation(
        freq_max=30e9, domain=(0.018, 0.016, 0.014), dx=0.001,
        boundary="cpml", cpml_layers=4,
    )
    waveform = "differentiated_gaussian"
    if kind == "custom":
        waveform = CustomWaveform(lambda t: jnp.sin(2 * jnp.pi * 15e9 * t))
    sim.add_tfsf_source(
        f0=15e9, bandwidth=0.8, polarization="ez", direction="+x", margin=3,
        waveform=waveform, closed_box=kind == "box",
    )
    sim.add_probe((0.003, 0.005, 0.004), component="ez")
    return sim


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


@pytest.mark.parametrize("transform", list(TRANSFORMS))
@pytest.mark.parametrize("kind", ("custom", "box"))
def test_run_external_transforms_refuse_extended_tfsf(kind, transform):
    sim = _sim(kind)

    def trace(eps):
        result = sim.run(n_steps=8, skip_preflight=True)
        return jnp.sum(result.time_series ** 2) * eps

    message = (
        "CustomWaveform and closed_box TFSF cannot be passed through run() "
        "under external JAX transformations; differentiate forward() instead."
    )
    with pytest.raises(NotImplementedError) as exc:
        TRANSFORMS[transform](trace)
    mismatch = int(str(exc.value) != message)
    print(f"{kind}/{transform}: refusal_message_mismatch={mismatch}; bar=0",
          file=sys.stderr)
    assert str(exc.value) == message

    if transform == "jit":
        actual = sim.run(n_steps=8, skip_preflight=True)
        expected = _sim(kind).run(n_steps=8, skip_preflight=True)
        assert actual.time_series.shape == expected.time_series.shape == (8, 1)
        # Same eager program: the specified bar is bitwise, including signed zero.
        arrays = [("time_series", actual.time_series, expected.time_series)]
        arrays += [(c, getattr(actual.state, c), getattr(expected.state, c))
                   for c in ("ex", "ey", "ez", "hx", "hy", "hz")]
        for name, got, want in arrays:
            got, want = np.asarray(got), np.asarray(want)
            assert got.shape == want.shape and got.dtype == want.dtype
            assert np.all(np.isfinite(got)) and np.all(np.isfinite(want))
            mismatches = np.count_nonzero(
                got.view(np.uint8) != want.view(np.uint8))
            print(
                f"{kind}/eager_after_jit/{name}: byte_mismatches={mismatches}; "
                f"bar=0; peak={np.max(np.abs(want)):.9g}; steps=8; "
                f"mesh={expected.grid.shape}", file=sys.stderr,
            )
            assert mismatches == 0, (
                f"{kind}/eager_after_jit/{name}: {mismatches} differing bytes; bar=0")
