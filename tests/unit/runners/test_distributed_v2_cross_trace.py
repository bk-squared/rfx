"""sim.run(devices=...) stays within nine peak float32 ULP in every trace context.

Sources and probes straddle the slab cut, witnessing the ghost exchange under
jvp, vjp, value_and_grad, vmap and stop_gradient.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from rfx import Box, Simulation

pytestmark = pytest.mark.distributed

_STEPS = 30


def _devices():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("needs two CPU devices")
    return devices[:2]


def _forward(model):
    devices = _devices()

    def build(amplitude):
        boundary = "cpml" if model == "cpml" else "pec"
        layers = 1 if boundary == "cpml" else 0
        sim = Simulation(freq_max=15e9, domain=((16 - 2 * layers) * 1e-3, 4e-3, 4e-3),
                         dx=1e-3, boundary=boundary, cpml_layers=layers)
        grid = sim._build_grid()
        width = (grid.shape[0] + len(devices) - 1) // len(devices)
        for i in (width - 1, width):   # the last real cell of slab 0, the first of slab 1
            position = ((i - grid.pad_x_lo) * grid.dx, 2e-3, 2e-3)
            sim.add_source(position, "ez", amplitude_kind="field",
                           waveform=lambda t: amplitude * jnp.cos(t * 2e10))
            for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
                sim.add_probe(position, component)
        if model == "pec_block":
            x = (width - grid.pad_x_lo) * grid.dx
            sim.add(Box((x, 3e-3, 1e-3), (x + 1e-3, 4e-3, 3e-3)), material="pec")
        return sim

    def forward(amplitude):
        result = build(amplitude).run(n_steps=_STEPS, devices=devices)
        return result.time_series, result.state.ez, result.state.hy

    return forward


def _contexts(forward):
    def value_and_grad_aux():
        def loss(amplitude):
            outputs = forward(amplitude)
            return jnp.sum(outputs[0] ** 2), outputs
        (_, outputs), _ = jax.value_and_grad(loss, has_aux=True)(1.0)
        return outputs

    return {
        "jvp": lambda: jax.jvp(forward, (1.0,), (1.0,))[0],
        "vjp": lambda: jax.vjp(forward, 1.0)[0],
        "value_and_grad": value_and_grad_aux,
        "vmap": lambda: tuple(o[0] for o in jax.vmap(forward)(
            jnp.array([1.0, 1.0], dtype=jnp.float32))),
        "stop_gradient": lambda: jax.jvp(
            lambda a: forward(lax.stop_gradient(a)), (1.0,), (1.0,))[0],
    }


@pytest.mark.parametrize("model", ["pec_block", "cpml"])
def test_every_trace_context_matches_plain_within_nine_ulp(model):
    forward = _forward(model)
    plain = [np.asarray(o) for o in forward(1.0)]
    assert np.isfinite(plain[0]).all() and np.max(np.abs(plain[0])) > 0
    for name, context in _contexts(forward).items():
        outputs = [np.asarray(o) for o in context()]
        for label, got, want in zip(("trace", "ez", "hy"), outputs, plain):
            assert got.shape == want.shape and got.dtype == want.dtype, (name, label)
            assert np.isfinite(got).all() and np.isfinite(want).all(), (name, label)
            # PI (2026-09-23): allow compiler rounding within nine ULP at each array's peak.
            ulp = float(np.spacing(np.max(np.abs(want))))
            error = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
            assert error <= 9 * ulp, (name, label, error / ulp)
