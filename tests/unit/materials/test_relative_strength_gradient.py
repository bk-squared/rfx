"""T1: float32 reverse/forward agreement, with a measured SI underflow margin.

Bar fixed before measurement: |rev-fwd| <= 1e-4 |fwd|. The amplitude is
calibrated from this reduced scene's a=1 SI derivative to 1e-3 * 1.1755e-38.
"""
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, LorentzPole, RelativeStrength, Simulation

KAPPA = float((2 * np.pi * 3e9) ** 2)
STEPS = 64
# Filled from the independent a=1 forward-mode measurement of this scene.
CALIBRATION = {"uniform": -2.3828923545015776e-33, "graded": -1.8987599431685053e-35}


def objective(lane, amplitude, *, relative):
    def evaluate(p):
        k = RelativeStrength(KAPPA, jnp.exp(p)) if relative else jnp.exp(p)
        options = {"dx_profile": np.r_[.002, np.linspace(.001, .003, 13), .002]} if lane == "graded" else {}
        sim = Simulation(freq_max=6e9, domain=(.030, .016, .016), dx=.002,
                         cpml_layers=2, precision="float32", **options)
        sim.add_material("plasma", eps_r=1., lorentz_poles=[LorentzPole(0., 2e9, k)])
        sim.add(Box((.01035, .00435, .00435), (.01965, .01165, .01165)), material="plasma")
        sim.add_source((.00585, .00835, .00815), "ez",
                       waveform=GaussianPulse(f0=4e9, amplitude=amplitude))
        sim.add_probe((.02435, .00785, .00835), "ez")
        grid = sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()
        result = sim.forward(n_steps=STEPS, checkpoint=True, skip_preflight=True)
        trace = jnp.asarray(result.time_series).reshape(STEPS, -1)[:, 0]
        kernel = (jnp.exp(-2j * jnp.pi * jnp.asarray([3.5e9, 4.5e9])[:, None]
                         * jnp.arange(STEPS)[None, :] * grid.dt) * grid.dt)
        return jnp.sum(jnp.abs(kernel @ trace) ** 2)
    return evaluate


def amplitude(lane):
    return float(np.sqrt(1e-3 * 1.1755e-38 / abs(CALIBRATION[lane])))


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("compiled", [False, True], ids=["grad", "jit_grad"])
def test_relative_strength_gradient(lane, compiled, monkeypatch):
    monkeypatch.delenv("RFX_POLE_UNDERFLOW_CHECK", raising=False)
    start = time.monotonic()
    fn = objective(lane, amplitude(lane), relative=True)
    p = jnp.float32(0.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reverse = (jax.jit(jax.grad(fn)) if compiled else jax.grad(fn))(p)
        primal, forward = jax.jvp(fn, (p,), (jnp.ones_like(p),))
        jax.block_until_ready((reverse, primal, forward))
        jax.effects_barrier()
    error = abs(float(reverse) - float(forward)) / abs(float(forward))
    underflow = [w for w in caught if hasattr(w.message, "count")]
    elapsed = time.monotonic() - start
    print(f"T1 {lane} compiled={compiled}: a1_dJ_dkappa={CALIBRATION[lane]:.12e} "
          f"amplitude={amplitude(lane):.12e} J={float(primal):.12e} "
          f"rev={float(reverse):.12e} fwd={float(forward):.12e} relative={error:.9e} "
          f"warnings={len(underflow)} seconds={elapsed:.3f}")
    assert reverse.dtype == forward.dtype == jnp.float32
    assert float(forward) != 0
    assert error <= 1e-4
    assert not underflow
    assert elapsed < 20
