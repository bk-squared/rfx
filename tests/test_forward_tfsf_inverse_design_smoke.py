"""End-to-end inverse-design smoke for the differentiable TFSF plane-wave forward.

Proves the capability the PEFM lab needs (RIS / FSS / RCS inverse design, anyscale
validation ladder): a plane-wave-driven objective can actually be OPTIMIZED by
gradient descent through Simulation.forward() — the loss goes down and the design
variable moves toward the target. This is the "it optimizes" capstone beyond a
single-point gradient check (test_forward_tfsf_differentiable.py) and the gradient
doctrine (test_forward_tfsf_gradient_doctrine.py).
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation


def _adam(v_and_g, x0, steps, lr=0.25, lo=1.0, hi=8.0):
    """Minimal Adam (no optax dependency — optax is an optional extra not in [dev]).
    Returns the value history (last entry is the final objective)."""
    x, m, v = float(x0), 0.0, 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for i in range(1, steps + 1):
        val, g = v_and_g(x)
        hist.append(float(val))
        g = float(g)
        assert np.isfinite(g), "P0: non-finite gradient during optimization"
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** i), v / (1 - b2 ** i)
        x = float(np.clip(x - lr * mh / (np.sqrt(vh) + eps), lo, hi))
    return x, hist


@pytest.mark.slow
def test_forward_tfsf_inverse_design_optimizes():
    """A plane-wave-driven objective is minimized by Adam through forward():
    jax.grad drives the scatterer permittivity so the response drops — the
    end-to-end inverse-design loop the PEFM RIS/FSS/RCS goals need. Asserts the
    objective meaningfully decreases and every gradient stays finite (NaN = P0)."""
    sim = Simulation(freq_max=10e9, domain=(0.24, 0.08, 0.006), dx=0.002,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=5e9, bandwidth=0.5, polarization="ez", direction="+x",
                        waveform="differentiated_gaussian")
    sim.add_probe((0.0, 0.0, 0.0), component="ez")
    shp = sim.run(n_steps=1).state.ez.shape
    xi = shp[0] // 2

    def objective(eps_val):
        eps = jnp.ones(shp, jnp.float32).at[xi:xi + 12, :, :].set(eps_val)
        fr = sim.forward(eps_override=eps, n_steps=500, checkpoint=True,
                         skip_preflight=True)
        return jnp.sum(jnp.abs(fr.time_series) ** 2).real

    v_and_g = jax.value_and_grad(objective)
    eps, history = _adam(lambda e: v_and_g(e), 4.0, steps=8, lr=0.25)
    history.append(float(objective(eps)))
    assert history[-1] < history[0] * 0.9, (
        f"Adam did not reduce the plane-wave objective through forward(): {history}"
    )
    assert eps != 4.0, "design variable never moved"
