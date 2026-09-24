"""``rfx.optimize(jit=True)`` compiles the loss and its gradient once per call (#1225).

Without ``jit=True`` each iteration of ``optimize()`` calls an un-jitted
``jax.value_and_grad`` over ``forward()``, which compiles the whole FDTD solve
again every time — on a board-sized model most of the iteration's wall time.
With it, the first iteration compiles and the rest reuse the program. The
default stays eager (an objective that reads a traced value on the host cannot
run under ``jax.jit``); ``test_optimize_multistart.py``'s legacy-loop
bit-identity gate covers that default.

The model is a 15 mm PEC-walled box with a port, a probe, a small permittivity
design region and 20 steps: the physics is incidental, the compile count is
the subject. Compiles are counted with ``jax.monitoring``
backend-compile events, with a positive control (the call must compile at
least once) so a counter that saw nothing cannot pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.optimize import DesignRegion, optimize

_BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"
_COUNTER = {"on": False, "n": 0}
_LISTENING = []


def _listener(event, duration, **kwargs):
    if _COUNTER["on"] and event == _BACKEND_COMPILE:
        _COUNTER["n"] += 1


def _compiles(fn, *args):
    """Run ``fn(*args)`` to completion; return (result, backend compiles)."""
    if not _LISTENING:
        jax.monitoring.register_event_duration_secs_listener(_listener)
        _LISTENING.append(True)
    _COUNTER["n"] = 0
    _COUNTER["on"] = True
    try:
        out = jax.block_until_ready(fn(*args))
    finally:
        _COUNTER["on"] = False
    return out, _COUNTER["n"]


def _tiny_sim():
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.011, 0.0075, 0.0075), "ez")
    return sim


def _tiny_region():
    return DesignRegion(corner_lo=(0.006, 0.006, 0.006),
                        corner_hi=(0.010, 0.010, 0.009), eps_range=(1.0, 4.4))


def _optimize(n_iters):
    return optimize(_tiny_sim(), _tiny_region(),
                    lambda r: -jnp.sum(r.time_series ** 2),
                    n_iters=n_iters, lr=0.2, n_steps=20, verbose=False,
                    skip_preflight=True, jit=True)


def test_optimize_iterations_after_the_first_do_not_compile():
    """``optimize(jit=True, n_iters=4)`` compiles as often as ``n_iters=1``.

    Each ``optimize()`` call builds its own objective closure and so compiles
    it once; the iterations after the first reuse that program. Without
    ``jit=True`` every iteration compiles the solve again. A warm-up call
    first takes the one-time compiles of the eager Adam arithmetic out of
    both counts.
    """
    _optimize(1)
    res_1, n_1 = _compiles(_optimize, 1)
    res_4, n_4 = _compiles(_optimize, 4)
    assert n_1 >= 1, "the compile counter saw no compile in optimize()"
    assert n_4 == n_1, (
        f"optimize(n_iters=4) did {n_4} backend compiles against {n_1} for "
        f"n_iters=1: iterations after the first compiled again")
    assert len(res_4.loss_history) == 4
    assert np.all(np.isfinite(res_4.loss_history))
