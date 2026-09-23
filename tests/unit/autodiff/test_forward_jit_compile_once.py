"""An optimisation step through ``Simulation.forward()`` compiles once (#1225).

The physics: a patch on a thin substrate, meshed with a graded z profile, fed
by a 50 ohm wire port from the ground plane to the patch, with a small
permittivity design box in the substrate beside the feed. The objective is
|S11|^2 over a few bins, differentiated with respect to the box permittivity.
Neither the value nor the gradient is the subject here; the wall time of an
optimisation iteration is.

What a user saw: ``jax.value_and_grad(loss)`` compiles the whole solve again on
every call, and the obvious remedy, ``jax.jit(jax.value_and_grad(loss))``,
failed while tracing on this board. The wire port decides which of its edges
are live by reading the realized PEC edge mask on the host, and under an outer
``jax.jit`` that mask — built from the geometry alone — was a tracer. The fix
evaluates the set-up of a model WITH A WIRE PORT at trace time (``forward()``
enters ``jax.ensure_compile_time_eval()`` when it finds itself staged).
``rfx.optimize(jit=True)`` is tested in ``test_optimize_jit_compile_once.py``.

Every model without a wire port keeps the code path it had before, so its
jitted program — and every plain-versus-jitted comparison of it — is
unchanged (the PI's cross-trace rule, ledger decision of 2026-09-23). The
predicate test below pins that scope: widening the trace-time evaluation to
other models turns it red.

Compiles are counted with ``jax.monitoring`` backend-compile events, the same
counter the issue's measurements used. Each count is guarded by a positive
control (a first call that must compile), so a counter that saw nothing
cannot pass a "0 compiles" assertion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, GaussianPulse, Simulation

F0 = 8e9
DX = 1e-3
# Graded z: six 0.5 mm cells through the ground and the substrate, then a
# 1.25x growth into the air above the patch.
DZ = np.concatenate([np.full(6, 0.5e-3), 0.5e-3 * 1.25 ** np.arange(1, 9)])
Z_GND = 1.0e-3
SUB_H = 1.5e-3
BOX = ((10e-3, 7e-3, Z_GND + 0.25e-3), (11e-3, 9e-3, Z_GND + 1.25e-3))
N_STEPS = 30

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


def _board():
    sim = Simulation(freq_max=2 * F0, domain=(18e-3, 16e-3, float(DZ.sum())),
                     dx=DX, boundary="cpml", cpml_layers=5, dz_profile=DZ)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((2e-3, 2e-3, Z_GND), (16e-3, 14e-3, Z_GND + SUB_H)),
            material="sub")
    sim.add(Box((2e-3, 2e-3, Z_GND - 0.5e-3), (16e-3, 14e-3, Z_GND)),
            material="pec")
    sim.add(Box((6e-3, 5e-3, Z_GND + SUB_H),
                (12e-3, 11e-3, Z_GND + SUB_H + 0.5e-3)), material="pec")
    sim.add_port(position=(7e-3, 8e-3, Z_GND), component="ez", extent=SUB_H,
                 impedance=50.0, direction="-x",
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    return sim


def _box_eps(sim, seed=4):
    from rfx.nonuniform import position_to_index
    grid = sim._build_nonuniform_grid()
    lo, hi = position_to_index(grid, BOX[0]), position_to_index(grid, BOX[1])
    shape = tuple(int(hi[d]) - int(lo[d]) + 1 for d in range(3))
    return jnp.asarray(2.0 + np.random.default_rng(seed).random(shape),
                       jnp.float32)


def _s11_loss(sim):
    def loss(eps_box):
        r = sim.forward(design_box=BOX, design_eps_override=eps_box,
                        n_steps=N_STEPS, checkpoint=False, skip_preflight=True)
        return jnp.sum(jnp.abs(r.s_params[0, 0, 3:8]) ** 2)
    return loss


def test_the_board_realizes_the_conductors_the_port_reads():
    """The ground and the patch are realized as PEC cells on this mesh.

    Without them the port's live-edge test has no mask to read and the board
    would not exercise the trace failure at all.
    """
    sim = _board()
    grid = sim._build_nonuniform_grid()
    _, _, _, pec_mask = sim._assemble_materials_nu(
        grid, pec_sheets=[], pec_wires=[])
    assert pec_mask is not None and int(np.asarray(pec_mask).sum()) > 0


def test_graded_board_with_a_wire_port_lowers_under_jit():
    """``jax.jit(jax.value_and_grad(|S11|^2))`` traces and lowers on the board.

    Before #1225 this raised TracerArrayConversionError from the wire port's
    live-edge test on the realized PEC edge mask.
    """
    sim = _board()
    lowered = jax.jit(jax.value_and_grad(_s11_loss(sim))).lower(_box_eps(sim))
    assert lowered is not None


def test_repeat_calls_of_the_jitted_gradient_do_not_compile():
    """First call compiles; the next two, at new design values, compile nothing."""
    sim = _board()
    step = jax.jit(jax.value_and_grad(_s11_loss(sim)))
    eps = _box_eps(sim)

    (v0, g0), n0 = _compiles(step, eps)
    assert n0 >= 1, "the compile counter saw no compile on the first call"
    (v1, g1), n1 = _compiles(step, eps + 0.1)
    (v2, g2), n2 = _compiles(step, eps + 0.2)
    assert (n1, n2) == (0, 0), f"repeat calls compiled {n1} and {n2} times"
    # The repeat calls ran the solve at the new design, not a cached answer.
    assert np.isfinite(float(v0)) and float(v1) != float(v0)
    assert float(jnp.max(jnp.abs(g0))) > 0.0


def _conductor_without_a_wire_port():
    """A PEC block, a 50 ohm lumped port (no extent) and a probe."""
    sim = Simulation(freq_max=2 * F0, domain=(16e-3, 14e-3, 12e-3), dx=DX,
                     boundary="cpml", cpml_layers=5)
    sim.add(Box((9e-3, 5e-3, 4e-3), (10e-3, 9e-3, 8e-3)), material="pec")
    sim.add_port(position=(5e-3, 7e-3, 6e-3), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((12e-3, 7e-3, 6e-3), "ez")
    return sim


def test_trace_time_setup_is_used_only_for_a_model_with_a_wire_port(monkeypatch):
    """A model without a wire port traces exactly as before #1225.

    ``forward()`` asks ``_forward_needs_trace_time_setup`` whether to evaluate
    its set-up at trace time. Under ``jax.jit`` the answer must be False for a
    model with a conductor and a lumped port but no wire port, and True for
    the wire-port board (positive control, so a spy that saw no call cannot
    pass).
    """
    import rfx.api._execute as ex

    answers = []
    real = ex._forward_needs_trace_time_setup

    def spy(sim, **kwargs):
        answers.append(real(sim, **kwargs))
        return answers[-1]

    monkeypatch.setattr(ex, "_forward_needs_trace_time_setup", spy)

    sim = _conductor_without_a_wire_port()
    eps = jnp.ones(tuple(sim._build_grid().shape), jnp.float32)

    def loss(e):
        r = sim.forward(eps_override=e, n_steps=10, checkpoint=False,
                        skip_preflight=True)
        return jnp.sum(r.time_series ** 2)

    jax.jit(jax.value_and_grad(loss)).lower(eps)
    assert answers == [False], (
        f"a model without a wire port: predicate answered {answers}")

    answers.clear()
    board = _board()
    jax.jit(jax.value_and_grad(_s11_loss(board))).lower(_box_eps(board))
    # The staged call answers True; the re-call inside the trace-time
    # context answers False and runs the plain body.
    assert answers == [True, False], (
        f"the wire-port board: predicate answered {answers}")
