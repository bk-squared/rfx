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

The jitted value and gradient are compared with the plain call under the PI's
cross-trace rule (2026-09-23): bit-identical is the aim, and a difference up
to single-digit float32 ULP is allowed. The unit is ULP AT THE PEAK of each
compared array — ``max|plain - jitted|`` over the float32 spacing at
``max|plain|`` — expressed once, in ``_ulp_at_peak``; per-element ULP near
zero says nothing about the result.

Compiles are counted with ``jax.monitoring`` backend-compile events, the same
counter the issue's measurements used. Each count is guarded by a positive
control (a first call that must compile), so a counter that saw nothing
cannot pass a "0 compiles" assertion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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

# The PI's cross-trace tolerance (2026-09-23): single-digit ULP at the peak.
MAX_ULP_AT_PEAK = 9


def _ulp_at_peak(plain, other):
    """``max|plain - other|`` in float32 spacings at ``max|plain|``."""
    plain = np.asarray(plain, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    peak = np.float32(np.max(np.abs(plain)))
    return float(np.max(np.abs(plain - other)) / np.spacing(peak))


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


def _graded_lumped_board():
    """The graded substrate on its ground plane, fed by a 50 ohm LUMPED port.

    One cell in the middle of the substrate, no extent; a probe at the same
    height 5 mm away records the field the objective reads.
    """
    sim = Simulation(freq_max=2 * F0, domain=(18e-3, 16e-3, float(DZ.sum())),
                     dx=DX, boundary="cpml", cpml_layers=5, dz_profile=DZ)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((2e-3, 2e-3, Z_GND), (16e-3, 14e-3, Z_GND + SUB_H)),
            material="sub")
    sim.add(Box((2e-3, 2e-3, Z_GND - 0.5e-3), (16e-3, 14e-3, Z_GND)),
            material="pec")
    sim.add_port(position=(7e-3, 8e-3, Z_GND + 0.75e-3), component="ez",
                 impedance=50.0, waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((12e-3, 8e-3, Z_GND + 0.75e-3), "ez")
    return sim


def _uniform_board():
    """The wire-port board on the uniform lane: PEC ground and patch, 3 mm substrate."""
    sim = Simulation(freq_max=2 * F0, domain=(18e-3, 16e-3, 12e-3), dx=DX,
                     boundary="cpml", cpml_layers=5)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((2e-3, 2e-3, 4e-3), (16e-3, 14e-3, 7e-3)), material="sub")
    sim.add(Box((2e-3, 2e-3, 3e-3), (16e-3, 14e-3, 4e-3)), material="pec")
    sim.add(Box((6e-3, 5e-3, 7e-3), (12e-3, 11e-3, 8e-3)), material="pec")
    sim.add_port(position=(7e-3, 8e-3, 4e-3), component="ez", extent=3e-3,
                 impedance=50.0, direction="-x",
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    return sim


UNIFORM_BOX = ((10e-3, 7e-3, 5e-3), (11e-3, 9e-3, 6e-3))


def _uniform_s11_loss(sim, checkpoint=False):
    freqs = jnp.asarray([6e9, 8e9, 10e9], jnp.float32)

    def loss(eps_box):
        r = sim.forward(design_box=UNIFORM_BOX, design_eps_override=eps_box,
                        n_steps=N_STEPS, checkpoint=checkpoint,
                        skip_preflight=True, port_s11_freqs=freqs)
        return jnp.sum(jnp.abs(r.s_params) ** 2)
    return loss


def _uniform_box_eps(sim, seed=2):
    grid = sim._build_grid()
    lo = grid.position_to_index(UNIFORM_BOX[0])
    hi = grid.position_to_index(UNIFORM_BOX[1])
    shape = tuple(int(hi[d]) - int(lo[d]) + 1 for d in range(3))
    return jnp.asarray(2.0 + np.random.default_rng(seed).random(shape),
                       jnp.float32)


def _box_eps(sim, seed=4):
    from rfx.nonuniform import position_to_index
    grid = sim._build_nonuniform_grid()
    lo, hi = position_to_index(grid, BOX[0]), position_to_index(grid, BOX[1])
    shape = tuple(int(hi[d]) - int(lo[d]) + 1 for d in range(3))
    return jnp.asarray(2.0 + np.random.default_rng(seed).random(shape),
                       jnp.float32)


def _s11_loss(sim, checkpoint=False):
    def loss(eps_box):
        r = sim.forward(design_box=BOX, design_eps_override=eps_box,
                        n_steps=N_STEPS, checkpoint=checkpoint,
                        skip_preflight=True)
        return jnp.sum(jnp.abs(r.s_params[0, 0, 3:8]) ** 2)
    return loss


def _probe_energy_loss(sim, checkpoint=False):
    def loss(eps_box):
        r = sim.forward(design_box=BOX, design_eps_override=eps_box,
                        n_steps=N_STEPS, checkpoint=checkpoint,
                        skip_preflight=True)
        return jnp.sum(r.time_series ** 2)
    return loss


# The models that could not be traced under jax.jit before #1225:
# name -> (builder, objective(sim, checkpoint), design-box permittivity(sim)).
BOARDS = {
    "graded wire port": (_board, _s11_loss, _box_eps),
    "uniform wire port": (_uniform_board, _uniform_s11_loss, _uniform_box_eps),
    "graded lumped port": (_graded_lumped_board, _probe_energy_loss, _box_eps),
}


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


@pytest.mark.parametrize("board", sorted(BOARDS))
def test_a_board_with_a_loaded_port_lowers_under_jit(board):
    """``jax.jit(jax.value_and_grad(objective))`` traces and lowers.

    Before #1225 the wire-port boards raised TracerArrayConversionError from
    the port's live-edge test on the realized PEC edge mask, and the graded
    lumped-port board ConcretizationTypeError from the port's cell sizes.
    """
    build, objective, design = BOARDS[board]
    sim = build()
    lowered = jax.jit(jax.value_and_grad(objective(sim))).lower(design(sim))
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


def _conductor_and_uniform_lumped_port():
    """A PEC block, a 50 ohm lumped port (no extent) and a probe, uniform mesh."""
    sim = Simulation(freq_max=2 * F0, domain=(16e-3, 14e-3, 12e-3), dx=DX,
                     boundary="cpml", cpml_layers=5)
    sim.add(Box((9e-3, 5e-3, 4e-3), (10e-3, 9e-3, 8e-3)), material="pec")
    sim.add_port(position=(5e-3, 7e-3, 6e-3), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((12e-3, 7e-3, 6e-3), "ez")
    return sim


def _graded_source_without_a_port():
    """The graded substrate and ground driven by a plain source, no port."""
    sim = Simulation(freq_max=2 * F0, domain=(18e-3, 16e-3, float(DZ.sum())),
                     dx=DX, boundary="cpml", cpml_layers=5, dz_profile=DZ)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((2e-3, 2e-3, Z_GND), (16e-3, 14e-3, Z_GND + SUB_H)),
            material="sub")
    sim.add(Box((2e-3, 2e-3, Z_GND - 0.5e-3), (16e-3, 14e-3, Z_GND)),
            material="pec")
    sim.add_source((7e-3, 8e-3, Z_GND + 0.75e-3), "ez", amplitude_kind="current",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((12e-3, 8e-3, Z_GND + 0.75e-3), "ez")
    return sim


def test_trace_time_setup_is_used_only_where_the_trace_failed(monkeypatch):
    """Models that traced before #1225 trace exactly as before.

    ``forward()`` asks ``_forward_needs_trace_time_setup`` whether to evaluate
    its set-up at trace time. Under ``jax.jit`` it must answer False for a
    uniform-mesh model with a conductor and a lumped port and for a graded
    model with no port, and True for the graded wire-port and lumped-port
    boards — the staged call; the re-call inside the trace-time context then
    answers False and runs the plain body. The positive controls keep a spy
    that saw no call from passing.
    """
    import rfx.api._execute as ex

    answers = []
    real = ex._forward_needs_trace_time_setup

    def spy(sim, **kwargs):
        answers.append(real(sim, **kwargs))
        return answers[-1]

    monkeypatch.setattr(ex, "_forward_needs_trace_time_setup", spy)

    for build in (_conductor_and_uniform_lumped_port,
                  _graded_source_without_a_port):
        sim = build()
        grid = sim._build_realized_grid()
        eps = jnp.ones(tuple(grid.shape), jnp.float32)

        def loss(e, sim=sim):
            r = sim.forward(eps_override=e, n_steps=10, checkpoint=False,
                            skip_preflight=True)
            return jnp.sum(r.time_series ** 2)

        answers.clear()
        jax.jit(jax.value_and_grad(loss)).lower(eps)
        assert answers == [False], (
            f"{build.__name__}: predicate answered {answers}")

    for board in ("graded wire port", "graded lumped port"):
        build, objective, design = BOARDS[board]
        sim = build()
        answers.clear()
        jax.jit(jax.value_and_grad(objective(sim))).lower(design(sim))
        assert answers == [True, False], (
            f"{board}: predicate answered {answers}")


def test_a_trace_time_context_that_cannot_make_constants_does_not_recurse(
        monkeypatch):
    """If the trace-time context leaves the set-up staged, fall through.

    Inside an eager ``jax.shard_map`` the staging probe still reads True
    within ``jax.ensure_compile_time_eval()``. ``forward()`` must then run its
    ordinary body — returning, or raising the error it raised before #1225 —
    never re-call itself until RecursionError. The probe is forced True here,
    which reproduces that state on a plain call.
    """
    import rfx.api._execute as ex

    monkeypatch.setattr(ex, "_staged_by_an_outer_trace", lambda: True)
    sim = _board()
    r = sim.forward(design_box=BOX, design_eps_override=_box_eps(sim),
                    n_steps=5, checkpoint=False, skip_preflight=True)
    assert np.all(np.isfinite(np.asarray(r.s_params)))


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("board", sorted(BOARDS))
def test_jitted_step_equals_the_plain_call_within_single_digit_ulp(board,
                                                                   checkpoint):
    """Objective and design-box gradient: jitted vs plain, <= 9 ULP at peak."""
    build, objective, design = BOARDS[board]
    sim = build()
    loss, eps = objective(sim, checkpoint=checkpoint), design(sim)

    v, g = jax.value_and_grad(loss)(eps)
    vj, gj = jax.jit(jax.value_and_grad(loss))(eps)
    assert float(v) > 0.0 and float(jnp.max(jnp.abs(g))) > 0.0
    du_v, du_g = _ulp_at_peak(v, vj), _ulp_at_peak(g, gj)
    print(f"{board}, checkpoint={checkpoint}, jitted vs plain: value "
          f"{du_v:.1f}, gradient {du_g:.1f} ULP at peak")
    assert du_v <= MAX_ULP_AT_PEAK and du_g <= MAX_ULP_AT_PEAK, (
        f"{board} (checkpoint={checkpoint}): jitted value {du_v:.1f} and "
        f"gradient {du_g:.1f} ULP at peak from the plain call "
        f"(allowed {MAX_ULP_AT_PEAK})")
