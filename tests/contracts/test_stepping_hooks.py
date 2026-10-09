"""The production uniform step exposes the physical hook order when compiled."""
from dataclasses import dataclass, fields, MISSING, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.drives import StepDrives, drive_layout
from rfx.core.yee import init_state, init_materials
from rfx.grid import Grid
from rfx.simulation import _StepContext, make_core_step
from rfx.stepping import HookPoint, Kernel, compose


ORDER = (HookPoint.AFTER_H, HookPoint.AFTER_E_UPDATE,
         HookPoint.BEFORE_SOURCES, HookPoint.AFTER_SOURCES, HookPoint.STEP_END)


def _context():
    grid = Grid(freq_max=10e9, domain=(5e-3, 5e-3, 5e-3), dx=1e-3, cpml_layers=0)
    required = {f.name: False for f in fields(_StepContext)
                if f.default is MISSING and f.default_factory is MISSING}
    required.update(grid=grid, materials=init_materials(grid.shape),
                    dt=grid.dt, dx=grid.dx, periodic=(False, False, False),
                    pec_axes="", stencil_order=2)
    return _StepContext(**required, drives=StepDrives(
        electric=drive_layout([(2, 2, 2, "ez")], jnp.float32), magnetic=None))


def _check_hook_observations(code, samples):
    assert int(code) == 12345
    np.testing.assert_array_equal(samples, [0, 0, 0, 2, 2])


def test_hook_checks_are_live():
    with pytest.raises(AssertionError):
        _check_hook_observations(21345, [0, 0, 0, 2, 2])
    with pytest.raises(AssertionError):
        _check_hook_observations(12345, [0, 0, 2, 0, 2])


def test_uniform_compiled_hooks_read_the_physical_phases():
    ctx = _context()
    # All observations become JIT outputs. These are executable observations,
    # not a comparison of the composer metadata against itself.
    def attach(digit):
        def observe(frame):
            code = frame.carry.get("hook_code", jnp.int32(0))
            frame.carry = {**frame.carry, "hook_code": code * 10 + digit}
            frame.carry[f"sample_{digit}"] = frame.st.ez[2, 2, 2]
            if digit == 5:
                frame.extras["hook_code"] = frame.carry["hook_code"]
                frame.extras["hook_samples"] = jnp.stack([
                    frame.carry[f"sample_{i}"] for i in range(1, 6)])
        return observe

    core = make_core_step(ctx, hooks={p: (attach(i),) for i, p in enumerate(ORDER, 1)})
    _, probe, extras = jax.jit(core)(
        {"fdtd": init_state(ctx.grid.shape)}, jnp.int32(0),
        jnp.array([2.0], jnp.float32), jnp.zeros(0))
    _check_hook_observations(extras["hook_code"], extras["hook_samples"])
    assert probe.shape == (0,)


def test_empty_hooks_add_no_jaxpr_operation():
    @dataclass
    class Frame:
        value: jax.Array

    def update(frame):
        frame.value = frame.value * 2

    empty = compose({Kernel.H_UPDATE: update}, {point: () for point in ORDER})
    absent = compose({Kernel.H_UPDATE: update}, {})
    def direct(value):
        frame = Frame(value)
        update(frame)
        return frame.value

    reference = str(jax.make_jaxpr(direct)(jnp.float32(3)))
    for step in (empty, absent):
        assert str(jax.make_jaxpr(lambda value: step(Frame(value)).value)(jnp.float32(3))) == reference


@pytest.mark.parametrize("point", [HookPoint.AFTER_H, HookPoint.AFTER_E_UPDATE])
def test_fast_uniform_rejects_internal_attachments_at_construction(point):
    ctx = _context()
    ctx = replace(ctx, use_fast_he=True)
    with pytest.raises(ValueError, match=f"fast path requires empty {point.value}"):
        make_core_step(ctx, hooks={point: (lambda frame: None,)})


def test_fast_uniform_rejects_builtin_internal_attachments():
    ctx = replace(_context(), use_fast_he=True)
    with pytest.raises(ValueError, match="empty after_e_update"):
        make_core_step(ctx, design_hook=lambda previous, updated: (updated, None))
    magnetic = drive_layout([(2, 2, 2, "hz")], jnp.float32)
    ctx = replace(ctx, drives=ctx.drives._replace(magnetic=magnetic))
    with pytest.raises(ValueError, match="empty after_h"):
        make_core_step(ctx)


def test_fast_uniform_compiled_remaining_hooks(monkeypatch):
    from rfx.core.yee import precompute_coeffs
    import rfx.stepping.uniform as uniform
    import rfx.simulation as simulation

    ctx = _context()
    ctx = replace(ctx, use_fast_he=True)
    faces = frozenset(f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))
    ctx = replace(ctx, use_pec_faces=True, pec_faces_frozen=faces,
                  fast_coeffs=precompute_coeffs(ctx.materials, ctx.dt, ctx.dx,
                                               pec_faces=faces))
    calls = []
    combined = simulation.update_he_fast
    def record(*args, **kwargs):
        calls.append("combined")
        return combined(*args, **kwargs)
    def no_pec(*args, **kwargs):
        raise AssertionError("fast path must not add a PEC pass")
    monkeypatch.setattr(simulation, "update_he_fast", record)
    monkeypatch.setattr(uniform, "apply_pec_faces", no_pec)
    def attach(digit):
        def observe(frame):
            frame.carry = {**frame.carry,
                           "code": frame.carry.get("code", jnp.int32(0)) * 10 + digit,
                           f"sample_{digit}": frame.st.ez[2, 2, 2]}
            if digit == 5:
                frame.extras["code"] = frame.carry["code"]
                frame.extras["samples"] = jnp.stack([
                    frame.carry[f"sample_{i}"] for i in (3, 4, 5)])
        return observe
    core = make_core_step(ctx, hooks={p: (attach(i),)
                                     for i, p in enumerate(ORDER, 1) if i >= 3})
    _, _, extras = jax.jit(core)(
        {"fdtd": init_state(ctx.grid.shape)}, jnp.int32(0),
        jnp.array([2.0], jnp.float32), jnp.zeros(0))
    assert calls == ["combined"]
    assert int(extras["code"]) == 345
    np.testing.assert_array_equal(extras["samples"], [0, 2, 2])
