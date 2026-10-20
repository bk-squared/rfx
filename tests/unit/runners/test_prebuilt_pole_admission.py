"""Low-level runners judge the ADE coefficients they actually receive (#1934)."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rfx.core.yee import init_materials
from rfx.materials.lorentz import drude_pole, init_lorentz
from rfx.simulation import run
from tests.unit.materials.test_pole_stability_admission import _box


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("decay", [False, True])
def test_prebuilt_lorentz_is_judged(graded, decay):
    sim = _box(.25, graded=graded)
    grid = sim._build_realized_grid()
    materials = init_materials(grid.shape)
    if graded:
        from rfx.nonuniform import run_nonuniform, run_nonuniform_until_decay
        runner = run_nonuniform_until_decay if decay else run_nonuniform
    elif decay:
        from rfx.stepping.uniform_loop import run_until_decay
        runner = run_until_decay
    else:
        runner = run
    kwargs = dict(max_steps=2, min_steps=1, check_interval=1) if decay else dict(n_steps=2)
    stable = init_lorentz([drude_pole(.25 / grid.dt, 0.)], materials, grid.dt)
    runner(grid, materials, lorentz=stable, **kwargs)
    unstable = init_lorentz([drude_pole(.45 / grid.dt, 0.)], materials, grid.dt)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        runner(grid, materials, lorentz=unstable, **kwargs)


@pytest.mark.parametrize("transform", ["jit", "vmap", "jvp", "grad"])
def test_prebuilt_runner_transforms_refuse(transform):
    grid = _box(.25)._build_realized_grid()
    materials = init_materials(grid.shape)
    coeffs, state = init_lorentz([drude_pole(.45 / grid.dt, 0.)], materials, grid.dt)
    def loss(scale):
        scaled = coeffs._replace(c=tuple(v * scale for v in coeffs.c))
        result = run(grid, materials, n_steps=1, lorentz=(scaled, state))
        return jnp.sum(result.state.ez ** 2)
    with pytest.raises(Exception, match="unstable at this model's time step"):
        if transform == "jit":
            result = jax.jit(loss)(1.)
        elif transform == "vmap":
            result = jax.vmap(loss)(jnp.ones(2))
        elif transform == "jvp":
            result = jax.jvp(loss, (1.,), (1.,))
        else:
            result = jax.grad(loss)(1.)
        jax.block_until_ready(result)
