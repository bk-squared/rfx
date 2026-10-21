"""Pole admission keeps the occupied cells' metric under JAX transforms (#1934)."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import MaterialArrays
from rfx.model.pole_stability import C0, courant_budget, refuse_unstable_cells


@pytest.mark.parametrize("operand", ["cells", "mask", "both"])
@pytest.mark.parametrize("transform", ["eager", "jvp", "grad", "jit", "vmap"])
def test_traced_poles_keep_local_budget_and_refuse(operand, transform):
    shape = (8, 3, 3)
    widths = np.array([1., 1., 1., 1., .25, .25, .25, .25])
    occupied = np.zeros(shape, bool)
    occupied[:2] = True
    dt = .99 / (C0 * np.sqrt(18.))

    def checked(value, wp):
        sizes = jnp.asarray(widths) * value if operand != "mask" else widths
        mask = jnp.asarray(occupied) & (value > 0) if operand != "cells" else occupied
        grid = SimpleNamespace(cells=lambda a: sizes if a == 0 else np.ones(3))
        materials = MaterialArrays(jnp.ones(shape), jnp.zeros(shape), None)
        refuse_unstable_cells(materials, ([(0., 0., (wp / dt) ** 2)], [mask]),
                              dt, courant_budget(grid), grid_kind="NonUniformGrid")
        return value ** 2

    def call(wp):
        f = lambda x: checked(x, wp)
        if transform == "jvp":
            result = jax.jvp(f, (1.,), (1.,))
        elif transform == "grad":
            result = jax.grad(f)(1.)
        elif transform == "jit":
            result = jax.jit(f)(1.)
        elif transform == "vmap":
            result = jax.vmap(f)(jnp.ones(2))
        else:
            result = f(1.)
        jax.block_until_ready(result)

    call(1.5)  # global budget would wrongly refuse: local limit is 1.829...
    with pytest.raises(Exception, match="unstable at this model's time step"):
        call(2.05)


def test_abstract_disjoint_and_empty_pole_sets_are_not_summed():
    shape, dt = (4, 2, 2), 1.
    materials = MaterialArrays(jnp.ones(shape), jnp.zeros(shape), None)
    # Each pole is stable alone at x=0; their sum is unstable. Dynamic masks
    # choose disjoint sets and leave the possible overlap empty.
    @jax.jit
    def checked(left, right):
        refuse_unstable_cells(materials, ([(0., 0., 3.), (0., 0., 3.)], [left, right]),
                              dt, 0., grid_kind="Grid")
        return jnp.sum(left)
    left = jnp.indices(shape)[0] < 2
    jax.block_until_ready(checked(left, ~left))
    with pytest.raises(Exception, match="unstable at this model's time step"):
        jax.block_until_ready(checked(left, left))


@pytest.mark.parametrize("transform", ["eager", "jvp", "grad", "jit", "vmap"])
def test_out_of_program_admission_keeps_concrete_primals(transform):
    def checked(eps):
        materials = MaterialArrays(jnp.full((2, 2, 2), eps), jnp.zeros((2, 2, 2)), None)
        # At dt=1, budget=0, the undamped Drude condition is kappa<=4*eps:
        # literal kappa=9 is stable at eps=3 and unstable at eps=1.
        refuse_unstable_cells(materials, ([(0., 0., 9.)], None), 1., 0.,
                              grid_kind="Grid", in_program=False)
        return eps ** 2

    def call(eps):
        if transform == "jvp":
            return jax.jvp(checked, (eps,), (1.,))[0]
        if transform == "grad":
            return jax.grad(checked)(eps)
        if transform == "jit":
            return jax.jit(checked)(eps)
        if transform == "vmap":
            return jax.vmap(checked)(jnp.full((2,), eps))
        return checked(eps)

    np.testing.assert_array_equal(call(3.), 6. if transform == "grad" else
                                  np.full((2,), 9.) if transform == "vmap" else 9.)
    if transform in ("eager", "jvp", "grad"):
        with pytest.raises(ValueError, match="unstable at this model's time step"):
            call(1.)
    else:
        np.testing.assert_array_equal(call(1.), np.full((2,), 1.) if transform == "vmap" else 1.)


@pytest.mark.parametrize("transform", ["jit", "vmap", "jvp", "grad"])
@pytest.mark.parametrize("n_poles", [8, 9])
def test_traced_mask_pattern_cap(n_poles, transform):
    from rfx.materials.debye import DebyePole
    from rfx.model.pole_stability import judge_pole_sets, pole_set_extremes
    shape = (1, 1, 1)
    materials = MaterialArrays(jnp.ones(shape), jnp.zeros(shape), None)

    def checked(value):
        mask = jnp.full(shape, value > 0)
        # Count both families; concrete eager primals must remain uncapped.
        records = pole_set_extremes(materials, ([(0., 0., 1.)] * (n_poles - 1), mask),
                                   debye_spec=([DebyePole(1., 1.)], mask))
        judge_pole_sets(records, .1, 0., grid_kind="Grid")
        return value ** 2 + sum(r.count for r in records)

    def call():
        if transform == "jit":
            result = jax.jit(checked)(1.)
        elif transform == "vmap":
            result = jax.vmap(checked)(jnp.ones(1))
        elif transform == "jvp":
            result = jax.jvp(checked, (1.,), (1.,))
        else:
            result = jax.grad(checked)(1.)
        jax.block_until_ready(result)

    if n_poles == 9 and transform in ("jit", "vmap"):
        with pytest.raises(ValueError, match=r"cannot judge: 9 poles give 2\^9 patterns"):
            call()
    else:
        call()
