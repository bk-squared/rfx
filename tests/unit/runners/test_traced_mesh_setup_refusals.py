"""Traced geometry/mesh admission precedes local two-device time loops."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.runners.distributed_nu import build_sharded_nu_grid
from tests.unit.autodiff.test_graded_wire_live_cell_admission import _board
from rfx._traced_refusal import TracedStructureRefusal


def _distributed(value, kind):
    if kind == "wire":
        sim = _board(value)
    else:
        dz = (jnp.full(8, .001).at[3].add(value).at[4].add(-value)
              if kind == "flux" else np.full(8, .001))
        sim = Simulation(freq_max=10e9, domain=(.008, .008, .008),
                         dx=.001, boundary="pec", dz_profile=dz)
        if kind == "flux":
            sim.add_flux_monitor(axis="z", coordinate=.002, freqs=[1e9])
        else:
            shape = (Box((.002, .002, .002), (.005 + value, .005, .005))
                     if kind == "Box" else Sphere((.004, .004, .004), .002 + value))
            sim.add(shape, material="pec")
    return jnp.sum(sim.forward(
        distributed=True, devices=jax.devices("cpu")[:2], n_steps=2,
        skip_preflight=True).time_series)


@pytest.mark.parametrize("kind", ["wire", "flux", "Box", "Sphere"])
@pytest.mark.parametrize("route", ["jit", "vmap", "jvp", "grad"])
def test_traced_distributed_setup_refuses_before_loop(kind, route):
    assert len(jax.devices("cpu")) >= 2
    fn = lambda value: _distributed(value, kind)
    refusal = TracedStructureRefusal
    if kind in ("wire", "flux"):
        message = "graded multi-device solve.*traced mesh metrics.*concrete mesh"
    elif route in ("jvp", "grad"):
        # The shape-gradient refusal on main precedes structure admission.
        refusal, message = NotImplementedError, "gradient.*declared shape dimension"
    else:
        message = f"{kind}.*traced.*concrete"
    with pytest.raises(refusal, match=message):
        if route == "jit":
            jax.jit(fn)(jnp.float32(0.))
        elif route == "vmap":
            jax.vmap(fn)(jnp.zeros(2))
        elif route == "jvp":
            jax.jvp(fn, (jnp.float32(0.),), (jnp.float32(1.),))
        else:
            jax.grad(fn)(jnp.float32(0.))


def test_concrete_distributed_mesh_still_builds():
    grid = _board(0.)._build_nonuniform_grid()
    sharded = build_sharded_nu_grid(grid, 2)
    np.testing.assert_array_equal(sharded.inv_dz, grid.inv_dz)
    assert sharded.dt == float(grid.dt)
